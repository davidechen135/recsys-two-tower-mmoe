"""End-to-end training and evaluation pipeline for RecSys Two-Tower Recall & MMoE Ranking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from ..data.generator import generate_synthetic_user_behavior
from ..features.extractor import extract_features, RecSysDataBundle
from ..metrics import calculate_auc, calculate_gauc, calculate_ranking_metrics
from ..models.deepfm import SingleTaskDeepFM
from ..models.dssm import TwoTowerDSSM
from ..models.mmoe_din import MMoEDINRanking
from ..recall.faiss_index import (
    FaissDSSMRecall,
    GlobalPopularityRecall,
    ItemCFRecall,
    evaluate_recall_metrics,
)


def train_two_tower_dssm(
    bundle: RecSysDataBundle,
    embed_dim: int = 64,
    out_dim: int = 64,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 0.002,
    device: str = "cpu",
) -> Tuple[TwoTowerDSSM, dict]:
    print("  [Two-Tower] Training DSSM on positive interactions...")
    train_df = bundle.train_recall_df
    n_users = bundle.n_users
    n_items = bundle.n_items
    n_cats = bundle.n_categories

    # 1-indexed item lookup tables (0 is PAD)
    cat_lut = torch.zeros(n_items + 1, dtype=torch.long)
    dense_lut = torch.zeros(n_items + 1, 4, dtype=torch.float32)
    log_q_lut = torch.zeros(n_items + 1, dtype=torch.float32)

    item_meta = bundle.item_meta_df
    item_ids_arr = item_meta["item_id"].values.astype(int)
    cat_lut[item_ids_arr] = torch.tensor(item_meta["category_id"].values, dtype=torch.long)

    dense_cols = ["pv_to_cart_rate", "pv_to_buy_rate", "log_pv", "log_buy"]
    dense_lut[item_ids_arr] = torch.tensor(item_meta[dense_cols].values, dtype=torch.float32)

    # In-batch sampled softmax popularity logQ correction
    item_counts = item_meta["total_interactions"].values + 1.0
    item_probs = item_counts / item_counts.sum()
    log_q_lut[item_ids_arr] = torch.tensor(np.log(item_probs), dtype=torch.float32)

    u_ids = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    i_ids = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    h_items = torch.tensor(np.vstack(train_df["history_items"].values), dtype=torch.long)
    h_lens = torch.tensor(train_df["history_len"].values, dtype=torch.long)

    dataset = TensorDataset(u_ids, i_ids, h_items, h_lens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TwoTowerDSSM(n_users, n_items, n_cats, embed_dim=embed_dim, out_dim=out_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cat_lut = cat_lut.to(device)
    dense_lut = dense_lut.to(device)
    log_q_lut = log_q_lut.to(device)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch in loader:
            b_u, b_i, b_hist, b_lens = [t.to(device) for t in batch]
            b_cats = cat_lut[b_i]
            b_dense = dense_lut[b_i]
            b_log_q = log_q_lut[b_i]

            optimizer.zero_grad()
            loss, _, _ = model(b_u, b_hist, b_lens, b_i, b_cats, b_dense, item_log_q=b_log_q)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Ground truth for test users
    test_df = bundle.test_recall_df
    gt_dict = {}
    for uid, grp in test_df.groupby("user_id"):
        gt_dict[uid] = set(grp["item_id"].unique())

    test_u_df = test_df.drop_duplicates("user_id")
    test_uids = test_u_df["user_id"].values

    # 1. Global Popularity baseline
    pop_model = GlobalPopularityRecall(item_meta)
    pop_retrievals, pop_latency_ms = pop_model.retrieve_batch(len(test_uids), top_k=100)
    pop_retrievals = {uid: pop_retrievals[idx] for idx, uid in enumerate(test_uids)}
    pop_metrics = evaluate_recall_metrics(pop_retrievals, gt_dict)
    pop_metrics["query_latency_ms"] = float(pop_latency_ms)

    # 2. ItemCF baseline
    item_cf = ItemCFRecall(train_df, n_items)
    user_test_hist = dict(zip(test_u_df["user_id"], test_u_df["history_items"]))
    cf_retrievals, cf_latency_ms = item_cf.retrieve_batch(user_test_hist, top_k=100)
    cf_metrics = evaluate_recall_metrics(cf_retrievals, gt_dict)
    cf_metrics["query_latency_ms"] = float(cf_latency_ms)

    # 3. Two-Tower DSSM with FAISS FlatIP & HNSW
    faiss_dssm = FaissDSSMRecall(model, item_meta, dim=out_dim, index_type="FlatIP")
    model.eval()
    t_u = torch.tensor(test_uids, dtype=torch.long, device=device)
    t_h = torch.tensor(np.vstack(test_u_df["history_items"].values), dtype=torch.long, device=device)
    t_l = torch.tensor(test_u_df["history_len"].values, dtype=torch.long, device=device)
    with torch.no_grad():
        test_u_vecs = model.user_tower(t_u, t_h, t_l).cpu().numpy()

    dssm_indices, dssm_latency_ms = faiss_dssm.retrieve_batch(test_u_vecs, top_k=100)
    dssm_retrievals = {uid: dssm_indices[idx] for idx, uid in enumerate(test_uids)}
    dssm_metrics = evaluate_recall_metrics(dssm_retrievals, gt_dict)
    dssm_metrics["query_latency_ms"] = float(dssm_latency_ms)

    # HNSW Index verification
    faiss_hnsw = FaissDSSMRecall(model, item_meta, dim=out_dim, index_type="HNSW")
    hnsw_indices, hnsw_latency_ms = faiss_hnsw.retrieve_batch(test_u_vecs, top_k=100)
    hnsw_retrievals = {uid: hnsw_indices[idx] for idx, uid in enumerate(test_uids)}
    hnsw_metrics = evaluate_recall_metrics(hnsw_retrievals, gt_dict)
    hnsw_metrics["query_latency_ms"] = float(hnsw_latency_ms)
    dssm_metrics["hnsw_recall@100"] = hnsw_metrics["Recall@100"]
    dssm_metrics["hnsw_latency_ms"] = hnsw_metrics["query_latency_ms"]

    lift_vs_pop = (dssm_metrics["Recall@100"] - pop_metrics["Recall@100"]) / max(1e-6, pop_metrics["Recall@100"])
    lift_vs_cf = (dssm_metrics["Recall@100"] - cf_metrics["Recall@100"]) / max(1e-6, cf_metrics["Recall@100"])

    return model, {
        "Popularity": pop_metrics,
        "ItemCF": cf_metrics,
        "TwoTower_DSSM": dssm_metrics,
        "relative_lift_vs_pop_pct": float(lift_vs_pop * 100),
        "relative_lift_vs_cf_pct": float(lift_vs_cf * 100),
    }


def train_ranking_models(
    bundle: RecSysDataBundle,
    embed_dim: int = 32,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 0.003,
    device: str = "cpu",
) -> Tuple[MMoEDINRanking, dict]:
    print("  [Ranking] Training Baselines and MMoE + DIN Multi-Task Model...")
    train_df = bundle.train_rank_df
    test_df = bundle.test_rank_df
    n_users = bundle.n_users
    n_items = bundle.n_items
    n_cats = bundle.n_categories

    dense_cols = ["pv_to_cart_rate", "pv_to_buy_rate", "log_pv", "log_buy"]

    # Test set evaluation masks
    click_mask_te = test_df["ctr_label"].values == 1
    y_true_ctr_te = test_df["ctr_label"].values
    y_true_cvr_clicked = test_df.loc[click_mask_te, "cvr_label"].values
    y_true_ctcvr_te = (test_df["ctr_label"] * test_df["cvr_label"]).values

    # 1. Baseline: LightGBM
    lgb_metrics = {}
    if _HAS_LGBM:
        feat_cols = ["user_id", "item_id", "category_id"] + dense_cols
        X_tr = train_df[feat_cols]
        X_te = test_df[feat_cols]

        clf_ctr = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, verbose=-1, n_jobs=2, random_state=42)
        clf_ctr.fit(X_tr, train_df["ctr_label"])
        pred_ctr_lgb = clf_ctr.predict_proba(X_te)[:, 1]

        clf_cvr = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, verbose=-1, n_jobs=2, random_state=42)
        click_mask_tr = train_df["ctr_label"] == 1
        clf_cvr.fit(X_tr[click_mask_tr], train_df.loc[click_mask_tr, "cvr_label"])
        pred_cvr_lgb_all = clf_cvr.predict_proba(X_te)[:, 1]
        pred_cvr_lgb_clicked = pred_cvr_lgb_all[click_mask_te]

        lgb_metrics = calculate_ranking_metrics(
            y_true_ctr_te, pred_ctr_lgb,
            y_true_cvr_clicked, pred_cvr_lgb_clicked,
            user_ids_ctr=test_df["user_id"].values,
            user_ids_cvr=test_df.loc[click_mask_te, "user_id"].values,
        )
        lgb_metrics["ctcvr_auc"] = calculate_auc(y_true_ctcvr_te, pred_ctr_lgb * pred_cvr_lgb_all)

    # 2. Baseline: Single-Task DeepFM
    def train_deepfm(target_col: str, click_only: bool = False):
        sub_df = train_df[train_df["ctr_label"] == 1] if click_only else train_df
        u_t = torch.tensor(sub_df["user_id"].values, dtype=torch.long)
        i_t = torch.tensor(sub_df["item_id"].values, dtype=torch.long)
        c_t = torch.tensor(sub_df["category_id"].values, dtype=torch.long)
        d_t = torch.tensor(sub_df[dense_cols].values, dtype=torch.float32)
        y_t = torch.tensor(sub_df[target_col].values, dtype=torch.float32)

        loader = DataLoader(TensorDataset(u_t, i_t, c_t, d_t, y_t), batch_size=batch_size, shuffle=True)
        model = SingleTaskDeepFM(n_users, n_items, n_cats, dense_dim=4, embed_dim=embed_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for b_u, b_i, b_c, b_d, b_y in loader:
                b_u, b_i, b_c, b_d, b_y = b_u.to(device), b_i.to(device), b_c.to(device), b_d.to(device), b_y.to(device)
                opt.zero_grad()
                p = model(b_u, b_i, b_c, b_d)
                loss = torch.nn.functional.binary_cross_entropy(p, b_y)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            te_u = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
            te_i = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
            te_c = torch.tensor(test_df["category_id"].values, dtype=torch.long, device=device)
            te_d = torch.tensor(test_df[dense_cols].values, dtype=torch.float32, device=device)
            preds = model(te_u, te_i, te_c, te_d).cpu().numpy()
        return preds

    pred_ctr_deepfm = train_deepfm("ctr_label", click_only=False)
    pred_cvr_deepfm_all = train_deepfm("cvr_label", click_only=True)
    pred_cvr_deepfm_clicked = pred_cvr_deepfm_all[click_mask_te]

    deepfm_metrics = calculate_ranking_metrics(
        y_true_ctr_te, pred_ctr_deepfm,
        y_true_cvr_clicked, pred_cvr_deepfm_clicked,
        user_ids_ctr=test_df["user_id"].values,
        user_ids_cvr=test_df.loc[click_mask_te, "user_id"].values,
    )
    deepfm_metrics["ctcvr_auc"] = calculate_auc(y_true_ctcvr_te, pred_ctr_deepfm * pred_cvr_deepfm_all)

    # 3. MMoE + DIN Multi-Task Model
    u_tr = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    i_tr = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    c_tr = torch.tensor(train_df["category_id"].values, dtype=torch.long)
    h_tr = torch.tensor(np.vstack(train_df["history_items"].values), dtype=torch.long)
    d_tr = torch.tensor(train_df[dense_cols].values, dtype=torch.float32)
    y_ctr_tr = torch.tensor(train_df["ctr_label"].values, dtype=torch.float32)
    y_cvr_tr = torch.tensor(train_df["cvr_label"].values, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(u_tr, i_tr, c_tr, h_tr, d_tr, y_ctr_tr, y_cvr_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    mmoe_model = MMoEDINRanking(
        n_users, n_items, n_cats, dense_dim=4, embed_dim=embed_dim, n_experts=4
    ).to(device)
    mmoe_opt = torch.optim.Adam(mmoe_model.parameters(), lr=lr)

    mmoe_model.train()
    for ep in range(epochs):
        for b_u, b_i, b_c, b_h, b_d, b_ctr, b_cvr in train_loader:
            b_u, b_i, b_c, b_h, b_d = b_u.to(device), b_i.to(device), b_c.to(device), b_h.to(device), b_d.to(device)
            b_ctr, b_cvr = b_ctr.to(device), b_cvr.to(device)

            mmoe_opt.zero_grad()
            p_ctr, p_cvr, _, _ = mmoe_model(b_u, b_i, b_c, b_h, b_d)

            # Uncertainty-weighted multi-task loss
            loss, _, _ = mmoe_model.compute_multi_task_loss(p_ctr, p_cvr, b_ctr, b_cvr)
            loss.backward()
            mmoe_opt.step()

    # Inference on test set
    mmoe_model.eval()
    with torch.no_grad():
        u_te = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
        i_te = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
        c_te = torch.tensor(test_df["category_id"].values, dtype=torch.long, device=device)
        h_te = torch.tensor(np.vstack(test_df["history_items"].values), dtype=torch.long, device=device)
        d_te = torch.tensor(test_df[dense_cols].values, dtype=torch.float32, device=device)

        pred_ctr_mmoe, pred_cvr_mmoe, w_ctr_gate, w_cvr_gate = mmoe_model(u_te, i_te, c_te, h_te, d_te)
        p_ctr_arr = pred_ctr_mmoe.cpu().numpy()
        p_cvr_arr_all = pred_cvr_mmoe.cpu().numpy()
        p_cvr_arr_clicked = p_cvr_arr_all[click_mask_te]

        w_ctr_mean = w_ctr_gate.mean(dim=0).cpu().numpy().tolist()
        w_cvr_mean = w_cvr_gate.mean(dim=0).cpu().numpy().tolist()

    mmoe_metrics = calculate_ranking_metrics(
        y_true_ctr_te, p_ctr_arr,
        y_true_cvr_clicked, p_cvr_arr_clicked,
        user_ids_ctr=test_df["user_id"].values,
        user_ids_cvr=test_df.loc[click_mask_te, "user_id"].values,
    )
    mmoe_metrics["ctcvr_auc"] = calculate_auc(y_true_ctcvr_te, p_ctr_arr * p_cvr_arr_all)
    mmoe_metrics["gate_weights_ctr_mean"] = [round(float(w), 4) for w in w_ctr_mean]
    mmoe_metrics["gate_weights_cvr_mean"] = [round(float(w), 4) for w in w_cvr_mean]
    mmoe_metrics["learned_task_weight_ctr"] = float(torch.exp(-mmoe_model.log_var_ctr).item())
    mmoe_metrics["learned_task_weight_cvr"] = float(torch.exp(-mmoe_model.log_var_cvr).item())

    all_ranking = {
        "LightGBM": lgb_metrics,
        "DeepFM": deepfm_metrics,
        "MMoE_DIN": mmoe_metrics,
    }
    return mmoe_model, all_ranking

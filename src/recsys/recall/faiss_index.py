"""FAISS Vector Indexing & Candidate Recall Baselines (ItemCF, Global Popularity)."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple
import faiss
import numpy as np
import scipy.sparse as sp
import torch

from ..models.dssm import TwoTowerDSSM


class GlobalPopularityRecall:
    """Baseline 1: Global Popularity."""

    def __init__(self, item_counts: np.ndarray):
        self.ranked_items = np.argsort(-item_counts)

    def retrieve(self, top_k: int = 100) -> np.ndarray:
        return self.ranked_items[:top_k]


class ItemCFRecall:
    """Baseline 2: Item-based Collaborative Filtering (Cosine Co-occurrence)."""

    def __init__(self, train_events_df, n_items: int):
        self.n_items = n_items
        # Build user-item sparse interaction matrix
        users = train_events_df["user_id"].values
        items = train_events_df["item_id"].values
        data = np.ones(len(users), dtype=np.float32)
        n_users = int(users.max() + 1)
        # Sparse matrix (n_items, n_users)
        item_user = sp.csr_matrix((data, (items, users)), shape=(n_items, n_users))

        # Item-item co-occurrence: A * A^T
        co_occ = item_user @ item_user.T
        # Diagonal degrees
        degrees = np.array(co_occ.diagonal()).flatten()
        inv_sqrt_deg = 1.0 / np.maximum(np.sqrt(degrees), 1e-9)
        # Cosine normalization: D^{-1/2} * co_occ * D^{-1/2}
        D = sp.diags(inv_sqrt_deg)
        self.sim_matrix = D @ co_occ @ D
        self.sim_matrix.setdiag(0.0)

    def retrieve(self, user_history: List[int], top_k: int = 100) -> np.ndarray:
        if len(user_history) == 0:
            return np.arange(top_k)
        active_items = [i for i in user_history if i < self.n_items]
        if len(active_items) == 0:
            return np.arange(top_k)

        # Aggregate similarity scores
        sub_sim = self.sim_matrix[active_items, :].sum(axis=0)
        scores = np.array(sub_sim).flatten()
        # Zero out history
        scores[active_items] = -1.0
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        sorted_top = top_indices[np.argsort(-scores[top_indices])]
        return sorted_top


class FaissDSSMRecall:
    """DSSM Item Vector Index with FAISS (FlatIP & HNSW)."""

    def __init__(
        self,
        model: TwoTowerDSSM,
        item_meta_df,
        dim: int = 64,
        index_type: str = "FlatIP",
    ):
        self.model = model
        self.dim = dim
        self.index_type = index_type

        # Compute all item embeddings
        self.model.eval()
        with torch.no_grad():
            item_ids = torch.tensor(item_meta_df["item_id"].values, dtype=torch.long)
            cat_ids = torch.tensor(item_meta_df["category_id"].values, dtype=torch.long)
            dense_cols = ["pv_to_cart_rate", "pv_to_buy_rate", "log_pv", "log_buy"]
            dense = torch.tensor(item_meta_df[dense_cols].values, dtype=torch.float32)
            item_vecs = self.model.item_tower(item_ids, cat_ids, dense).cpu().numpy().astype(np.float32)

        # Build FAISS index
        if index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(item_vecs)

    def retrieve_batch(self, user_vecs: np.ndarray, top_k: int = 100) -> tuple[np.ndarray, float]:
        user_vecs = user_vecs.astype(np.float32)
        start = time.perf_counter()
        distances, indices = self.index.search(user_vecs, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(user_vecs)
        return indices, elapsed_ms


def evaluate_recall_metrics(
    retrieved_dict: Dict[int, np.ndarray],
    ground_truth_dict: Dict[int, set[int]],
    top_ks: Tuple[int, ...] = (50, 100),
) -> Dict[str, float]:
    """Calculate Recall@K, HitRate@K, and NDCG@K."""
    results = {}
    for k in top_ks:
        recalls = []
        hits = []
        ndcgs = []

        for uid, gt_items in ground_truth_dict.items():
            if len(gt_items) == 0:
                continue
            preds = retrieved_dict.get(uid, np.array([]))[:k]
            matched = [1 if item in gt_items else 0 for item in preds]
            n_matched = sum(matched)

            recalls.append(n_matched / len(gt_items))
            hits.append(1.0 if n_matched > 0 else 0.0)

            # NDCG
            dcg = sum(m / np.log2(idx + 2) for idx, m in enumerate(matched))
            ideal_matched = min(len(gt_items), k)
            idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_matched))
            ndcgs.append(dcg / max(idcg, 1e-9))

        results[f"Recall@{k}"] = float(np.mean(recalls))
        results[f"HitRate@{k}"] = float(np.mean(hits))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs))

    return results

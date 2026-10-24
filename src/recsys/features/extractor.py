"""Feature extraction pipeline using DuckDB for leakage-safe sequence & statistical features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import duckdb
import numpy as np
import pandas as pd


@dataclass
class RecSysDataBundle:
    train_recall_df: pd.DataFrame
    val_recall_df: pd.DataFrame
    test_recall_df: pd.DataFrame

    train_rank_df: pd.DataFrame
    val_rank_df: pd.DataFrame
    test_rank_df: pd.DataFrame

    item_meta_df: pd.DataFrame
    n_users: int
    n_items: int
    n_categories: int
    max_seq_len: int


def extract_features(
    raw_df: pd.DataFrame,
    max_seq_len: int = 20,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> RecSysDataBundle:
    """Extract features with DuckDB engine, strictly partitioned by time."""
    con = duckdb.connect(":memory:")

    # Map IDs to continuous 0-indexed integers
    user_map = {uid: i for i, uid in enumerate(raw_df["user_id"].unique())}
    item_map = {iid: i for i, iid in enumerate(raw_df["item_id"].unique())}
    cat_map = {cid: i for i, cid in enumerate(raw_df["category_id"].unique())}

    raw_df = raw_df.copy()
    raw_df["u_idx"] = raw_df["user_id"].map(user_map)
    raw_df["i_idx"] = raw_df["item_id"].map(item_map)
    raw_df["c_idx"] = raw_df["category_id"].map(cat_map)

    n_users = len(user_map)
    n_items = len(item_map)
    n_cats = len(cat_map)

    con.register("events", raw_df)

    # Time split thresholds
    times = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events").fetchone()
    min_t, max_t = times[0], times[1]
    span = max_t - min_t
    t_train = min_t + span * train_frac
    t_val = min_t + span * (train_frac + val_frac)

    # 1. Item features computed on train set only to prevent leakage!
    item_meta_df = con.execute(f"""
        SELECT
            i_idx AS item_id,
            MODE(c_idx) AS category_id,
            COUNT(*) AS total_interactions,
            COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) AS item_pv,
            COUNT(CASE WHEN behavior_type = 'cart' THEN 1 END) AS item_cart,
            COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) AS item_buy,
            (COUNT(CASE WHEN behavior_type = 'cart' THEN 1 END) + 1.0) /
                (COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) + 10.0) AS pv_to_cart_rate,
            (COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) + 1.0) /
                (COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) + 30.0) AS pv_to_buy_rate
        FROM events
        WHERE timestamp <= {t_train}
        GROUP BY i_idx
    """).fetchdf()

    # Fill items that appeared only in val/test with global averages
    all_items_df = pd.DataFrame({"item_id": np.arange(n_items)})
    item_meta_df = pd.merge(all_items_df, item_meta_df, on="item_id", how="left")
    # For missing categories, fill from raw events
    item_cat_dict = dict(zip(raw_df["i_idx"], raw_df["c_idx"]))
    item_meta_df["category_id"] = item_meta_df["item_id"].map(item_cat_dict).fillna(0).astype(int)
    item_meta_df["total_interactions"] = item_meta_df["total_interactions"].fillna(0)
    item_meta_df["item_pv"] = item_meta_df["item_pv"].fillna(0)
    item_meta_df["item_cart"] = item_meta_df["item_cart"].fillna(0)
    item_meta_df["item_buy"] = item_meta_df["item_buy"].fillna(0)
    item_meta_df["pv_to_cart_rate"] = item_meta_df["pv_to_cart_rate"].fillna(0.05)
    item_meta_df["pv_to_buy_rate"] = item_meta_df["pv_to_buy_rate"].fillna(0.02)

    # Item log-transformed dense features
    item_meta_df["log_pv"] = np.log1p(item_meta_df["item_pv"])
    item_meta_df["log_buy"] = np.log1p(item_meta_df["item_buy"])

    # 2. Extract sequences & interaction triplets for Recall and Ranking
    # For every event, get user's preceding clicked item history using window function
    event_seq_df = con.execute(f"""
        WITH ranked AS (
            SELECT
                u_idx AS user_id,
                i_idx AS item_id,
                c_idx AS category_id,
                behavior_type,
                timestamp,
                ROW_NUMBER() OVER (PARTITION BY u_idx ORDER BY timestamp ASC) AS user_seq_step
            FROM events
        )
        SELECT * FROM ranked
    """).fetchdf()

    # Construct user history sequences efficiently in memory
    user_histories: Dict[int, List[int]] = {u: [] for u in range(n_users)}
    user_cat_histories: Dict[int, List[int]] = {u: [] for u in range(n_users)}

    rows = []
    # Iterate through events chronologically
    for row in event_seq_df.itertuples():
        u = row.user_id
        i = row.item_id
        c = row.category_id
        b = row.behavior_type
        ts = row.timestamp

        # Preceding history (fixed length padded with 0)
        hist = user_histories[u][-max_seq_len:]
        pad_hist = [0] * (max_seq_len - len(hist)) + hist
        hist_len = min(len(hist), max_seq_len)

        # Labels
        # CTR label: 1 for any interaction (pv/cart/fav/buy)
        ctr_label = 1
        # CVR label: 1 if buy, 0 otherwise
        cvr_label = 1 if b == "buy" else 0

        rows.append({
            "user_id": u,
            "item_id": i,
            "category_id": c,
            "behavior_type": b,
            "timestamp": ts,
            "history_items": pad_hist,
            "history_len": max(1, hist_len),
            "ctr_label": ctr_label,
            "cvr_label": cvr_label,
        })

        # Update user history on clicks/interactions
        user_histories[u].append(i)
        user_cat_histories[u].append(c)

    full_samples_df = pd.DataFrame(rows)

    # Generate ranking negative samples for CTR (impressions not clicked)
    # E.g. For each positive event, sample 1-2 negative unclicked items
    rng = np.random.default_rng(42)
    pop_probs = item_meta_df["total_interactions"].values + 1.0
    pop_probs = pop_probs / pop_probs.sum()

    n_samples = len(full_samples_df)
    neg_items = rng.choice(n_items, size=n_samples, p=pop_probs)
    neg_rows = []
    for idx, row in enumerate(full_samples_df.itertuples()):
        neg_item = int(neg_items[idx])
        neg_cat = int(item_meta_df.loc[neg_item, "category_id"])
        neg_rows.append({
            "user_id": row.user_id,
            "item_id": neg_item,
            "category_id": neg_cat,
            "behavior_type": "impression_unclicked",
            "timestamp": row.timestamp,
            "history_items": row.history_items,
            "history_len": row.history_len,
            "ctr_label": 0,
            "cvr_label": 0,
        })
    neg_df = pd.DataFrame(neg_rows)

    all_ranking_df = pd.concat([full_samples_df, neg_df], ignore_index=True)
    all_ranking_df = all_ranking_df.sort_values("timestamp").reset_index(drop=True)

    # Join item features to ranking DataFrame
    item_cols_to_join = item_meta_df[["item_id", "pv_to_cart_rate", "pv_to_buy_rate", "log_pv", "log_buy"]]
    all_ranking_df = pd.merge(all_ranking_df, item_cols_to_join, on="item_id", how="left")

    # Time-based Train / Val / Test split
    train_recall_df = full_samples_df[full_samples_df["timestamp"] <= t_train].reset_index(drop=True)
    val_recall_df = full_samples_df[(full_samples_df["timestamp"] > t_train) & (full_samples_df["timestamp"] <= t_val)].reset_index(drop=True)
    test_recall_df = full_samples_df[full_samples_df["timestamp"] > t_val].reset_index(drop=True)

    train_rank_df = all_ranking_df[all_ranking_df["timestamp"] <= t_train].reset_index(drop=True)
    val_rank_df = all_ranking_df[(all_ranking_df["timestamp"] > t_train) & (all_ranking_df["timestamp"] <= t_val)].reset_index(drop=True)
    test_rank_df = all_ranking_df[all_ranking_df["timestamp"] > t_val].reset_index(drop=True)

    con.close()

    return RecSysDataBundle(
        train_recall_df=train_recall_df,
        val_recall_df=val_recall_df,
        test_recall_df=test_recall_df,
        train_rank_df=train_rank_df,
        val_rank_df=val_rank_df,
        test_rank_df=test_rank_df,
        item_meta_df=item_meta_df,
        n_users=n_users,
        n_items=n_items,
        n_categories=n_cats,
        max_seq_len=max_seq_len,
    )

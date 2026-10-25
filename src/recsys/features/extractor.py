"""Feature extraction pipeline using DuckDB for leakage-safe sequence & statistical features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    sql_path: str | Path | None = None,
) -> RecSysDataBundle:
    """Extract features with DuckDB engine, strictly partitioned by time without label leakage.

    Item IDs and Category IDs are 1-indexed (1..N). Index 0 is strictly reserved for <PAD>.
    """
    con = duckdb.connect(":memory:")

    # Map IDs: User IDs are 0..n_users-1; Items and Categories are 1-indexed (0 reserved for PAD)
    raw_user_ids = sorted(raw_df["user_id"].unique())
    raw_item_ids = sorted(raw_df["item_id"].unique())
    raw_cat_ids = sorted(raw_df["category_id"].unique())

    user_map = {uid: i for i, uid in enumerate(raw_user_ids)}
    item_map = {iid: i + 1 for i, iid in enumerate(raw_item_ids)}  # 1-indexed
    cat_map = {cid: i + 1 for i, cid in enumerate(raw_cat_ids)}    # 1-indexed

    n_users = len(user_map)
    n_items = len(item_map)
    n_cats = len(cat_map)

    events_df = raw_df.copy()
    events_df["user_id"] = events_df["user_id"].map(user_map)
    events_df["item_id"] = events_df["item_id"].map(item_map)
    events_df["category_id"] = events_df["category_id"].map(cat_map)

    # Time split thresholds
    min_t = int(events_df["timestamp"].min())
    max_t = int(events_df["timestamp"].max())
    span = max_t - min_t
    t_train = min_t + span * train_frac
    t_val = min_t + span * (train_frac + val_frac)

    con.register("raw_events", events_df)
    con.register("train_events", events_df[events_df["timestamp"] <= t_train])

    # 1. Execute external SQL feature engineering file
    sql_file = Path(sql_path) if sql_path else Path(__file__).resolve().parent.parent.parent.parent / "sql" / "feature_engineering.sql"
    if sql_file.exists():
        sql_content = sql_file.read_text()
        for statement in sql_content.split(";"):
            statement = statement.strip()
            if statement:
                con.execute(statement)
    else:
        # Fallback if sql file missing
        con.execute(f"""
            CREATE OR REPLACE TABLE item_features AS
            SELECT
                item_id,
                MODE(category_id) AS category_id,
                COUNT(*) AS total_interactions,
                COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) AS item_pv_count,
                COUNT(CASE WHEN behavior_type = 'cart' THEN 1 END) AS item_cart_count,
                COUNT(CASE WHEN behavior_type = 'fav' THEN 1 END) AS item_fav_count,
                COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) AS item_buy_count,
                (COUNT(CASE WHEN behavior_type = 'cart' THEN 1 END) + 1.0) /
                    (COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) + 20.0) AS item_pv_to_cart_rate,
                (COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) + 1.0) /
                    (COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) + 50.0) AS item_pv_to_buy_rate
            FROM train_events
            GROUP BY item_id;

            CREATE OR REPLACE TABLE user_event_windows AS
            SELECT
                user_id, item_id, category_id, behavior_type, timestamp,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) AS seq_rank_desc,
                timestamp - LAG(timestamp, 1) OVER (PARTITION BY user_id ORDER BY timestamp ASC) AS time_since_prev_event,
                COUNT(*) OVER (PARTITION BY user_id ORDER BY timestamp ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS user_cum_actions
            FROM raw_events;

            CREATE OR REPLACE TABLE user_sessions AS
            SELECT *,
                SUM(CASE WHEN time_since_prev_event > 1800 OR time_since_prev_event IS NULL THEN 1 ELSE 0 END) OVER (
                    PARTITION BY user_id ORDER BY timestamp ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS session_id
            FROM user_event_windows;

            CREATE OR REPLACE TABLE session_candidate_units AS
            SELECT
                user_id, session_id, item_id, category_id,
                MIN(timestamp) AS session_timestamp,
                MAX(CASE WHEN behavior_type IN ('pv', 'cart', 'fav', 'buy') THEN 1 ELSE 0 END) AS is_clicked,
                MAX(CASE WHEN behavior_type = 'buy' THEN 1 ELSE 0 END) AS is_bought
            FROM user_sessions
            GROUP BY user_id, session_id, item_id, category_id;
        """)

    item_meta_df = con.execute("SELECT * FROM item_features").fetchdf()

    # Complete full 1..n_items mapping (0 is PAD, items 1..n_items)
    all_items_df = pd.DataFrame({"item_id": np.arange(1, n_items + 1)})
    item_meta_df = pd.merge(all_items_df, item_meta_df, on="item_id", how="left")

    raw_item_cat_dict = dict(zip(events_df["item_id"], events_df["category_id"]))
    item_meta_df["category_id"] = item_meta_df["item_id"].map(raw_item_cat_dict).fillna(1).astype(int)
    item_meta_df["total_interactions"] = item_meta_df["total_interactions"].fillna(0)
    item_meta_df["item_pv_count"] = item_meta_df["item_pv_count"].fillna(0)
    item_meta_df["item_cart_count"] = item_meta_df["item_cart_count"].fillna(0)
    item_meta_df["item_buy_count"] = item_meta_df["item_buy_count"].fillna(0)
    item_meta_df["pv_to_cart_rate"] = item_meta_df["item_pv_to_cart_rate"].fillna(0.05)
    item_meta_df["pv_to_buy_rate"] = item_meta_df["item_pv_to_buy_rate"].fillna(0.02)
    item_meta_df["log_pv"] = np.log1p(item_meta_df["item_pv_count"])
    item_meta_df["log_buy"] = np.log1p(item_meta_df["item_buy_count"])

    # 2. Extract session candidate units
    session_units = con.execute("""
        SELECT user_id, session_id, item_id, category_id, session_timestamp, is_clicked, is_bought
        FROM session_candidate_units
        ORDER BY session_timestamp ASC
    """).fetchdf()

    # 3. Construct user history strictly from preceding sessions (NO intra-session leakage!)
    user_histories: Dict[int, List[int]] = {u: [] for u in range(n_users)}
    user_current_session: Dict[int, int] = {u: -1 for u in range(n_users)}
    session_staged_items: Dict[int, List[int]] = {u: [] for u in range(n_users)}

    rows = []
    for row in session_units.itertuples():
        u = row.user_id
        s = row.session_id
        i = row.item_id
        c = row.category_id
        ts = row.session_timestamp

        # When moving to a new session, commit items from previous session to user history
        if s != user_current_session[u]:
            user_histories[u].extend(session_staged_items[u])
            session_staged_items[u] = []
            user_current_session[u] = s

        hist = user_histories[u][-max_seq_len:]
        pad_hist = [0] * (max_seq_len - len(hist)) + hist
        hist_len = min(len(hist), max_seq_len)

        rows.append({
            "user_id": u,
            "session_id": s,
            "item_id": i,
            "category_id": c,
            "timestamp": ts,
            "history_items": pad_hist,
            "history_len": max(1, hist_len),
            "ctr_label": int(row.is_clicked),
            "cvr_label": int(row.is_bought),
        })

        session_staged_items[u].append(i)

    pos_df = pd.DataFrame(rows)

    # 4. Negative sampling for unclicked impression candidates
    # For each clicked session item, sample unclicked negative candidate with same user history
    rng = np.random.default_rng(42)
    pop_probs = item_meta_df["total_interactions"].values + 1.0
    pop_probs = pop_probs / pop_probs.sum()
    all_item_indices = np.arange(1, n_items + 1)

    n_pos = len(pos_df)
    neg_items = rng.choice(all_item_indices, size=n_pos, p=pop_probs)
    item_cat_lut = dict(zip(item_meta_df["item_id"], item_meta_df["category_id"]))

    neg_rows = []
    for idx, row in enumerate(pos_df.itertuples()):
        neg_item = int(neg_items[idx])
        neg_cat = int(item_cat_lut.get(neg_item, 1))
        neg_rows.append({
            "user_id": row.user_id,
            "session_id": row.session_id,
            "item_id": neg_item,
            "category_id": neg_cat,
            "timestamp": row.timestamp,
            "history_items": row.history_items,
            "history_len": row.history_len,
            "ctr_label": 0,
            "cvr_label": 0,
        })
    neg_df = pd.DataFrame(neg_rows)

    all_ranking_df = pd.concat([pos_df, neg_df], ignore_index=True)
    all_ranking_df = all_ranking_df.sort_values("timestamp").reset_index(drop=True)

    # Join item features
    item_cols_to_join = item_meta_df[["item_id", "pv_to_cart_rate", "pv_to_buy_rate", "log_pv", "log_buy"]]
    all_ranking_df = pd.merge(all_ranking_df, item_cols_to_join, on="item_id", how="left")

    # Time-based split: strictly partition by timestamp
    train_recall_df = pos_df[pos_df["timestamp"] <= t_train].reset_index(drop=True)
    val_recall_df = pos_df[(pos_df["timestamp"] > t_train) & (pos_df["timestamp"] <= t_val)].reset_index(drop=True)
    test_recall_df = pos_df[pos_df["timestamp"] > t_val].reset_index(drop=True)

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

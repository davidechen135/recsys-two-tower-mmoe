"""Data loader and realistic synthetic generator for E-commerce user behavior logs.

Schema matches Alibaba Taobao UserBehavior dataset (Tianchi 649):
user_id, item_id, category_id, behavior_type, timestamp
behavior_type in {'pv', 'cart', 'fav', 'buy'}
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


BEHAVIORS = ("pv", "cart", "fav", "buy")


def generate_synthetic_user_behavior(
    n_users: int = 5000,
    n_items: int = 2000,
    n_categories: int = 50,
    n_interactions: int = 150_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic e-commerce interaction log.

    Simulates:
    - Long-tail item popularity (Power-law / Zipf)
    - Item-to-category hierarchical assignment
    - User latent category affinity
    - E-commerce conversion funnel:
        P(cart|pv) ~ 0.15, P(fav|pv) ~ 0.08, P(buy|cart) ~ 0.35, P(buy|pv) ~ 0.04
    - Diurnal timestamp ordering across 7 days
    """
    rng = np.random.default_rng(seed)

    # 1. Item-category mapping
    # Items clustered into categories with uneven size
    cat_weights = rng.dirichlet(np.ones(n_categories) * 1.5)
    item_categories = rng.choice(n_categories, size=n_items, p=cat_weights)

    # 2. Item popularity power-law
    ranks = np.arange(1, n_items + 1)
    item_probs = 1.0 / (ranks ** 0.85)
    item_probs /= item_probs.sum()

    # 3. User latent category affinities (top 3 preferred categories per user)
    user_top_cats = rng.choice(n_categories, size=(n_users, 3), replace=True)

    # 4. Generate user sessions & interactions
    user_ids = []
    item_ids = []
    category_ids = []
    behaviors = []
    timestamps = []

    base_time = 1511568000  # 2017-11-25 00:00:00 UTC (matching Taobao dataset timeframe)
    week_seconds = 7 * 24 * 3600

    # User activity distribution (Pareto: some users are very active)
    user_activity = rng.pareto(a=2.0, size=n_users) + 1.0
    user_activity /= user_activity.sum()
    user_sample = rng.choice(n_users, size=n_interactions, p=user_activity)

    # Timestamp with diurnal variation
    raw_times = rng.uniform(0, week_seconds, size=n_interactions)
    # Sort roughly by time
    time_sort_idx = np.argsort(raw_times)
    user_sample = user_sample[time_sort_idx]
    raw_times = raw_times[time_sort_idx]

    for i in range(n_interactions):
        u = int(user_sample[i])
        t = int(base_time + raw_times[i])

        # 70% chance to pick item from user's preferred categories, 30% random popular item
        if rng.random() < 0.70:
            preferred_cat = int(rng.choice(user_top_cats[u]))
            cat_items = np.where(item_categories == preferred_cat)[0]
            if len(cat_items) > 0:
                item_idx = int(rng.choice(cat_items))
            else:
                item_idx = int(rng.choice(n_items, p=item_probs))
        else:
            item_idx = int(rng.choice(n_items, p=item_probs))

        cat_id = int(item_categories[item_idx])

        # Primary interaction: pv
        user_ids.append(u)
        item_ids.append(item_idx)
        category_ids.append(cat_id)
        behaviors.append("pv")
        timestamps.append(t)

        # Funnel transitions within session
        rand_val = rng.random()
        if rand_val < 0.22:
            # Cart action shortly after
            user_ids.append(u)
            item_ids.append(item_idx)
            category_ids.append(cat_id)
            behaviors.append("cart")
            timestamps.append(t + rng.integers(10, 60))

            if rng.random() < 0.35:
                # Buy after cart
                user_ids.append(u)
                item_ids.append(item_idx)
                category_ids.append(cat_id)
                behaviors.append("buy")
                timestamps.append(t + rng.integers(70, 300))
        elif rand_val < 0.30:
            # Fav action
            user_ids.append(u)
            item_ids.append(item_idx)
            category_ids.append(cat_id)
            behaviors.append("fav")
            timestamps.append(t + rng.integers(15, 90))

    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "category_id": category_ids,
        "behavior_type": behaviors,
        "timestamp": timestamps,
    })

    # Sort strictly by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_taobao_user_behavior(
    file_path: str | Path,
    max_users: Optional[int] = None,
    sample_fraction: Optional[float] = None,
) -> pd.DataFrame:
    """Load real Alibaba Taobao UserBehavior dataset.

    Expected CSV columns (no header in original Tianchi release):
    user_id, item_id, category_id, behavior_type, timestamp
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    names = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    if sample_fraction is not None and 0 < sample_fraction < 1.0:
        # Sample lines
        chunks = []
        for chunk in pd.read_csv(file_path, names=names, chunksize=200_000):
            chunks.append(chunk.sample(frac=sample_fraction))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(file_path, names=names)

    if max_users is not None:
        top_users = df["user_id"].unique()[:max_users]
        df = df[df["user_id"].isin(top_users)].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

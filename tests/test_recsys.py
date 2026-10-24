"""Unit and integration tests for RecSys Two-Tower Recall & MMoE Ranking."""
import numpy as np
import pytest
import torch

from recsys.data.generator import generate_synthetic_user_behavior
from recsys.features.extractor import extract_features
from recsys.models.dssm import TwoTowerDSSM
from recsys.models.mmoe_din import MMoEDINRanking
from recsys.models.deepfm import SingleTaskDeepFM


def test_synthetic_generator_schema():
    df = generate_synthetic_user_behavior(n_users=100, n_items=50, n_interactions=500, seed=42)
    expected_cols = {"user_id", "item_id", "category_id", "behavior_type", "timestamp"}
    assert set(df.columns) == expected_cols
    assert len(df) > 0
    assert set(df["behavior_type"].unique()).issubset({"pv", "cart", "fav", "buy"})
    # Strictly non-decreasing timestamp
    assert np.all(np.diff(df["timestamp"].values) >= 0)


def test_duckdb_feature_extraction_no_leakage():
    df = generate_synthetic_user_behavior(n_users=200, n_items=100, n_interactions=1000, seed=42)
    bundle = extract_features(df, max_seq_len=10, train_frac=0.7, val_frac=0.15)

    assert bundle.n_users == len(df["user_id"].unique())
    assert bundle.n_items == len(df["item_id"].unique())
    # Train timestamps strictly <= Val timestamps strictly <= Test timestamps
    max_train_t = bundle.train_recall_df["timestamp"].max()
    min_val_t = bundle.val_recall_df["timestamp"].min()
    max_val_t = bundle.val_recall_df["timestamp"].max()
    min_test_t = bundle.test_recall_df["timestamp"].min()

    assert max_train_t <= min_val_t
    assert max_val_t <= min_test_t


def test_two_tower_dssm_forward_and_norm():
    n_users = 50
    n_items = 100
    n_cats = 10
    model = TwoTowerDSSM(n_users, n_items, n_cats, embed_dim=32, out_dim=32)

    batch_size = 8
    u = torch.randint(0, n_users, (batch_size,))
    hist = torch.randint(0, n_items, (batch_size, 10))
    lens = torch.randint(1, 10, (batch_size,))
    i = torch.randint(0, n_items, (batch_size,))
    c = torch.randint(0, n_cats, (batch_size,))
    dense = torch.randn(batch_size, 4)

    loss, u_vec, i_vec = model(u, hist, lens, i, c, dense)

    assert loss.item() > 0
    assert u_vec.shape == (batch_size, 32)
    assert i_vec.shape == (batch_size, 32)
    # L2 norm of embeddings should be 1.0
    u_norms = torch.norm(u_vec, p=2, dim=-1)
    i_norms = torch.norm(i_vec, p=2, dim=-1)
    assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-5)
    assert torch.allclose(i_norms, torch.ones_like(i_norms), atol=1e-5)


def test_mmoe_din_forward_and_gate_weights():
    n_users = 50
    n_items = 100
    n_cats = 10
    model = MMoEDINRanking(n_users, n_items, n_cats, dense_dim=4, embed_dim=16, n_experts=3)

    batch_size = 6
    u = torch.randint(0, n_users, (batch_size,))
    i = torch.randint(0, n_items, (batch_size,))
    c = torch.randint(0, n_cats, (batch_size,))
    hist = torch.randint(0, n_items, (batch_size, 8))
    dense = torch.randn(batch_size, 4)

    p_ctr, p_cvr, w_ctr, w_cvr = model(u, i, c, hist, dense)

    assert p_ctr.shape == (batch_size,)
    assert p_cvr.shape == (batch_size,)
    assert torch.all((p_ctr >= 0.0) & (p_ctr <= 1.0))
    assert torch.all((p_cvr >= 0.0) & (p_cvr <= 1.0))
    # Gate weights must sum to 1.0 per sample
    assert torch.allclose(w_ctr.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(w_cvr.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

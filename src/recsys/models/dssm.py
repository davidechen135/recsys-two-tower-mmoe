"""PyTorch Two-Tower DSSM (Deep Structured Semantic Model) for Vector Recall."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UserTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, user_id: torch.Tensor, hist_items: torch.Tensor, hist_len: torch.Tensor) -> torch.Tensor:
        # user_id: (B,), hist_items: (B, L)
        u_e = self.user_emb(user_id)  # (B, D)
        seq_e = self.item_emb(hist_items)  # (B, L, D)

        # Masked average pooling
        mask = (hist_items > 0).unsqueeze(-1).float()  # (B, L, 1)
        sum_e = torch.sum(seq_e * mask, dim=1)  # (B, D)
        lengths = torch.clamp(hist_len.unsqueeze(-1).float(), min=1.0)
        hist_pool = sum_e / lengths  # (B, D)

        concat = torch.cat([u_e, hist_pool], dim=-1)  # (B, 2D)
        out = self.mlp(concat)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_cats: int,
        dense_dim: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.cat_emb = nn.Embedding(n_cats, embed_dim // 2)
        in_dim = embed_dim + embed_dim // 2 + dense_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, item_id: torch.Tensor, cat_id: torch.Tensor, dense_feats: torch.Tensor) -> torch.Tensor:
        i_e = self.item_emb(item_id)
        c_e = self.cat_emb(cat_id)
        concat = torch.cat([i_e, c_e, dense_feats], dim=-1)
        out = self.mlp(concat)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerDSSM(nn.Module):
    """Two-Tower DSSM with in-batch sampled softmax and learnable temperature."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cats: int,
        embed_dim: int = 64,
        out_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(n_users, n_items, embed_dim=embed_dim, out_dim=out_dim)
        self.item_tower = ItemTower(n_items, n_cats, embed_dim=embed_dim, out_dim=out_dim)
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(
        self,
        user_id: torch.Tensor,
        hist_items: torch.Tensor,
        hist_len: torch.Tensor,
        item_id: torch.Tensor,
        cat_id: torch.Tensor,
        dense_feats: torch.Tensor,
        item_log_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_vec = self.user_tower(user_id, hist_items, hist_len)  # (B, out_dim)
        i_vec = self.item_tower(item_id, cat_id, dense_feats)  # (B, out_dim)

        # In-batch cosine similarity matrix (B, B)
        sim_matrix = torch.matmul(u_vec, i_vec.T) / torch.clamp(self.temperature, min=0.01)

        # LogQ correction for item popularity
        if item_log_q is not None:
            # item_log_q: (B,)
            sim_matrix = sim_matrix - item_log_q.unsqueeze(0)

        # Labels are diagonal (index 0..B-1)
        labels = torch.arange(len(user_id), device=user_id.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss, u_vec, i_vec

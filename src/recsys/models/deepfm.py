"""Single-task DeepFM (Factorization Machine + Deep Neural Network) baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskDeepFM(nn.Module):
    """DeepFM for single-task ranking (CTR or CVR)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cats: int,
        dense_dim: int = 4,
        embed_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # 1st order linear weights
        self.u_w = nn.Embedding(n_users, 1)
        self.i_w = nn.Embedding(n_items + 1, 1, padding_idx=0)
        self.c_w = nn.Embedding(n_cats, 1)
        self.dense_w = nn.Linear(dense_dim, 1, bias=False)

        # 2nd order factorization embeddings
        self.u_e = nn.Embedding(n_users, embed_dim)
        self.i_e = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.c_e = nn.Embedding(n_cats, embed_dim)
        self.dense_e = nn.Linear(dense_dim, embed_dim, bias=False)

        # Deep component MLP
        in_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        cat_id: torch.Tensor,
        dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        # 1st order linear sum
        linear_part = (
            self.u_w(user_id)
            + self.i_w(item_id)
            + self.c_w(cat_id)
            + self.dense_w(dense_feats)
            + self.bias
        ).squeeze(-1)

        # 2nd order FM interaction
        # Stack embeddings: (B, 4, embed_dim)
        embs = torch.stack(
            [self.u_e(user_id), self.i_e(item_id), self.c_e(cat_id), self.dense_e(dense_feats)],
            dim=1,
        )
        # Sum of embeddings squared
        sum_embs = torch.sum(embs, dim=1)  # (B, D)
        sum_sq = torch.sum(sum_embs ** 2, dim=-1)  # (B,)
        # Square of sum of embeddings
        sq_embs = torch.sum(embs ** 2, dim=1)  # (B, D)
        sq_sum = torch.sum(sq_embs, dim=-1)  # (B,)
        fm_part = 0.5 * (sum_sq - sq_sum)

        # Deep component
        flattened = embs.view(embs.size(0), -1)  # (B, 4D)
        deep_part = self.mlp(flattened).squeeze(-1)

        logit = linear_part + fm_part + deep_part
        return torch.sigmoid(logit)

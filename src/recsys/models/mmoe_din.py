"""MMoE (Multi-gate Mixture-of-Experts) + DIN (Deep Interest Network) for Multi-Task Ranking."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINAttentionUnit(nn.Module):
    """Deep Interest Network (DIN) local activation unit.

    Computes query-key attention: [query, key, query - key, query * key].
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # query: (B, D), keys: (B, L, D), mask: (B, L)
        B, L, D = keys.size()
        q_expanded = query.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)

        # Concatenate interaction features: [q, k, q - k, q * k]
        interaction = torch.cat(
            [q_expanded, keys, q_expanded - keys, q_expanded * keys], dim=-1
        )  # (B, L, 4D)

        scores = self.mlp(interaction).squeeze(-1)  # (B, L)

        # Mask padding tokens
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)

        # Attention weighted pooling
        pooled = torch.sum(attn_weights * keys, dim=1)  # (B, D)
        return pooled


class MMoEDINRanking(nn.Module):
    """MMoE with DIN attention for joint CTR and CVR ranking."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cats: int,
        dense_dim: int = 4,
        embed_dim: int = 32,
        n_experts: int = 4,
        expert_dim: int = 64,
        tower_dim: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_experts = n_experts

        # Embeddings
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats + 1, embed_dim // 2, padding_idx=0)

        # Multi-task loss dynamic uncertainty weighting (Kendall et al., CVPR 2018)
        self.log_var_ctr = nn.Parameter(torch.zeros(1))
        self.log_var_cvr = nn.Parameter(torch.zeros(1))

        # DIN Attention Unit
        self.din_attention = DINAttentionUnit(embed_dim=embed_dim)

        # Dense feature projection
        self.dense_proj = nn.Linear(dense_dim, embed_dim // 2)

        # Input dimension to Experts
        # user_emb(D) + item_emb(D) + cat_emb(D/2) + din_pooled(D) + dense(D/2) = 4D
        self.shared_in_dim = embed_dim * 3 + (embed_dim // 2) * 2

        # 1. Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.shared_in_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])

        # 2. Gating Networks
        self.gate_ctr = nn.Linear(self.shared_in_dim, n_experts)
        self.gate_cvr = nn.Linear(self.shared_in_dim, n_experts)

        # 3. Task Towers
        self.tower_ctr = nn.Sequential(
            nn.Linear(expert_dim, tower_dim),
            nn.ReLU(),
            nn.Linear(tower_dim, 1),
        )
        self.tower_cvr = nn.Sequential(
            nn.Linear(expert_dim, tower_dim),
            nn.ReLU(),
            nn.Linear(tower_dim, 1),
        )

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        cat_id: torch.Tensor,
        hist_items: torch.Tensor,
        dense_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Embeddings
        u_e = self.user_emb(user_id)
        i_e = self.item_emb(item_id)
        c_e = self.cat_emb(cat_id)
        d_e = F.relu(self.dense_proj(dense_feats))

        # DIN Attention over user history
        hist_e = self.item_emb(hist_items)
        mask = hist_items > 0
        din_pooled = self.din_attention(i_e, hist_e, mask)

        # Concatenate shared input representation
        x = torch.cat([u_e, i_e, c_e, din_pooled, d_e], dim=-1)

        # Compute Expert outputs: (B, n_experts, expert_dim)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        # Gates: (B, n_experts, 1)
        w_ctr = F.softmax(self.gate_ctr(x), dim=-1).unsqueeze(-1)
        w_cvr = F.softmax(self.gate_cvr(x), dim=-1).unsqueeze(-1)

        # Task specific representations
        h_ctr = torch.sum(w_ctr * expert_stack, dim=1)  # (B, expert_dim)
        h_cvr = torch.sum(w_cvr * expert_stack, dim=1)  # (B, expert_dim)

        # Predictions (logits)
        logit_ctr = self.tower_ctr(h_ctr).squeeze(-1)
        logit_cvr = self.tower_cvr(h_cvr).squeeze(-1)

        p_ctr = torch.sigmoid(logit_ctr)
        p_cvr = torch.sigmoid(logit_cvr)

        return p_ctr, p_cvr, w_ctr.squeeze(-1), w_cvr.squeeze(-1)

    def compute_multi_task_loss(
        self,
        p_ctr: torch.Tensor,
        p_cvr: torch.Tensor,
        y_ctr: torch.Tensor,
        y_cvr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint CTR and CVR loss with dynamic Uncertainty Weighting (Kendall et al., CVPR 2018)."""
        loss_ctr = F.binary_cross_entropy(p_ctr, y_ctr)

        # CVR is conditioned on click (y_ctr == 1)
        click_mask = y_ctr == 1
        if click_mask.any():
            loss_cvr = F.binary_cross_entropy(p_cvr[click_mask], y_cvr[click_mask])
        else:
            loss_cvr = torch.tensor(0.0, device=p_ctr.device)

        # Dynamic uncertainty weighting
        precision_ctr = torch.exp(-self.log_var_ctr)
        precision_cvr = torch.exp(-self.log_var_cvr)
        loss = (
            0.5 * precision_ctr * loss_ctr + 0.5 * self.log_var_ctr
            + 0.5 * precision_cvr * loss_cvr + 0.5 * self.log_var_cvr
        )
        return loss, loss_ctr, loss_cvr

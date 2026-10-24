"""Evaluation metrics for RecSys Ranking: AUC, GAUC (Group AUC), LogLoss."""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def calculate_gauc(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_impressions: int = 2,
) -> float:
    """Group AUC (GAUC) averaged across users, weighted by user impression count."""
    df = pd.DataFrame({"user_id": user_ids, "y_true": y_true, "y_pred": y_pred})
    auc_sum = 0.0
    weight_sum = 0.0

    for uid, group in df.groupby("user_id"):
        y_u = group["y_true"].values
        # User must have both positive and negative samples to compute AUC
        if len(np.unique(y_u)) == 2 and len(y_u) >= min_impressions:
            auc_u = roc_auc_score(y_u, group["y_pred"].values)
            weight = len(y_u)
            auc_sum += auc_u * weight
            weight_sum += weight

    if weight_sum == 0.0:
        return calculate_auc(y_true, y_pred)
    return float(auc_sum / weight_sum)


def calculate_ranking_metrics(
    y_true_ctr: np.ndarray,
    y_pred_ctr: np.ndarray,
    y_true_cvr: np.ndarray,
    y_pred_cvr: np.ndarray,
    user_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {
        "ctr_auc": calculate_auc(y_true_ctr, y_pred_ctr),
        "ctr_logloss": float(log_loss(y_true_ctr, np.clip(y_pred_ctr, 1e-6, 1 - 1e-6))),
        "cvr_auc": calculate_auc(y_true_cvr, y_pred_cvr),
        "cvr_logloss": float(log_loss(y_true_cvr, np.clip(y_pred_cvr, 1e-6, 1 - 1e-6))),
    }
    if user_ids is not None:
        metrics["ctr_gauc"] = calculate_gauc(user_ids, y_true_ctr, y_pred_ctr)
        metrics["cvr_gauc"] = calculate_gauc(user_ids, y_true_cvr, y_pred_cvr)

    return metrics

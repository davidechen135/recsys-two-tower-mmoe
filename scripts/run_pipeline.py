"""Main entry point to execute the RecSys Two-Tower Recall & MMoE Ranking Pipeline."""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path

# OpenMP fallback for LightGBM on macOS
_sklearn_dylibs = Path(__file__).resolve().parent.parent.parent / ".venv" / "lib" / "python3.12" / "site-packages" / "sklearn" / ".dylibs"
if _sklearn_dylibs.exists() and "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = str(_sklearn_dylibs)

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from recsys.data.generator import generate_synthetic_user_behavior, load_taobao_user_behavior
from recsys.features.extractor import extract_features
from recsys.pipeline.trainer import train_ranking_models, train_two_tower_dssm


def run_pipeline(data_path: str | None = None, output_dir: Path | None = None):
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/4] Loading / Generating E-commerce Interaction Data ===")
    if data_path and Path(data_path).exists():
        print(f"  Loading real dataset from: {data_path}")
        raw_df = load_taobao_user_behavior(data_path, max_users=10_000)
    else:
        print("  Generating realistic synthetic Taobao UserBehavior data (5k users, 2k items)...")
        raw_df = generate_synthetic_user_behavior(
            n_users=3000, n_items=1200, n_categories=40, n_interactions=80_000, seed=42
        )
    print(f"  Total raw interactions: {len(raw_df):,}")

    print("\n=== [2/4] SQL Feature Engineering via DuckDB ===")
    bundle = extract_features(raw_df, max_seq_len=20)
    print(f"  Users: {bundle.n_users:,} | Items: {bundle.n_items:,} | Categories: {bundle.n_categories}")
    print(f"  Train Recall Samples: {len(bundle.train_recall_df):,} | Test: {len(bundle.test_recall_df):,}")
    print(f"  Train Ranking Samples: {len(bundle.train_rank_df):,} | Test: {len(bundle.test_rank_df):,}")

    print("\n=== [3/4] Vector Recall: Two-Tower DSSM + FAISS vs Baselines ===")
    dssm_model, recall_metrics = train_two_tower_dssm(bundle, embed_dim=48, out_dim=48, epochs=3)
    print("  Recall Results:")
    for k, v in [("Popularity", recall_metrics["Popularity"]),
                 ("ItemCF", recall_metrics["ItemCF"]),
                 ("TwoTower_DSSM", recall_metrics["TwoTower_DSSM"])]:
        print(f"    {k:15s} | Recall@50: {v['Recall@50']:.4f} | Recall@100: {v['Recall@100']:.4f} | NDCG@100: {v['NDCG@100']:.4f}")
    print(f"    DSSM Lift vs ItemCF: {recall_metrics['relative_lift_vs_cf_pct']:+.1f}% | Latency: {recall_metrics['TwoTower_DSSM']['query_latency_ms']:.3f} ms")

    # Plot Recall Comparison
    plt.figure(figsize=(8, 4.5))
    models = ["Popularity", "ItemCF", "TwoTower_DSSM"]
    r50 = [recall_metrics[m]["Recall@50"] for m in models]
    r100 = [recall_metrics[m]["Recall@100"] for m in models]
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, r50, width, label="Recall@50", color="steelblue")
    plt.bar(x + width/2, r100, width, label="Recall@100", color="darkorange")
    plt.xticks(x, models)
    plt.ylabel("Recall Score")
    plt.title("Candidate Recall: Two-Tower DSSM + FAISS vs Baselines")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "recall_comparison.png", dpi=200)
    plt.close()

    print("\n=== [4/4] Multi-Task Ranking: MMoE + DIN vs Baselines ===")
    mmoe_model, rank_metrics = train_ranking_models(bundle, embed_dim=32, epochs=3)
    print("  Ranking Results:")
    for m in ["LightGBM", "DeepFM", "MMoE_DIN"]:
        v = rank_metrics[m]
        if v:
            print(f"    {m:15s} | CTR AUC: {v['ctr_auc']:.4f} | CVR AUC: {v['cvr_auc']:.4f} | CTR LogLoss: {v['ctr_logloss']:.4f}")

    # Plot Ranking AUC Comparison
    plt.figure(figsize=(8, 4.5))
    rk_models = [m for m in ["LightGBM", "DeepFM", "MMoE_DIN"] if rank_metrics[m]]
    ctr_aucs = [rank_metrics[m]["ctr_auc"] for m in rk_models]
    cvr_aucs = [rank_metrics[m]["cvr_auc"] for m in rk_models]
    x = np.arange(len(rk_models))
    plt.bar(x - width/2, ctr_aucs, width, label="CTR AUC", color="royalblue")
    plt.bar(x + width/2, cvr_aucs, width, label="CVR AUC", color="seagreen")
    plt.xticks(x, rk_models)
    plt.ylim(0.5, 1.0)
    plt.ylabel("AUC")
    plt.title("Multi-Task Ranking: MMoE + DIN vs Single-Task Baselines")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "ranking_auc_comparison.png", dpi=200)
    plt.close()

    # Plot Gate Weights
    if "gate_weights_ctr_mean" in rank_metrics["MMoE_DIN"]:
        plt.figure(figsize=(7, 4))
        w_ctr = rank_metrics["MMoE_DIN"]["gate_weights_ctr_mean"]
        w_cvr = rank_metrics["MMoE_DIN"]["gate_weights_cvr_mean"]
        exp_idx = np.arange(len(w_ctr))
        plt.bar(exp_idx - width/2, w_ctr, width, label="CTR Task Gate", color="crimson")
        plt.bar(exp_idx + width/2, w_cvr, width, label="CVR Task Gate", color="purple")
        plt.xticks(exp_idx, [f"Expert {i+1}" for i in exp_idx])
        plt.ylabel("Mean Softmax Weight")
        plt.title("MMoE Dynamic Gate Weights Divergence across Tasks")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(fig_dir / "mmoe_gate_weights.png", dpi=200)
        plt.close()

    all_metrics = {
        "recall": recall_metrics,
        "ranking": rank_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll Project 1 results saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(data_path=args.data_path)

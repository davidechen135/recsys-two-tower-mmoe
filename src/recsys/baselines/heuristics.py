"""Heuristic and traditional collaborative filtering recall baselines."""
from ..recall.faiss_index import GlobalPopularityRecall, ItemCFRecall

__all__ = ["GlobalPopularityRecall", "ItemCFRecall"]

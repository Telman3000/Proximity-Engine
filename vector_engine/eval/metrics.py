"""Per-query retrieval metrics (returned per query for bootstrap CIs)."""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


def _dcg(gains: List[int]) -> float:
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def evaluate(
    retrieved: List[np.ndarray],
    relevant: List[Set[int]],
    ks: List[int],
) -> Dict[str, np.ndarray]:
    """Return per-query metric arrays keyed by metric name."""
    max_k = max(ks)
    out: Dict[str, List[float]] = {f"recall@{k}": [] for k in ks}
    for k in ks:
        out[f"precision@{k}"] = []
    out["map"] = []
    out[f"ndcg@{max_k}"] = []

    for ret, rel in zip(retrieved, relevant):
        ret = list(ret)
        nrel = len(rel)
        for k in ks:
            topk = ret[:k]
            hits = sum(1 for r in topk if r in rel)
            out[f"recall@{k}"].append(hits / nrel if nrel else 0.0)
            out[f"precision@{k}"].append(hits / k if k else 0.0)

        # Average precision over the retrieved ranking
        ap, hit = 0.0, 0
        for i, r in enumerate(ret):
            if r in rel:
                hit += 1
                ap += hit / (i + 1)
        out["map"].append(ap / nrel if nrel else 0.0)

        # nDCG@max_k (binary relevance)
        gains = [1 if r in rel else 0 for r in ret[:max_k]]
        idcg = _dcg([1] * min(nrel, max_k))
        out[f"ndcg@{max_k}"].append(_dcg(gains) / idcg if idcg > 0 else 0.0)

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

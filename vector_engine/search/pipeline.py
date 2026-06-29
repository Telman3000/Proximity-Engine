"""Query execution: single-stage ANN or a coarse->rerank cascade.

The cascade retrieves `coarse_k` candidates from the (possibly quantised) index,
then re-ranks them with full-precision cosine similarity. This is the classic
engineering pattern: cheap recall-oriented first stage + precise second stage.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from ..config import Config
from ..store import faiss_store


def run_queries(
    cfg: Config,
    index,
    query_emb: np.ndarray,
    ids: np.ndarray,
    corpus_emb: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    top_k = int(cfg.get("search.top_k", 10))
    cascade = bool(cfg.get("search.cascade.enabled", False))
    coarse_k = int(cfg.get("search.cascade.coarse_k", 100))
    rerank = cfg.get("search.cascade.rerank", "exact")

    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
    retrieved: List[np.ndarray] = []
    latencies = np.empty(query_emb.shape[0], dtype=np.float64)

    for i in range(query_emb.shape[0]):
        q = query_emb[i:i + 1]
        t0 = time.perf_counter()
        if cascade:
            _, I = faiss_store.search(index, q, coarse_k)
            cand = I[0]
            cand = cand[cand >= 0]
            if rerank == "exact" and corpus_emb is not None and cand.size > 0:
                scores = corpus_emb[cand] @ q[0]
                order = np.argsort(-scores)[:top_k]
                top_pos = cand[order]
            else:
                top_pos = cand[:top_k]
        else:
            _, I = faiss_store.search(index, q, top_k)
            top_pos = I[0]
            top_pos = top_pos[top_pos >= 0]
        latencies[i] = (time.perf_counter() - t0) * 1000.0
        retrieved.append(ids[top_pos])

    return retrieved, latencies

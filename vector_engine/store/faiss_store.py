"""FAISS index construction and search.

Supports the iteration ladder:
  flat   -> exact brute force (recall ceiling)
  hnsw   -> graph ANN
  ivfpq  -> IVF + Product Quantisation (the storage-compression study)
  opq    -> OPQ rotation + IVF + PQ
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, Tuple

import faiss
import numpy as np

from ..config import Config


def _metric(cfg: Config) -> int:
    m = cfg.get("index.metric", "ip")
    return faiss.METRIC_INNER_PRODUCT if m == "ip" else faiss.METRIC_L2


def _factory_metric(cfg: Config) -> str:
    return "IP" if cfg.get("index.metric", "ip") == "ip" else "L2"


def build_index(cfg: Config, emb: np.ndarray) -> Tuple[faiss.Index, Dict]:
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    n, d = emb.shape
    itype = cfg.get("index.type", "flat")
    metric = _metric(cfg)
    t0 = time.time()

    if itype == "flat":
        index = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT \
            else faiss.IndexFlatL2(d)
        index.add(emb)

    elif itype == "hnsw":
        M = int(cfg.get("index.hnsw.M", 32))
        index = faiss.IndexHNSWFlat(d, M, metric)
        index.hnsw.efConstruction = int(cfg.get("index.hnsw.ef_construction", 200))
        index.add(emb)
        index.hnsw.efSearch = int(cfg.get("index.hnsw.ef_search", 64))

    elif itype == "ivfpq":
        nlist = min(int(cfg.get("index.ivfpq.nlist", 256)), max(1, n // 2))
        m = int(cfg.get("index.ivfpq.m", 32))
        nbits = int(cfg.get("index.ivfpq.nbits", 8))
        quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT \
            else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric)
        index.train(emb)
        index.add(emb)
        index.nprobe = int(cfg.get("index.ivfpq.nprobe", 16))

    elif itype == "opq":
        nlist = min(int(cfg.get("index.opq.nlist", 256)), max(1, n // 2))
        m = int(cfg.get("index.opq.m", 32))
        nbits = int(cfg.get("index.opq.nbits", 8))
        factory = f"OPQ{m},IVF{nlist},PQ{m}x{nbits}"
        index = faiss.index_factory(d, factory, metric)
        index.train(emb)
        index.add(emb)
        faiss.extract_index_ivf(index).nprobe = int(cfg.get("index.opq.nprobe", 16))

    else:
        raise ValueError(f"Unknown index.type: {itype}")

    build_seconds = time.time() - t0
    size_bytes = _index_size_bytes(index)
    info = {
        "type": itype,
        "n": int(n),
        "dim": int(d),
        "build_seconds": round(build_seconds, 3),
        "index_size_bytes": int(size_bytes),
        "bits_per_vector": round(size_bytes * 8 / max(1, n), 2),
    }
    return index, info


def _index_size_bytes(index: faiss.Index) -> int:
    fd, path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, path)
        return os.path.getsize(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def search(index: faiss.Index, queries: np.ndarray, k: int):
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    scores, idx = index.search(queries, k)
    return scores, idx

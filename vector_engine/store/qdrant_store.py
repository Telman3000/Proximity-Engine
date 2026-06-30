"""Qdrant vector storage (local on-disk mode, no server required).

Supports the same index ladder as FAISS for ablation:
  flat  -> exact search (SearchParams.exact=True)
  hnsw  -> HNSW graph index
  scalar -> scalar int8 quantisation (storage economy analogue of PQ)
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np

from ..config import Config


def _storage_path(cfg: Config) -> str:
    rel = cfg.get("qdrant.path", "artifacts/qdrant")
    return cfg.path(rel)


def _collection_name(cfg: Config) -> str:
    return str(cfg.get("qdrant.collection", "corpus"))


def _index_type(cfg: Config) -> str:
    # Map FAISS index.type to Qdrant mode for fair comparison.
    t = cfg.get("index.type", "flat")
    if t in ("ivfpq", "opq"):
        return "scalar"
    return t


def build_index(cfg: Config, emb: np.ndarray, ids: np.ndarray | List[int]) -> Tuple[object, Dict]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        PointStruct,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        VectorParams,
    )

    emb = np.ascontiguousarray(emb, dtype=np.float32)
    n, d = emb.shape
    ids_arr = np.asarray(ids, dtype=np.int64)
    itype = _index_type(cfg)
    storage = _storage_path(cfg)
    collection = _collection_name(cfg)

    if os.path.isdir(storage):
        shutil.rmtree(storage, ignore_errors=True)
    os.makedirs(os.path.dirname(storage) or ".", exist_ok=True)

    client = QdrantClient(path=storage)
    t0 = time.time()

    hnsw_cfg = None
    quant_cfg = None
    if itype == "hnsw":
        hnsw_cfg = HnswConfigDiff(
            m=int(cfg.get("qdrant.hnsw.m", cfg.get("index.hnsw.M", 32))),
            ef_construct=int(cfg.get("qdrant.hnsw.ef_construct",
                                     cfg.get("index.hnsw.ef_construction", 200))),
            full_scan_threshold=int(cfg.get("qdrant.hnsw.full_scan_threshold", 10000)),
        )
    elif itype == "scalar":
        qtype = str(cfg.get("qdrant.quantization.scalar_type", "int8")).lower()
        stype = ScalarType.INT8 if qtype == "int8" else ScalarType.FLOAT16
        quant_cfg = ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=stype, always_ram=True),
        )

    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=d, distance=Distance.COSINE),
        hnsw_config=hnsw_cfg,
        quantization_config=quant_cfg,
    )

    batch = int(cfg.get("qdrant.upload_batch", 512))
    for start in range(0, n, batch):
        end = min(start + batch, n)
        points = [
            PointStruct(id=int(ids_arr[i]), vector=emb[i].tolist())
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=points)

    build_seconds = time.time() - t0
    info_obj = client.get_collection(collection)
    size_bytes = _collection_size_bytes(storage)
    info = {
        "type": itype,
        "backend": "qdrant",
        "n": int(n),
        "dim": int(d),
        "build_seconds": round(build_seconds, 3),
        "index_size_bytes": int(size_bytes),
        "bits_per_vector": round(size_bytes * 8 / max(1, n), 2),
        "storage_path": storage,
        "collection": collection,
        "points_count": info_obj.points_count,
    }
    handle = {"client": client, "collection": collection, "ids": ids_arr, "itype": itype}
    return handle, info


def _collection_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def apply_search_params(cfg: Config, store: dict) -> None:
    itype = store.get("itype", _index_type(cfg))
    if itype == "hnsw":
        store["ef_search"] = int(cfg.get("index.hnsw.ef_search", 64))
    if itype in ("ivfpq", "opq", "scalar"):
        store["exact"] = bool(cfg.get("qdrant.search.exact_rerank", False))


def search(cfg: Config, store: dict, queries: np.ndarray, k: int):
    from qdrant_client.models import SearchParams

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    client = store["client"]
    collection = store["collection"]
    itype = store.get("itype", _index_type(cfg))

    exact = itype == "flat"
    ef = int(store.get("ef_search", cfg.get("index.hnsw.ef_search", 64)))
    params = SearchParams(
        hnsw_ef=ef if itype == "hnsw" else None,
        exact=exact,
    )

    scores = np.empty((queries.shape[0], k), dtype=np.float32)
    idx = np.full((queries.shape[0], k), -1, dtype=np.int64)
    id_to_pos = {int(rid): pos for pos, rid in enumerate(store["ids"])}

    for i, q in enumerate(queries):
        response = client.query_points(
            collection_name=collection,
            query=q.tolist(),
            limit=k,
            search_params=params,
        )
        hits = response.points
        for j, hit in enumerate(hits):
            scores[i, j] = hit.score
            idx[i, j] = id_to_pos.get(int(hit.id), -1)
    return scores, idx

"""Vector storage backends (FAISS, Qdrant)."""
from __future__ import annotations

from ..config import Config


def build_index(cfg: Config, emb, ids=None):
    backend = cfg.get("store.backend", "faiss")
    if backend == "faiss":
        from . import faiss_store
        return faiss_store.build_index(cfg, emb)
    if backend == "qdrant":
        from . import qdrant_store
        return qdrant_store.build_index(cfg, emb, ids)
    raise ValueError(f"Unknown store.backend: {backend}")


def search(cfg: Config, store, queries, k: int):
    backend = cfg.get("store.backend", "faiss")
    if backend == "faiss":
        from . import faiss_store
        return faiss_store.search(store, queries, k)
    if backend == "qdrant":
        from . import qdrant_store
        return qdrant_store.search(cfg, store, queries, k)
    raise ValueError(f"Unknown store.backend: {backend}")


def apply_search_params(cfg: Config, store) -> None:
    """Apply runtime search parameters (ef_search, nprobe, exact, etc.)."""
    backend = cfg.get("store.backend", "faiss")
    if backend == "faiss":
        from . import faiss_store
        faiss_store.apply_search_params(cfg, store)
    elif backend == "qdrant":
        from . import qdrant_store
        qdrant_store.apply_search_params(cfg, store)

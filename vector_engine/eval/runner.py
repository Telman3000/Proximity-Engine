"""End-to-end experiment runner: corpus -> index -> search -> metrics + CIs."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict

import numpy as np

from ..config import Config
from ..data import corpus as corpus_mod
from ..represent import encoder
from ..search import pipeline
from ..store import faiss_store
from . import bootstrap, metrics


def _peak_rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        return None


def _tag(cfg: Config) -> str:
    t = cfg.get("index.type", "flat")
    if cfg.get("search.cascade.enabled", False):
        t += "_cascade"
    return t


def run_experiment(cfg: Config) -> Dict:
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    df = corpus_mod.load_corpus(cfg)
    queries = corpus_mod.load_queries(cfg)
    ids = df["id"].to_numpy()

    emb_meta = encoder.embed_corpus(cfg, df["text"].tolist())
    corpus_emb = encoder.load_corpus_embeddings(cfg)

    index, index_info = faiss_store.build_index(cfg, corpus_emb)

    q_texts = [q["text"] for q in queries]
    t0 = time.perf_counter()
    query_emb = encoder.embed_queries(cfg, q_texts)
    query_encode_ms = (time.perf_counter() - t0) * 1000.0

    retrieved, latencies = pipeline.run_queries(
        cfg, index, query_emb, ids, corpus_emb=corpus_emb)

    relevant = [set(q["relevant_ids"]) for q in queries]
    ks = list(cfg.get("eval.ks", [1, 5, 10]))
    per_query = metrics.evaluate(retrieved, relevant, ks)

    boot_cfg = cfg.get("eval.bootstrap", {}) or {}
    if boot_cfg.get("enabled", True):
        summary = bootstrap.summarize(
            per_query,
            n_resamples=int(boot_cfg.get("n_resamples", 1000)),
            level=float(boot_cfg.get("ci", 0.95)),
            seed=seed,
        )
    else:
        summary = {k: {"mean": round(float(v.mean()), 4)} for k, v in per_query.items()}

    lat = np.sort(latencies)
    latency = {
        "mean_ms": round(float(lat.mean()), 3),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "query_encode_ms_total": round(query_encode_ms, 1),
        "n_queries": int(len(queries)),
    }

    result = {
        "tag": _tag(cfg),
        "config": {
            "embed": cfg.get("embed"),
            "index": {"type": cfg.get("index.type"), "metric": cfg.get("index.metric")},
            "search": cfg.get("search"),
            "corpus_size": int(len(df)),
            "seed": seed,
        },
        "embedding": {k: emb_meta[k] for k in ("backend", "model", "dim", "n_vectors", "encode_seconds") if k in emb_meta},
        "index_info": index_info,
        "metrics": summary,
        "latency": latency,
        "peak_rss_mb": _peak_rss_mb(),
    }

    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"run_{_tag(cfg)}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

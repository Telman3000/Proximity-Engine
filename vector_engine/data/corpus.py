"""Build and persist the retrieval corpus + evaluation queries."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import pandas as pd

from ..config import Config
from . import ingest_real, synthetic


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build(cfg: Config) -> Dict:
    seed = cfg.get("seed", 42)
    target = int(cfg.get("corpus.target_size", 20000))
    frac = float(cfg.get("corpus.synthetic_fraction", 1.0))
    mult_min = int(cfg.get("corpus.key_multiplicity_min", 3))
    mult_max = int(cfg.get("corpus.key_multiplicity_max", 25))
    n_queries = int(cfg.get("queries.n_queries", 500))

    n_synth = int(round(target * frac))
    n_real_cap = max(0, target - n_synth)

    syn_records, key_to_ids = synthetic.generate(
        target_size=n_synth, mult_min=mult_min, mult_max=mult_max, seed=seed,
        start_id=0,
    )

    real_records: List[Dict] = []
    scenes_dir = cfg.get("paths.real_scenes_dir", "")
    if scenes_dir and not os.path.isabs(scenes_dir):
        scenes_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scenes_dir))
    if n_real_cap > 0:
        real_records = ingest_real.ingest(
            scenes_dir, start_id=len(syn_records), limit=n_real_cap)

    records = syn_records + real_records
    queries = synthetic.make_queries(key_to_ids, n_queries=n_queries, seed=seed)

    # ---- persist ----------------------------------------------------------
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    _ensure_dir(art)
    _ensure_dir(out)

    df = pd.DataFrame(records)
    corpus_path = os.path.join(art, "corpus.parquet")
    df.to_parquet(corpus_path, index=False)

    queries_path = os.path.join(art, "queries.json")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False)

    stats = {
        "n_records": len(records),
        "n_synthetic": len(syn_records),
        "n_real": len(real_records),
        "n_keys": len(key_to_ids),
        "n_queries": len(queries),
        "avg_relevant_per_query": round(
            sum(len(q["relevant_ids"]) for q in queries) / max(1, len(queries)), 2),
        "real_scenes_dir": scenes_dir,
        "corpus_path": corpus_path,
        "queries_path": queries_path,
    }
    with open(os.path.join(out, "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def load_corpus(cfg: Config) -> pd.DataFrame:
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    return pd.read_parquet(os.path.join(art, "corpus.parquet"))


def load_queries(cfg: Config) -> List[Dict]:
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    with open(os.path.join(art, "queries.json"), "r", encoding="utf-8") as f:
        return json.load(f)

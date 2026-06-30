"""Compare FAISS vs Qdrant on the same embeddings and queries."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from ..config import load_config
from ..eval import runner
from .run_all import _plots


STORE_EXPERIMENTS = [
    {"name": "faiss_flat", "ov": ["store.backend=faiss", "index.type=flat"]},
    {"name": "faiss_hnsw", "ov": ["store.backend=faiss", "index.type=hnsw"]},
    {"name": "faiss_ivfpq", "ov": ["store.backend=faiss", "index.type=ivfpq"]},
    {"name": "qdrant_flat", "ov": ["store.backend=qdrant", "index.type=flat"]},
    {"name": "qdrant_hnsw", "ov": ["store.backend=qdrant", "index.type=hnsw"]},
    {"name": "qdrant_scalar", "ov": ["store.backend=qdrant", "index.type=ivfpq"]},
]


def _row(name: str, res: Dict) -> Dict:
    m = res["metrics"]
    ii = res["index_info"]
    lat = res["latency"]
    r10 = m.get("recall@10", {})
    return {
        "name": name,
        "store_backend": res["config"].get("store", {}).get("backend", "faiss"),
        "index_type": res["config"]["index"]["type"],
        "bits_per_vector": ii["bits_per_vector"],
        "index_size_mb": round(ii["index_size_bytes"] / 1e6, 3),
        "build_seconds": ii["build_seconds"],
        "recall@10": r10.get("mean"),
        "recall@10_lo": r10.get("lo"),
        "recall@10_hi": r10.get("hi"),
        "p50_ms": lat["p50_ms"],
        "p95_ms": lat["p95_ms"],
    }


def run(config_path=None, overrides=None, experiments=None) -> Dict:
    overrides = overrides or []
    experiments = experiments or STORE_EXPERIMENTS
    base = load_config(config_path, overrides)
    out = base.path(base.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)

    rows: List[Dict] = []
    full: Dict[str, Dict] = {}
    for exp in experiments:
        cfg = load_config(config_path, overrides + exp["ov"])
        res = runner.run_experiment(cfg)
        rows.append(_row(exp["name"], res))
        full[exp["name"]] = res
        print(f"[store] {exp['name']:16} backend={res['config']['store']['backend']:6} "
              f"bits={res['index_info']['bits_per_vector']:>7} "
              f"recall@10={res['metrics']['recall@10']['mean']:.4f}")

    with open(os.path.join(out, "store_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    cols = list(rows[0].keys())
    with open(os.path.join(out, "store_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    _plots(out, rows)
    return {"rows": rows, "out_dir": out}

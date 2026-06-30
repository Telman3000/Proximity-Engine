"""Compare neural vs tfidf-svd embedding backends on flat exact search."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from ..config import load_config
from ..eval import runner


def _row(name: str, res: Dict) -> Dict:
    m = res["metrics"]
    emb = res.get("embedding", {})
    r10 = m.get("recall@10", {})
    return {
        "name": name,
        "backend": emb.get("backend"),
        "model": emb.get("model"),
        "dim": emb.get("dim"),
        "encode_seconds": emb.get("encode_seconds"),
        "recall@10": r10.get("mean"),
        "recall@10_lo": r10.get("lo"),
        "recall@10_hi": r10.get("hi"),
        "map": m.get("map", {}).get("mean"),
        "p95_ms": res["latency"]["p95_ms"],
    }


def run(config_path=None, overrides=None) -> Dict:
    cfg = load_config(config_path, overrides)
    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)

    backends = cfg.get("experiments.embedding_backends", [
        {"name": "tfidf-svd", "ov": ["embed.backend=tfidf-svd"]},
        {"name": "fastembed", "ov": ["embed.backend=fastembed"]},
        {"name": "sentence-transformers",
         "ov": ["embed.backend=sentence-transformers", "embed.model=sentence-transformers/all-MiniLM-L6-v2"]},
    ])

    rows: List[Dict] = []
    full: Dict[str, Dict] = {}
    base_ov = list(overrides or []) + ["index.type=flat", "store.backend=faiss"]

    for spec in backends:
        name = spec["name"]
        try:
            res = runner.run_experiment(load_config(config_path, base_ov + spec["ov"]))
        except Exception as exc:
            print(f"[embed] {name:22} SKIPPED ({exc})")
            rows.append({"name": name, "backend": name, "error": str(exc)})
            continue
        rows.append(_row(name, res))
        full[name] = res
        print(f"[embed] {name:22} recall@10={res['metrics']['recall@10']['mean']:.4f}")

    with open(os.path.join(out, "embedding_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    ok_rows = [r for r in rows if "error" not in r]
    if ok_rows:
        with open(os.path.join(out, "embedding_comparison.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ok_rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return {"rows": rows, "out_dir": out}

"""Run the full index ladder and produce the comparative study.

Outputs:
  outputs/comparison.csv          tabular summary (one row per index variant)
  outputs/comparison.json         full results
  outputs/compression_tradeoff.png   bits/vector vs recall@10 (with CIs)
  outputs/latency_tradeoff.png       p95 latency vs recall@10
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from ..config import load_config
from ..eval import runner

DEFAULT_EXPERIMENTS = [
    {"name": "flat", "ov": ["index.type=flat"]},
    {"name": "hnsw", "ov": ["index.type=hnsw"]},
    {"name": "ivfpq", "ov": ["index.type=ivfpq"]},
    {"name": "ivfpq_cascade", "ov": ["index.type=ivfpq", "search.cascade.enabled=true"]},
    {"name": "opq", "ov": ["index.type=opq"]},
]


def _row(name: str, res: Dict) -> Dict:
    m = res["metrics"]
    ii = res["index_info"]
    lat = res["latency"]

    def cell(metric):
        d = m.get(metric, {})
        return d.get("mean"), d.get("lo"), d.get("hi")

    r10 = cell("recall@10")
    row = {
        "name": name,
        "index_type": res["config"]["index"]["type"],
        "cascade": res["config"]["search"]["cascade"]["enabled"],
        "bits_per_vector": ii["bits_per_vector"],
        "index_size_mb": round(ii["index_size_bytes"] / 1e6, 3),
        "build_seconds": ii["build_seconds"],
        "recall@1": m.get("recall@1", {}).get("mean"),
        "recall@5": m.get("recall@5", {}).get("mean"),
        "recall@10": r10[0],
        "recall@10_lo": r10[1],
        "recall@10_hi": r10[2],
        "map": m.get("map", {}).get("mean"),
        "ndcg@10": m.get("ndcg@10", {}).get("mean"),
        "p50_ms": lat["p50_ms"],
        "p95_ms": lat["p95_ms"],
    }
    return row


def run(config_path=None, overrides=None, experiments=None) -> Dict:
    overrides = overrides or []
    experiments = experiments or DEFAULT_EXPERIMENTS
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
        print(f"[done] {exp['name']:16} bits/vec={res['index_info']['bits_per_vector']:>9} "
              f"recall@10={res['metrics']['recall@10']['mean']:.4f} "
              f"p95={res['latency']['p95_ms']}ms")

    with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    cols = list(rows[0].keys())
    with open(os.path.join(out, "comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    _plots(out, rows)
    return {"rows": rows, "out_dir": out}


def _plots(out: str, rows: List[Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compression trade-off: bits/vector vs recall@10 (with CI bars)
    plt.figure(figsize=(7, 4.5))
    for r in rows:
        yerr = [[r["recall@10"] - r["recall@10_lo"]], [r["recall@10_hi"] - r["recall@10"]]]
        plt.errorbar(r["bits_per_vector"], r["recall@10"], yerr=yerr, fmt="o",
                     capsize=4, ms=7)
        plt.annotate(r["name"], (r["bits_per_vector"], r["recall@10"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("bits per vector (index size / n)")
    plt.ylabel("recall@10 (95% CI)")
    plt.title("Storage vs quality trade-off")
    plt.grid(True, ls="--", lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "compression_tradeoff.png"), dpi=130)
    plt.close()

    # Latency trade-off
    plt.figure(figsize=(7, 4.5))
    for r in rows:
        plt.scatter(r["p95_ms"], r["recall@10"], s=55)
        plt.annotate(r["name"], (r["p95_ms"], r["recall@10"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("p95 search latency (ms)")
    plt.ylabel("recall@10")
    plt.title("Latency vs quality trade-off")
    plt.grid(True, ls="--", lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "latency_tradeoff.png"), dpi=130)
    plt.close()

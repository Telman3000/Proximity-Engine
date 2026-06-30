"""Parameter sweeps for the storage-compression and latency/recall studies.

  sweep-pq    : index.ivfpq.m × index.ivfpq.nbits  -> bits/vector vs recall curve
  sweep-hnsw  : index.hnsw.M × index.hnsw.ef_search -> Pareto latency/recall
  sweep-ivf   : index.ivfpq.nprobe                  -> recall vs latency trade-off
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

from ..config import load_config
from ..eval import runner


def _metric_cell(res: Dict, name: str) -> Tuple[float | None, float | None, float | None]:
    d = res["metrics"].get(name, {})
    return d.get("mean"), d.get("lo"), d.get("hi")


def _base_row(name: str, res: Dict, extra: Dict | None = None) -> Dict:
    m = res["metrics"]
    ii = res["index_info"]
    lat = res["latency"]
    r10 = _metric_cell(res, "recall@10")
    row = {
        "name": name,
        "index_type": res["config"]["index"]["type"],
        "store_backend": res["config"].get("store", {}).get("backend", "faiss"),
        "bits_per_vector": ii["bits_per_vector"],
        "index_size_mb": round(ii["index_size_bytes"] / 1e6, 3),
        "build_seconds": ii["build_seconds"],
        "recall@10": r10[0],
        "recall@10_lo": r10[1],
        "recall@10_hi": r10[2],
        "map": m.get("map", {}).get("mean"),
        "ndcg@10": m.get("ndcg@10", {}).get("mean"),
        "p50_ms": lat["p50_ms"],
        "p95_ms": lat["p95_ms"],
    }
    if extra:
        row.update(extra)
    return row


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _valid_pq_m(dim: int, m_values: Iterable[int]) -> List[int]:
    out = []
    for m in m_values:
        m = int(m)
        if m > 0 and dim % m == 0:
            out.append(m)
    return out


def sweep_pq(config_path=None, overrides=None) -> Dict:
    cfg = load_config(config_path, overrides)
    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)

    dim = int(cfg.get("embed.tfidf.dim", 256))
    m_values = _valid_pq_m(dim, cfg.get("sweeps.pq.m", [8, 16, 32, 64]))
    nbits_values = [int(x) for x in cfg.get("sweeps.pq.nbits", [4, 6, 8])]

    rows: List[Dict] = []
    full: Dict[str, Dict] = {}
    for m in m_values:
        for nbits in nbits_values:
            name = f"ivfpq_m{m}_nb{nbits}"
            ov = list(overrides or []) + [
                "index.type=ivfpq",
                f"index.ivfpq.m={m}",
                f"index.ivfpq.nbits={nbits}",
            ]
            res = runner.run_experiment(load_config(config_path, ov))
            rows.append(_base_row(name, res, {"m": m, "nbits": nbits}))
            full[name] = res
            print(f"[pq] {name:18} bits={res['index_info']['bits_per_vector']:>7} "
                  f"recall@10={res['metrics']['recall@10']['mean']:.4f}")

    _write_csv(os.path.join(out, "sweep_pq.csv"), rows)
    with open(os.path.join(out, "sweep_pq.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    _plot_pq(out, rows)
    return {"rows": rows, "out_dir": out}


def sweep_hnsw(config_path=None, overrides=None) -> Dict:
    cfg = load_config(config_path, overrides)
    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)

    m_values = [int(x) for x in cfg.get("sweeps.hnsw.M", [16, 32, 64])]
    ef_values = [int(x) for x in cfg.get("sweeps.hnsw.ef_search", [16, 32, 64, 128])]

    rows: List[Dict] = []
    full: Dict[str, Dict] = {}
    for m in m_values:
        for ef in ef_values:
            name = f"hnsw_M{m}_ef{ef}"
            ov = list(overrides or []) + [
                "index.type=hnsw",
                f"index.hnsw.M={m}",
                f"index.hnsw.ef_search={ef}",
            ]
            res = runner.run_experiment(load_config(config_path, ov))
            rows.append(_base_row(name, res, {"M": m, "ef_search": ef}))
            full[name] = res
            print(f"[hnsw] {name:18} recall@10={res['metrics']['recall@10']['mean']:.4f} "
                  f"p95={res['latency']['p95_ms']}ms")

    _write_csv(os.path.join(out, "sweep_hnsw.csv"), rows)
    with open(os.path.join(out, "sweep_hnsw.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    _plot_pareto(out, rows, "sweep_hnsw_pareto.png", "HNSW: latency vs recall")
    return {"rows": rows, "out_dir": out}


def sweep_ivf(config_path=None, overrides=None) -> Dict:
    cfg = load_config(config_path, overrides)
    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)

    nprobe_values = [int(x) for x in cfg.get("sweeps.ivf.nprobe", [1, 4, 8, 16, 32, 64])]

    rows: List[Dict] = []
    full: Dict[str, Dict] = {}
    for nprobe in nprobe_values:
        name = f"ivfpq_nprobe{nprobe}"
        ov = list(overrides or []) + [
            "index.type=ivfpq",
            f"index.ivfpq.nprobe={nprobe}",
        ]
        res = runner.run_experiment(load_config(config_path, ov))
        rows.append(_base_row(name, res, {"nprobe": nprobe}))
        full[name] = res
        print(f"[ivf] {name:18} recall@10={res['metrics']['recall@10']['mean']:.4f} "
              f"p95={res['latency']['p95_ms']}ms")

    _write_csv(os.path.join(out, "sweep_ivf.csv"), rows)
    with open(os.path.join(out, "sweep_ivf.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    _plot_pareto(out, rows, "sweep_ivf_pareto.png", "IVF-PQ: nprobe latency vs recall")
    return {"rows": rows, "out_dir": out}


def _plot_pq(out: str, rows: List[Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for r in rows:
        yerr = [[r["recall@10"] - r["recall@10_lo"]], [r["recall@10_hi"] - r["recall@10"]]]
        plt.errorbar(r["bits_per_vector"], r["recall@10"], yerr=yerr, fmt="o", capsize=4, ms=6)
        plt.annotate(f"m={r['m']},nb={r['nbits']}",
                     (r["bits_per_vector"], r["recall@10"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("bits per vector")
    plt.ylabel("recall@10 (95% CI)")
    plt.title("PQ quantisation: storage vs quality")
    plt.grid(True, ls="--", lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "sweep_pq_curve.png"), dpi=130)
    plt.close()


def _plot_pareto(out: str, rows: List[Dict], fname: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for r in rows:
        plt.scatter(r["p95_ms"], r["recall@10"], s=50)
        plt.annotate(r["name"], (r["p95_ms"], r["recall@10"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("p95 search latency (ms)")
    plt.ylabel("recall@10")
    plt.title(title)
    plt.grid(True, ls="--", lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out, fname), dpi=130)
    plt.close()

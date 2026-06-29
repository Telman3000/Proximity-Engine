"""Analyse the embedding space to justify dimension shrinking.

Runs on the already-computed corpus embedding matrix (backend-agnostic): reports
norm statistics and the PCA explained-variance spectrum, i.e. how many
dimensions are actually needed. Directly supports the "analyse vectors and
shrink the model" requirement.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from ..config import Config
from . import encoder


def run(cfg: Config) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)

    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    emb_path = os.path.join(art, "corpus_emb.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            "corpus_emb.npy not found. Run `embed` before `analyze`.")
    emb = np.load(emb_path).astype(np.float32)

    sample = int(cfg.get("analyze.sample", 20000))
    if emb.shape[0] > sample:
        rng = np.random.RandomState(int(cfg.get("seed", 42)))
        idx = rng.choice(emb.shape[0], size=sample, replace=False)
        emb = emb[idx]

    norms = np.linalg.norm(emb, axis=1)
    max_comp = int(min(emb.shape[0], emb.shape[1]))
    pca = PCA(n_components=max_comp, random_state=int(cfg.get("seed", 42)))
    pca.fit(emb)
    cum = np.cumsum(pca.explained_variance_ratio_)

    def dim_for(th: float) -> int:
        return int(np.searchsorted(cum, th) + 1)

    dims = {f"var_{int(th*100)}": dim_for(th) for th in (0.90, 0.95, 0.99)}

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum, color="#1f6fb2")
    for th in (0.90, 0.95, 0.99):
        plt.axhline(th, ls="--", lw=0.7, color="#999999")
        plt.axvline(dim_for(th), ls="--", lw=0.7, color="#0e7c66")
    plt.xlabel("PCA components")
    plt.ylabel("cumulative explained variance")
    plt.title("Embedding spectrum")
    plt.tight_layout()
    plot_path = os.path.join(out, "vector_analysis.png")
    plt.savefig(plot_path, dpi=130)
    plt.close()

    info = {
        "n_analysed": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "effective_dim": dims,
        "plot_path": plot_path,
    }
    with open(os.path.join(out, "vector_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info

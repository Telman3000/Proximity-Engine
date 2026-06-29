"""CPU text encoder with disk caching and pluggable backends.

Backends (embed.backend):
  - tfidf-svd            : pure scikit-learn (no native ML DLLs). Always works,
                           fully CPU, deterministic. Default on this machine
                           because torch / onnxruntime fail to load under the
                           Microsoft Store Python sandbox.
  - fastembed            : ONNX runtime (no torch). Best quality/speed when the
                           native DLLs load.
  - sentence-transformers: torch backend.

Pipeline:
    text -> [backend encode/transform] -> [optional PCA] -> [optional L2-norm]

Trainable backends (tfidf-svd, and PCA reduction) persist their fitted state so
queries are transformed identically to the corpus.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np

from ..config import Config

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MODEL_CACHE: Dict[str, object] = {}


def _neural_model(backend: str, name: str):
    key = f"{backend}:{name}"
    if key not in _MODEL_CACHE:
        if backend == "fastembed":
            from fastembed import TextEmbedding
            _MODEL_CACHE[key] = TextEmbedding(model_name=name)
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[key] = SentenceTransformer(name, device="cpu")
        else:
            raise ValueError(f"Unknown neural backend: {backend}")
    return _MODEL_CACHE[key]


def _texts_hash(texts: List[str]) -> str:
    h = hashlib.sha1()
    h.update(str(len(texts)).encode())
    step = max(1, len(texts) // 1000)
    for t in texts[::step]:
        h.update(t.encode("utf-8", "ignore"))
    if texts:
        h.update(texts[0].encode("utf-8", "ignore"))
        h.update(texts[-1].encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _cache_key(cfg: Config, texts: List[str]) -> str:
    parts = [
        cfg.get("embed.backend", "tfidf-svd"),
        cfg.get("embed.model", ""),
        str(cfg.get("embed.normalize", True)),
        str(cfg.get("embed.dim_reduction.method", "none")),
        str(cfg.get("embed.dim_reduction.dim", 0)),
        str(cfg.get("embed.tfidf.dim", 0)),
        str(cfg.get("embed.tfidf.max_features", 0)),
        str(cfg.get("embed.tfidf.ngram_max", 1)),
        _texts_hash(texts),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _neural_encode(cfg: Config, texts: List[str]) -> np.ndarray:
    backend = cfg.get("embed.backend")
    bs = int(cfg.get("embed.batch_size", 256))
    model = _neural_model(backend, cfg.get("embed.model"))
    if backend == "fastembed":
        emb = np.asarray(list(model.embed(texts, batch_size=bs)), dtype=np.float32)
    else:
        emb = model.encode(texts, batch_size=bs, convert_to_numpy=True,
                           show_progress_bar=len(texts) > 5000,
                           normalize_embeddings=False)
    return np.asarray(emb, dtype=np.float32)


def _encode_raw(cfg: Config, texts: List[str]) -> np.ndarray:
    """Raw (pre-reduction) embeddings. For analysis / neural backends only."""
    backend = cfg.get("embed.backend", "tfidf-svd")
    if backend == "tfidf-svd":
        return _tfidf_fit_transform(cfg, texts, fit=True)[0]
    return _neural_encode(cfg, texts)


# ---- tfidf-svd backend ------------------------------------------------------
def _tfidf_paths(cfg: Config):
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    return os.path.join(art, "tfidf.joblib"), os.path.join(art, "svd.joblib")


def _tfidf_fit_transform(cfg: Config, texts: List[str], fit: bool):
    import joblib
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_path, svd_path = _tfidf_paths(cfg)
    if fit:
        vec = TfidfVectorizer(
            max_features=int(cfg.get("embed.tfidf.max_features", 50000)),
            ngram_range=(1, int(cfg.get("embed.tfidf.ngram_max", 2))),
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts)
        dim = int(cfg.get("embed.tfidf.dim", 256))
        dim = min(dim, X.shape[1] - 1, max(2, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=dim, random_state=int(cfg.get("seed", 42)))
        emb = svd.fit_transform(X).astype(np.float32)
        joblib.dump(vec, tfidf_path)
        joblib.dump(svd, svd_path)
        return emb, dim
    else:
        vec = joblib.load(tfidf_path)
        svd = joblib.load(svd_path)
        emb = svd.transform(vec.transform(texts)).astype(np.float32)
        return emb, emb.shape[1]


# ---- public API -------------------------------------------------------------
def embed_corpus(cfg: Config, texts: List[str]) -> Dict:
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    os.makedirs(art, exist_ok=True)
    emb_path = os.path.join(art, "corpus_emb.npy")
    pca_path = os.path.join(art, "pca.joblib")
    meta_path = os.path.join(art, "emb_meta.json")

    key = _cache_key(cfg, texts)
    if os.path.exists(emb_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("cache_key") == key:
            meta["cached"] = True
            return meta

    backend = cfg.get("embed.backend", "tfidf-svd")
    t0 = time.time()

    if backend == "tfidf-svd":
        final, out_dim = _tfidf_fit_transform(cfg, texts, fit=True)
        raw_dim = out_dim
        if os.path.exists(pca_path):
            os.remove(pca_path)
    else:
        raw = _neural_encode(cfg, texts)
        raw_dim = int(raw.shape[1])
        final = raw
        out_dim = raw_dim
        method = cfg.get("embed.dim_reduction.method", "none")
        if method == "pca":
            from sklearn.decomposition import PCA
            import joblib
            dim = int(cfg.get("embed.dim_reduction.dim", 128))
            dim = min(dim, raw_dim, raw.shape[0])
            pca = PCA(n_components=dim, random_state=int(cfg.get("seed", 42)))
            final = pca.fit_transform(raw).astype(np.float32)
            out_dim = dim
            joblib.dump(pca, pca_path)
        elif os.path.exists(pca_path):
            os.remove(pca_path)

    if cfg.get("embed.normalize", True):
        final = _l2norm(final)
    final = np.ascontiguousarray(final, dtype=np.float32)

    np.save(emb_path, final)
    meta = {
        "cache_key": key,
        "backend": backend,
        "model": cfg.get("embed.model") if backend != "tfidf-svd" else "tfidf-svd",
        "n_vectors": int(final.shape[0]),
        "raw_dim": raw_dim,
        "dim": int(final.shape[1]),
        "normalize": bool(cfg.get("embed.normalize", True)),
        "dim_reduction": cfg.get("embed.dim_reduction.method", "none"),
        "encode_seconds": round(time.time() - t0, 2),
        "emb_path": emb_path,
        "cached": False,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_corpus_embeddings(cfg: Config) -> np.ndarray:
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    return np.load(os.path.join(art, "corpus_emb.npy"))


def embed_queries(cfg: Config, texts: List[str]) -> np.ndarray:
    backend = cfg.get("embed.backend", "tfidf-svd")
    if backend == "tfidf-svd":
        final, _ = _tfidf_fit_transform(cfg, texts, fit=False)
    else:
        raw = _neural_encode(cfg, texts)
        final = raw
        art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
        pca_path = os.path.join(art, "pca.joblib")
        if cfg.get("embed.dim_reduction.method", "none") == "pca" and os.path.exists(pca_path):
            import joblib
            pca = joblib.load(pca_path)
            final = pca.transform(raw).astype(np.float32)
    if cfg.get("embed.normalize", True):
        final = _l2norm(final)
    return np.ascontiguousarray(final, dtype=np.float32)

# Proximity-Engine

A scalable, **compressed vector-search engine** for semantic retrieval, built for
the DLS course project. It implements the full pipeline — **representation model →
vector storage → indexing → cascaded search** — and evaluates every stage with
**bootstrap confidence intervals**, focusing on **storage economy** (fewer bits
per indexed vector).

It is the thematic counterpart to the `beyond-proximity` research project: where
that work goes *beyond* embedding proximity, this engine builds the proper,
scalable *proximity* (vector) search arm — a strong, compressed retrieval baseline
plus a storage-compression study.

## Highlights

- Iteration ladder: `flat` (exact) → `hnsw` → `ivfpq` / `opq` (quantisation) →
  `ivfpq + cascade` (coarse ANN → exact re-rank).
- Scales to **500k+** vectors; corpus = parametric synthetic records with exact
  ground truth (+ optional real VLM-JSON anchor records).
- Every metric reported with 95% bootstrap CIs; storage measured as **bits/vector**.
- Fully parametrised via YAML; reproducible (fixed seed).

## Results (CPU, `tfidf-svd` 256-dim embeddings)

500k records — compression is the headline:

| Index | bits/vector | recall@10 | p95 latency |
|---|---|---|---|
| flat (exact) | 8192 | 0.257 | 14.9 ms |
| hnsw | 10369 | 0.224 | 0.37 ms |
| ivfpq | **328** (×25) | 0.210 | 0.79 ms |
| ivfpq + cascade | 328 | 0.217 | 1.15 ms |
| opq | 333 | 0.197 | 0.77 ms |

The quantised index is **~25× smaller** than exact while the cascade recovers part
of the recall lost to compression. (Absolute recall is bounded by the lightweight
`tfidf-svd` representation; swapping to a neural backend is a one-line config change
— see Environment note.)

## Install

```powershell
pip install -r vector_engine/requirements.txt
```

## Usage (run from the repo root)

```powershell
python -m vector_engine.cli build-corpus            # corpus + eval queries
python -m vector_engine.cli embed                   # encode once (disk-cached)
python -m vector_engine.cli analyze                 # PCA spectrum -> dim shrinking
python -m vector_engine.cli run --set index.type=ivfpq
python -m vector_engine.cli compare                 # full ladder + plots
```

Scale up:

```powershell
python -m vector_engine.cli build-corpus --set corpus.target_size=500000
python -m vector_engine.cli compare       --set corpus.target_size=500000
```

Override any config key with `--set key.subkey=value`. Outputs land in
`vector_engine/outputs/` (`comparison.csv`, `run_*.json`, trade-off plots).

## Environment note

The default embedding backend is **`tfidf-svd`** (pure scikit-learn, no native ML
DLLs) so the engine runs anywhere on CPU. Neural backends (`fastembed`,
`sentence-transformers`) are available — set `embed.backend` in
`vector_engine/configs/default.yaml` once `torch`/`onnxruntime` load in your
environment. On Windows, `faiss` is imported first (in `vector_engine/__init__.py`)
to avoid a native runtime conflict with `pandas`/`pyarrow`.

## Layout

```
vector_engine/
  config.py            YAML config + dotted CLI overrides
  configs/default.yaml all parameters
  data/                synthetic generator, real ingest, corpus builder
  represent/           CPU encoder + PCA analysis
  store/               FAISS indexes (flat/hnsw/ivfpq/opq)
  search/              single-stage + coarse->rerank cascade
  eval/                metrics, bootstrap CIs, runner
  experiments/         comparative study + plots
  cli.py               entry point
```

## License

MIT — see [LICENSE](LICENSE).

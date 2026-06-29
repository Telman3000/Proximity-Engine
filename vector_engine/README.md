# DLS Vector-Search Engine

Scalable, compressed semantic-retrieval engine for the DLS course project:
**representation model → vector storage → indexing → cascaded search**, evaluated
with **bootstrap confidence intervals**. Thematically connected to the
`beyond-proximity` research project (it consumes the scenes' VLM-JSON object
records as "anchor" data and serves as a proper, scalable embedding-retrieval
baseline + storage-compression study).

## Why this design

The course wants a vector-search engineering project at scale (≥50k, target
~500k) with a focus on storage economy (fewer bits per indexed row) and rigorous
evaluation. `beyond-proximity` deliberately avoids vector similarity search, so
this module builds that missing arm cleanly and independently.

## Environment notes (important)

This machine uses **Microsoft Store Python**, whose sandbox breaks several native
ML wheels:

- `torch` and `onnxruntime` fail to load their DLLs → the neural embedding
  backends (`sentence-transformers`, `fastembed`) do not run here.
- `faiss` works, but its native runtime conflicts with `pandas`/`pyarrow` unless
  **faiss is imported first** (handled automatically in `vector_engine/__init__.py`).

Therefore the **default embedding backend is `tfidf-svd`** (pure scikit-learn,
deterministic, fully CPU). It is a strong fit for this benchmark because queries
and records share attribute words. Switch to a neural backend via config once a
non-Store Python (or a fixed environment) is available.

## Install

```powershell
pip install -r vector_engine/requirements.txt
```

## Run (from the DLS_proj root)

```powershell
# 1) Build the corpus + evaluation queries (synthetic + real VLM-JSON anchors)
python -m vector_engine.cli build-corpus

# 2) Encode the corpus once (cached on disk for all later experiments)
python -m vector_engine.cli embed

# 3) Analyse the embedding space (PCA spectrum -> dimension shrinking)
python -m vector_engine.cli analyze

# 4) Run a single experiment
python -m vector_engine.cli run --set index.type=ivfpq

# 5) Run the full comparative study (flat / hnsw / ivfpq / ivfpq+cascade / opq)
python -m vector_engine.cli compare
```

### Scaling up (the professor wants ~500k)

```powershell
python -m vector_engine.cli build-corpus --set corpus.target_size=500000
python -m vector_engine.cli compare       --set corpus.target_size=500000
```

Everything is parametrised; override any config key with `--set key.subkey=value`.

## Outputs

- `outputs/corpus_stats.json` — corpus composition
- `outputs/vector_analysis.{json,png}` — embedding spectrum / effective dim
- `outputs/run_<index>.json` — per-experiment metrics (recall@k, mAP, nDCG,
  latency p50/p95, bits/vector) with 95% CIs
- `outputs/comparison.{csv,json}` — side-by-side comparative study
- `outputs/compression_tradeoff.png` — bits/vector vs recall@10 (with CIs)
- `outputs/latency_tradeoff.png` — p95 latency vs recall@10

## Iteration ladder (the comparative study)

| Iter | Index | Point |
|---|---|---|
| 1 | `flat` | exact brute force, recall ceiling |
| 2 | `hnsw` | graph ANN, latency/recall |
| 3 | `ivfpq` / `opq` | **quantisation — fewer bits per vector** |
| 4 | `ivfpq + cascade` | coarse ANN → exact re-rank recovers recall |

## Module layout

```
vector_engine/
  config.py            YAML config + dotted CLI overrides
  configs/default.yaml all parameters
  data/                synthetic generator, real VLM-JSON ingest, corpus builder
  represent/           CPU encoder (tfidf-svd | fastembed | sentence-transformers), PCA analysis
  store/               FAISS index construction (flat/hnsw/ivfpq/opq)
  search/              single-stage + coarse→rerank cascade
  eval/                metrics, bootstrap CIs, experiment runner
  experiments/         run_all (comparative study + plots)
  cli.py               entry point
```

# DLS Vector-Search Engine — Proximity-Engine

Scalable, compressed semantic-retrieval engine for the DLS course project:
**representation model → vector storage → indexing → cascaded search**, evaluated
with **bootstrap confidence intervals**. Thematically connected to the
`beyond-proximity` research project.

## Course requirements coverage

| Requirement | Implementation |
|---|---|
| Representation model | `represent/encoder.py` — tfidf-svd (default), fastembed, sentence-transformers |
| Vector storage | FAISS + **Qdrant** (local on-disk) |
| Indexing | flat, HNSW, IVF-PQ, OPQ |
| Search algorithm | single-stage ANN + **coarse→rerank cascade** |
| Scale ≥ 50k (target 500k) | parametric synthetic corpus (`configs/scale_500k.yaml`) |
| Storage economy | **bits/vector** metric on every run |
| Bootstrap CIs | recall@k, precision@k, mAP, nDCG@k |
| Everything parametrised | YAML + `--set key=value` |
| Optional Whisper | `whisper-demo` command |

## Install

```bash
pip install -r vector_engine/requirements.txt
# optional multimodal queries:
pip install -r vector_engine/requirements-whisper.txt
```

## Quick start (from repo root)

```bash
python3 -m vector_engine.cli build-corpus
python3 -m vector_engine.cli embed
python3 -m vector_engine.cli analyze
python3 -m vector_engine.cli compare                 # FAISS index ladder + plots
```

## Full experiment suite

```bash
# Parameter sweeps (central rubric graphs)
python3 -m vector_engine.cli sweep-pq                # bits/vector vs recall curve
python3 -m vector_engine.cli sweep-hnsw              # HNSW Pareto latency/recall
python3 -m vector_engine.cli sweep-ivf               # IVF nprobe Pareto

# Storage ablation: FAISS vs Qdrant
python3 -m vector_engine.cli compare-stores

# Embedding ablation: tfidf-svd vs neural backends (skips if DLLs missing)
python3 -m vector_engine.cli compare-embeddings

# Optional: audio query → Whisper → same index
python3 -m vector_engine.cli whisper-demo
python3 -m vector_engine.cli whisper-demo --audio path/to/query.wav
```

## Scale to 500k

```bash
python3 -m vector_engine.cli build-corpus --config vector_engine/configs/scale_500k.yaml
python3 -m vector_engine.cli embed       --config vector_engine/configs/scale_500k.yaml
python3 -m vector_engine.cli compare     --config vector_engine/configs/scale_500k.yaml
```

Fast smoke-test grid (5k records):

```bash
python3 -m vector_engine.cli sweep-pq --config vector_engine/configs/sweeps_fast.yaml
```

## Outputs (`vector_engine/outputs/`)

| File | Content |
|---|---|
| `comparison.csv/json` | Index ladder (flat/hnsw/ivfpq/cascade/opq) |
| `sweep_pq_curve.png` | **bits/vector vs recall@10** (with CIs) |
| `sweep_hnsw_pareto.png` | HNSW latency vs recall |
| `sweep_ivf_pareto.png` | IVF nprobe trade-off |
| `store_comparison.csv` | FAISS vs Qdrant ablation |
| `embedding_comparison.csv` | tfidf-svd vs neural backends |
| `run_<index>.json` | Per-experiment metrics + CIs + latency |
| `vector_analysis.png` | PCA spectrum / effective dimensionality |

## Environment notes

- Default backend **`tfidf-svd`** — pure scikit-learn, runs on any CPU.
- On Windows Store Python, torch/onnxruntime may fail → use tfidf-svd or install
  python.org CPython for neural backends.
- **faiss is imported first** in `vector_engine/__init__.py` to avoid OpenMP
  conflicts with pandas/pyarrow on Windows.

## Module layout

```
vector_engine/
  config.py / configs/     YAML configs (default, sweeps_fast, scale_500k)
  data/                    synthetic generator, real VLM-JSON ingest
  represent/               encoder, PCA analysis, Whisper ASR
  store/                   FAISS + Qdrant backends
  search/                  single-stage + cascade rerank
  eval/                    metrics, bootstrap CIs, runner
  experiments/             compare, sweeps, store/embedding ablations
  cli.py                   entry point
```

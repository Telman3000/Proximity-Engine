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
- **FAISS + Qdrant** storage backends with ablation study.
- Parameter sweeps: PQ (`m`, `nbits`), HNSW (`M`, `ef_search`), IVF (`nprobe`).
- Scales to **500k+** vectors; corpus = parametric synthetic records with exact
  ground truth (+ optional real VLM-JSON anchor records).
- Every metric reported with 95% bootstrap CIs; storage measured as **bits/vector**.
- Optional **Whisper** audio-query path (multimodal retrieval demo).
- Fully parametrised via YAML; reproducible (fixed seed).

## Results (CPU, `tfidf-svd` 256-dim embeddings)

50k records (verified end-to-end):

| Index | bits/vector | recall@10 | p95 latency |
|---|---|---|---|
| flat (exact) | 8192 | 0.455 | 1.14 ms |
| hnsw | 10370 | 0.438 | 0.13 ms |
| ivfpq | **404** (×20) | 0.346 | 0.06 ms |
| ivfpq + cascade | 404 | 0.349 | 0.10 ms |
| opq | 446 | 0.360 | 0.07 ms |

At 500k the quantised index is **~25× smaller** than exact (see `configs/scale_500k.yaml`).

## Install

```bash
pip install -r vector_engine/requirements.txt
# optional: Whisper multimodal queries
pip install -r vector_engine/requirements-whisper.txt
```

## Usage (run from the repo root)

```bash
python3 -m vector_engine.cli build-corpus            # corpus + eval queries
python3 -m vector_engine.cli embed                   # encode once (disk-cached)
python3 -m vector_engine.cli analyze                 # PCA spectrum -> dim shrinking
python3 -m vector_engine.cli compare                 # FAISS index ladder + plots
```

### Parameter sweeps & ablations

```bash
python3 -m vector_engine.cli sweep-pq                # bits/vector vs recall curve
python3 -m vector_engine.cli sweep-hnsw              # HNSW Pareto latency/recall
python3 -m vector_engine.cli sweep-ivf               # IVF nprobe trade-off
python3 -m vector_engine.cli compare-stores          # FAISS vs Qdrant
python3 -m vector_engine.cli compare-embeddings      # tfidf-svd vs neural backends
python3 -m vector_engine.cli whisper-demo            # optional audio queries
```

### Scale up

```bash
# minimum course scale (50k)
python3 -m vector_engine.cli build-corpus --set corpus.target_size=50000
python3 -m vector_engine.cli compare       --set corpus.target_size=50000

# full benchmark (500k, run overnight on CPU)
python3 -m vector_engine.cli build-corpus --config vector_engine/configs/scale_500k.yaml
python3 -m vector_engine.cli compare       --config vector_engine/configs/scale_500k.yaml
```

Override any config key with `--set key.subkey=value`. Outputs land in
`vector_engine/outputs/` (`comparison.csv`, `sweep_pq_curve.png`, `run_*.json`, …).

See [vector_engine/README.md](vector_engine/README.md) for full documentation.

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
  config.py / configs/   YAML configs (default, sweeps_fast, scale_500k)
  data/                  synthetic generator, real ingest, corpus builder
  represent/             encoder, PCA analysis, Whisper ASR
  store/                 FAISS + Qdrant backends
  search/                single-stage + coarse->rerank cascade
  eval/                  metrics, bootstrap CIs, runner
  experiments/           compare, sweeps, store/embedding ablations
  cli.py                 entry point
```

## License

MIT — see [LICENSE](LICENSE).

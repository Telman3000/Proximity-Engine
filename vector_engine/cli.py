"""Command-line entry point for the DLS vector-search engine.

Usage (run from the repo root):
    python3 -m vector_engine.cli build-corpus [--config PATH] [--set k=v ...]
    python3 -m vector_engine.cli embed
    python3 -m vector_engine.cli analyze
    python3 -m vector_engine.cli run          [--set index.type=ivfpq]
    python3 -m vector_engine.cli compare     # FAISS index ladder
    python3 -m vector_engine.cli sweep-pq     # PQ m×nbits curve
    python3 -m vector_engine.cli sweep-hnsw   # HNSW Pareto
    python3 -m vector_engine.cli sweep-ivf    # IVF nprobe Pareto
    python3 -m vector_engine.cli compare-stores  # FAISS vs Qdrant
    python3 -m vector_engine.cli compare-embeddings
    python3 -m vector_engine.cli whisper-demo
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to YAML config")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Override config: key.subkey=value (repeatable)")


def cmd_build_corpus(args) -> None:
    from .data import corpus
    cfg = load_config(args.config, args.overrides)
    stats = corpus.build(cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_embed(args) -> None:
    from .data import corpus
    from .represent import encoder
    cfg = load_config(args.config, args.overrides)
    df = corpus.load_corpus(cfg)
    info = encoder.embed_corpus(cfg, df["text"].tolist())
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_analyze(args) -> None:
    from .represent import analyze
    cfg = load_config(args.config, args.overrides)
    info = analyze.run(cfg)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_run(args) -> None:
    from .eval import runner
    cfg = load_config(args.config, args.overrides)
    result = runner.run_experiment(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_compare(args) -> None:
    from .experiments import run_all
    out = run_all.run(args.config, args.overrides)
    print(f"\nWrote comparison to: {out['out_dir']}")


def cmd_sweep_pq(args) -> None:
    from .experiments import sweeps
    out = sweeps.sweep_pq(args.config, args.overrides)
    print(f"\nWrote PQ sweep to: {out['out_dir']}")


def cmd_sweep_hnsw(args) -> None:
    from .experiments import sweeps
    out = sweeps.sweep_hnsw(args.config, args.overrides)
    print(f"\nWrote HNSW sweep to: {out['out_dir']}")


def cmd_sweep_ivf(args) -> None:
    from .experiments import sweeps
    out = sweeps.sweep_ivf(args.config, args.overrides)
    print(f"\nWrote IVF sweep to: {out['out_dir']}")


def cmd_compare_stores(args) -> None:
    from .experiments import compare_stores
    out = compare_stores.run(args.config, args.overrides)
    print(f"\nWrote store comparison to: {out['out_dir']}")


def cmd_compare_embeddings(args) -> None:
    from .experiments import compare_embeddings
    out = compare_embeddings.run(args.config, args.overrides)
    print(f"\nWrote embedding comparison to: {out['out_dir']}")


def cmd_whisper_demo(args) -> None:
    from .represent import whisper_asr
    cfg = load_config(args.config, args.overrides)
    try:
        result = whisper_asr.run_demo(cfg, audio_paths=args.audio or None)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vector_engine")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("build-corpus", cmd_build_corpus),
        ("embed", cmd_embed),
        ("analyze", cmd_analyze),
        ("run", cmd_run),
        ("compare", cmd_compare),
        ("sweep-pq", cmd_sweep_pq),
        ("sweep-hnsw", cmd_sweep_hnsw),
        ("sweep-ivf", cmd_sweep_ivf),
        ("compare-stores", cmd_compare_stores),
        ("compare-embeddings", cmd_compare_embeddings),
        ("whisper-demo", cmd_whisper_demo),
    ]:
        sp = sub.add_parser(name)
        _add_common(sp)
        sp.set_defaults(func=fn)

    whisper_p = sub.choices["whisper-demo"]
    whisper_p.add_argument("--audio", nargs="+", default=None,
                           help="Paths to audio files (default: synthesise from queries)")

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line entry point for the DLS vector-search engine.

Usage (run from the DLS_proj root):
    python -m vector_engine.cli build-corpus [--config PATH] [--set k=v ...]
    python -m vector_engine.cli embed        [--config PATH] [--set k=v ...]
    python -m vector_engine.cli analyze      [--config PATH] [--set k=v ...]
    python -m vector_engine.cli run          [--config PATH] [--set k=v ...]
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vector_engine")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("build-corpus", cmd_build_corpus),
        ("embed", cmd_embed),
        ("analyze", cmd_analyze),
        ("run", cmd_run),
        ("compare", cmd_compare),
    ]:
        sp = sub.add_parser(name)
        _add_common(sp)
        sp.set_defaults(func=fn)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

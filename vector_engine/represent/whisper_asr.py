"""Optional Whisper ASR: audio query -> text -> same retrieval index.

Whisper is optional (see requirements-whisper.txt). When unavailable, the CLI
prints a clear install hint instead of failing the whole pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from ..config import Config
from ..data import corpus as corpus_mod
from ..represent import encoder
from ..search import pipeline
from ..store import apply_search_params, build_index


def _load_whisper(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed. Run: pip install -r vector_engine/requirements-whisper.txt"
        ) from exc
    return whisper.load_model(model_name)


def transcribe(cfg: Config, audio_paths: List[str]) -> List[str]:
    model_name = str(cfg.get("whisper.model", "base"))
    model = _load_whisper(model_name)
    texts: List[str] = []
    for path in audio_paths:
        result = model.transcribe(path, fp16=False)
        texts.append(str(result.get("text", "")).strip())
    return texts


def run_demo(cfg: Config, audio_paths: List[str] | None = None) -> Dict:
    """Transcribe audio queries and run them through the existing index."""
    df = corpus_mod.load_corpus(cfg)
    queries = corpus_mod.load_queries(cfg)
    ids = df["id"].to_numpy()

    encoder.embed_corpus(cfg, df["text"].tolist())
    corpus_emb = encoder.load_corpus_embeddings(cfg)
    index, index_info = build_index(cfg, corpus_emb, ids=ids)
    apply_search_params(cfg, index)

    mode = "audio"
    if audio_paths is None:
        n = int(cfg.get("whisper.sample_queries", 5))
        audio_paths, mode = _prepare_audio(queries[:n], cfg)

    if mode == "audio":
        texts = transcribe(cfg, audio_paths)
    else:
        n = len(audio_paths) if audio_paths else int(cfg.get("whisper.sample_queries", 5))
        texts = [q["text"] for q in queries[:n]]
        audio_paths = [""] * len(texts)

    query_emb = encoder.embed_queries(cfg, texts)
    retrieved, latencies = pipeline.run_queries(
        cfg, index, query_emb, ids, corpus_emb=corpus_emb)

    results = []
    for i, (path, text, hits) in enumerate(zip(audio_paths, texts, retrieved)):
        ref = queries[i] if i < len(queries) else None
        rel = set(ref["relevant_ids"]) if ref else set()
        hit_set = set(int(x) for x in hits[: int(cfg.get("search.top_k", 10))])
        results.append({
            "audio_path": path,
            "transcript": text,
            "reference_query": ref["text"] if ref else None,
            "retrieved_ids": [int(x) for x in hits],
            "recall@10": len(hit_set & rel) / len(rel) if rel else None,
            "latency_ms": round(float(latencies[i]), 3),
        })

    out = cfg.path(cfg.get("paths.outputs", "outputs"))
    os.makedirs(out, exist_ok=True)
    report = {
        "mode": mode,
        "n_audio_queries": len(results),
        "whisper_model": cfg.get("whisper.model", "base"),
        "index_info": index_info,
        "results": results,
    }
    path = os.path.join(out, "whisper_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = path
    return report


def _prepare_audio(queries: List[Dict], cfg: Config) -> tuple[List[str], str]:
    """Return (paths, mode). mode is 'audio' when real/synthetic speech exists."""
    art = cfg.path(cfg.get("paths.artifacts", "artifacts"))
    audio_dir = os.path.join(art, "whisper_demo")
    os.makedirs(audio_dir, exist_ok=True)
    paths: List[str] = []
    for q in queries:
        fp = os.path.join(audio_dir, f"q{q['qid']}.wav")
        if not os.path.exists(fp):
            if not _text_to_wav(q["text"], fp):
                return [], "text_proxy"
        paths.append(fp)
    return paths, "audio"


def _text_to_wav(text: str, path: str) -> bool:
    try:
        from gtts import gTTS
        import subprocess
        mp3 = path.replace(".wav", ".mp3")
        gTTS(text=text, lang="en").save(mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", path],
            check=True, capture_output=True,
        )
        os.remove(mp3)
        return True
    except Exception:
        return False

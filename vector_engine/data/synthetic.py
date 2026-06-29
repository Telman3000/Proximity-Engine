"""Parametric synthetic generator for scene-object records.

Design for ground-truth retrieval evaluation:
  - A *relevance key* = (label, color, material, size, room_type).
  - Records sharing the same key are mutually relevant.
  - Each record renders a *paraphrased* text (varied templates) so texts differ
    even within a key -> embedding quality actually matters.
  - Queries are rendered with a *different* template family than records, so we
    measure semantic matching rather than string overlap.

Synthetic data gives us a free, exact ground truth (query -> relevant ids),
which solves the "validation set (query + relevance)" requirement.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from . import vocab

RECORD_TEMPLATES = [
    "{color} {material} {label} ({size}) in the {room_type}, {location}.",
    "A {size} {label} made of {material}, {color} in colour, located in the {room_type}.",
    "In the {room_type}: a {color} {label}, {material}, {location}.",
    "{room_type} — {label}, {color} {material}, {size} size, {location}.",
    "There is a {color} {size} {label} ({material}) {location} in the {room_type}.",
    "{size} {color} {label} of {material} standing {location}, {room_type}.",
]

QUERY_TEMPLATES = [
    "find the {color} {material} {label} in the {room_type}",
    "where is the {size} {color} {label} in the {room_type}?",
    "locate a {material} {label}, {color}, in the {room_type}",
    "I am looking for a {color} {label} made of {material} in the {room_type}",
    "show me the {size} {color} {material} {label} in the {room_type}",
]


def _key_tuple() -> List[str]:
    return ["label", "color", "material", "size", "room_type"]


def key_to_str(rec: Dict) -> str:
    return "|".join(str(rec[k]) for k in _key_tuple())


def _sample_key(rng: random.Random) -> Dict:
    return {
        "label": rng.choice(vocab.LABELS),
        "color": rng.choice(vocab.COLORS),
        "material": rng.choice(vocab.MATERIALS),
        "size": rng.choice(vocab.SIZES),
        "room_type": rng.choice(vocab.ROOM_TYPES),
    }


def _render(template: str, key: Dict, rng: random.Random) -> str:
    return template.format(
        location=rng.choice(vocab.LOCATIONS),
        **key,
    )


def generate(
    target_size: int,
    mult_min: int,
    mult_max: int,
    seed: int,
    start_id: int = 0,
) -> Tuple[List[Dict], Dict[str, List[int]]]:
    """Generate up to `target_size` synthetic records.

    Returns (records, key_to_ids) where key_to_ids maps relevance-key -> record ids.
    """
    rng = random.Random(seed)
    records: List[Dict] = []
    key_to_ids: Dict[str, List[int]] = {}
    seen_keys = set()
    next_id = start_id

    while len(records) < target_size:
        key = _sample_key(rng)
        ks = key_to_str({**key})
        if ks in seen_keys:
            continue  # keep keys distinct so multiplicity is controlled
        seen_keys.add(ks)

        mult = rng.randint(mult_min, mult_max)
        mult = min(mult, target_size - len(records))
        ids_for_key: List[int] = []
        for _ in range(mult):
            tmpl = rng.choice(RECORD_TEMPLATES)
            state = rng.choice(vocab.STATES) if rng.random() < 0.3 else None
            text = _render(tmpl, key, rng)
            if state:
                text = text[:-1] + f", {state}."
            rec = {
                "id": next_id,
                "text": text,
                "label": key["label"],
                "color": key["color"],
                "material": key["material"],
                "size": key["size"],
                "room_type": key["room_type"],
                "state": state,
                "key": ks,
                "source": "synthetic",
            }
            records.append(rec)
            ids_for_key.append(next_id)
            next_id += 1
        key_to_ids[ks] = ids_for_key

    return records, key_to_ids


def make_queries(
    key_to_ids: Dict[str, List[int]],
    n_queries: int,
    seed: int,
) -> List[Dict]:
    """Build evaluation queries from existing corpus keys (exact ground truth)."""
    rng = random.Random(seed + 1)
    keys = list(key_to_ids.keys())
    rng.shuffle(keys)
    keys = keys[:n_queries]
    queries: List[Dict] = []
    for qi, ks in enumerate(keys):
        label, color, material, size, room_type = ks.split("|")
        key = {
            "label": label, "color": color, "material": material,
            "size": size, "room_type": room_type,
        }
        tmpl = rng.choice(QUERY_TEMPLATES)
        text = tmpl.format(**key)
        queries.append({
            "qid": qi,
            "text": text,
            "key": ks,
            "relevant_ids": list(key_to_ids[ks]),
        })
    return queries

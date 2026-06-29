"""Ingest real VLM-JSON view files from beyond-proximity scenes.

Each detected object becomes a text record. These "anchor" records tie the
corpus to the research domain and add realistic distractors. They are NOT used
for ground-truth metrics (synthetic provides exact relevance); they only enrich
the indexed corpus.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


def _obj_text(view: Dict, obj: Dict) -> str:
    attrs = obj.get("attributes") or {}
    parts = []
    color = attrs.get("color")
    material = attrs.get("material")
    size = attrs.get("size")
    label = obj.get("label", "object")
    room = view.get("room_type")
    loc = attrs.get("location_description")
    desc = []
    if size:
        desc.append(size)
    if color:
        desc.append(color)
    if material:
        desc.append(material)
    desc.append(label)
    parts.append(" ".join(desc))
    if room:
        parts.append(f"in the {room}")
    if loc:
        parts.append(str(loc))
    return ", ".join(parts) + "."


def ingest(scenes_dir: str, start_id: int, limit: int | None = None) -> List[Dict]:
    records: List[Dict] = []
    if not scenes_dir or not os.path.isdir(scenes_dir):
        return records
    next_id = start_id
    for root, _dirs, files in os.walk(scenes_dir):
        # Only parse per-view analysis JSONs (have an "objects" list).
        for fn in files:
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            objects = data.get("objects")
            if not isinstance(objects, list):
                continue
            scene_id = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            for obj in objects:
                if not isinstance(obj, dict) or "label" not in obj:
                    continue
                attrs = obj.get("attributes") or {}
                rec = {
                    "id": next_id,
                    "text": _obj_text(data, obj),
                    "label": obj.get("label"),
                    "color": attrs.get("color"),
                    "material": attrs.get("material"),
                    "size": attrs.get("size"),
                    "room_type": data.get("room_type"),
                    "state": attrs.get("state"),
                    "key": None,        # real records carry no synthetic gt key
                    "source": "real",
                    "scene_id": scene_id,
                }
                records.append(rec)
                next_id += 1
                if limit is not None and len(records) >= limit:
                    return records
    return records

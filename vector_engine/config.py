"""Config loading with YAML + dotted CLI overrides.

Everything in the project is parametrised through a single YAML file so that
every experiment is reproducible (fixed seed) and nothing is hard-coded.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "configs", "default.yaml")


def _coerce(value: str) -> Any:
    """Parse a CLI string override into a native YAML scalar."""
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def _set_dotted(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


class Config:
    """Dict-backed config with attribute and dotted access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        cur: Any = self._data
        for p in dotted_key.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # Convenience resolved paths -------------------------------------------
    def path(self, *parts: str) -> str:
        root = self.get("paths.root", HERE)
        if not os.path.isabs(root):
            root = os.path.join(HERE, root)
        return os.path.join(root, *parts)


def load_config(config_path: str | None = None, overrides: list[str] | None = None) -> Config:
    path = config_path or DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(data, k.strip(), _coerce(v.strip()))
    return Config(data)

"""YAML config loading with `inherit:` chains and dotted CLI overrides (key.sub=value)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PILOT_DIR = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PILOT_DIR / path
    doc = yaml.safe_load(path.read_text()) or {}
    parents = doc.pop('inherit', []) or []
    if isinstance(parents, str):
        parents = [parents]
    cfg: dict[str, Any] = {}
    for parent in parents:
        p = Path(parent)
        if not p.is_absolute():
            p = path.parent / p
        cfg = deep_merge(cfg, load_config(p))
    cfg = deep_merge(cfg, doc)
    for item in overrides or []:
        key, _, raw = item.partition('=')
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
        if isinstance(node, list):
            node[int(parts[-1])] = value
        else:
            node[parts[-1]] = value
    return cfg


def resolve_path(p: str | None) -> str | None:
    """Relative paths in configs are relative to the pilot directory."""
    if p is None:
        return None
    q = Path(p)
    return str(q if q.is_absolute() else (PILOT_DIR / q).resolve())

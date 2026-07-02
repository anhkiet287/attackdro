"""Config loading (with `_base_` inheritance) and JSON result IO.

Config-driven design (see CLAUDE.md): scripts are thin CLIs that load a YAML
config and hand it to src/robustdro. A method config inherits the shared
protocol via a top-level `_base_: path/to/base.yaml` key.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. `override` wins on leaf keys.

    Dicts are merged key-by-key; every other type (including lists) is replaced
    wholesale by the override value.
    """
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str, _seen: set[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving a single `_base_` parent (recursively).

    `_base_` is a path relative to the current working directory (repo root),
    matching how the configs reference `configs/base.yaml`.
    """
    _seen = _seen or set()
    path = os.path.normpath(path)
    if path in _seen:
        raise ValueError(f"Circular _base_ inheritance detected at {path}")
    _seen.add(path)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        base_cfg = load_config(base_path, _seen)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply flat dotted-key overrides, e.g. {'train.epochs': 2}."""
    cfg = deepcopy(cfg)
    for dotted, value in overrides.items():
        if value is None:
            continue
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)

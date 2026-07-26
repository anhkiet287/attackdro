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
    matching how active configs reference repo-root-relative base paths.
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


# --------------------------------------------------------------------------- #
# Per-run results layout (post-reset 2026-07-07): one folder per run, seed
# subfolders, self-documenting run_meta.json. Applies to NEW runs; results/archive/
# stays flat + untouched.
#   results/<run_name>/                 run_name = {method}_{variant}_{protocol}
#       run_meta.json                   recipe / attack-steps / config-hash / git / eps
#       s<seed>/
#           smoke.json  train.json  eval.json  eval_fullAA.json
#           ckpt/                       ep010.pt ... ep080.pt, val_best.pt, last.pt
# --------------------------------------------------------------------------- #
import hashlib as _hashlib
import re as _re
import subprocess as _subprocess


def _base_run_name(run_name: str) -> str:
    """Strip a trailing _s<seed> / _smoke so seeds/smokes of one run share a folder."""
    n = _re.sub(r"_s\d+$", "", run_name or "run")
    return _re.sub(r"_smoke$", "", n) or "run"


def run_paths(cfg: dict, results_root: str | None = None) -> dict:
    """Resolve the per-run results paths for cfg (run_name + seed). Returns a dict of
    absolute-ish paths; callers create dirs as needed."""
    root = results_root or cfg.get("results_dir", "results/")
    base = _base_run_name(cfg.get("run_name", "run"))
    seed = cfg.get("seed", 0)
    run_dir = os.path.join(root, base)
    seed_dir = os.path.join(run_dir, f"s{seed}")
    ckpt_dir = os.path.join(seed_dir, "ckpt")
    return {
        "base_run": base, "seed": seed,
        "run_dir": run_dir, "seed_dir": seed_dir, "ckpt_dir": ckpt_dir,
        "meta": os.path.join(run_dir, "run_meta.json"),
        "train": os.path.join(seed_dir, "train.json"),
        "eval": os.path.join(seed_dir, "eval.json"),
        "eval_fullAA": os.path.join(seed_dir, "eval_fullAA.json"),
        "smoke": os.path.join(seed_dir, "smoke.json"),
    }


def config_hash(cfg: dict) -> str:
    """Stable short hash of the resolved config (setup fingerprint for run_meta)."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return _hashlib.sha1(blob.encode()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return _subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_run_meta(cfg: dict, paths: dict, extra: dict | None = None) -> None:
    """Self-documenting run_meta.json — each run records its own setup (critical post-reset)."""
    tm = cfg.get("threat_model", {})
    tr = cfg.get("train", {})
    atk = [{"norm": a.get("norm"), "steps": a.get("steps")} for a in tr.get("attacks", [])]
    meta = {
        "run_name": paths["base_run"],
        "method": cfg.get("method"),
        "objective": tr.get("groupdro", {}).get("objective"),
        "recipe": {"epochs": tr.get("epochs"), "lr": tr.get("lr"),
                   "lr_schedule": tr.get("lr_schedule"), "milestones": tr.get("milestones"),
                   "save_freq": tr.get("save_freq")},
        "train_attack_steps": atk,
        "eps": {k: v.get("eps") for k, v in tm.items()},
        "split_protocol": {
            "train_core": 50000 - int((cfg.get("dataset", {}) or {}).get("val_holdout", 0)),
            "val_select": int((cfg.get("dataset", {}) or {}).get("val_holdout", 0)),
            "selection_metric": "val_select/worst_union",
            "selection_checkpoint": "val_best.pt",
            "test_monitor_n_examples": tr.get("test_monitor_n_examples"),
            "test_monitor_frequency": tr.get("test_monitor_frequency"),
        },
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        **(extra or {}),
    }
    save_json(meta, paths["meta"])

#!/usr/bin/env python
"""Norm-specific multi-norm audit harness.

This script is safe to run with --dry-run during setup: it only parses and
validates config, checks path safety, and reports whether attack packages appear
available. Without --dry-run it is an evaluation command for later RQ1/final
audit use; do not run normal mode while active writers are incomplete.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA_VERSION = "audit_cifar10_preactrn18_multinorm_v1"
MASK_SCHEMA_VERSION = "masks_multinorm_v1"
MASK_SIDECAR_FILENAME = "masks_multinorm_v1.npz"
MASK_CONVENTION = "true_means_robust_false_means_failed"
VALIDATION_TOL = 1e-6
CANONICAL_ATTACKS = {
    "linf": ["apgd_ce_linf", "apgd_dlr_linf", "fab_t_linf", "square_linf"],
    "l2": ["apgd_ce_l2", "apgd_dlr_l2", "fab_t_l2", "square_l2"],
    "l1": ["apgd_ce_l1", "apgd_dlr_l1", "fab_t_l1", "square_l1"],
}
NORM_MAP = {"linf": "Linf", "l2": "L2", "l1": "L1"}
AUTOATTACK_NAME = {
    "apgd_ce_linf": "apgd-ce",
    "apgd_dlr_linf": "apgd-dlr",
    "fab_t_linf": "fab-t",
    "square_linf": "square",
    "apgd_ce_l2": "apgd-ce",
    "apgd_dlr_l2": "apgd-dlr",
    "fab_t_l2": "fab-t",
    "square_l2": "square",
    "apgd_ce_l1": "apgd-ce",
    "apgd_dlr_l1": "apgd-dlr",
    "fab_t_l1": "fab-t",
    "square_l1": "square",
}
ACTIVE_WRITER_PREFIXES = (
    Path("results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0"),
)
MASKING_CAVEAT = (
    "{norm} audit tests alternative white-box/minimum-norm strength, "
    "not black-box gradient-masking behavior."
)
MASK_METADATA_REQUIRED = [
    "schema_version",
    "source_json_path",
    "run_id",
    "checkpoint_role",
    "checkpoint_path",
    "checkpoint_sha256",
    "audit_subset_path",
    "audit_subset_sha256",
    "config_path",
    "n_examples",
    "attack_names",
    "skipped_attacks",
    "mask_convention",
    "script_git_commit",
    "created_at",
    "validation_status",
]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def display_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    # Colab/zip fallback: no .git in the extracted tree -> read the sha baked into the zip at build time
    try:
        f = ROOT / ".git_commit"
        if f.exists():
            return f.read_text(encoding="utf-8").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_audit_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def iter_attacks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    attacks: list[dict[str, Any]] = []
    for norm in ("linf", "l2", "l1"):
        for attack in (cfg.get("attacks", {}) or {}).get(norm, []) or []:
            attacks.append(dict(attack))
    return attacks


def validate_config(cfg: dict[str, Any]) -> None:
    errors: list[str] = []
    if cfg.get("name") != SCHEMA_VERSION:
        errors.append(f"name must be {SCHEMA_VERSION!r}")
    if cfg.get("dataset") not in ("cifar10", "cifar100"):
        errors.append("dataset must be 'cifar10' or 'cifar100'")
    if cfg.get("split") != "test":
        errors.append("split must be 'test'")
    if (cfg.get("model", {}) or {}).get("arch") != "preact_resnet18":
        errors.append("model.arch must be 'preact_resnet18'")
    if (cfg.get("model", {}) or {}).get("normalize") is not False:
        errors.append("model.normalize must be false")

    eps = cfg.get("eps", {}) or {}
    for key, expected in {"linf": 8 / 255, "l2": 0.5, "l1": 12.0}.items():
        try:
            value = float(eps[key])
        except Exception:
            errors.append(f"eps.{key} is missing or non-numeric")
            continue
        if abs(value - expected) > 1e-12:
            errors.append(f"eps.{key}={value!r} does not match locked value {expected!r}")

    for norm, names in CANONICAL_ATTACKS.items():
        configured = [a.get("name") for a in (cfg.get("attacks", {}) or {}).get(norm, []) or []]
        if configured != names:
            errors.append(f"attacks.{norm} names must be exactly {names}, got {configured}")
        for attack in (cfg.get("attacks", {}) or {}).get(norm, []) or []:
            name = attack.get("name")
            if attack.get("norm") != norm:
                errors.append(f"{name}: norm must be {norm!r}")
            if attack.get("eps_key") != norm:
                errors.append(f"{name}: eps_key must be {norm!r}")
            if name not in AUTOATTACK_NAME:
                errors.append(f"{name}: unknown canonical attack name")
            if attack.get("type") == "square" and int(attack.get("queries", 0)) <= 0:
                errors.append(f"{name}: square attack requires positive queries")
            if attack.get("type") in {"apgd", "fab"} and int(attack.get("steps", 0)) <= 0:
                errors.append(f"{name}: attack requires positive steps")

    if errors:
        raise ValueError("Invalid audit config:\n- " + "\n- ".join(errors))


def attack_availability(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    has_autoattack = importlib.util.find_spec("autoattack") is not None
    out: dict[str, dict[str, Any]] = {}
    for attack in iter_attacks(cfg):
        name = str(attack["name"])
        if not has_autoattack:
            out[name] = {
                "available": False,
                "reason": "autoattack package is not importable",
            }
            continue
        out[name] = {
            "available": True,
            "reason": "appears available through AutoAttack; not executed in dry-run",
        }
    return out


def is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to((ROOT / parent).resolve())
        return True
    except ValueError:
        return False


def assert_write_path_safe(path: str | Path) -> None:
    p = repo_path(path)
    for active in ACTIVE_WRITER_PREFIXES:
        if is_under(p, active):
            raise ValueError(f"Refusing to write inside active writer directory: {display_path(p)}")


def check_parent_creatable(path: str | Path) -> tuple[bool, str]:
    parent = repo_path(path).parent
    cur = parent
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    if not cur.exists():
        return False, f"no existing ancestor for {display_path(parent)}"
    if os.access(cur, os.W_OK):
        return True, f"parent {display_path(parent)} can be created under {display_path(cur)}"
    return False, f"nearest existing ancestor is not writable: {display_path(cur)}"


def to_eval_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {
            "name": cfg["dataset"],
            "num_classes": 10,
            "root": "data/",
            "num_workers": 4,
        },
        "model": dict(cfg.get("model", {}) or {}),
        "threat_model": {
            norm: {"eps": float(value)}
            for norm, value in (cfg.get("eps", {}) or {}).items()
        },
    }


def infer_model_family(checkpoint: str) -> str:
    norm = checkpoint.replace("\\", "/")
    if "external/RAMP/" in norm or "/external/RAMP/" in norm:
        return "ramp"
    return "robustdro"


def load_subset(cfg: dict[str, Any]):
    import torch
    import torchvision
    import torchvision.transforms as T

    subset_cfg = cfg["subset"]
    indices_path = repo_path(subset_cfg["indices_path"])
    assert_write_path_safe(indices_path)

    # dataset from config (default cifar10 → byte-identical to the frozen behaviour); cifar100 for
    # the C100 matched-pair audit. Subset generation below (incl class_balanced) is dataset-agnostic.
    _DSETS = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}
    ds = _DSETS[cfg.get("dataset", "cifar10")](
        root=str(ROOT / "data"),
        train=False,
        download=False,
        transform=T.Compose([T.ToTensor()]),
    )
    n_examples = int(subset_cfg["n_examples"])
    seed = int(subset_cfg["seed"])
    class_balanced = bool(subset_cfg.get("class_balanced", False))

    if indices_path.exists():
        with open(indices_path, encoding="utf-8") as f:
            payload = json.load(f)
        indices = [int(i) for i in payload["indices"]]
    else:
        if class_balanced:
            targets = list(getattr(ds, "targets"))
            classes = sorted(set(targets))
            if n_examples % len(classes) != 0:
                raise ValueError("class_balanced subset requires n_examples divisible by class count")
            per_class = n_examples // len(classes)
            rng = random.Random(seed)
            indices = []
            for cls in classes:
                pool = [i for i, y in enumerate(targets) if int(y) == int(cls)]
                rng.shuffle(pool)
                indices.extend(pool[:per_class])
            indices.sort()
        else:
            rng = random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            indices = sorted(indices[:n_examples])

        indices_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset": cfg["dataset"],
            "split": cfg["split"],
            "n_examples": n_examples,
            "seed": seed,
            "class_balanced": class_balanced,
            "indices": indices,
        }
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    if len(indices) != n_examples:
        raise ValueError(f"subset length {len(indices)} != expected {n_examples}")

    xs = torch.stack([ds[i][0] for i in indices])
    ys = torch.tensor([ds[i][1] for i in indices], dtype=torch.long)
    return xs, ys, indices, sha256_file(indices_path)


def clean_mask(model, x, y, device: str, bs: int):
    import torch

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            xb = x[i:i + bs].to(device)
            preds.append(model(xb).argmax(1).cpu())
    return torch.cat(preds) == y.cpu()


def configure_autoattack(adversary: Any, attack: dict[str, Any]) -> None:
    steps = int(attack.get("steps", 100))
    restarts = int(attack.get("restarts", 1))
    if hasattr(adversary, "apgd"):
        adversary.apgd.n_iter = steps
        adversary.apgd.n_restarts = restarts
    if hasattr(adversary, "apgd_targeted"):
        adversary.apgd_targeted.n_iter = steps
        adversary.apgd_targeted.n_restarts = restarts
    if hasattr(adversary, "fab"):
        adversary.fab.n_iter = steps
        adversary.fab.n_restarts = restarts
    if hasattr(adversary, "square") and attack.get("type") == "square":
        adversary.square.n_queries = int(attack.get("queries", 5000))
        adversary.square.n_restarts = restarts


def run_attack_mask(model, x, y, attack: dict[str, Any], eps: float,
                    device: str, bs: int):
    import torch
    from autoattack import AutoAttack

    aa_name = AUTOATTACK_NAME[attack["name"]]
    adversary = AutoAttack(
        model,
        norm=NORM_MAP[attack["norm"]],
        eps=float(eps),
        version="custom",
        attacks_to_run=[aa_name],
        device=device,
        seed=int(attack.get("seed", 0)),
        verbose=False,
    )
    configure_autoattack(adversary, attack)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        x_adv = adversary.run_standard_evaluation(x.to(device), y.to(device), bs=bs)
    with torch.no_grad():
        pred = []
        for i in range(0, x_adv.shape[0], bs):
            pred.append(model(x_adv[i:i + bs]).argmax(1).detach().cpu())
    mask = torch.cat(pred) == y.cpu()

    actual_queries_mean = None
    actual_queries_max = None
    square = getattr(adversary, "square", None)
    if square is not None:
        queries = getattr(square, "n_queries_used", None)
        if queries is not None:
            try:
                q = torch.as_tensor(queries).float()
                actual_queries_mean = float(q.mean().item())
                actual_queries_max = float(q.max().item())
            except Exception:
                actual_queries_mean = None
                actual_queries_max = None
    return mask, actual_queries_mean, actual_queries_max


def mean_mask(mask: Any) -> float:
    return float(mask.float().mean().item())


def and_masks(masks: list[Any]):
    if not masks:
        raise ValueError("Cannot AND an empty mask list")
    out = masks[0].clone()
    for mask in masks[1:]:
        out &= mask
    return out


def mask_sidecar_path(args: argparse.Namespace) -> Path:
    if args.mask_out:
        return repo_path(args.mask_out)
    base = args.existing_json if getattr(args, "sidecar_only", False) else args.out
    if not base:
        raise ValueError("Cannot infer mask sidecar path without --mask-out, --out, or --existing-json")
    return repo_path(base).parent / MASK_SIDECAR_FILENAME


def numpy_masks_from_torch(masks: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {}
    for name, mask in masks.items():
        out[name] = np.asarray(mask.detach().cpu().numpy(), dtype=bool)
    return out


def and_numpy_masks(masks: list[Any]):
    import numpy as np

    if not masks:
        raise ValueError("Cannot AND an empty mask list")
    out = np.asarray(masks[0], dtype=bool).copy()
    for mask in masks[1:]:
        out &= np.asarray(mask, dtype=bool)
    return out


def numpy_mean(mask: Any) -> float:
    import numpy as np

    return float(np.asarray(mask, dtype=bool).mean())


def _compare_metric(mismatches: list[dict[str, Any]], field: str,
                    expected: Any, actual: float, tol: float) -> None:
    if expected is None:
        mismatches.append({
            "field": field,
            "json_value": expected,
            "recomputed_value": actual,
            "diff": None,
            "reason": "JSON value is null for an exported mask",
        })
        return
    expected_f = float(expected)
    diff = abs(expected_f - float(actual))
    if diff > tol:
        mismatches.append({
            "field": field,
            "json_value": expected_f,
            "recomputed_value": float(actual),
            "diff": diff,
        })


def metric_diffs_against_result(result: dict[str, Any],
                                masks: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(field: str, json_value: Any, recomputed_value: Any,
            category: str, attack_name: str | None = None) -> None:
        if json_value is None or recomputed_value is None:
            diff = None
        else:
            diff = abs(float(json_value) - float(recomputed_value))
        row: dict[str, Any] = {
            "field": field,
            "category": category,
            "json_value": None if json_value is None else float(json_value),
            "recomputed_value": None if recomputed_value is None else float(recomputed_value),
            "diff": diff,
        }
        if attack_name is not None:
            row["attack_name"] = attack_name
        rows.append(row)

    for name, mask in masks.items():
        attack_result = (result.get("per_attack", {}) or {}).get(name, {}) or {}
        add(
            f"per_attack.{name}.robust_acc",
            attack_result.get("robust_acc"),
            numpy_mean(mask),
            "per_attack",
            attack_name=name,
        )

    required_primary = {
        "linf": "apgd_ce_linf",
        "l2": "apgd_ce_l2",
        "l1": "apgd_ce_l1",
    }
    if all(name in masks for name in required_primary.values()):
        primary = result.get("primary_apgd", {}) or {}
        for norm, name in required_primary.items():
            metric = "robust_linf" if norm == "linf" else f"robust_{norm}"
            add(
                f"primary_apgd.{metric}",
                primary.get(metric),
                numpy_mean(masks[name]),
                "aggregate",
            )
        primary_union = and_numpy_masks([masks[name] for name in required_primary.values()])
        add(
            "primary_apgd.primary_apgd_union",
            primary.get("primary_apgd_union"),
            numpy_mean(primary_union),
            "aggregate",
        )

    per_norm = result.get("per_norm_audit", {}) or {}
    for norm, names in CANONICAL_ATTACKS.items():
        available = [name for name in names if name in masks]
        if not available:
            continue
        norm_union = and_numpy_masks([masks[name] for name in available])
        add(
            f"per_norm_audit.audit_acc_{norm}",
            per_norm.get(f"audit_acc_{norm}"),
            numpy_mean(norm_union),
            "aggregate",
        )

    if masks:
        full_union = and_numpy_masks([masks[name] for name in masks])
        add(
            "full_audit_union",
            result.get("full_audit_union"),
            numpy_mean(full_union),
            "aggregate",
        )
    return rows


def significant_metric_diffs(result: dict[str, Any], masks: dict[str, Any],
                             tol: float = VALIDATION_TOL) -> list[dict[str, Any]]:
    return [
        row for row in metric_diffs_against_result(result, masks)
        if row["diff"] is None or float(row["diff"]) > tol
    ]


def classify_metric_mismatch(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    attack_rows = [r for r in rows if r.get("category") == "per_attack"]
    attack_names = [str(r.get("attack_name", "")) for r in attack_rows]
    if attack_rows and all(name.startswith("square_") for name in attack_names):
        return "square_only_possible_stochasticity"
    if any(name.startswith("apgd_") or name.startswith("fab_t_") for name in attack_names):
        return "apgd_or_fab_reproducibility_issue"
    if not attack_rows and any(r.get("category") == "aggregate" for r in rows):
        return "union_logic_issue"
    return "unknown"


def format_metric_diffs(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, sort_keys=True)


def validate_masks_against_result(result: dict[str, Any],
                                  masks: dict[str, Any],
                                  metadata: dict[str, Any] | None = None,
                                  tol: float = VALIDATION_TOL) -> list[dict[str, Any]]:
    """Return validation mismatches for an audit mask sidecar payload.

    Mask convention is true=robust, false=failed. Aggregate robust accuracies are
    therefore means over boolean masks, and unions are boolean ANDs.
    """
    import numpy as np

    mismatches: list[dict[str, Any]] = []

    if metadata is not None:
        for key in MASK_METADATA_REQUIRED:
            if key not in metadata:
                mismatches.append({
                    "field": f"metadata.{key}",
                    "reason": "missing required metadata field",
                })
        if metadata.get("schema_version") != MASK_SCHEMA_VERSION:
            mismatches.append({
                "field": "metadata.schema_version",
                "json_value": metadata.get("schema_version"),
                "recomputed_value": MASK_SCHEMA_VERSION,
            })
        if metadata.get("mask_convention") != MASK_CONVENTION:
            mismatches.append({
                "field": "metadata.mask_convention",
                "json_value": metadata.get("mask_convention"),
                "recomputed_value": MASK_CONVENTION,
            })
        if sorted(metadata.get("attack_names") or []) != sorted(masks):
            mismatches.append({
                "field": "metadata.attack_names",
                "json_value": metadata.get("attack_names"),
                "recomputed_value": sorted(masks),
            })

    subset = result.get("subset", {}) or {}
    n_examples = int(subset.get("n_examples", metadata.get("n_examples", 0) if metadata else 0))
    if metadata is not None and int(metadata.get("n_examples", -1)) != n_examples:
        mismatches.append({
            "field": "metadata.n_examples",
            "json_value": metadata.get("n_examples"),
            "recomputed_value": n_examples,
        })

    for name, attack_result in (result.get("per_attack", {}) or {}).items():
        if not isinstance(attack_result, dict):
            continue
        if attack_result.get("robust_acc") is not None and not attack_result.get("skipped", False):
            if name not in masks:
                mismatches.append({
                    "field": f"per_attack.{name}.mask",
                    "reason": "JSON reports attack was run, but sidecar omits its mask",
                })

    for name, mask in masks.items():
        if np.asarray(mask).dtype != np.dtype(bool):
            mismatches.append({
                "field": name,
                "reason": f"mask dtype must be bool, got {np.asarray(mask).dtype}",
            })
        shape = getattr(mask, "shape", ())
        if len(shape) != 1:
            mismatches.append({
                "field": name,
                "reason": f"mask must be one-dimensional, got shape {tuple(shape)}",
            })
            continue
        if n_examples and int(shape[0]) != n_examples:
            mismatches.append({
                "field": name,
                "reason": f"mask length {shape[0]} != n_examples {n_examples}",
            })
        attack_result = (result.get("per_attack", {}) or {}).get(name)
        if attack_result is None:
            mismatches.append({
                "field": f"per_attack.{name}",
                "reason": "exported mask has no JSON per_attack entry",
            })
            continue
        _compare_metric(
            mismatches,
            f"per_attack.{name}.robust_acc",
            attack_result.get("robust_acc"),
            numpy_mean(mask),
            tol,
        )

    required_primary = {
        "linf": "apgd_ce_linf",
        "l2": "apgd_ce_l2",
        "l1": "apgd_ce_l1",
    }
    for norm, name in required_primary.items():
        if name not in masks:
            mismatches.append({
                "field": name,
                "reason": "required primary APGD-CE mask is missing",
            })

    if all(name in masks for name in required_primary.values()):
        primary = result.get("primary_apgd", {}) or {}
        for norm, name in required_primary.items():
            metric = "robust_linf" if norm == "linf" else f"robust_{norm}"
            _compare_metric(
                mismatches,
                f"primary_apgd.{metric}",
                primary.get(metric),
                numpy_mean(masks[name]),
                tol,
            )
        primary_union = and_numpy_masks([masks[name] for name in required_primary.values()])
        _compare_metric(
            mismatches,
            "primary_apgd.primary_apgd_union",
            primary.get("primary_apgd_union"),
            numpy_mean(primary_union),
            tol,
        )

    per_norm = result.get("per_norm_audit", {}) or {}
    for norm, names in CANONICAL_ATTACKS.items():
        available = [name for name in names if name in masks]
        if not available:
            continue
        norm_union = and_numpy_masks([masks[name] for name in available])
        _compare_metric(
            mismatches,
            f"per_norm_audit.audit_acc_{norm}",
            per_norm.get(f"audit_acc_{norm}"),
            numpy_mean(norm_union),
            tol,
        )

    if masks:
        full_union = and_numpy_masks([masks[name] for name in masks])
        _compare_metric(
            mismatches,
            "full_audit_union",
            result.get("full_audit_union"),
            numpy_mean(full_union),
            tol,
        )

    return mismatches


def load_mask_sidecar(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("Mask sidecar missing metadata_json")
        metadata = json.loads(str(data["metadata_json"].item()))
        masks = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    return metadata, masks


def validate_mask_sidecar_paths(json_path: str | Path,
                                masks_path: str | Path) -> list[dict[str, Any]]:
    with open(json_path, encoding="utf-8") as f:
        result = json.load(f)
    metadata, masks = load_mask_sidecar(masks_path)
    return validate_masks_against_result(result, masks, metadata=metadata)


def seed_provenance(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    top_level_seed_fields = {
        key: value for key, value in result.items()
        if "seed" in str(key).lower()
    }
    per_attack_seed_fields: dict[str, Any] = {}
    for name, attack_result in (result.get("per_attack", {}) or {}).items():
        if not isinstance(attack_result, dict):
            continue
        seeds: dict[str, Any] = {
            key: value for key, value in attack_result.items()
            if "seed" in str(key).lower()
        }
        params = attack_result.get("params")
        if isinstance(params, dict):
            for key, value in params.items():
                if "seed" in str(key).lower():
                    seeds[f"params.{key}"] = value
        if seeds:
            per_attack_seed_fields[name] = seeds

    config_attack_seeds = {
        attack["name"]: attack.get("seed")
        for attack in iter_attacks(cfg)
        if "name" in attack
    }
    subset = result.get("subset", {}) or {}
    expected_attacks = [
        name for name, attack_result in (result.get("per_attack", {}) or {}).items()
        if isinstance(attack_result, dict)
        and attack_result.get("robust_acc") is not None
        and not attack_result.get("skipped", False)
    ]
    json_records_all_attack_seeds = bool(expected_attacks) and all(
        name in per_attack_seed_fields for name in expected_attacks
    )
    return {
        "top_level_seed_fields": top_level_seed_fields,
        "per_attack_seed_fields": per_attack_seed_fields,
        "subset_seed": subset.get("seed"),
        "subset_path": subset.get("indices_path"),
        "subset_sha256": subset.get("indices_sha256"),
        "config_subset_seed": (cfg.get("subset", {}) or {}).get("seed"),
        "config_attack_seeds": config_attack_seeds,
        "rerun_seed": sorted(set(config_attack_seeds.values())),
        "provenance_status": (
            "json_attack_seed_recorded"
            if json_records_all_attack_seeds
            else "config_inferred_seed_not_json_recorded"
        ),
    }


def print_seed_provenance(provenance: dict[str, Any]) -> None:
    print("[sidecar-only] seed provenance:")
    print(json.dumps(provenance, indent=2, sort_keys=True))


def load_existing_result_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def export_mask_sidecar(args: argparse.Namespace, cfg: dict[str, Any],
                        validation_result: dict[str, Any], masks: dict[str, Any],
                        skipped: list[dict[str, str]], checkpoint_sha: str,
                        indices_sha: str, source_json_path: str | Path) -> Path:
    import numpy as np

    out_path = mask_sidecar_path(args)
    assert_write_path_safe(out_path)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing mask sidecar: {display_path(out_path)}")

    mask_arrays = numpy_masks_from_torch(masks)
    metadata = {
        "schema_version": MASK_SCHEMA_VERSION,
        "source_json_path": display_path(repo_path(source_json_path)),
        "run_id": args.run_id,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "audit_subset_path": cfg["subset"]["indices_path"],
        "audit_subset_sha256": indices_sha,
        "config_path": args.config,
        "n_examples": int(cfg["subset"]["n_examples"]),
        "attack_names": list(mask_arrays),
        "skipped_attacks": skipped,
        "mask_convention": MASK_CONVENTION,
        "script_git_commit": git_commit(),
        "created_at": utc_now(),
        "validation_status": "passed",
    }

    mismatches = validate_masks_against_result(validation_result, mask_arrays, metadata=metadata)
    if mismatches:
        significant = significant_metric_diffs(validation_result, mask_arrays)
        classification = classify_metric_mismatch(significant)
        details = json.dumps(mismatches, indent=2, sort_keys=True)
        metric_details = format_metric_diffs(significant)
        raise RuntimeError(
            "Mask sidecar validation failed before write:\n"
            f"classification={classification}\n"
            f"mismatches={details}\n"
            f"metric_diffs={metric_details}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(mask_arrays)
    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(out_path, **payload)
    return out_path


def bootstrap_ci(values_a: Any, values_b: Any | None, n_bootstrap: int,
                 ci: float, seed: int) -> list[float]:
    import numpy as np

    a = values_a.detach().cpu().numpy().astype(float)
    b = None if values_b is None else values_b.detach().cpu().numpy().astype(float)
    rng = np.random.default_rng(seed)
    n = len(a)
    samples = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        if b is None:
            samples.append(float(a[idx].mean()))
        else:
            samples.append(float(a[idx].mean() - b[idx].mean()))
    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return [float(lo), float(hi)]


def build_result(args, cfg: dict[str, Any], checkpoint_sha: str,
                 indices_sha: str, clean, per_attack: dict[str, dict[str, Any]],
                 masks: dict[str, Any], skipped: list[dict[str, str]]) -> dict[str, Any]:
    eps = {k: float(v) for k, v in cfg["eps"].items()}
    apgd_primary = {
        norm: masks[f"apgd_ce_{norm}"]
        for norm in ("linf", "l2", "l1")
    }
    primary_union = and_masks([apgd_primary[n] for n in ("linf", "l2", "l1")])

    per_norm_audit: dict[str, float | None] = {}
    masking_caveats = {"linf": None, "l2": None, "l1": None}
    black_box_masks: dict[str, Any] = {}
    norm_audit_masks: dict[str, Any] = {}
    for norm in ("linf", "l2", "l1"):
        available_names = [
            a["name"] for a in (cfg["attacks"].get(norm, []) or [])
            if a["name"] in masks
        ]
        norm_audit_masks[norm] = and_masks([masks[name] for name in available_names])
        per_norm_audit[f"audit_acc_{norm}"] = mean_mask(norm_audit_masks[norm])
        per_norm_audit[f"audit_gap_{norm}"] = (
            mean_mask(apgd_primary[norm]) - mean_mask(norm_audit_masks[norm])
        )

        bb_names = [
            a["name"] for a in (cfg["attacks"].get(norm, []) or [])
            if a.get("black_box") and a["name"] in masks
        ]
        if bb_names:
            black_box_masks[norm] = and_masks([masks[name] for name in bb_names])
            per_norm_audit[f"black_box_gap_{norm}"] = (
                mean_mask(apgd_primary[norm]) - mean_mask(black_box_masks[norm])
            )
        else:
            per_norm_audit[f"black_box_gap_{norm}"] = None

    skipped_names = {item["name"] for item in skipped}
    if "square_l2" in skipped_names:
        masking_caveats["l2"] = MASKING_CAVEAT.format(norm="l2")
    if "square_l1" in skipped_names:
        masking_caveats["l1"] = MASKING_CAVEAT.format(norm="l1")

    all_attack_union = and_masks(list(masks.values()))
    ci_cfg = (cfg.get("metrics", {}) or {}).get("bootstrap_ci", {}) or {}
    ci: dict[str, list[float]] = {}
    if ci_cfg.get("enabled", False):
        n_bootstrap = int(ci_cfg.get("n_bootstrap", 1000))
        ci_level = float(ci_cfg.get("ci", 0.95))
        seed = int(ci_cfg.get("seed", 0))
        for norm in ("linf", "l2", "l1"):
            ci[f"audit_gap_{norm}_95ci"] = bootstrap_ci(
                apgd_primary[norm], norm_audit_masks[norm], n_bootstrap, ci_level, seed
            )
        ci["full_audit_union_95ci"] = bootstrap_ci(
            all_attack_union, None, n_bootstrap, ci_level, seed
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "git_commit": git_commit(),
        "audit_config": args.config,
        "device": args.device,
        "subset": {
            "split": cfg["split"],
            "n_examples": int(cfg["subset"]["n_examples"]),
            "seed": int(cfg["subset"]["seed"]),
            "class_balanced": bool(cfg["subset"].get("class_balanced", False)),
            "indices_path": cfg["subset"]["indices_path"],
            "indices_sha256": indices_sha,
        },
        "eps": eps,
        "clean_acc": mean_mask(clean),
        "per_attack": per_attack,
        "primary_apgd": {
            "robust_linf": mean_mask(apgd_primary["linf"]),
            "robust_l2": mean_mask(apgd_primary["l2"]),
            "robust_l1": mean_mask(apgd_primary["l1"]),
            "primary_apgd_union": mean_mask(primary_union),
            "definition": "AND over apgd_ce_linf, apgd_ce_l2, apgd_ce_l1",
        },
        "per_norm_audit": per_norm_audit,
        "full_audit_union": mean_mask(all_attack_union),
        "ci": ci,
        "skipped_attacks": skipped,
        "masking_caveats": masking_caveats,
    }


def dry_run(args, cfg: dict[str, Any]) -> int:
    validate_config(cfg)
    if args.sidecar_only:
        if not args.existing_json:
            raise ValueError("--sidecar-only requires --existing-json")
        if not args.export_masks:
            raise ValueError("--sidecar-only requires --export-masks")
        if args.out:
            raise ValueError("--sidecar-only does not write --out; use --existing-json")
        output_ref = args.existing_json
    else:
        if not args.out:
            raise ValueError("--out is required unless --sidecar-only is used")
        output_ref = args.out
        assert_write_path_safe(args.out)
    assert_write_path_safe(cfg["subset"]["indices_path"])
    if args.mask_out:
        assert_write_path_safe(args.mask_out)
    ok, msg = check_parent_creatable(output_ref)
    if not ok:
        raise ValueError(msg)
    availability = attack_availability(cfg)

    print(f"[dry-run] config: {args.config}")
    print(f"[dry-run] schema: {cfg['name']}")
    print(f"[dry-run] checkpoint argument accepted but not loaded: {args.checkpoint}")
    print(f"[dry-run] output: {output_ref}")
    if args.export_masks:
        print(f"[dry-run] mask sidecar: {display_path(mask_sidecar_path(args))}")
    print(f"[dry-run] output parent: {msg}")
    print("[dry-run] eps:")
    for norm in ("linf", "l2", "l1"):
        print(f"  {norm}: {float(cfg['eps'][norm])}")
    print("[dry-run] attacks:")
    for attack in iter_attacks(cfg):
        info = availability[attack["name"]]
        status = "available" if info["available"] else "unavailable"
        req = "required" if attack.get("required", False) else "optional"
        print(f"  {attack['name']} ({req}): {status} - {info['reason']}")
    print("[dry-run] no checkpoint loaded")
    print("[dry-run] no dataset loaded")
    print("[dry-run] no attacks run")
    print("[dry-run] no result JSON written")
    return 0


def run_normal(args, cfg: dict[str, Any]) -> int:
    import torch
    from robustdro.eval import load_eval_checkpoint
    from robustdro.utils.io import save_json

    validate_config(cfg)
    existing_result = None
    existing_json_path = None
    existing_json_sha_before = None
    source_json_path = None

    if args.sidecar_only:
        if not args.existing_json:
            raise ValueError("--sidecar-only requires --existing-json")
        if not args.export_masks:
            raise ValueError("--sidecar-only requires --export-masks")
        if args.out:
            raise ValueError("--sidecar-only does not write --out; use --existing-json")
        existing_json_path = repo_path(args.existing_json)
        if not existing_json_path.exists():
            raise FileNotFoundError(
                f"--existing-json does not exist: {display_path(existing_json_path)}"
            )
        source_json_path = args.existing_json
        existing_json_sha_before = sha256_file(existing_json_path)
        existing_result = load_existing_result_json(existing_json_path)
        print_seed_provenance(seed_provenance(existing_result, cfg))
    else:
        if not args.out:
            raise ValueError("--out is required unless --sidecar-only is used")
        if args.existing_json:
            raise ValueError("--existing-json is only valid with --sidecar-only")
        source_json_path = args.out

    if args.mask_out and not args.export_masks:
        raise ValueError("--mask-out requires --export-masks")
    if args.validate_mask_sidecar and not args.export_masks:
        raise ValueError("--validate-mask-sidecar requires --export-masks")
    if args.export_masks and mask_sidecar_path(args).exists():
        raise FileExistsError(
            f"Refusing to overwrite existing mask sidecar: {display_path(mask_sidecar_path(args))}"
        )
    if not args.sidecar_only:
        assert_write_path_safe(args.out)
    assert_write_path_safe(cfg["subset"]["indices_path"])
    out_path = None
    if not args.sidecar_only:
        out_path = repo_path(args.out)
        if out_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result JSON: {display_path(out_path)}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    eval_cfg = to_eval_cfg(cfg)
    model_family = args.model_family or infer_model_family(args.checkpoint)
    model, _ = load_eval_checkpoint(
        args.checkpoint,
        eval_cfg,
        model_family=model_family,
        device=args.device,
    )
    x, y, _, indices_sha = load_subset(cfg)
    clean = clean_mask(model, x, y, args.device, args.bs)

    availability = attack_availability(cfg)
    per_attack: dict[str, dict[str, Any]] = {}
    masks: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for attack in iter_attacks(cfg):
        name = attack["name"]
        available = bool(availability[name]["available"])
        if not available:
            reason = availability[name]["reason"]
            if attack.get("required", False) or not args.allow_missing_optional_attacks:
                raise RuntimeError(f"Attack {name} unavailable: {reason}")
            skipped.append({"name": name, "reason": reason})
            per_attack[name] = {
                "robust_acc": None,
                "n": int(cfg["subset"]["n_examples"]),
                "available": False,
                "skipped": True,
                "params": dict(attack),
                "actual_queries_mean": None,
                "actual_queries_max": None,
            }
            continue
        try:
            mask, q_mean, q_max = run_attack_mask(
                model, x, y, attack, float(cfg["eps"][attack["eps_key"]]),
                args.device, args.bs
            )
        except Exception as exc:
            if attack.get("required", False) or not args.allow_missing_optional_attacks:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append({"name": name, "reason": reason})
            per_attack[name] = {
                "robust_acc": None,
                "n": int(cfg["subset"]["n_examples"]),
                "available": False,
                "skipped": True,
                "params": dict(attack),
                "actual_queries_mean": None,
                "actual_queries_max": None,
            }
            continue
        masks[name] = mask
        per_attack[name] = {
            "robust_acc": mean_mask(mask),
            "n": int(mask.numel()),
            "available": True,
            "skipped": False,
            "params": dict(attack),
            "actual_queries_mean": q_mean,
            "actual_queries_max": q_max,
        }

    for required in ("apgd_ce_linf", "apgd_ce_l2", "apgd_ce_l1"):
        if required not in masks:
            raise RuntimeError(f"Primary APGD attack missing after audit: {required}")

    result = build_result(
        args=args,
        cfg=cfg,
        checkpoint_sha=sha256_file(args.checkpoint),
        indices_sha=indices_sha,
        clean=clean,
        per_attack=per_attack,
        masks=masks,
        skipped=skipped,
    )
    mask_path = None
    checkpoint_sha = result["checkpoint_sha256"]
    if args.export_masks:
        mask_path = export_mask_sidecar(
            args=args,
            cfg=cfg,
            validation_result=existing_result if args.sidecar_only else result,
            masks=masks,
            skipped=skipped,
            checkpoint_sha=checkpoint_sha,
            indices_sha=indices_sha,
            source_json_path=source_json_path,
        )
    if args.sidecar_only:
        if mask_path is None:
            raise RuntimeError("--sidecar-only expected a mask sidecar but none was written")
        print(f"[audit] sidecar-only masks saved -> {display_path(mask_path)}")
        mismatches = validate_mask_sidecar_paths(existing_json_path, mask_path)
        if mismatches:
            metadata, mask_arrays = load_mask_sidecar(mask_path)
            significant = significant_metric_diffs(existing_result, mask_arrays)
            classification = classify_metric_mismatch(significant)
            details = json.dumps(mismatches, indent=2, sort_keys=True)
            metric_details = format_metric_diffs(significant)
            raise RuntimeError(
                "Mask sidecar validation failed after write:\n"
                f"classification={classification}\n"
                f"mismatches={details}\n"
                f"metric_diffs={metric_details}"
            )
        existing_json_sha_after = sha256_file(existing_json_path)
        if existing_json_sha_after != existing_json_sha_before:
            raise RuntimeError(
                "JSON mutation guard failed: existing audit JSON SHA256 changed "
                f"from {existing_json_sha_before} to {existing_json_sha_after}"
            )
        print("[audit] mask sidecar validation: PASS")
        print(
            "[audit] JSON mutation guard: PASS "
            f"sha256={existing_json_sha_after}"
        )
        return 0

    save_json(result, str(out_path))
    print(f"[audit] saved -> {display_path(out_path)}")
    if mask_path is not None:
        print(f"[audit] masks saved -> {display_path(mask_path)}")
        if args.validate_mask_sidecar:
            mismatches = validate_mask_sidecar_paths(out_path, mask_path)
            if mismatches:
                details = json.dumps(mismatches, indent=2, sort_keys=True)
                raise RuntimeError(f"Mask sidecar validation failed after write:\n{details}")
            print("[audit] mask sidecar validation: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--checkpoint-role", required=True)
    p.add_argument("--out", default=None,
                   help="new audit JSON path; required unless --sidecar-only is used")
    p.add_argument("--existing-json", default=None,
                   help="existing audit JSON used as validation target in --sidecar-only mode")
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--model-family", choices=["robustdro", "ramp"], default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sidecar-only", action="store_true",
                   help="rerun attacks to write only masks; validate against --existing-json")
    p.add_argument("--allow-missing-optional-attacks", action="store_true")
    p.add_argument("--export-masks", action="store_true",
                   help="write masks_multinorm_v1.npz next to the audit JSON")
    p.add_argument("--mask-out", default=None,
                   help="optional explicit mask sidecar path (requires --export-masks)")
    p.add_argument("--validate-mask-sidecar", action="store_true",
                   help="reload and validate the written sidecar against the result JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_audit_config(args.config)
    if args.dry_run:
        return dry_run(args, cfg)
    return run_normal(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())

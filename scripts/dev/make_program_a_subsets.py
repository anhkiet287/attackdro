#!/usr/bin/env python
"""Program-A (pre-reg v3-A) frozen subset generator — CODE ONLY, no model runs.

Produces two on-disk, hashed index artifacts under results/audit/subsets/:

1. cal index   : the frozen calibration split = CIFAR-10 TEST indices 9000-9999
                 (explicit, contiguous). Used by the locked calibration in
                 G2-RUN; must exist and be hashed BEFORE any calibration.
2. audit subset: a class-balanced 1000 drawn ONLY from test_final
                 (TEST indices 1000-8999), fixed seed, disjoint from
                 cal (9000-9999) and test_monitor (0-999). Supersedes the old
                 full-test subset that overlapped cal/monitor.

This only reads CIFAR-10 test LABELS to class-balance; it crafts no adversarial
examples and loads no checkpoint. Deterministic given the seed. It writes new
files only and never overwrites an existing artifact (fail-closed).

Split index ranges are imported from the single source of truth
(robustdro.data.CANONICAL_SPLIT_RANGES) so this generator can never drift from
the loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.data import CANONICAL_SPLIT_RANGES  # noqa: E402

SUBSET_DIR = "results/audit/subsets"
CAL_PATH = os.path.join(SUBSET_DIR, "cal_cifar10_test_9000_9999.json")
AUDIT_SEED = 20260709
N_CLASSES = 10
AUDIT_PER_CLASS = 100  # 10 classes * 100 = 1000 class-balanced


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_frozen(payload: dict, path: str, *, force: bool = False) -> str:
    """Write a subset artifact deterministically and return its SHA-256.
    Refuses to overwrite an existing file unless force=True (fail-closed)."""
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact {path!r} (frozen provenance)."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return _sha256_file(path)


def _range(name: str) -> tuple[int, int]:
    src, lo, hi = CANONICAL_SPLIT_RANGES[name]
    assert src == "test", f"{name} is not a test split"
    return lo, hi


def build_cal_index(force: bool = False) -> tuple[str, str]:
    lo, hi = _range("cal")
    indices = list(range(lo, hi))
    payload = {
        "schema_version": "program_a_v3A_cal_index",
        "dataset": "cifar10",
        "split": "cal",
        "role": "calibration_only",
        "source_range": [lo, hi],  # half-open [lo, hi) over CIFAR-10 TEST
        "contiguous": True,
        "n_examples": len(indices),
        "indices": indices,
        "notes": "Frozen calibration split; may set ONLY eps/labels/E_id, never "
                 "checkpoints/thresholds/reported accuracies.",
    }
    sha = _write_json_frozen(payload, CAL_PATH, force=force)
    return CAL_PATH, sha


def build_audit_subset(seed: int = AUDIT_SEED, force: bool = False) -> tuple[str, str]:
    lo, hi = _range("test_final")
    mon_lo, mon_hi = _range("test_monitor")
    cal_lo, cal_hi = _range("cal")

    ds = torchvision.datasets.CIFAR10(root="data/", train=False, download=False)
    targets = np.asarray(ds.targets)
    assert targets.shape[0] == 10000, targets.shape

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    per_class_counts = {}
    for c in range(N_CLASSES):
        # candidate absolute test indices for class c within test_final only
        cand = np.nonzero((targets == c))[0]
        cand = cand[(cand >= lo) & (cand < hi)]
        cand = np.sort(cand)
        if cand.shape[0] < AUDIT_PER_CLASS:
            raise ValueError(
                f"class {c}: only {cand.shape[0]} candidates in test_final "
                f"[{lo},{hi}); need {AUDIT_PER_CLASS}"
            )
        pick = rng.choice(cand, size=AUDIT_PER_CLASS, replace=False)
        chosen.extend(int(i) for i in pick)
        per_class_counts[str(c)] = AUDIT_PER_CLASS
    indices = sorted(chosen)

    # Fail-closed disjointness + provenance assertions
    assert len(indices) == N_CLASSES * AUDIT_PER_CLASS == 1000, len(indices)
    assert len(set(indices)) == len(indices), "duplicate indices"
    assert all(lo <= i < hi for i in indices), "index outside test_final"
    assert all(not (mon_lo <= i < mon_hi) for i in indices), "overlaps test_monitor"
    assert all(not (cal_lo <= i < cal_hi) for i in indices), "overlaps cal"

    path = os.path.join(
        SUBSET_DIR, f"cifar10_testfinal_1000_seed{seed}_v3A.json"
    )
    payload = {
        "schema_version": "program_a_v3A_audit_subset",
        "dataset": "cifar10",
        "split": "test_final",
        "source_range": [lo, hi],  # drawn ONLY from test_final
        "class_balanced": True,
        "n_per_class": AUDIT_PER_CLASS,
        "n_examples": len(indices),
        "seed": seed,
        "disjoint_from": {
            "test_monitor": [mon_lo, mon_hi],
            "cal": [cal_lo, cal_hi],
        },
        "supersedes": "results/audit/subsets/cifar10_test_1000_seed20260709.json",
        "indices": indices,
    }
    sha = _write_json_frozen(payload, path, force=force)
    return path, sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=AUDIT_SEED)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing artifacts (default: fail-closed).")
    args = ap.parse_args()

    cal_path, cal_sha = build_cal_index(force=args.force)
    aud_path, aud_sha = build_audit_subset(seed=args.seed, force=args.force)

    print("=== Program-A frozen subsets ===")
    print(f"cal index    : {cal_path}")
    print(f"  sha256     : {cal_sha}")
    print(f"audit subset : {aud_path}")
    print(f"  sha256     : {aud_sha}")


if __name__ == "__main__":
    main()

"""Sanitize audited artifacts (eval.json + masks npz) and copy them into the artifact repo.

DEV-REPO TOOL — must never ship inside the artifact repo: it encodes the internal path
vocabulary we are stripping out.

Reads a raw audit output pair (eval.json + masks_multinorm_v1.npz) produced by the frozen
harness, rewrites ONLY the identity-bearing/path fields, and writes the pair into the
artifact tree. Every metric is copied verbatim: robust accuracies, per_attack, per_norm_audit,
primary_apgd, full_audit_union, clean_acc, ci, checkpoint_sha256, subset.indices_sha256 and
all twelve mask arrays are untouched (bit-identical). Only strings that leak internal
structure are replaced.

Field mapping (matches what already shipped for M1a):
  run_id                  -> "<arm>_<grade>"
  checkpoint              -> "<arm>/seed<seed>/val_best.pt"
  audit_config            -> "configs/eval/audit_<grade>.yaml"
  git_commit              -> "withheld_for_anonymity"
  subset.indices_path     -> "results/subsets/<clean subset name>"
  npz metadata_json: audit_subset_path, checkpoint_path, config_path, run_id,
                     script_git_commit, source_json_path  -> the same clean values

Usage (one arm at a time):
  python scripts/dev/export_sanitize.py \
      --src results/eval/union_bench/M0/10k \
      --arm M0 --seed 0 --grade 10k

  # seed replicate: keep the arm folder name, record the true seed
  python scripts/dev/export_sanitize.py \
      --src results/eval/union_bench/M1a_seed2/10k \
      --arm M1a_seed2 --seed 2 --grade 10k

Verifies after writing that the union recomputed from the sanitized npz still equals the
`full_audit_union` recorded in the sanitized eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARTIFACT = Path("/mnt/c/Users/ADMIN/Documents/Claude/Projects/pullpush-artifact")
MASK_FILE = "masks_multinorm_v1.npz"

# clean names for the audit subsets, keyed by the subset's sha256 prefix
SUBSET_CLEAN = {
    "42ceae55": "results/subsets/cifar10_test_10000_full.json",
    "af0b037a": "results/subsets/cifar10_test_1000.json",
}


def clean_subset_path(sha: str | None, fallback_grade: str) -> str:
    if sha:
        for pref, name in SUBSET_CLEAN.items():
            if sha.startswith(pref):
                return name
    return f"results/subsets/cifar10_test_{'10000_full' if fallback_grade == '10k' else '1000'}.json"


def sanitize(src: Path, arm: str, seed: int, grade: str, out_root: Path) -> Path:
    ej, mz = src / "eval.json", src / MASK_FILE
    for p in (ej, mz):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    e = json.loads(ej.read_text())
    subset_sha = (e.get("subset") or {}).get("indices_sha256")
    clean_subset = clean_subset_path(subset_sha, grade)
    ckpt_clean = f"{arm}/seed{seed}/val_best.pt"
    run_clean = f"{arm}_{grade}"
    cfg_clean = f"configs/eval/audit_{grade}.yaml"
    dst = out_root / "results" / arm / grade
    dst.mkdir(parents=True, exist_ok=True)

    # ---- eval.json: rewrite identity/path strings only ----
    before = dict(e)
    e["run_id"] = run_clean
    e["checkpoint"] = ckpt_clean
    e["audit_config"] = cfg_clean
    if "git_commit" in e:
        e["git_commit"] = "withheld_for_anonymity"
    if isinstance(e.get("subset"), dict):
        e["subset"]["indices_path"] = clean_subset
    for guard in ("full_audit_union", "clean_acc", "checkpoint_sha256"):
        if before.get(guard) != e.get(guard):
            raise AssertionError(f"metric {guard} changed during sanitize -- refusing")
    (dst / "eval.json").write_text(json.dumps(e, indent=2))

    # ---- masks npz: identical arrays, sanitized metadata ----
    z = np.load(mz)
    arrays = {k: z[k] for k in z.files if k != "metadata_json"}
    md = {}
    if "metadata_json" in z.files:
        raw = z["metadata_json"]
        md = json.loads(raw.item() if raw.shape == () else str(raw))
    md.update({
        "audit_subset_path": clean_subset,
        "checkpoint_path": ckpt_clean,
        "config_path": cfg_clean,
        "run_id": run_clean,
        "script_git_commit": "withheld_for_anonymity",
        "source_json_path": f"results/{arm}/{grade}/eval.json",
    })
    np.savez_compressed(dst / MASK_FILE, metadata_json=json.dumps(md), **arrays)

    # ---- verify: arrays bit-identical + union still matches eval.json ----
    z2 = np.load(dst / MASK_FILE)
    for k, v in arrays.items():
        if not np.array_equal(z2[k], v):
            raise AssertionError(f"mask array {k} changed during re-save -- refusing")
    u = np.ones(len(next(iter(arrays.values()))), bool)
    for k in arrays:
        u &= z2[k].astype(bool)
    recorded = e["full_audit_union"]
    if abs(float(u.mean()) - float(recorded)) > 1e-6:
        raise AssertionError(f"union mismatch: recomputed {u.mean():.6f} vs recorded {recorded:.6f}")
    print(f"  {arm}/{grade}: union {u.mean():.4f} (matches eval.json) | "
          f"{len(arrays)} masks bit-identical | -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir holding the RAW eval.json + masks npz")
    ap.add_argument("--arm", required=True, help="clean arm name, e.g. M0, M1b, M1a_seed2")
    ap.add_argument("--seed", type=int, required=True, help="true training seed of the checkpoint")
    ap.add_argument("--grade", default="10k", choices=["1k", "10k"])
    ap.add_argument("--artifact-root", default=str(ARTIFACT))
    a = ap.parse_args()
    sanitize(Path(a.src), a.arm, a.seed, a.grade, Path(a.artifact_root))


if __name__ == "__main__":
    main()

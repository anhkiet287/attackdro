#!/usr/bin/env python
"""Re-evaluate the main-table models at the RE-LOCKED protocol eps_inf=8/255.

⚠ QUICK ESTIMATE / LOWER BOUND for our in-house rows: these checkpoints were
TRAINED at eps_inf=0.03 and are here EVALUATED at the harsher 8/255 (train/eval
mismatch), so their numbers are artificially low and are NOT paper numbers and
NOT fairly comparable to train-at-8/255 models (e.g. RAMP repro 46.1). Paper-grade
8/255 numbers require RETRAINING at 8/255 (train==eval) — folded into the next
post-Gate-α clean run, not done here. evaluate.py auto-records the train/eval eps
mismatch into each JSON so the table/dashboard flag these rows.
Official MSD/AVG/MAX re-eval is a valid REFERENCE (tiered as such); note the
released MSD ckpt was also trained at 0.03 (same flag, as a reference not a rival).

Reuses checkpoints (NO retraining). For each model:
  1. If results/eval_<stem>.json is already 8/255 -> skip (idempotent).
  2. Else archive the existing 0.03 JSON -> results/archive/eval_<stem>_eps003.json
     (provenance; never appears in comparisons), then
  3. run scripts/evaluate.py --config configs/base.yaml (= 8/255) writing the
     canonical results/eval_<stem>.json, and log a W&B eval summary to the run.
Official MSD/AVG/MAX are re-eval'd via scripts/dev/eval_baselines.py (regenerates
results/baseline_table.json at 8/255; the 0.03 one is archived first).

Phases (APGD is the cheap "cluster"; standard-AA is the expensive final pass):
  --phase apgd  (default)  APGD CE+T, n=1000, all models
  --phase std              full AutoAttack, n=1000, finalists + MSD only
  --phase all              apgd then std

Meant to be QUEUED behind the live gate_bind run (do not disturb it). Rebuild the
views afterwards with: make_experiment_table.py + make_dashboard.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
ARCHIVE = os.path.join(RESULTS, "archive")
EPS_8255 = 8 / 255
BASE = "configs/base.yaml"

# (stem, checkpoint, model_family, is_finalist_for_std)
INHOUSE = [
    ("bindaware_sample_T025",    "checkpoints/bindaware_sample_T025_best.pt",    "robustdro", True),
    ("bindaware_sample_T025_s1", "checkpoints/bindaware_sample_T025_s1_best.pt", "robustdro", False),
    ("bindaware_sample_T025_s2", "checkpoints/bindaware_sample_T025_s2_best.pt", "robustdro", False),
    ("bindaware_sample_T05",     "checkpoints/bindaware_sample_T05_best.pt",     "robustdro", True),
    ("bindaware_sample_T05_s1",  "checkpoints/bindaware_sample_T05_s1_best.pt",  "robustdro", False),
    ("bindaware_sample_T05_s2",  "checkpoints/bindaware_sample_T05_s2_best.pt",  "robustdro", False),
    ("bindaware_s0",             "checkpoints/bindaware_s0_best.pt",             "robustdro", True),
    ("bindaware_s1",             "checkpoints/bindaware_s1_best.pt",             "robustdro", False),
    ("bindaware_s2",             "checkpoints/bindaware_s2_best.pt",             "robustdro", False),
    ("max_inhouse_s0",           "checkpoints/max_inhouse_s0_best.pt",           "robustdro", False),
    ("msd_inhouse_s0",           "checkpoints/msd_inhouse_s0_best.pt",           "robustdro", True),
    ("avg_frozen_s0",            "checkpoints/avg_frozen_s0_best.pt",            "robustdro", True),
    ("avg_frozen_s1",            "checkpoints/avg_frozen_s1_best.pt",            "robustdro", False),
    ("avg_frozen_s2",            "checkpoints/avg_frozen_s2_best.pt",            "robustdro", False),
    ("attackdro_union_s0",       "checkpoints/attackdro_union_s0_best.pt",       "robustdro", False),
    ("attackdro_union_s1",       "checkpoints/attackdro_union_s1_best.pt",       "robustdro", False),
    ("attackdro_union_s2",       "checkpoints/attackdro_union_s2_best.pt",       "robustdro", False),
]


def json_eps(path):
    try:
        d = json.load(open(path))
        return d.get("protocol", {}).get("threat_model", {}).get("linf", {}).get("eps")
    except Exception:
        return None


def is_8255(path):
    e = json_eps(path)
    return e is not None and abs(e - EPS_8255) < 1e-6


def archive_003(path, tag):
    """Move a 0.03 provenance JSON into results/archive/ (once)."""
    os.makedirs(ARCHIVE, exist_ok=True)
    dst = os.path.join(ARCHIVE, tag)
    if os.path.exists(path) and not os.path.exists(dst):
        shutil.move(path, dst)
        print(f"  archived 0.03 -> {os.path.relpath(dst, REPO)}")


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def eval_inhouse(stem, ckpt, family, version, n, dry):
    out = os.path.join(RESULTS, f"eval_{stem}.json") if version == "apgd" \
        else os.path.join(RESULTS, f"eval_std_{stem}.json")
    tag = f"eval_{stem}_eps003.json" if version == "apgd" else f"eval_std_{stem}_eps003.json"
    if not os.path.exists(os.path.join(REPO, ckpt)):
        print(f"[{stem}] MISSING CKPT {ckpt} -> RETRAIN NEEDED (flagged, not run)")
        return "retrain"
    if os.path.exists(out) and is_8255(out):
        print(f"[{stem}] already 8/255 ({version}) -> skip")
        return "skip"
    print(f"[{stem}] re-eval {version} @8/255")
    if dry:
        return "dry"
    archive_003(out, tag)
    cmd = [sys.executable, "scripts/evaluate.py", "--config", BASE,
           "--checkpoint", ckpt, "-n", str(n), "--version", version,
           "--model-family", family, "--run-name", stem, "--tier", "in-house",
           "--out", out]
    return "ok" if run(cmd) == 0 else "fail"


def eval_official(version, n, dry):
    bt = os.path.join(RESULTS, "baseline_table.json")
    if os.path.exists(bt) and is_8255(bt):
        print("[official MSD/AVG/MAX] baseline_table already 8/255 -> skip")
        return "skip"
    print(f"[official MSD/AVG/MAX] re-eval {version} @8/255 (eval_baselines.py)")
    if dry:
        return "dry"
    archive_003(bt, "baseline_table_eps003.json")
    md = os.path.join(RESULTS, "baseline_table.md")
    archive_003(md, "baseline_table_eps003.md")
    cmd = [sys.executable, "scripts/dev/eval_baselines.py", "--config", BASE,
           "--version", version, "-n", str(n), "--models", "MSD", "AVG", "MAX"]
    return "ok" if run(cmd) == 0 else "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="apgd", choices=["apgd", "std", "all"])
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="print plan, run nothing")
    args = ap.parse_args()
    phases = ["apgd", "std"] if args.phase == "all" else [args.phase]
    summary = {}
    for ph in phases:
        print(f"\n========== PHASE {ph} @8/255 (n={args.n}) ==========")
        for stem, ckpt, fam, is_final in INHOUSE:
            if ph == "std" and not is_final:
                continue
            summary[f"{stem}:{ph}"] = eval_inhouse(stem, ckpt, fam, ph, args.n, args.dry_run)
        summary[f"official:{ph}"] = eval_official(ph, args.n, args.dry_run)
    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    retrain = [k for k, v in summary.items() if v == "retrain"]
    if retrain:
        print(f"\nRETRAIN-NEEDED (no usable ckpt, NOT re-eval'd): {retrain}")
    print("\nNext: rebuild views ->")
    print("  .venv/bin/python scripts/make_experiment_table.py")
    print("  .venv/bin/python scripts/make_dashboard.py")


if __name__ == "__main__":
    main()

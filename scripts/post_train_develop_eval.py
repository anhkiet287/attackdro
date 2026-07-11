#!/usr/bin/env python
"""Run the four required post-train develop eval events.

This script evaluates ckpt/val_best.pt and ckpt/last.pt on both develop splits:
val_select and test_monitor. Each event writes a distinct eval JSON and logs to
the same training W&B run. It never runs test_final/full-AA.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robustdro.utils.io import apply_overrides, load_config, run_paths  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--version", default="apgd", choices=["apgd", "standard"])
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--n-examples", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, {"run_name": args.run_name, "seed": args.seed})
    paths = run_paths(cfg)
    train_run_name = cfg["run_name"]
    events = [
        ("val_select", "val_best"),
        ("val_select", "last"),
        ("test_monitor", "val_best"),
        ("test_monitor", "last"),
    ]
    for split, role in events:
        ckpt = os.path.join(paths["ckpt_dir"], f"{role}.pt")
        out = os.path.join(paths["seed_dir"], f"eval_{split}_{role}.json")
        if not os.path.exists(ckpt):
            msg = f"Missing checkpoint for {split}/{role}: {ckpt}"
            if args.dry_run:
                print(f"# {msg}", flush=True)
            else:
                raise FileNotFoundError(msg)
        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--config", args.config,
            "--checkpoint", ckpt,
            "--run-name", train_run_name,
            "--eval-split", split,
            "--checkpoint-role", role,
            "--version", args.version,
            "--n-examples", str(args.n_examples),
            "--bs", str(args.bs),
            "--out", out,
        ]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

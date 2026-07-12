#!/usr/bin/env python
"""Backfill a W&B run from a COMPLETED train.json (post-hoc online logging).

When a training run's live W&B init failed (e.g. a run name whose auto `run:`
tag exceeded W&B's 64-char limit) and it fell back to stdout-only while
train.json — the canonical source of truth — completed normally, this re-creates
the W&B run and replays the per-epoch history + end summary through the SAME
WandbLogger path a live run uses (identical project/name/id/tags/config).

Reads JSON only; recomputes nothing; touches no checkpoints or results.
Fail-closed: aborts (nonzero) if online W&B does not initialize — never leaves a
silent stdout-only no-op.

Usage:
    .venv/bin/python scripts/dev/backfill_wandb_from_trainjson.py \
        results/<run>/s0/train.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.utils.wandb_log import WandbLogger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("train_json", help="path to results/<run>/s<seed>/train.json")
    ap.add_argument("--expect-records", type=int, default=None,
                    help="Fail if history length != this (e.g. 80 for a full run).")
    args = ap.parse_args()

    with open(args.train_json, encoding="utf-8") as f:
        d = json.load(f)
    cfg = d.get("cfg") or {}
    hist = d.get("history") or []
    run_name = cfg.get("run_name")
    if not run_name or not hist:
        raise SystemExit(f"[backfill] missing run_name/history in {args.train_json!r}")
    if args.expect_records is not None and len(hist) != args.expect_records:
        raise SystemExit(
            f"[backfill] history has {len(hist)} records, expected "
            f"{args.expect_records}; refusing to backfill a partial run."
        )

    logger = WandbLogger(cfg, run_name)
    if logger.run is None:
        raise SystemExit(
            "[backfill] W&B did not initialize online (fail-closed) — check "
            "login/mode/tags. Nothing backfilled."
        )

    for row in hist:
        logger.log(dict(row), step=row.get("epoch"))

    best = d.get("best_val_select_worst_union")
    if best is not None:
        logger.summary({
            "best/val_select_worst_union": best,
            "best/selection_source": "val_select",
            "best/checkpoint": "val_best.pt",
        })
    url = getattr(logger.run, "url", "?")
    logger.finish()
    print(f"[backfill] replayed {len(hist)} epochs into W&B run '{run_name}' ({url})")


if __name__ == "__main__":
    main()

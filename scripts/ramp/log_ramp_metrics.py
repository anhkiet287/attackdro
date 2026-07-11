#!/usr/bin/env python
"""Convert RAMP metrics.pth to JSON and optional W&B logs.

RAMP does not use W&B natively. This script bridges its saved `metrics.pth`
artifact into the same W&B project used by our trainers, while keeping all keys
prefixed with `ramp_internal/` so they are not mistaken for our independent
eval_union results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.utils.io import save_json  # noqa: E402
from robustdro.utils.wandb_log import WandbLogger  # noqa: E402


def _to_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _to_list(x: Any):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    return x


def _key(name: str) -> str:
    return {"Linf": "linf", "L2": "l2", "L1": "l1", "union": "worst_union"}.get(name, name)


def _ramp_commit(repo_root: Path) -> str | None:
    ramp_root = repo_root / "external" / "RAMP"
    try:
        return subprocess.check_output(
            ["git", "-C", str(ramp_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def _num_epochs(stats: dict) -> int:
    for section in ("loss_train_dets", "rob_acc_test_dets", "rob_acc_train_dets"):
        values = stats.get(section, {})
        for v in values.values():
            if torch.is_tensor(v):
                return int(v.numel())
    return 0


def _epoch_metric(stats: dict, section: str, name: str, epoch: int) -> float | None:
    v = stats.get(section, {}).get(name)
    if not torch.is_tensor(v) or epoch >= v.numel():
        return None
    return _to_float(v[epoch])


def build_history(stats: dict) -> list[dict[str, float | int]]:
    history = []
    n_epochs = _num_epochs(stats)
    for epoch in range(n_epochs):
        row: dict[str, float | int] = {"epoch": epoch}

        for name in sorted(stats.get("loss_train_dets", {})):
            value = _epoch_metric(stats, "loss_train_dets", name, epoch)
            if value is not None:
                row[f"ramp_internal/train_loss_{_key(name)}"] = value

        # RAMP leaves unevaluated epochs at zero. Log robust metrics only on
        # epochs where at least one test metric is nonzero.
        test_values = {
            name: _epoch_metric(stats, "rob_acc_test_dets", name, epoch)
            for name in sorted(stats.get("rob_acc_test_dets", {}))
        }
        if any(v not in (None, 0.0) for v in test_values.values()):
            for name, value in test_values.items():
                if value is not None:
                    row[f"ramp_internal/test_{_key(name)}"] = value
            for name in sorted(stats.get("rob_acc_train_dets", {})):
                value = _epoch_metric(stats, "rob_acc_train_dets", name, epoch)
                if value is not None:
                    row[f"ramp_internal/train_{_key(name)}"] = value

        for name in sorted(stats.get("freq_in_at", {})):
            value = _epoch_metric(stats, "freq_in_at", name, epoch)
            if value not in (None, 0.0):
                row[f"ramp_internal/freq_{_key(name)}"] = value

        history.append(row)
    return history


def final_metrics(stats: dict) -> dict[str, float]:
    out = {}
    for item in stats.get("final_acc_dets", []) or []:
        if len(item) != 2:
            continue
        name, value = item
        out[f"ramp_internal/final_{_key(str(name))}"] = _to_float(value)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="RAMP run dir containing metrics.pth")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--out", default=None, help="Default: results/ramp_<run-dir-name>.json")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.pth"
    if not metrics_path.exists():
        raise FileNotFoundError(f"RAMP metrics not found: {metrics_path}")

    stats = torch.load(metrics_path, map_location="cpu", weights_only=False)
    history = build_history(stats)
    summary = final_metrics(stats)
    run_name = args.run_name or f"ramp_{run_dir.name}"
    out = Path(args.out or (repo_root / "results" / f"ramp_{run_dir.name}.json"))

    cfg = {
        "method": "ramp",
        "run_name": run_name,
        "seed": args.seed,
        "external": {
            "ramp_commit": _ramp_commit(repo_root),
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
        },
        "wandb": {
            "mode": args.wandb_mode or os.environ.get("WANDB_MODE", "online"),
            "project": "union-robustness-dro",
            "entity": None,
            "group": "ramp",
            "tags": ["ramp", "external", "internal-eval"],
        },
    }

    logger = WandbLogger(cfg, run_name=run_name)
    for row in history:
        step = int(row["epoch"])
        logger.log(row, step=step)
    if summary:
        logger.summary(summary)
    logger.finish()

    save_json(
        {
            "cfg": cfg,
            "history": history,
            "summary": summary,
            "raw_metrics": _to_list(stats),
        },
        str(out),
    )
    print(f"[ramp-log] saved -> {out}")


if __name__ == "__main__":
    main()

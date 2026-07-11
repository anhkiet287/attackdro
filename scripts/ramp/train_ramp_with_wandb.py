#!/usr/bin/env python
"""Run upstream RAMP.py with live W&B logging from its stdout.

This keeps external/RAMP unmodified. RAMP prints one structured line per epoch:

  [epoch] 10 [time] ... [train] loss ... [eval train] ... [eval test] ...

We echo every line unchanged and log parsed epoch/test metrics to the same W&B
project as our native trainers. At the end, if metrics.pth exists, we also write
the JSON artifact used by the dashboard/paper bookkeeping.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.utils.io import save_json  # noqa: E402
from robustdro.utils.wandb_log import WandbLogger  # noqa: E402

from log_ramp_metrics import _ramp_commit, _to_list, final_metrics  # noqa: E402


EPOCH_RE = re.compile(
    r"\[epoch\]\s+(?P<epoch>\d+)\s+"
    r"\[time\]\s+(?P<time>[0-9.]+)\s+s\s+"
    r"\[train\]\s+loss\s+(?P<loss>[0-9.eE+-]+)"
)
PAIR_RE = re.compile(r"\b(?P<name>Linf|L2|L1|clean|union)\s+(?P<pct>[0-9.]+)%")


def _key(name: str) -> str:
    return {"Linf": "linf", "L2": "l2", "L1": "l1", "union": "worst_union"}.get(name, name)


def _parse_pairs(block: str, prefix: str) -> dict[str, float]:
    out = {}
    for m in PAIR_RE.finditer(block):
        out[f"{prefix}_{_key(m.group('name'))}"] = float(m.group("pct")) / 100.0
    return out


def parse_epoch_line(line: str) -> tuple[int, dict[str, float | int]] | None:
    m = EPOCH_RE.search(line)
    if not m:
        return None
    epoch = int(m.group("epoch"))
    metrics: dict[str, float | int] = {
        "epoch": epoch - 1,
        "ramp_live/epoch_time_s": float(m.group("time")),
        "ramp_live/train_loss": float(m.group("loss")),
    }

    train_start = line.find("[eval train]")
    test_start = line.find("[eval test]")
    if train_start >= 0 and test_start >= 0:
        train_block = line[train_start + len("[eval train]"):test_start]
        test_block = line[test_start + len("[eval test]"):]
        metrics.update(_parse_pairs(train_block, "ramp_live/train"))
        metrics.update(_parse_pairs(test_block, "ramp_live/test"))
    return epoch - 1, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ramp-root", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fname", required=True)
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    ramp_root = Path(args.ramp_root).resolve()
    run_dir = ramp_root / "trained_models" / args.fname
    metrics_path = run_dir / "metrics.pth"

    cfg = {
        "method": "ramp",
        "run_name": f"ramp_seed{args.seed}",
        "seed": args.seed,
        "external": {
            "ramp_commit": _ramp_commit(repo_root),
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
        },
        "wandb": {
            "mode": args.wandb_mode,
            "project": "union-robustness-dro",
            "entity": None,
            "group": "ramp",
            "tags": ["ramp", "external", "live"],
        },
    }
    logger = WandbLogger(cfg, run_name=f"ramp_seed{args.seed}")

    cmd = [
        args.python,
        "-u",
        "RAMP.py",
        "--lr-max", "0.05",
        "--lr-schedule=static",
        "--at_iter", "10",
        "--epochs", "80",
        "--save_freq", "10",
        "--eval_freq", "10",
        "--fname", args.fname,
        "--kl",
        "--max",
        "--final_eval",
        "--gp",
        "--lbd", "5",
        "--seed", str(args.seed),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("PYTHONUNBUFFERED", "1")

    parsed_history = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(ramp_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        parsed = parse_epoch_line(line)
        if parsed is None:
            continue
        step, metrics = parsed
        parsed_history.append(metrics)
        logger.log(metrics, step=step)

    code = proc.wait()
    summary = {}
    raw_metrics = None
    if metrics_path.exists():
        stats = torch.load(metrics_path, map_location="cpu", weights_only=False)
        summary = final_metrics(stats)
        raw_metrics = _to_list(stats)
        if summary:
            logger.summary(summary)
    logger.finish()

    save_json(
        {
            "cfg": cfg,
            "history": parsed_history,
            "summary": summary,
            "raw_metrics": raw_metrics,
            "return_code": code,
        },
        args.out,
    )
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

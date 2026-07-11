#!/usr/bin/env python
"""Thin training CLI. Reads a YAML config and calls src/robustdro.

Examples
--------
    # Paper reactive run under the locked RAMP recipe
    python scripts/train.py --config configs/paper/reactive_ramprecipe.yaml

    # After training, evaluate ckpt/val_best.pt and ckpt/last.pt on both
    # val_select and test_monitor, logging back to the same W&B training run.
    python scripts/post_train_develop_eval.py --config configs/paper/reactive_ramprecipe.yaml

    # 2-iteration smoke test on GPU (offline wandb, tiny eval)
    python scripts/train.py --config configs/paper/reactive_ramprecipe.yaml --smoke

    # Ad-hoc overrides
    python scripts/train.py --config configs/paper/reactive_ramprecipe.yaml --epochs 10 --wandb-mode offline
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `import robustdro` work without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robustdro.training import Trainer          # noqa: E402
from robustdro.training.groupdro import GroupDROTrainer  # noqa: E402
from robustdro.utils.io import apply_overrides, load_config  # noqa: E402
from robustdro.utils.seed import set_seed        # noqa: E402

# method -> trainer class
_TRAINERS = {"pgd_at": Trainer, "attackdro": GroupDROTrainer}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint path, or 'auto' (=<run>/s<seed>/ckpt/last.pt). "
                        "Restores model+optimizer+scheduler+epoch+RNG for a valid continuation.")
    p.add_argument("--save-freq", type=int, default=None,
                   help="Checkpoint every N epochs (ep010.pt ...) for epoch curves + continuation.")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Generic dotted-key config override, repeatable. "
             "e.g. --set train.groupdro.temperature=0.5 (YAML-parsed value)",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="2 train steps + 2 eval batches, 1 epoch, wandb offline. Proves the "
             "full path (data->model->GPU->backward->log) without a long run.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {
        "train.epochs": args.epochs,
        "seed": args.seed,
        "wandb.mode": args.wandb_mode,
        "run_name": args.run_name,
        "train.resume": args.resume,
        "train.save_freq": args.save_freq,
    }
    # Generic --set KEY=VALUE overrides (YAML-parsed, so numbers/bools work).
    import yaml
    for kv in args.set:
        key, _, raw = kv.partition("=")
        if not _:
            raise ValueError(f"--set expects KEY=VALUE, got {kv!r}")
        overrides[key.strip()] = yaml.safe_load(raw)
    cfg = apply_overrides(cfg, overrides)

    smoke_kwargs = {}
    if args.smoke:
        cfg["train"]["epochs"] = args.epochs or 1
        cfg["wandb"]["mode"] = args.wandb_mode or "offline"
        cfg["run_name"] = (cfg.get("run_name") or "run") + "_smoke"
        smoke_kwargs = {"max_steps_per_epoch": 2, "eval_max_batches": 2}
        print(f"[smoke] {cfg['train']['epochs']} epoch(s), 2 train steps, "
              "2 eval batches, wandb offline unless overridden")

    set_seed(cfg["seed"])
    method = cfg.get("method", "pgd_at")
    trainer_cls = _TRAINERS.get(method)
    if trainer_cls is None:
        raise ValueError(f"Unknown method '{method}'. Known: {list(_TRAINERS)}")
    print(f"[train] method={method} -> {trainer_cls.__name__}")
    trainer = trainer_cls(cfg)
    result = trainer.fit(**smoke_kwargs)
    print(f"[done] {result}")


if __name__ == "__main__":
    main()

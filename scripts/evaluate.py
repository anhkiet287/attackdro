#!/usr/bin/env python
"""Evaluate a checkpoint's worst-case UNION robustness (linf, l2, l1).

The evaluation linchpin — works on both self-trained checkpoints and downloaded
baselines, always under the shared protocol (configs/base.yaml). Uses AutoAttack
per norm and the union rule.

Examples
--------
    # Quick eval on 1000 test images (APGD-CE + APGD-T), our PGD-AT checkpoint:
    python scripts/evaluate.py --checkpoint checkpoints/pgd_at_linf_best.pt -n 1000

    # Final numbers: full AutoAttack on the whole test set:
    python scripts/evaluate.py --checkpoint checkpoints/pgd_at_linf_best.pt \
        --version standard --n-examples 10000

    # Just linf (e.g. sanity vs the in-training PGD probe):
    python scripts/evaluate.py --checkpoint ckpt.pt --norms linf

    # Upstream RAMP checkpoint under our independent union harness:
    python scripts/evaluate.py --model_family ramp --config configs/base.yaml \
        --checkpoint external/RAMP/models/pretr_Linf.pth -n 1000
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robustdro.eval import evaluate_union, load_eval_checkpoint, load_test_subset  # noqa: E402
from robustdro.utils.io import apply_overrides, load_config, save_json            # noqa: E402
from robustdro.utils.seed import set_seed                     # noqa: E402
from robustdro.utils.wandb_log import log_eval_summary        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pgd_at.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-family", "--model_family", dest="model_family",
                   default="robustdro", choices=["robustdro", "ramp"],
                   help="Checkpoint/model format. 'robustdro' preserves existing behavior; "
                        "'ramp' loads external/RAMP/model_zoo.fast_models.PreActResNet18.")
    p.add_argument("-n", "--n-examples", type=int, default=1000,
                   help="Number of test images (first n; deterministic). Use 10000 for full.")
    p.add_argument("--norms", nargs="+", default=["linf", "l2", "l1"],
                   choices=["linf", "l2", "l1"])
    p.add_argument("--version", default="apgd", choices=["apgd", "standard"],
                   help="apgd = APGD-CE+APGD-T (fast); standard = full AutoAttack (final).")
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON output path (default results/eval_<ckpt>.json)")
    p.add_argument("--run-name", default=None,
                   help="W&B run to attach eval summary to (default: checkpoint stem sans _best/_last). "
                        "Our trainers use {method}_{variant}_s{seed}; pass it to land eval on the SAME run.")
    p.add_argument("--tier", default=None, choices=["in-house", "repro", "official"],
                   help="Comparison-lane tag for W&B (default: cfg wandb.tier; RAMP evals should pass 'repro').")
    p.add_argument("--no-wandb", action="store_true", help="Skip the W&B eval-summary log (JSON still written).")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Generic dotted-key config override, repeatable. "
             "e.g. --set threat_model.linf.eps=0.03137254901960784 "
             "--set eval_attack.linf.restarts=10",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    import yaml
    overrides = {}
    for kv in args.set:
        key, _, raw = kv.partition("=")
        if not _:
            raise ValueError(f"--set expects KEY=VALUE, got {kv!r}")
        overrides[key.strip()] = yaml.safe_load(raw)
    cfg = apply_overrides(cfg, overrides)
    device = args.device if torch.cuda.is_available() else "cpu"

    model, ckpt = load_eval_checkpoint(
        args.checkpoint, cfg, model_family=args.model_family, device=device
    )
    epoch = ckpt.get("epoch", "?") if ckpt is not None else "external"
    print(f"[eval] model_family={args.model_family}  checkpoint={args.checkpoint} "
          f"(epoch {epoch})  device={device}")

    x, y = load_test_subset(cfg, n_examples=args.n_examples)
    metrics = evaluate_union(model, x, y, cfg, norms=tuple(args.norms),
                             version=args.version, device=device, bs=args.bs, seed=args.seed)

    print("\n===== UNION ROBUSTNESS =====")
    print(f"clean          : {metrics['clean_acc']:.4f}")
    for k, v in metrics["per_norm_robust_acc"].items():
        print(f"robust {k:>4s}   : {v:.4f}")
    print(f"average-case   : {metrics['avg_robust_acc']:.4f}")
    print(f"WORST-UNION    : {metrics['worst_union_acc']:.4f}   <-- primary metric")

    if args.out:
        out = args.out
    else:
        # Default into the per-run layout: results/<run>/s<seed>/eval[_fullAA].json
        # (screening APGD -> eval.json; full AutoAttack -> eval_fullAA.json). Explicit
        # --out still wins (e.g. flat RAMP-reference evals).
        from robustdro.utils.io import run_paths as _rp
        _rn = args.run_name or re.sub(r"_(best|last|ep\d+)$", "",
                                      os.path.splitext(os.path.basename(args.checkpoint))[0])
        _p = _rp({"run_name": _rn, "seed": args.seed, "results_dir": "results/"})
        out = _p["eval_fullAA"] if args.version == "standard" else _p["eval"]
    # Training protocol recorded FROM THE CHECKPOINT'S OWN cfg (if present), so
    # the views can auto-flag a train/eval eps MISMATCH (e.g. a 0.03-trained ckpt
    # evaluated at 8/255 is a lower bound, not a paper number). None for external
    # ckpts (ramp/official) that carry no cfg.
    train_tm = None
    if isinstance(ckpt, dict):
        train_tm = (ckpt.get("cfg") or {}).get("threat_model")
    train_eps = (train_tm or {}).get("linf", {}).get("eps") if train_tm else None
    eval_eps = cfg["threat_model"]["linf"]["eps"]
    mismatch = train_eps is not None and abs(float(train_eps) - float(eval_eps)) > 1e-6

    save_json({
        "checkpoint": args.checkpoint,
        "model_family": args.model_family,
        "epoch": ckpt.get("epoch") if ckpt is not None else None,
        "n_examples": args.n_examples,
        "protocol": {"threat_model": cfg["threat_model"], "eval_attack": cfg.get("eval_attack")},
        "train_protocol": {"threat_model": train_tm} if train_tm else None,
        "train_eval_eps_mismatch": mismatch,
        "metrics": metrics,
    }, out)
    print(f"\n[eval] saved -> {out}")
    if mismatch:
        print(f"[eval] ⚠ train/eval eps MISMATCH: trained@{float(train_eps):g}, "
              f"eval@{float(eval_eps):g} — this row is a LOWER BOUND, not a paper number.")

    # W&B view: attach union + per-norm to the run (JSON above is source of truth).
    if not args.no_wandb:
        run_name = args.run_name or re.sub(r"_(best|last)$", "",
                                           os.path.splitext(os.path.basename(args.checkpoint))[0])
        tier = args.tier or ("repro" if args.model_family == "ramp" else None)
        log_eval_summary(cfg, run_name, metrics, version=args.version,
                         tier=tier, n_examples=args.n_examples)


if __name__ == "__main__":
    main()

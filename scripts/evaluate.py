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
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robustdro.eval import evaluate_union, load_test_subset   # noqa: E402
from robustdro.models import build_model                      # noqa: E402
from robustdro.utils.io import load_config, save_json         # noqa: E402
from robustdro.utils.seed import set_seed                     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pgd_at.yaml")
    p.add_argument("--checkpoint", required=True)
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
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("cfg", cfg)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[eval] checkpoint={args.checkpoint} (epoch {ckpt.get('epoch', '?')})  device={device}")

    x, y = load_test_subset(cfg, n_examples=args.n_examples)
    metrics = evaluate_union(model, x, y, cfg, norms=tuple(args.norms),
                             version=args.version, device=device, bs=args.bs, seed=args.seed)

    print("\n===== UNION ROBUSTNESS =====")
    print(f"clean          : {metrics['clean_acc']:.4f}")
    for k, v in metrics["per_norm_robust_acc"].items():
        print(f"robust {k:>4s}   : {v:.4f}")
    print(f"average-case   : {metrics['avg_robust_acc']:.4f}")
    print(f"WORST-UNION    : {metrics['worst_union_acc']:.4f}   <-- primary metric")

    out = args.out or os.path.join(
        "results", f"eval_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json"
    )
    save_json({
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "n_examples": args.n_examples,
        "protocol": {"threat_model": cfg["threat_model"], "eval_attack": cfg.get("eval_attack")},
        "metrics": metrics,
    }, out)
    print(f"\n[eval] saved -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate our eval_union harness against the official locuslab/robust_union.

Golden rule #3: validate against the authors' reported numbers before using a
baseline in any comparison. This loads an official robust_union checkpoint with
THEIR model definition (avoids state_dict / preprocessing mismatch) and runs OUR
eval_union under THEIR protocol (eps_inf=0.03, eps_2=0.5, eps_1=12, first 1000
test images). We expect our AutoAttack numbers to be <= their reported PGD
numbers (APGD is a stronger attack); a large gap signals a harness bug.

Reported (Maini et al. 2020, CIFAR10, first 1000, 10 restarts):
  model   clean   linf(0.03)  l2(0.5)  l1(12)   All(union)
  MSD_V0  81.7    47.6        64.3     53.4     46.1
  LINF    83.3    50.7        57.3     16.0     15.6
  L2      90.2    28.3        61.6     46.6     27.5
  L1      73.3     0.2         0.0      7.9      0.0

Example
-------
    python scripts/validate_baseline.py --model MSD_V0 -n 1000 --version apgd
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from robustdro.eval import evaluate_union, load_test_subset   # noqa: E402
from robustdro.utils.io import load_config                    # noqa: E402
from robustdro.utils.seed import set_seed                     # noqa: E402

# Their reported numbers, for an automatic sanity comparison.
# (Drive filenames: MSD, LINF, L2, L1, AVG=PGD-Aug, MAX=Worst-PGD.)
REPORTED = {
    "MSD":  {"clean": 81.7, "linf": 47.6, "l2": 64.3, "l1": 53.4, "union": 46.1},
    "LINF": {"clean": 83.3, "linf": 50.7, "l2": 57.3, "l1": 16.0, "union": 15.6},
    "L2":   {"clean": 90.2, "linf": 28.3, "l2": 61.6, "l1": 46.6, "union": 27.5},
    "L1":   {"clean": 73.3, "linf": 0.2,  "l2": 0.0,  "l1": 7.9,  "union": 0.0},
    "AVG":  {"clean": 84.6, "linf": 42.5, "l2": 65.0, "l1": 54.0, "union": 40.6},
    "MAX":  {"clean": 81.0, "linf": 44.9, "l2": 61.7, "l1": 39.4, "union": 34.9},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="MSD",
                   choices=["MSD", "LINF", "L2", "L1", "AVG", "MAX"])
    p.add_argument("--repo", default=os.path.join(REPO_ROOT, "external/robust_union/CIFAR10"))
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("-n", "--n-examples", type=int, default=1000)
    p.add_argument("--version", default="apgd", choices=["apgd", "standard"])
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Their model definition + checkpoint. Import the module directly (their
    # models/__init__.py uses a py2-style implicit import that breaks on py3).
    sys.path.insert(0, os.path.join(args.repo, "models"))
    from preact_resnet import PreActResNet18  # noqa: E402  (their class)

    ckpt_path = os.path.join(args.repo, "Selected", args.model + ".pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PreActResNet18()
    model.load_state_dict(state)          # fp16 ckpt casts into fp32 params
    model.float().to(device).eval()
    print(f"[validate] loaded {args.model} from {ckpt_path}")

    # THEIR protocol: eps_inf = 0.03 (NOT 8/255).
    cfg = load_config(args.config)
    cfg["threat_model"]["linf"]["eps"] = 0.03
    cfg["threat_model"]["l2"]["eps"] = 0.5
    cfg["threat_model"]["l1"]["eps"] = 12.0

    x, y = load_test_subset(cfg, n_examples=args.n_examples)
    metrics = evaluate_union(model, x, y, cfg, norms=("linf", "l2", "l1"),
                             version=args.version, device=device, bs=args.bs, seed=args.seed)

    ours = {
        "clean": metrics["clean_acc"] * 100,
        "linf": metrics["per_norm_robust_acc"]["linf"] * 100,
        "l2": metrics["per_norm_robust_acc"]["l2"] * 100,
        "l1": metrics["per_norm_robust_acc"]["l1"] * 100,
        "union": metrics["worst_union_acc"] * 100,
    }
    rep = REPORTED.get(args.model, {})
    print(f"\n===== {args.model}: OURS (AutoAttack-{args.version}, n={args.n_examples}) "
          f"vs REPORTED (PGD, 10 restarts) =====")
    print(f"{'metric':<8}{'ours':>8}{'reported':>10}{'delta':>8}")
    for k in ["clean", "linf", "l2", "l1", "union"]:
        r = rep.get(k)
        d = f"{ours[k]-r:+.1f}" if r is not None else "  n/a"
        rs = f"{r:.1f}" if r is not None else "n/a"
        print(f"{k:<8}{ours[k]:>8.1f}{rs:>10}{d:>8}")
    print("\nExpectation: ours <= reported (APGD stronger than their PGD). "
          "A large positive delta or wildly-low ours => harness/model bug.")


if __name__ == "__main__":
    main()

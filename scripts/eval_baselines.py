#!/usr/bin/env python
"""Evaluate ALL official robust_union baselines under the LOCKED protocol.

Produces THE baseline table (clean, linf, l2, l1, worst-union) that everything
downstream is compared against. Uses their PreActResNet18 + our eval_union at the
locked eps (configs/base.yaml: eps_inf=0.03, l2=0.5, l1=12), APGD version.

Emits results/baseline_table.json and results/baseline_table.md.

    python scripts/eval_baselines.py -n 1000 --version apgd
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from robustdro.eval import evaluate_union, load_test_subset   # noqa: E402
from robustdro.utils.io import load_config, save_json         # noqa: E402
from robustdro.utils.seed import set_seed                     # noqa: E402

# Drive filenames -> human label / method family.
MODELS = ["MSD", "AVG", "MAX", "LINF", "L2", "L1"]
# Reported union (their PGD/10-restart) for context only.
REPORTED_UNION = {"MSD": 46.1, "AVG": 40.6, "MAX": 34.9, "LINF": 15.6, "L2": 27.5, "L1": 0.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.path.join(REPO_ROOT, "external/robust_union/CIFAR10"))
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("-n", "--n-examples", type=int, default=1000)
    p.add_argument("--version", default="apgd", choices=["apgd", "standard"])
    p.add_argument("--bs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=MODELS)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, os.path.join(args.repo, "models"))
    from preact_resnet import PreActResNet18  # noqa: E402  (their class)

    cfg = load_config(args.config)
    tm = cfg["threat_model"]
    print(f"[baselines] LOCKED protocol: eps_inf={tm['linf']['eps']} "
          f"l2={tm['l2']['eps']} l1={tm['l1']['eps']}  version={args.version}  n={args.n_examples}")

    x, y = load_test_subset(cfg, n_examples=args.n_examples)

    rows = {}
    for name in args.models:
        ckpt = os.path.join(args.repo, "Selected", name + ".pt")
        if not os.path.exists(ckpt):
            print(f"[baselines] SKIP {name}: {ckpt} missing")
            continue
        model = PreActResNet18()
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.float().to(device).eval()
        print(f"\n[baselines] === {name} ===")
        m = evaluate_union(model, x, y, cfg, norms=("linf", "l2", "l1"),
                           version=args.version, device=device, bs=args.bs, seed=args.seed)
        rows[name] = {
            "clean": m["clean_acc"] * 100,
            "linf": m["per_norm_robust_acc"]["linf"] * 100,
            "l2": m["per_norm_robust_acc"]["l2"] * 100,
            "l1": m["per_norm_robust_acc"]["l1"] * 100,
            "union": m["worst_union_acc"] * 100,
        }

    # ---- Emit table -------------------------------------------------------
    header = f"| model | clean | l_inf | l2 | **l1** | **worst-union** | (reported union) |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for name in args.models:
        if name not in rows:
            continue
        r = rows[name]
        lines.append(f"| {name} | {r['clean']:.1f} | {r['linf']:.1f} | {r['l2']:.1f} | "
                     f"**{r['l1']:.1f}** | **{r['union']:.1f}** | {REPORTED_UNION.get(name,'-')} |")
    table_md = "\n".join(lines)

    print("\n" + "=" * 70)
    print(f"BASELINE TABLE (locked: eps_inf={tm['linf']['eps']}, n={args.n_examples}, {args.version})")
    print("=" * 70)
    print(table_md)

    os.makedirs("results", exist_ok=True)
    save_json({
        "protocol": {"threat_model": tm, "version": args.version, "n_examples": args.n_examples},
        "rows": rows,
    }, "results/baseline_table.json")
    with open("results/baseline_table.md", "w") as f:
        f.write(f"# Baseline table (locked protocol)\n\n"
                f"eps_inf={tm['linf']['eps']}, l2={tm['l2']['eps']}, l1={tm['l1']['eps']}; "
                f"n={args.n_examples}; AutoAttack version={args.version}; "
                f"models = locuslab/robust_union official checkpoints.\n\n{table_md}\n")
    print("\n[baselines] saved -> results/baseline_table.{json,md}")


if __name__ == "__main__":
    main()

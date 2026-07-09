#!/usr/bin/env python
"""Standard-AutoAttack pack: final-number evals (version=standard) + APGD delta.

Targets: our finalist seed-0 checkpoints + MSD official ckpt (+ avg_frozen_s0 if
its checkpoint exists). Skips any target whose eval_std_*.json already exists.
Writes results/eval_std_<name>.json per model and results/standard_vs_apgd.md
with the APGD -> standard delta per model.

    python scripts/eval_standard_pack.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO, "src"))

from robustdro.eval import evaluate_union, load_test_subset  # noqa: E402
from robustdro.models import build_model                     # noqa: E402
from robustdro.utils.io import load_config, save_json        # noqa: E402
from robustdro.utils.seed import set_seed                    # noqa: E402

# (name, kind, path) — kind: ours | locuslab
TARGETS = [
    ("bindaware_sample_T025", "ours", "checkpoints/bindaware_sample_T025_best.pt"),
    ("bindaware_sample_T05", "ours", "checkpoints/bindaware_sample_T05_best.pt"),
    ("bindaware_s0", "ours", "checkpoints/bindaware_s0_best.pt"),
    ("MSD_official", "locuslab", "external/robust_union/CIFAR10/Selected/MSD.pt"),
    ("avg_frozen_s0", "ours", "checkpoints/avg_frozen_s0_best.pt"),  # skipped if absent
]
APGD_JSON = {  # for the delta column
    "bindaware_sample_T025": "results/eval_bindaware_sample_T025.json",
    "bindaware_sample_T05": "results/eval_bindaware_sample_T05.json",
    "bindaware_s0": "results/eval_bindaware_s0.json",
    "MSD_official": None,  # baseline_table.json holds it
    "avg_frozen_s0": "results/eval_avg_frozen_s0.json",
}


def load_model(kind, path, device):
    if kind == "ours":
        ck = torch.load(path, map_location=device, weights_only=False)
        m = build_model(ck["cfg"])
        m.load_state_dict(ck["model"])
    else:
        sys.path.insert(0, os.path.join(REPO, "external/robust_union/CIFAR10/models"))
        from preact_resnet import PreActResNet18  # noqa: E402
        m = PreActResNet18()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    return m.float().to(device).eval()


def apgd_union(name):
    p = APGD_JSON.get(name)
    if p and os.path.exists(p):
        return json.load(open(p))["metrics"]["worst_union_acc"] * 100
    if name == "MSD_official":
        try:
            return json.load(open("results/baseline_table.json"))["rows"]["MSD"]["union"]
        except Exception:
            return None
    return None


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("configs/base.yaml")
    x, y = load_test_subset(cfg, n_examples=1000)

    rows = []
    for name, kind, path in TARGETS:
        out = f"results/eval_std_{name}.json"
        if os.path.exists(out):
            print(f"[std] {name}: already done -> skip")
            m = json.load(open(out))["metrics"]
        elif not os.path.exists(path):
            print(f"[std] {name}: checkpoint missing ({path}) -> skip")
            continue
        else:
            print(f"[std] {name}: running standard AutoAttack (n=1000)...")
            model = load_model(kind, path, device)
            m = evaluate_union(model, x, y, cfg, norms=("linf", "l2", "l1"),
                               version="standard", device=device, bs=250, seed=0)
            save_json({"checkpoint": path, "n_examples": 1000,
                       "protocol": {"threat_model": cfg["threat_model"]},
                       "metrics": m}, out)
        std_u = m["worst_union_acc"] * 100
        ap = apgd_union(name)
        rows.append((name, std_u, ap, None if ap is None else std_u - ap))

    lines = ["# standard AutoAttack vs APGD (n=1000, locked protocol)", "",
             "| model | standard union | apgd union | delta (std−apgd) |", "|---|---|---|---|"]
    for name, s, a, d in rows:
        lines.append(f"| {name} | {s:.1f} | {'-' if a is None else f'{a:.1f}'} | "
                     f"{'-' if d is None else f'{d:+.1f}'} |")
    open("results/standard_vs_apgd.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

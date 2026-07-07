#!/usr/bin/env python
"""F5 figure: probe (weak attack) vs APGD per-norm robust accuracy.

Grouped bars per norm; grayscale-safe via hatching (probe = hatched white,
APGD = solid). Value labels on every bar (relief rule). Shows both the per-norm
bias and the ranking inversion.

    python scripts/figures/plot_probe_bias.py \
        --log logs/attackdro_union_s0.log --epoch 33 \
        --eval results/eval_attackdro_union_s0.json \
        --out results/figures/probe_bias_f5.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt  # noqa: E402
from fig_style import NORM_LABEL, NORMS, apply_style  # noqa: E402


def probe_at_epoch(log_path, epoch):
    for line in open(log_path):
        if line.startswith(f"[step {epoch}]") and f"epoch={epoch}" in line:
            vals = dict(re.findall(r"probe/robust_(\w+)=([0-9.]+)", line))
            return {n: float(vals[n]) * 100 for n in NORMS}
    raise SystemExit(f"epoch {epoch} probe line not found in {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--epoch", type=int, required=True, help="ckpt epoch to read probe at")
    p.add_argument("--eval", required=True, help="eval_union JSON for the same ckpt")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    probe = probe_at_epoch(args.log, args.epoch)
    m = json.load(open(args.eval))["metrics"]["per_norm_robust_acc"]
    apgd = {n: m[n] * 100 for n in NORMS}

    apply_style()
    fig, ax = plt.subplots()
    x = range(len(NORMS))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], [probe[n] for n in NORMS], w,
                facecolor="white", edgecolor="0.2", hatch="///",
                label=f"probe (PGD-{20})")
    b2 = ax.bar([i + w / 2 for i in x], [apgd[n] for n in NORMS], w,
                facecolor="0.35", edgecolor="0.2", label="APGD (CE+T, 100 it)")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
    for i, n in enumerate(NORMS):
        d = probe[n] - apgd[n]
        ax.annotate(f"+{d:.1f}", (i, max(probe[n], apgd[n]) + 6),
                    ha="center", fontsize=7, color="0.1")
    ax.set_xticks(list(x))
    ax.set_xticklabels([NORM_LABEL[n] for n in NORMS])
    ax.set_ylabel("robust accuracy (%)")
    ax.set_ylim(0, max(probe.values()) + 14)
    # Legend above the axes so it never collides with bar labels.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")
    # Ranking-inversion note to stdout (goes in the caption).
    pw = min(probe, key=probe.get)
    aw = min(apgd, key=apgd.get)
    print(f"weakest by probe = {pw}; weakest by APGD = {aw}; inverted = {pw != aw}")


if __name__ == "__main__":
    main()

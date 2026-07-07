#!/usr/bin/env python
"""Worst-union vs temperature T (CARD-3b sweep figure).

X = log-scale T with the two analytic anchors: hard-MAX baseline (T -> 0,
locuslab MAX = 25.0) and AVG (T -> inf, locuslab AVG = 38.7). Points = the
1-seed T-sweep pilots; MSD = 42.5 reference line.

    python scripts/figures/plot_union_vs_T.py --out results/figures/union_vs_T.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt  # noqa: E402
from fig_style import apply_style  # noqa: E402

MAX_ANCHOR = 25.0   # locuslab MAX *checkpoint* under our harness = T->0 reference.
# NOTE (POSITIONING.md par.3): Croce&Hein's own retrained MAX reports 44.0 (their
# recipe/protocol) — 25.0 reflects that specific checkpoint, NOT hard-max in principle.
AVG_ANCHOR = 38.7   # locuslab AVG = T->inf limit
MSD_REF = 42.5

TAG2T = {"T025": 0.25, "T05": 0.5, "T1": 1.0, "T2": 2.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures/union_vs_T.png")
    args = p.parse_args()

    pts = []
    for path in glob.glob("results/eval_bindaware_sample_T*.json"):
        tag = re.search(r"_T(\w+)\.json", path).group(0)[1:-5]
        T = TAG2T.get(tag)
        if T is None:
            continue
        m = json.load(open(path))["metrics"]
        pts.append((T, m["worst_union_acc"] * 100))
    pts.sort()
    if not pts:
        raise SystemExit("no sweep evals found yet")

    apply_style()
    fig, ax = plt.subplots()
    xs = [t for t, _ in pts]
    ys = [u for _, u in pts]
    ax.plot(xs, ys, "o-", color="#0072B2", label="per-sample soft (1 seed)")
    for t, u in pts:
        ax.annotate(f"{u:.1f}", (t, u), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=7)
    # Fixed frame: anchors sit INSIDE the plot area at fixed x positions.
    x_lo, x_hi = 0.25 / 2.2, 2 * 2.2
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.scatter([x_lo * 1.15], [MAX_ANCHOR], marker="v", facecolor="white",
               edgecolor="0.2", zorder=3, clip_on=False)
    ax.annotate(f"locuslab MAX ckpt {MAX_ANCHOR:.1f} (T→0 ref.)", (x_lo * 1.15, MAX_ANCHOR),
                textcoords="offset points", xytext=(8, 0), fontsize=6.5, va="center")
    ax.scatter([x_hi / 1.15], [AVG_ANCHOR], marker="^", facecolor="white",
               edgecolor="0.2", zorder=3, clip_on=False)
    ax.annotate(f"AVG {AVG_ANCHOR:.1f} (T→∞)", (x_hi / 1.15, AVG_ANCHOR),
                textcoords="offset points", xytext=(-8, 0), fontsize=6.5,
                ha="right", va="center")
    ax.axhline(MSD_REF, color="0.4", lw=0.9, ls="--")
    ax.text(x_lo * 1.05, MSD_REF + 0.3, f"MSD {MSD_REF}", fontsize=6.5, color="0.3")
    # Fixed ticks at the sweep's T values only (auto log ticks collide).
    ax.set_xticks([0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xlabel("temperature $T$ (log scale)")
    ax.set_ylabel("worst-case union acc. (%)")
    ax.legend(loc="center left")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}  points={pts}")


if __name__ == "__main__":
    main()

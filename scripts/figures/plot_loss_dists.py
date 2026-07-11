#!/usr/bin/env python
"""Per-norm per-sample loss distributions over epochs (CARD-3b amendment #1).

Reads dist/<norm>/{p10,p50,p90} from a run's results JSON history and plots one
small-multiple panel per norm: p50 line + p10-p90 band. This is the scale-bias
check: if one norm's whole band sits above the others, the per-sample softmax
in CARD-3b inherits that bias (-> zscore variant).

    python scripts/figures/plot_loss_dists.py \
        --results results/bindaware_sample_T05.json \
        --out results/figures/loss_dists_T05.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt  # noqa: E402
from fig_style import FIG_W, NORM_COLOR, NORM_LABEL, NORM_LS, NORMS, apply_style  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="results/<run>.json (with history)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    hist = json.load(open(args.results))["history"]
    hist = [h for h in hist if f"dist/{NORMS[0]}/p50" in h]
    if not hist:
        raise SystemExit(f"no dist/* keys in {args.results}")
    epochs = [h.get("epoch", i) for i, h in enumerate(hist)]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 2, 2.2), sharey=True)
    for ax, n in zip(axes, NORMS):
        p10 = [h[f"dist/{n}/p10"] for h in hist]
        p50 = [h[f"dist/{n}/p50"] for h in hist]
        p90 = [h[f"dist/{n}/p90"] for h in hist]
        if len(epochs) > 1:
            ax.fill_between(epochs, p10, p90, color=NORM_COLOR[n], alpha=0.25, lw=0)
            ax.plot(epochs, p50, color=NORM_COLOR[n], ls=NORM_LS[n])
        else:  # single epoch (smoke) — draw point + error bar instead of a band
            ax.errorbar(epochs, p50, yerr=[[p50[0] - p10[0]], [p90[0] - p50[0]]],
                        fmt="o", color=NORM_COLOR[n], capsize=3)
        ax.set_title(NORM_LABEL[n])
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("per-sample CE loss\n(p50, band = p10–p90)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out} ({len(epochs)} epochs)")


if __name__ == "__main__":
    main()

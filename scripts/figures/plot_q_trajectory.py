#!/usr/bin/env python
"""q-trajectory figure + CSV dump (mechanism Figure 1 candidate).

Parses a training log's per-epoch `q/<norm>=` values and plots the full
trajectory of the group weights. Also writes the raw trajectory to CSV so the
data survives independent of the log file.

    python scripts/figures/plot_q_trajectory.py --log logs/bindaware_s0.log \
        --out results/figures/q_traj_bindaware_s0.png --title "CARD-3a pilot"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt  # noqa: E402
from fig_style import NORM_COLOR, NORM_LABEL, NORM_LS, NORM_MARKER, NORMS, apply_style  # noqa: E402


def parse_log(path):
    rows = []  # (epoch, {norm: q})
    for line in open(path):
        if not line.startswith("[step"):
            continue
        ep = re.search(r"epoch=(\d+)", line)
        qs = dict(re.findall(r"q/(\w+)=([0-9.]+)", line))
        if ep and len(qs) == len(NORMS):
            rows.append((int(ep.group(1)), {n: float(qs[n]) for n in NORMS}))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--csv", default=None, help="default: <out>.csv")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"no q/* epoch lines found in {args.log}")

    csv_path = args.csv or os.path.splitext(args.out)[0] + ".csv"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + [f"q_{n}" for n in NORMS])
        for ep, q in rows:
            w.writerow([ep] + [f"{q[n]:.6f}" for n in NORMS])

    apply_style()
    fig, ax = plt.subplots()
    epochs = [r[0] for r in rows]
    mark_every = max(1, len(epochs) // 10)
    for n in NORMS:
        ax.plot(epochs, [r[1][n] for r in rows], color=NORM_COLOR[n],
                ls=NORM_LS[n], marker=NORM_MARKER[n], markevery=mark_every,
                label=NORM_LABEL[n])
    ax.axhline(1 / 3, color="0.5", lw=0.8, ls=":", zorder=0)
    ax.text(epochs[-1], 1 / 3, " uniform", va="center", fontsize=6, color="0.4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("group weight $q_g$")
    ax.set_ylim(0, 1)
    if args.title:
        ax.set_title(args.title)
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(args.out)
    print(f"saved {args.out} (+ {csv_path}, {len(rows)} epochs)")


if __name__ == "__main__":
    main()

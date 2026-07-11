#!/usr/bin/env python
"""State-of-play figures from every eval result on disk (CPU-only).

1. Prints an inventory table of all results/eval_*.json + baseline_table.json
   (malformed files are FLAGGED, never guessed).
2. fig_state_of_play.png  — worst-union bar chart, ours vs official baselines.
3. fig_pernorm_finalists.png — per-norm grouped bars for the finalists.

    python scripts/figures/plot_state_of_play.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fig_style import FIG_W, apply_style  # noqa: E402

OUT = "results/figures/presentation"

# run-name -> display label (single-seed marking added at draw time)
LABELS = {
    "attackdro_union_s0": ("P1", "AttackDRO++ P1"),
    "attackdro_union_s1": ("P1", "AttackDRO++ P1"),
    "attackdro_union_s2": ("P1", "AttackDRO++ P1"),
    "bindaware_s0": ("solo", "3a binding-aware"),
    "bindaware_v2_s0": ("solo", "3a-v2 val-calibrated"),
    "bindaware_sample_T025": ("solo", "3b per-sample T=0.25"),
    "bindaware_sample_T05": ("solo", "3b per-sample T=0.5"),
    "bindaware_sample_T1": ("solo", "3b per-sample T=1"),
    "bindaware_sample_T2": ("solo", "3b per-sample T=2"),
    "pgd_at_linf_best": ("solo", "PGD-AT ℓ∞ (8/255 recipe)"),
}
BASELINE_ORDER = ["MSD", "AVG", "MAX", "LINF", "L2", "L1"]


def inventory():
    rows, flags = [], []
    for path in sorted(glob.glob("results/eval_*.json")):
        name = os.path.basename(path)[5:-5]
        try:
            d = json.load(open(path))
            m = d["metrics"]
            rows.append({
                "run": name, "clean": m["clean_acc"] * 100,
                "linf": m["per_norm_robust_acc"]["linf"] * 100,
                "l2": m["per_norm_robust_acc"]["l2"] * 100,
                "l1": m["per_norm_robust_acc"]["l1"] * 100,
                "union": m["worst_union_acc"] * 100,
                "n": d.get("n_examples", m.get("n", "?")),
                "version": m.get("version", "?"), "src": path,
            })
        except (KeyError, json.JSONDecodeError) as e:
            flags.append(f"MALFORMED {path}: {e}")
    base = {}
    try:
        base = json.load(open("results/baseline_table.json"))["rows"]
    except Exception as e:
        flags.append(f"MALFORMED results/baseline_table.json: {e}")

    hdr = f"{'run':<28}{'clean':>7}{'linf':>7}{'l2':>7}{'l1':>7}{'union':>7}{'n':>6}  {'ver':<6}{'source'}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<28}{r['clean']:>7.1f}{r['linf']:>7.1f}{r['l2']:>7.1f}"
              f"{r['l1']:>7.1f}{r['union']:>7.1f}{r['n']:>6}  {r['version']:<6}{r['src']}")
    for b in BASELINE_ORDER:
        if b in base:
            v = base[b]
            print(f"{b+' (official)':<28}{v['clean']:>7.1f}{v['linf']:>7.1f}{v['l2']:>7.1f}"
                  f"{v['l1']:>7.1f}{v['union']:>7.1f}{'1000':>6}  {'apgd':<6}results/baseline_table.json")
    for f in flags:
        print("FLAG:", f)
    return rows, base, flags


# ---------------------------------------------------------------- figure 2 --
def fig_state_of_play(rows, base):
    p1 = [r["union"] for r in rows if LABELS.get(r["run"], ("", ""))[0] == "P1"]
    bars = []  # (label, value, err, is_ours)
    if p1:
        bars.append((f"AttackDRO++ P1 (mean of {len(p1)} seeds)",
                     float(np.mean(p1)), float(np.std(p1)), True))
    for r in rows:
        kind, lab = LABELS.get(r["run"], ("solo", r["run"]))
        if kind == "P1":
            continue
        bars.append((f"{lab} (1 seed)", r["union"], 0.0, True))
    for b in BASELINE_ORDER:
        if b in base:
            bars.append((f"{b} (official ckpt)", base[b]["union"], 0.0, False))
    bars.sort(key=lambda x: x[1])  # ascending -> descending top-first in barh

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W * 1.7, 0.28 * len(bars) + 0.9))
    y = range(len(bars))
    for i, (lab, v, err, ours) in enumerate(bars):
        ax.barh(i, v, xerr=err if err else None, capsize=2.5,
                color="#0072B2" if ours else "0.62", height=0.62,
                error_kw={"lw": 0.9})
        ax.annotate(f"{v:.1f}" + (f"±{err:.1f}" if err else ""),
                    (v + (err or 0), i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=6.5)
    ax.axvline(42.5, color="0.25", lw=0.9, ls="--")
    ax.axvline(38.7, color="0.45", lw=0.9, ls=":")
    ax.text(42.5, len(bars) - 0.1, " MSD 42.5", fontsize=6.5, color="0.25", va="bottom")
    ax.text(38.7, -0.55, " AVG 38.7", fontsize=6.5, color="0.45", va="top")
    # RAMP (NeurIPS'24) reported from-scratch number — DIFFERENT protocol (eps_inf=8/255).
    ax.axvline(44.6, color="#B2182B", lw=0.9, ls="-.", alpha=0.8)
    ax.text(44.6, len(bars) - 1.6, " RAMP 44.6 ‡\n (reported, ε∞=8/255\n — not protocol-aligned)",
            fontsize=5.8, color="#B2182B", va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_xlabel("worst-case union robust acc. (%, APGD, n=1000)")
    ax.set_xlim(0, 50)
    path = f"{OUT}/fig_state_of_play.png"
    fig.savefig(path)
    print(f"\nsaved {path}")
    return path


# ---------------------------------------------------------------- figure 3 --
FINALISTS = [
    ("bindaware_sample_T025", "3b T=0.25", "#0072B2", "//"),
    ("bindaware_sample_T05", "3b T=0.5", "#56B4E9", "\\\\"),
    ("bindaware_s0", "3a", "#009E73", "xx"),
    ("MSD", "MSD", "0.35", None),
    ("AVG", "AVG", "0.70", None),
]
METRICS = [("clean", "clean"), ("linf", r"$\ell_\infty$"), ("l2", r"$\ell_2$"),
           ("l1", r"$\ell_1$"), ("union", "worst-∪")]


def fig_pernorm_finalists(rows, base):
    byrun = {r["run"]: r for r in rows}
    data = {}
    for key, lab, _, _ in FINALISTS:
        if key in byrun:
            data[lab] = byrun[key]
        elif key in base:
            data[lab] = base[key]
        else:
            print(f"FLAG: finalist {key} missing on disk -> omitted from fig_pernorm")
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W * 1.9, 2.5))
    nm, ns = len(METRICS), len(data)
    w = 0.8 / ns
    for si, (key, lab, color, hatch) in enumerate([f for f in FINALISTS if f[1] in data]):
        vals = [data[lab][mk] for mk, _ in METRICS]
        xs = [mi + (si - ns / 2 + 0.5) * w for mi in range(nm)]
        b = ax.bar(xs, vals, w * 0.94, label=lab + (" (1 seed)" if hatch else ""),
                   facecolor=color if hatch is None else "white",
                   edgecolor=color if hatch else "0.2", hatch=hatch, linewidth=0.8)
        ax.bar_label(b, fmt="%.1f", fontsize=5.2, padding=1, rotation=90)
    ax.set_xticks(range(nm))
    ax.set_xticklabels([ml for _, ml in METRICS])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 97)
    # The recorded l1/linf trade annotation — above the l1/worst-∪ groups, clear of bars.
    ax.annotate("ℓ1: ours 48.3 / 50.1 vs MSD 46.7\n— bought for <1pp ℓ∞",
                xy=(4.45, 74), ha="right", va="bottom", fontsize=6.5, style="italic")
    ax.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    path = f"{OUT}/fig_pernorm_finalists.png"
    fig.savefig(path)
    print(f"saved {path}")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, base, flags = inventory()
    fig_state_of_play(rows, base)
    fig_pernorm_finalists(rows, base)
    if flags:
        print("\n".join(flags))


if __name__ == "__main__":
    main()

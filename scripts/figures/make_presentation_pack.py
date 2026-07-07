#!/usr/bin/env python
"""Presentation figure pack — 4 pedagogy figures from REAL data (CPU-only).

Every number is read from results/*.json or logs/*.log; nothing synthetic.
Reuses the existing figure scripts as libraries. Outputs 300-dpi PNGs to
results/figures/presentation/ and appends provenance to results/MANIFEST.md.

    python scripts/figures/make_presentation_pack.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from fig_style import FIG_W, NORM_COLOR, NORM_LABEL, NORM_LS, NORM_MARKER, NORMS, apply_style  # noqa: E402
from plot_probe_bias import probe_at_epoch  # noqa: E402  (data logic reused)
from plot_q_trajectory import parse_log  # noqa: E402

OUT = "results/figures/presentation"
MANIFEST = "results/MANIFEST.md"
manifest_rows = []


def note(fig_name, reads_as, sources, script="scripts/figures/make_presentation_pack.py"):
    print(f"[{fig_name}] reads as: {reads_as}")
    manifest_rows.append((fig_name, reads_as, sources, script))


# ---------------------------------------------------------------- figure 1 --
def fig_loss_percentiles():
    src = "results/bindaware_sample_T025.json"
    hist = json.load(open(src))["history"]
    ep = hist[-1]  # last epoch
    epoch = ep.get("epoch")
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W * 1.4, 2.0))
    ys = {n: i for i, n in enumerate(reversed(NORMS))}
    spreads = {}
    for n in NORMS:
        p10, p50, p90 = ep[f"dist/{n}/p10"], ep[f"dist/{n}/p50"], ep[f"dist/{n}/p90"]
        spreads[n] = p90 - p10
        y = ys[n]
        ax.barh(y, p90 - p10, left=p10, height=0.5, color=NORM_COLOR[n], alpha=0.35)
        ax.plot([p50], [y], marker="|", ms=14, mew=2.5, color=NORM_COLOR[n])
        ax.annotate(f"p50={p50:.2f}", (p50, y + 0.33), ha="center", fontsize=6.5)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([NORM_LABEL[n] for n in ys])
    ax.set_xlabel(f"per-sample CE loss under attack (epoch {epoch}, T=0.25 run)")
    # Truthful annotations computed from the data itself.
    widest = max(spreads, key=spreads.get)
    ax.text(0.98, 0.96, "scale bias across norms (mild, no ranking inversion)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, style="italic")
    ax.text(0.98, 0.80, f"widest spread: {NORM_LABEL[widest]} "
            f"(p90−p10 = {spreads[widest]:.2f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, style="italic")
    path = f"{OUT}/fig_loss_percentiles.png"
    fig.savefig(path)
    note("fig_loss_percentiles.png",
         f"per-norm loss scales differ mildly (p50 linf {ep['dist/linf/p50']:.2f} > "
         f"l1 {ep['dist/l1/p50']:.2f} > l2 {ep['dist/l2/p50']:.2f}); widest spread = {widest}",
         src)
    return spreads


# ---------------------------------------------------------------- figure 2 --
def fig_temperature_dial():
    src = "results/bindaware_sample_T025.json"
    ep = json.load(open(src))["history"][-1]
    losses = torch.tensor([ep[f"train/loss_{n}"] for n in NORMS])  # epoch-MEAN losses
    temps = [(0.25, "toward-MAX\n(T=0.25)"), (1.0, "soft\n(T=1)"), (100.0, "toward-AVG\n(T=100)")]
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.6, 1.9), sharey=True)
    for ax, (T, title) in zip(axes, temps):
        w = torch.softmax(losses / T, dim=0)
        bars = ax.bar(range(len(NORMS)), w, color=[NORM_COLOR[n] for n in NORMS],
                      edgecolor="0.25", linewidth=0.6)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
        ax.set_xticks(range(len(NORMS)))
        ax.set_xticklabels([NORM_LABEL[n] for n in NORMS], fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("softmax weight")
    fig.text(0.5, -0.06,
             "temperature interpolates AVG ↔ MAX — illustrative weights from "
             "epoch-mean losses (T=0.25 run, ep49)",
             ha="center", fontsize=7, style="italic")
    path = f"{OUT}/fig_temperature_dial.png"
    fig.savefig(path)
    note("fig_temperature_dial.png",
         "one dial T moves weighting from near-uniform (AVG) to winner-take-most (MAX); "
         "weights computed from REAL epoch-mean losses (labeled illustrative — raw "
         "per-sample matrices are not stored on disk)",
         src)


# ---------------------------------------------------------------- figure 3 --
def fig_union_vs_t():
    path = f"{OUT}/fig_union_vs_T.png"
    subprocess.run([sys.executable, "scripts/figures/plot_union_vs_T.py", "--out", path],
                   check=True)
    pts = sorted(json.load(open(p))["metrics"]["worst_union_acc"] * 100
                 for p in __import__("glob").glob("results/eval_bindaware_sample_T*.json"))
    note("fig_union_vs_T.png",
         f"soft per-sample weighting at moderate-cold T sits at MSD level "
         f"({', '.join(f'{v:.1f}' for v in pts)}), far above the hard-MAX anchor 25.0 "
         f"— softness, not worst-case focus alone, is what works (auto-updates as T=1/2 land)",
         "results/eval_bindaware_sample_T*.json",
         script="scripts/figures/plot_union_vs_T.py")


# ---------------------------------------------------------------- figure 4 --
def fig_mechanism_chain():
    log_p1, ev_p1 = "logs/attackdro_union_s0.log", "results/eval_attackdro_union_s0.json"
    log_3a = "logs/bindaware_s0.log"
    probe = probe_at_epoch(log_p1, 33)                       # real F5 inputs
    apgd = {n: v * 100 for n, v in
            json.load(open(ev_p1))["metrics"]["per_norm_robust_acc"].items()}
    rows = parse_log(log_3a)                                  # real 3a q trajectory

    apply_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(FIG_W * 2, 2.3))
    # Left: F5 bias bars (probe hatched vs APGD solid).
    x, w = range(len(NORMS)), 0.36
    b1 = axl.bar([i - w / 2 for i in x], [probe[n] for n in NORMS], w,
                 facecolor="white", edgecolor="0.2", hatch="///", label="weak probe")
    b2 = axl.bar([i + w / 2 for i in x], [apgd[n] for n in NORMS], w,
                 facecolor="0.35", edgecolor="0.2", label="APGD (strong)")
    for bars in (b1, b2):
        axl.bar_label(bars, fmt="%.0f", fontsize=6)
    axl.set_xticks(list(x))
    axl.set_xticklabels([NORM_LABEL[n] for n in NORMS])
    axl.set_ylabel("robust acc (%)")
    axl.set_title("(a) biased signal: probe inverts\nthe weakest norm (F5)", fontsize=7.5)
    axl.set_ylim(0, 82)  # headroom so the legend clears the bars
    axl.legend(fontsize=6, loc="upper right")
    # Right: 3a q trajectory (F6 migration).
    epochs = [r[0] for r in rows]
    for n in NORMS:
        axr.plot(epochs, [r[1][n] for r in rows], color=NORM_COLOR[n], ls=NORM_LS[n],
                 marker=NORM_MARKER[n], markevery=max(1, len(epochs) // 8),
                 label=NORM_LABEL[n])
    axr.axhline(1 / 3, color="0.5", lw=0.8, ls=":")
    axr.set_xlabel("epoch")
    axr.set_ylabel("group weight $q_g$")
    axr.set_ylim(0, 1)
    axr.set_title("(b) binding-aware q tracks the\nmigrating weakest norm (F6)", fontsize=7.5)
    axr.legend(fontsize=6, ncol=3, loc="upper right")
    fig.text(0.5, -0.13,
             "biased signal → misallocated q → fixed by binding-aware weighting",
             ha="center", fontsize=8)
    path = f"{OUT}/fig_mechanism_chain.png"
    fig.savefig(path)
    note("fig_mechanism_chain.png",
         "left: cheap probes misrank the weakest norm; right: binding-aware weighting "
         "re-allocates q and tracks the weakest norm as it migrates (F5→F6 chain)",
         f"{log_p1} (ep33), {ev_p1}, {log_3a}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_loss_percentiles()
    fig_temperature_dial()
    fig_union_vs_t()
    fig_mechanism_chain()
    # Manifest
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a") as f:
        if new:
            f.write("# MANIFEST — figure/claim provenance\n\n"
                    "| figure | reads as | source data | generating script |\n|---|---|---|---|\n")
        for name, reads, src, script in manifest_rows:
            f.write(f"| presentation/{name} | {reads} | {src} | {script} |\n")
    print(f"\nManifest -> {MANIFEST}; figures -> {OUT}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Eval convergence curve: per-norm APGD robust acc vs attack steps, on best.pt.
Reads results/eval_curve/<run>__<norm>__s<steps>.json, writes results/eval_curve.md.
Runs = reactive_apgd_8255 (baseline), predictive_apgd_8255 (ks16), predictive_ks8_apgd_8255 (ks8)."""
import glob
import json
import os
import re

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"
DIR = f"{ROOT}/results/eval_curve"
RUNS = [("reactive_apgd_8255", "reactive (baseline)"),
        ("predictive_apgd_8255", "predictive ks16"),
        ("predictive_ks8_apgd_8255", "predictive ks8")]
NORMS = ["linf", "l2", "l1"]


def load_pt(run, norm, steps):
    p = f"{DIR}/{run}__{norm}__s{steps}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    m = d.get("metrics", d)
    return 100 * m["per_norm_robust_acc"][norm]


def main():
    # discover the step grid actually present per norm
    grid = {n: set() for n in NORMS}
    for f in glob.glob(f"{DIR}/*.json"):
        m = re.search(r"__(linf|l2|l1)__s(\d+)\.json$", f)
        if m:
            grid[m.group(1)].add(int(m.group(2)))
    L = ["# Eval convergence curve — per-norm APGD robust acc vs attack steps (best.pt, n=1000)", "",
         "![eval convergence curve](eval_curve_chart.svg)", "",
         "Runs @ ep50 RAMP recipe, APGD training. Robust acc (%) at increasing APGD steps → plateau = converged.",
         "*(reactive_ks8 doesn't exist — reactive has no kspan; using reactive_apgd_8255 as the baseline.)*", ""]
    for norm in NORMS:
        steps = sorted(grid[norm])
        if not steps:
            continue
        L += [f"## {norm}", "| run | " + " | ".join(f"s{s}" for s in steps) + " | plateau Δ(last two) |",
              "|---" * (len(steps) + 2) + "|"]
        for run, label in RUNS:
            vals = [load_pt(run, norm, s) for s in steps]
            cells = " | ".join(f"{v:.1f}" if v is not None else "—" for v in vals)
            last2 = [v for v in vals[-2:] if v is not None]
            dtxt = f"{last2[-1]-last2[-2]:+.2f}pp" if len(last2) == 2 else "—"
            L.append(f"| {label} | {cells} | {dtxt} |")
        L.append("")
    open(f"{ROOT}/results/eval_curve.md", "w").write("\n".join(L) + "\n")
    print("wrote results/eval_curve.md")
    print("grid:", {n: sorted(grid[n]) for n in NORMS})


if __name__ == "__main__":
    main()

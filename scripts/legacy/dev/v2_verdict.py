#!/usr/bin/env python
"""CARD-PB v2 sweep table + pre-registered verdict.

Reads each kspan arm's eval JSON (union/l1) + training JSON (measured FLOPs, final
per-norm floor + miss_rate), applies the pre-registered rule, and rewrites
results/cardpb_v2_sweep.md (header preserved, table + verdict filled). Idempotent —
re-run after each arm lands.

Pre-reg (frozen): SUCCESS = union∈[41.60,42.20] AND l1≥46.10 AND flops≤0.60.
Winner = lowest-FLOPs passing arm. No passing arm => finding
"l1 predictability floors the efficiency".
"""
from __future__ import annotations
import json
import os

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"
KSPANS = (8, 12, 16)
COLD = 5
U_LO, U_HI = 41.60, 42.20
L1_MIN = 46.10
FLOPS_MAX = 0.60
BASE_U, BASE_L1 = 41.90, 48.10

HEADER = open(os.path.join(ROOT, "results/cardpb_v2_sweep.md")).read().split(
    "## Results")[0].rstrip() if os.path.exists(
    os.path.join(ROOT, "results/cardpb_v2_sweep.md")) else "# CARD-PB v2 sweep"


def _load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


def _arm(k):
    ev = _load(f"results/eval_predictive_v2_ks{k}_s0.json")
    tr = _load(f"results/predictive_v2_ks{k}_s0.json")
    if not ev:
        return {"k": k, "ready": False}
    m = ev["metrics"]
    union = 100 * m["worst_union_acc"]
    l1 = 100 * m["per_norm_robust_acc"]["l1"]
    linf = 100 * m["per_norm_robust_acc"]["linf"]
    l2 = 100 * m["per_norm_robust_acc"]["l2"]
    flops = floor = miss = None
    if tr:
        h = tr.get("history", [])
        post = [e["pb/attack_flops_ratio"] for e in h
                if e.get("epoch", 0) >= COLD and "pb/attack_flops_ratio" in e]
        flops = sum(post) / len(post) if post else None
        last = h[-1] if h else {}
        floor = {n: last.get(f"floor/{n}") for n in ("linf", "l2", "l1")}
        # v2 driver = miss VOLUME (population-aware); fall back to 1-recall for older runs
        miss = {n: (last.get(f"phi/miss_vol_{n}", last.get(f"phi/miss_rate_{n}")))
                for n in ("linf", "l2", "l1")}
    ok = (U_LO <= union <= U_HI) and (l1 >= L1_MIN) and (flops is not None and flops <= FLOPS_MAX)
    return {"k": k, "ready": True, "union": union, "l1": l1, "linf": linf, "l2": l2,
            "flops": flops, "floor": floor, "miss": miss, "pass": ok}


def _fmt_floor(d):
    return "/".join(str(d[n]) if d and d.get(n) is not None else "—" for n in ("linf", "l2", "l1")) if d else "—"


def _fmt_miss(d):
    return "/".join(f"{d[n]:.2f}" if d and d.get(n) is not None else "—" for n in ("linf", "l2", "l1")) if d else "—"


def main():
    arms = [_arm(k) for k in KSPANS]
    rows = ["| kspan | union | l1 | FLOPs | floor li/l2/l1 | miss li/l2/l1 | pass? |",
            "|---|---|---|---|---|---|---|"]
    for a in arms:
        if not a["ready"]:
            rows.append(f"| {a['k']} | _running_ | | | | | |")
            continue
        fl = f"{a['flops']:.3f}" if a["flops"] is not None else "—"
        rows.append(f"| {a['k']} | {a['union']:.1f} | {a['l1']:.1f} | {fl} | "
                    f"{_fmt_floor(a['floor'])} | {_fmt_miss(a['miss'])} | "
                    f"{'✅' if a['pass'] else '❌'} |")

    ready = [a for a in arms if a["ready"]]
    passing = [a for a in ready if a["pass"]]
    lines = [HEADER, "", "## Results (filled per arm as they land)", ""]
    lines += rows
    lines.append("")
    lines.append(f"_Baseline: union {BASE_U} (target [{U_LO},{U_HI}]), l1 {BASE_L1} "
                 f"(target ≥{L1_MIN}), FLOPs ≤{FLOPS_MAX}. RAMP 46.1._")
    lines.append("")
    if len(ready) < len(KSPANS):
        lines.append(f"**Verdict PENDING** — {len(ready)}/{len(KSPANS)} arms evaluated.")
    elif passing:
        w = min(passing, key=lambda a: a["flops"])
        lines.append(f"## ✅ WINNER: kspan={w['k']} — union {w['union']:.1f}, l1 {w['l1']:.1f}, "
                     f"**FLOPs {w['flops']:.3f}** (lowest-FLOPs arm meeting all three pre-registered "
                     f"criteria). v2's confidence floor recovers l1 at ≤60% attack-FLOPs — the "
                     f"efficiency-at-equal-robustness claim HOLDS at seed 0. Next: 3-seed confirm "
                     f"(Kiet-gated).")
    else:
        # closest arm for the finding narrative
        best_l1 = max(ready, key=lambda a: a["l1"])
        lines.append(f"## ⚠ FINDING (pre-registered): \"l1 predictability floors the efficiency\"")
        lines.append("")
        lines.append(f"No arm met all three criteria. Best l1 recovery = kspan={best_l1['k']} "
                     f"(l1 {best_l1['l1']:.1f}, union {best_l1['union']:.1f}, FLOPs "
                     f"{best_l1['flops']:.3f}). l1 recovers toward the baseline only as FLOPs rise "
                     f"— the least-predictable minority norm cannot be cheaply skipped. This is the "
                     f"pre-registered NEGATIVE-BUT-SHARPENING result (not a failure): predictive "
                     f"budgeting works when binding is predictable and fails on trait-heavy minority "
                     f"norms. Reactive baseline (41.90, full FLOPs) stands as the headline row.")
    with open(os.path.join(ROOT, "results/cardpb_v2_sweep.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[v2_verdict] {len(ready)}/{len(KSPANS)} arms; passing={len(passing)}")
    for a in ready:
        print(f"  kspan={a['k']}: union {a['union']:.1f} l1 {a['l1']:.1f} flops "
              f"{a['flops'] if a['flops'] is None else round(a['flops'],3)} pass={a['pass']}")


if __name__ == "__main__":
    main()

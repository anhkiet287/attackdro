#!/usr/bin/env python
"""PHASE-1 reactive first-run checks + PRE-REGISTERED GAP GATE (P-11).
After APGD training, the l1 train-vs-eval gap must CLOSE (was +41.7pp under plain-PGD;
target ~l∞/l2 level) and l1 eval must CONVERGE (l1@100≈l1@200). If not, the port interacts
badly with full training → STOP before spending predictive hours.

Reads reactive train.json (final-epoch train adv-acc) + eval.json (per-norm eval robust) +
/tmp/rl1_{100,200}.json. Writes results/reports/phase1_firstrun_checks.md and prints GAPCHECK=PASS|FAIL."""
import json
import os

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"
REPORTS = f"{ROOT}/results/reports"
L1_GAP_MAX = 18.0     # was 41.7 under plain-PGD; l∞/l2 were 13.7/8.2 -> ≤18 = closed
L1_CONV_MAX = 0.7     # |l1@100 - l1@200| pp
L1_FLOOR = 38.0       # l1 eval must be clearly above the broken ~30 (RAMP@50 = 47.1)


def load(p):
    p = p if os.path.isabs(p) else f"{ROOT}/{p}"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tr = load("results/reactive_apgd_8255/s0/train.json")
    ev = load("results/reactive_apgd_8255/s0/eval.json")
    a = load("/tmp/rl1_100.json")
    b = load("/tmp/rl1_200.json")
    L = ["# PHASE-1 reactive first-run checks + gap gate (APGD training, RAMP recipe, ep50)", ""]
    verdict = "FAIL"
    reasons = []

    if not (tr and ev):
        L.append(f"⚠ INCOMPLETE — train.json:{bool(tr)} eval.json:{bool(ev)}")
        _w(L); print("GAPCHECK=FAIL"); return

    h = tr["history"][-1]
    pn = ev["metrics"]["per_norm_robust_acc"]
    aa = {n: 100 * h[f"train/adv_acc_{n}"] for n in ("linf", "l2", "l1")}
    ee = {n: 100 * pn[n] for n in ("linf", "l2", "l1")}
    gap = {n: aa[n] - ee[n] for n in ("linf", "l2", "l1")}
    union = 100 * ev["metrics"]["worst_union_acc"]

    L += ["## Train-attack robust vs EVAL robust (the gap that was +41.7pp on l1 under plain-PGD)",
          "| norm | train-attack robust | eval robust | gap |",
          "|---|---|---|---|"]
    for n in ("linf", "l2", "l1"):
        L.append(f"| {n} | {aa[n]:.1f} | {ee[n]:.1f} | **{gap[n]:+.1f}pp** |")
    L += ["", f"- **union {union:.1f}** · vs RAMP@50 42.9 (pre-drop). l1 eval = **{ee['l1']:.1f}** (was 30.7 broken; RAMP 47.1).", ""]

    # l1 convergence
    conv_ok, conv_txt = False, "l1 100/200 evals missing"
    if a and b:
        l1a = 100 * a["metrics"]["per_norm_robust_acc"]["l1"]
        l1b = 100 * b["metrics"]["per_norm_robust_acc"]["l1"]
        conv_ok = abs(l1b - l1a) <= L1_CONV_MAX
        conv_txt = f"l1 APGD-100 {l1a:.1f} vs APGD-200 {l1b:.1f} (Δ {l1b-l1a:+.2f}pp) → {'CONVERGED' if conv_ok else 'NOT converged'}"
    L += [f"## l1 eval convergence — {conv_txt}", ""]

    # gate
    gap_ok = gap["l1"] <= L1_GAP_MAX
    floor_ok = ee["l1"] >= L1_FLOOR
    if not gap_ok:
        reasons.append(f"l1 gap {gap['l1']:+.1f} > {L1_GAP_MAX} (not closed)")
    if not conv_ok:
        reasons.append("l1 eval not converged")
    if not floor_ok:
        reasons.append(f"l1 eval {ee['l1']:.1f} < {L1_FLOOR} (still weak — port problem?)")
    verdict = "PASS" if (gap_ok and conv_ok and floor_ok) else "FAIL"

    L += ["## GAP GATE (pre-registered)",
          f"- l1 gap ≤ {L1_GAP_MAX}: **{gap_ok}** (l1 {gap['l1']:+.1f}pp vs l∞ {gap['linf']:+.1f} / l2 {gap['l2']:+.1f})",
          f"- l1 converges (Δ ≤ {L1_CONV_MAX}): **{conv_ok}**",
          f"- l1 eval ≥ {L1_FLOOR} (not the broken ~30): **{floor_ok}** ({ee['l1']:.1f})",
          "",
          (f"**GATE PASS — APGD closed the l1 train/eval mismatch on full training. Proceed to predictive de-risk.**"
           if verdict == "PASS" else
           f"**GATE FAIL ({'; '.join(reasons)}) — STOP. The APGD port interacts badly with full training; do NOT run predictive. Report.**")]
    _w(L)
    print(f"GAPCHECK={verdict}")
    print(f"  l1 gap {gap['l1']:+.1f}pp (l∞ {gap['linf']:+.1f}/l2 {gap['l2']:+.1f}) · l1 eval {ee['l1']:.1f} · union {union:.1f}")


def _w(lines):
    os.makedirs(REPORTS, exist_ok=True)
    open(f"{REPORTS}/phase1_firstrun_checks.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

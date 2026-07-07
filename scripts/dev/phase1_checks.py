#!/usr/bin/env python
"""PHASE-1 first-run checks on the reactive model (item 1a/1b):
(a) l1 TRAINING strength — is l1 being attacked comparably to linf/l2 at step_size 0.10/10-step
    (not trivially robust = under-attacked)? Reads train.json final epoch.
(b) l1 EVAL convergence for OUR model — l1@100 vs l1@200 on the reactive ckpt (RAMP's plateau
    may not transfer). Reads /tmp/rl1_{100,200}.json.
Writes results/phase1_firstrun_checks.md."""
import json
import os

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tr = load(f"{ROOT}/results/reactive_T025_8255/s0/train.json")
    L = ["# PHASE-1 first-run checks — reactive @ RAMP recipe, ep50", ""]
    # (a) l1 training strength
    if tr and tr.get("history"):
        h = tr["history"][-1]
        aa = {n: h.get(f"train/adv_acc_{n}") for n in ("linf", "l2", "l1")}
        ls = {n: h.get(f"train/loss_{n}") for n in ("linf", "l2", "l1")}
        L += ["## (a) l1 TRAINING strength (step_size 0.10, 10 steps)",
              f"- final-epoch train adv-acc: linf {aa['linf']:.3f} / l2 {aa['l2']:.3f} / l1 {aa['l1']:.3f}",
              f"- final-epoch train loss:    linf {ls['linf']:.3f} / l2 {ls['l2']:.3f} / l1 {ls['l1']:.3f}"]
        # l1 under-attacked if its train adv-acc is far ABOVE the others (model barely challenged on l1)
        if aa["l1"] is not None and aa["linf"] is not None:
            gap = aa["l1"] - max(aa["linf"], aa["l2"])
            verdict = ("⚠ l1 train adv-acc >0.15 above linf/l2 — l1 may be UNDER-attacked (raise l1 step_size/steps)"
                       if gap > 0.15 else "✓ l1 train adv-acc comparable to linf/l2 — l1 is being genuinely attacked")
            L.append(f"- **{verdict}** (l1 − max(linf,l2) = {gap:+.3f}).")
        L.append("")
    # (b) l1 eval convergence for our model
    a = load("/tmp/rl1_100.json")
    b = load("/tmp/rl1_200.json")
    L.append("## (b) l1 EVAL convergence — OUR reactive model")
    if a and b:
        l1a = 100 * a["metrics"]["per_norm_robust_acc"]["l1"]
        l1b = 100 * b["metrics"]["per_norm_robust_acc"]["l1"]
        d = l1b - l1a
        L += [f"- reactive l1: APGD-100 = {l1a:.1f} · APGD-200 = {l1b:.1f} · Δ = {d:+.2f}pp",
              f"- **{'✓ l1@100 CONVERGED for our model — screening 20/20/100 holds' if abs(d) <= 0.1 else '⚠ l1 NOT converged at 100 for our model — RAISE l1 eval steps'}**"]
    else:
        L.append("- (l1 100/200 evals missing)")
    open(f"{ROOT}/results/phase1_firstrun_checks.md", "w").write("\n".join(L) + "\n")
    print("[phase1_checks] wrote results/phase1_firstrun_checks.md")


if __name__ == "__main__":
    main()

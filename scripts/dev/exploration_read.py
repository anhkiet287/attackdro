#!/usr/bin/env python
"""Read the 3 exploration probes into verdicts (EXPLORATION lane — NOT paper-grade).
Reads <exp_dir>/<run>/s0/{train,eval}.json, writes <exp_dir>/idea{1,2,3}_*.md + SUMMARY.md.
Usage: exploration_read.py [exp_dir]   (default results/exploration)

Reference points (paper lane, for context only): reactive union 38.8 / l1 45.6; CARD-PB ks16
FLOPs 0.665; de-risk predictive l1 45.6. These probes are alternatives to CARD-PB, not
confirmed better — the verdict is a cheap signal for Kiet, who decides on a proper port."""
import json
import os
import sys

EXP = sys.argv[1] if len(sys.argv) > 1 else "results/exploration"
CARDPB_FLOPS = 0.665
REACT_L1, REACT_UNION = 45.6, 38.8
GAP_BLOWUP = 25.0   # l1 train-vs-eval gap this large = under-attacked (paper closed it to ~11)


def load(run, which):
    p = f"{EXP}/{run}/s0/{which}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def evm(run):
    d = load(run, "eval")
    if not d:
        return None
    m = d.get("metrics", d); pn = m.get("per_norm_robust_acc", {})
    return {"union": 100 * m["worst_union_acc"], "l1": 100 * pn.get("l1", 0),
            "linf": 100 * pn.get("linf", 0), "l2": 100 * pn.get("l2", 0),
            "clean": 100 * m.get("clean_acc", 0)}


def hist(run):
    d = load(run, "train")
    return d.get("history", []) if d else []


def w(name, lines):
    open(f"{EXP}/{name}", "w").write("\n".join(lines) + "\n")


def idea1():
    h = hist("idea1_failrate"); ev = evm("idea1_failrate")
    L = ["# Idea 1 — batch fail-rate threshold allocation (EXPLORATION, ep20, 1 seed)", ""]
    if not h or not ev:
        L += ["⚠ INCOMPLETE — run idea1_failrate first."]; w("idea1_failrate.md", L); return ("Idea 1", "INCOMPLETE", "")
    last = h[-1]
    steps = {n: last.get(f"exp/steps_{n}") for n in ("linf", "l2", "l1")}
    flops = last.get("efficiency/attack_flops_ratio", last.get("exp/attack_flops_ratio"))
    tr_l1 = 100 * last.get("train/adv_acc_l1", 0)
    gap = tr_l1 - ev["l1"]
    L += [f"- **steps the threshold self-selected** (final epoch): l∞ {steps['linf']:.1f} · l2 {steps['l2']:.1f} · l1 {steps['l1']:.1f}",
          f"  → self-adapt claim: {'YES l1 > l∞ (hard norm gets more)' if (steps['l1'] or 0) > (steps['linf'] or 0) else 'NO — l1 did NOT get more steps'}",
          f"- **attack-FLOPs ratio = {flops:.3f}** vs CARD-PB 0.665 → {'cheaper' if flops < CARDPB_FLOPS else 'NOT cheaper'}",
          f"- **l1 train-vs-eval gap = {gap:+.1f}pp** (train {tr_l1:.1f} / eval {ev['l1']:.1f}); paper-lane closed gap ≈ +11pp",
          f"- eval: union {ev['union']:.1f} · l∞ {ev['linf']:.1f} · l2 {ev['l2']:.1f} · l1 {ev['l1']:.1f}", ""]
    kill = gap > GAP_BLOWUP or (flops is not None and flops >= CARDPB_FLOPS)
    pos = (not kill) and (steps['l1'] or 0) > (steps['linf'] or 0) and flops < CARDPB_FLOPS
    verdict = "KILL" if kill else ("POSITIVE" if pos else "MIXED")
    why = ("l1 gap blew up (under-attacked)" if gap > GAP_BLOWUP else
           "no FLOPs saving vs CARD-PB" if (flops or 1) >= CARDPB_FLOPS else
           "l1 holds, FLOPs < 0.665, steps self-adapt" if pos else "l1 holds but steps didn't clearly self-adapt")
    L += [f"## Verdict: **{verdict}** — {why}",
          "Caveat (record even if positive): batch-level stop leaves each norm's HARDEST samples under-attacked "
          "(they never fail at the threshold); and the threshold is a new hyperparameter. Per-sample repack = future work."]
    w("idea1_failrate.md", L)
    return ("Idea 1 fail-rate", verdict, f"steps l∞/l2/l1={steps['linf']:.1f}/{steps['l2']:.1f}/{steps['l1']:.1f}, FLOPs {flops:.3f}, l1-gap {gap:+.1f}")


def idea2():
    h = hist("idea2_kmin_recovery"); ev = evm("idea2_kmin_recovery")
    L = ["# Idea 2 — k_min self-recovery (EXPLORATION, kspan=4 k_min=1, ep30, 1 seed)", ""]
    if not h:
        L += ["⚠ INCOMPLETE — run idea2_kmin_recovery first."]; w("idea2_kmin_recovery.md", L); return ("Idea 2", "INCOMPLETE", "")
    traj = [(e.get("epoch"), e.get("floor/l1"), e.get("floor/linf"), e.get("floor/l2")) for e in h]
    L += ["**floor trajectory (the result) — floor/l1 should dip toward 1 then climb as miss_vol rises:**",
          "| epoch | floor l∞ | floor l2 | floor l1 |", "|---|---|---|---|"]
    for ep, fl1, fli, fl2 in traj:
        L.append(f"| {ep} | {fli} | {fl2} | **{fl1}** |")
    l1floors = [f for _, f, _, _ in traj if f is not None]
    climbs = len(l1floors) >= 3 and max(l1floors[1:]) > l1floors[0]
    stuck = l1floors and max(l1floors) <= 1
    final_l1 = ev["l1"] if ev else None
    L += ["", f"- floor_l1: min {min(l1floors) if l1floors else '—'} → max {max(l1floors) if l1floors else '—'} "
          f"→ {'CLIMBS (self-recovery)' if climbs else 'STUCK low' if stuck else 'flat/other'}",
          f"- final l1 eval = {final_l1:.1f} vs de-risk 45.6 → {'≈ held' if final_l1 and final_l1 >= REACT_L1 - 2 else 'DEGRADED' if final_l1 else 'pending'}" if final_l1 is not None else "- l1 eval pending", ""]
    kill = stuck or (final_l1 is not None and final_l1 < REACT_L1 - 2)
    pos = climbs and (final_l1 is not None and final_l1 >= REACT_L1 - 2)
    verdict = "KILL" if kill else ("POSITIVE" if pos else "MIXED")
    why = ("floor stuck low + l1 damaged → feedback doesn't save robustness" if kill else
           "floor climbs & stabilizes, l1 ≈ de-risk → k_min can be lowered for extra saving" if pos else
           "floor moves but l1 outcome ambiguous")
    L += [f"## Verdict: **{verdict}** — {why}",
          "Purpose was a MECHANISM probe (does miss_vol feedback self-correct?), not an operating point. "
          "k_min=3 stays the safe default unless recovery is clean."]
    w("idea2_kmin_recovery.md", L)
    return ("Idea 2 k_min-recovery", verdict, f"floor_l1 {min(l1floors) if l1floors else '—'}→{max(l1floors) if l1floors else '—'}, l1 {final_l1:.1f}" if final_l1 else "pending eval")


def _cum_flops_curriculum():
    h = hist("idea3_curriculum")
    if not h:
        return None
    per = [e.get("exp/curriculum_steps") for e in h if e.get("exp/curriculum_steps") is not None]
    # each epoch spends (curriculum_steps * 3 norms) / 30 of a full-10/10/10 epoch
    return sum((s * 3) / 30.0 for s in per) / max(len(per), 1)


def idea3():
    cur = evm("idea3_curriculum"); ctl = evm("idea3_control_fixed10")
    L = ["# Idea 3 — curriculum step ramp vs fixed-10 control (EXPLORATION, ep30, 1 seed)", ""]
    if not cur or not ctl:
        L += [f"⚠ INCOMPLETE — curriculum:{bool(cur)} control:{bool(ctl)}."]; w("idea3_curriculum.md", L); return ("Idea 3", "INCOMPLETE", "")
    cum = _cum_flops_curriculum()
    L += ["| run | union | l1 | l∞ | l2 | clean | avg attack-FLOPs/epoch |",
          "|---|---|---|---|---|---|---|",
          f"| curriculum | {cur['union']:.1f} | **{cur['l1']:.1f}** | {cur['linf']:.1f} | {cur['l2']:.1f} | {cur['clean']:.1f} | **{cum:.3f}** |" if cum else f"| curriculum | {cur['union']:.1f} | {cur['l1']:.1f} | … | … | … | … |",
          f"| control fixed-10 | {ctl['union']:.1f} | **{ctl['l1']:.1f}** | {ctl['linf']:.1f} | {ctl['l2']:.1f} | {ctl['clean']:.1f} | 1.000 |", "",
          f"- Δunion (curr − ctl) = {cur['union']-ctl['union']:+.2f}pp · Δl1 = {cur['l1']-ctl['l1']:+.2f}pp · "
          f"curriculum avg FLOPs/epoch {cum:.3f} (< 1.0 = cheaper early)" if cum else "", ""]
    l1_lag = cur['l1'] < ctl['l1'] - 2
    saves = cum is not None and cum < 0.9
    kill = l1_lag or not saves
    pos = (not l1_lag) and saves and cur['union'] >= ctl['union'] - 1
    verdict = "KILL" if kill else ("POSITIVE" if pos else "MIXED")
    why = ("curriculum l1 lags fixed-10 (weak-early hurt l1)" if l1_lag else
           "no meaningful FLOPs saving" if not saves else
           "matches fixed-10 union/l1 at lower cumulative FLOPs" if pos else "ambiguous")
    L += [f"## Verdict: **{verdict}** — {why}",
          "Note: curriculum only saves EARLY (late epochs full) → modest, supplementary; low novelty alone "
          "(FAT/CAT). Only interesting combined with predictive allocation."]
    w("idea3_curriculum.md", L)
    return ("Idea 3 curriculum", verdict, f"Δunion {cur['union']-ctl['union']:+.2f}, Δl1 {cur['l1']-ctl['l1']:+.2f}, cumFLOPs {cum:.3f}" if cum else "pending")


def main():
    os.makedirs(EXP, exist_ok=True)
    rows = [idea1(), idea2(), idea3()]
    S = ["# EXPLORATION SUMMARY — 3 allocation/schedule probes (Colab, NOT paper-grade)", "",
         "*These are alternatives to CARD-PB (the working mechanism: FLOPs 0.665, l1 holds), not confirmed better. "
         "Signal for Kiet to decide a proper port. Paper lane untouched.*", "",
         "| idea | verdict | key numbers |", "|---|---|---|"]
    for name, verdict, nums in rows:
        S.append(f"| {name} | **{verdict}** | {nums} |")
    S += ["", "## Recommendation", ""]
    for name, verdict, _ in rows:
        rec = {"POSITIVE": "worth a proper port (still cheap to validate first)",
               "KILL": "drop — kill signal fired",
               "MIXED": "inconclusive — needs a different/longer test",
               "INCOMPLETE": "not run yet"}.get(verdict, "—")
        S.append(f"- **{name}: {verdict}** → {rec}")
    w("SUMMARY.md", S)
    print("wrote", EXP + "/SUMMARY.md and idea{1,2,3}_*.md")
    for name, verdict, nums in rows:
        print(f"  {name}: {verdict} ({nums})")


if __name__ == "__main__":
    main()

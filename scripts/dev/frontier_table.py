#!/usr/bin/env python
"""PHASE-1 frontier sweep table (P-12): kspan -> (union, l1, FLOPs, floor, miss_vol).
Pre-registered read (unchanged from de-risk):
  Winner = LOWEST-FLOPs arm with l1 drop <2pp (dL1 >= -2) AND union within noise
           (dUnion >= -1.0, the 1pp seed-noise band) AND FLOPs <= 0.60.
  If ks8 hits <=0.60 AND l1 holds -> conditional efficiency win (operating point).
  If NO arm gets both l1-hold AND <=0.60 -> floors-finding: 'l1 predictability floors
    efficiency at ~min-FLOPs-with-l1-hold' (the curve IS the result).
Runs incrementally: only tabulates arms whose eval.json exists. Writes results/frontier_sweep.md."""
import json
import os

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"
COLD = 5
RAMP50 = 42.9
DUNION_NOISE = -1.0   # 1pp seed-noise band (selection rule)
DL1_HOLD = -2.0
FLOPS_TARGET = 0.60

# kspan -> run name (ks16 = the de-risk run already on disk)
ARMS = {8: "predictive_ks8_apgd_8255", 16: "predictive_apgd_8255",
        24: "predictive_ks24_apgd_8255", 32: "predictive_ks32_apgd_8255"}
REACT = "reactive_apgd_8255"


def _load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


def _ev(run):
    d = _load(f"results/{run}/s0/eval.json")
    if not d:
        return None
    m = d.get("metrics", d)
    pn = m.get("per_norm_robust_acc", {})
    return {"union": 100 * m["worst_union_acc"], "clean": 100 * m.get("clean_acc", 0),
            "linf": 100 * pn.get("linf", 0), "l2": 100 * pn.get("l2", 0), "l1": 100 * pn.get("l1", 0)}


def _pb(run):
    d = _load(f"results/{run}/s0/train.json")
    if not d:
        return {}
    h = d.get("history", [])
    post = [e["pb/attack_flops_ratio"] for e in h if e.get("epoch", 0) >= COLD and "pb/attack_flops_ratio" in e]
    last = h[-1] if h else {}
    return {"flops": (sum(post) / len(post)) if post else None,
            "floor": {n: last.get(f"floor/{n}") for n in ("linf", "l2", "l1")},
            "miss_vol": {n: last.get(f"phi/miss_vol_{n}") for n in ("linf", "l2", "l1")}}


def main():
    react = _ev(REACT)
    if not react:
        open(f"{ROOT}/results/frontier_sweep.md", "w").write("⚠ reactive_apgd_8255 eval missing.\n")
        print("NO REACTIVE"); return
    rows = []
    for ks in sorted(ARMS):
        ev = _ev(ARMS[ks])
        if not ev:
            continue
        pb = _pb(ARMS[ks])
        rows.append({"ks": ks, "ev": ev, "pb": pb,
                     "du": ev["union"] - react["union"], "dl1": ev["l1"] - react["l1"],
                     "flops": pb.get("flops")})

    L = ["# PHASE-1 frontier sweep — CARD-PB v2 kspan ∈ {8,16,24,32} @ ep50 (RAMP recipe, APGD, 1 seed, screening 20/20/100)",
         "",
         f"Reactive ref (ks n/a): union **{react['union']:.1f}**, l1 **{react['l1']:.1f}**, clean {react['clean']:.1f}, FLOPs 1.00 (30-step denom).",
         f"RAMP ep50 ref: union {RAMP50}.", "",
         "| kspan | union | Δunion | l1 | Δl1 | l∞ | l2 | clean | FLOPs | floor(l∞/l2/l1) | miss_vol(l∞/l2/l1) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e, pb = r["ev"], r["pb"]
        fl = r["flops"]
        flr = pb.get("floor", {}) or {}
        mv = pb.get("miss_vol", {}) or {}
        floor_s = "/".join(str(flr.get(n)) for n in ("linf", "l2", "l1"))
        mv_s = "/".join((f"{mv.get(n):.2f}" if isinstance(mv.get(n), (int, float)) else "—") for n in ("linf", "l2", "l1"))
        L.append(f"| {r['ks']} | {e['union']:.1f} | {r['du']:+.2f} | **{e['l1']:.1f}** | {r['dl1']:+.2f} | "
                 f"{e['linf']:.1f} | {e['l2']:.1f} | {e['clean']:.1f} | "
                 f"{'**'+format(fl,'.3f')+'**' if fl is not None else '—'} | {floor_s} | {mv_s} |")
    L.append("")

    # pre-registered outcome
    qual = [r for r in rows if r["flops"] is not None and r["dl1"] >= DL1_HOLD
            and r["du"] >= DUNION_NOISE and r["flops"] <= FLOPS_TARGET]
    L.append("## Pre-registered read")
    L.append(f"Criteria: Δl1 ≥ {DL1_HOLD} (l1 holds) · Δunion ≥ {DUNION_NOISE} (1pp noise band) · FLOPs ≤ {FLOPS_TARGET}.")
    missing = [ks for ks in ARMS if not any(r["ks"] == ks for r in rows)]
    if missing:
        L.append(f"*(arms not yet run: {missing})*")
    if qual:
        win = min(qual, key=lambda r: r["flops"])
        L.append(f"**WINNER = kspan {win['ks']}** — lowest-FLOPs arm meeting all three: "
                 f"FLOPs **{win['flops']:.3f}** ≤ {FLOPS_TARGET}, Δunion {win['du']:+.2f}, l1 holds (Δl1 {win['dl1']:+.2f}). "
                 f"**Operating point.** Next (Kiet GO): escalate ks{win['ks']} reactive-vs-predictive to 3→20 seeds; β-ablation {{0.3,0.5,0.8}} at ks{win['ks']}.")
        print(f"FRONTIER WINNER ks={win['ks']} flops={win['flops']:.3f} du={win['du']:.2f} dl1={win['dl1']:.2f}")
    elif not missing:
        holds = [r for r in rows if r["dl1"] >= DL1_HOLD and r["du"] >= DUNION_NOISE and r["flops"] is not None]
        floor_pt = min((r["flops"] for r in holds), default=None)
        note = (f"min FLOPs with l1 still holding = **{floor_pt:.3f}** (> {FLOPS_TARGET} target)"
                if floor_pt is not None else "no arm holds l1 within noise")
        L.append(f"**FLOORS-FINDING** — no arm hits both l1-hold AND FLOPs ≤ {FLOPS_TARGET}. "
                 f"Characterized result: *l1 predictability floors predictive efficiency at ~{floor_pt:.2f}× FLOPs* ({note}). "
                 f"The curve is the finding — publishable. Next (Kiet decision): report the floor, or accept the ≤0.60 arm's l1 cost.")
        print(f"FRONTIER FLOORS-FINDING floor={floor_pt}")
    else:
        L.append("*(incomplete — waiting on remaining arms before firing the outcome.)*")
        print("FRONTIER INCOMPLETE")
    L.append("")
    open(f"{ROOT}/results/frontier_sweep.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

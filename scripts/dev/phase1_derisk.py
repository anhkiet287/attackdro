#!/usr/bin/env python
"""PHASE-1 de-risk read: reactive vs predictive CARD-PB v2 @ ep50, RAMP recipe, 1 seed.
The critical question after the R1/R3/v2 fixes: does l1 HOLD, or is this F9 again?
Reads the nested eval.json + train.json, writes results/phase1_derisk.md."""
import json
import os

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"
COLD = 5
RAMP50 = 42.9  # matched pre-drop budget comparator


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


def _pb_stats(run):
    d = _load(f"results/{run}/s0/train.json")
    if not d:
        return {}
    h = d.get("history", [])
    post = [e["pb/attack_flops_ratio"] for e in h if e.get("epoch", 0) >= COLD and "pb/attack_flops_ratio" in e]
    last = h[-1] if h else {}
    return {"flops": (sum(post) / len(post)) if post else None,
            "floor": {n: last.get(f"floor/{n}") for n in ("linf", "l2", "l1")},
            "miss_vol": {n: last.get(f"phi/miss_vol_{n}") for n in ("linf", "l2", "l1")},
            "phi_mis": last.get("phi/misprediction_rate")}


def main():
    r, p = _ev("reactive_T025_8255"), _ev("predictive_pb_8255")
    pb = _pb_stats("predictive_pb_8255")
    L = ["# PHASE-1 de-risk — reactive vs predictive CARD-PB v2 @ ep50 (RAMP recipe, 1 seed, screening 20/20/100)", ""]
    if not r or not p:
        L += [f"⚠ INCOMPLETE — reactive:{bool(r)} predictive:{bool(p)}. Re-run.", ""]
        _w(L); print("INCOMPLETE"); return
    du = p["union"] - r["union"]
    dl1 = p["l1"] - r["l1"]
    flops = pb.get("flops")
    L += ["| method | union | clean | linf | l2 | l1 | attack-FLOPs |",
          "|---|---|---|---|---|---|---|",
          f"| reactive T=0.25 | **{r['union']:.1f}** | {r['clean']:.1f} | {r['linf']:.1f} | {r['l2']:.1f} | **{r['l1']:.1f}** | 1.00 (ref, 30-step) |",
          f"| predictive v2 | **{p['union']:.1f}** | {p['clean']:.1f} | {p['linf']:.1f} | {p['l2']:.1f} | **{p['l1']:.1f}** | **{flops:.3f}** |" if flops is not None else
          f"| predictive v2 | **{p['union']:.1f}** | {p['clean']:.1f} | {p['linf']:.1f} | {p['l2']:.1f} | **{p['l1']:.1f}** | — |",
          f"| RAMP ep_50 (ref) | {RAMP50} | — | — | — | — | 1.00 |", "",
          f"- **Δunion (pred − react) = {du:+.2f}pp** · **Δl1 = {dl1:+.2f}pp** · measured attack-FLOPs = **{flops:.3f}** (30-step denom)." if flops is not None else
          f"- **Δunion = {du:+.2f}pp** · **Δl1 = {dl1:+.2f}pp** · FLOPs pending.",
          f"- predictive floor li/l2/l1 = {pb.get('floor')} · miss_vol = "
          f"{ {k: (round(v,2) if isinstance(v,(int,float)) else v) for k,v in (pb.get('miss_vol') or {}).items()} } · φ mispred = {pb.get('phi_mis')}.",
          ""]
    # F9-recurrence verdict
    l1_holds = dl1 >= -2.0
    union_ok = du >= -0.5
    flops_ok = flops is not None and flops <= 0.60
    L.append("## De-risk verdict")
    if l1_holds and union_ok:
        L.append(f"**l1 HOLDS — the R1/R3/v2 fixes worked** (Δl1 {dl1:+.2f} ≥ −2pp, Δunion {du:+.2f} ≥ −0.5). "
                 f"NOT F9 again. " + (f"FLOPs {flops:.3f} ≤ 0.60 → efficiency signal → **proceed to the "
                 f"{{8,16,24,32}} frontier sweep + β-ablation** (Kiet GO)." if flops_ok else
                 f"FLOPs {flops:.3f} > 0.60 at this kspan → run the frontier sweep to find the ≤0.60 point (Kiet GO)."))
    else:
        why = []
        if not l1_holds:
            why.append(f"l1 collapsed {dl1:+.2f}pp (>−2)")
        if not union_ok:
            why.append(f"union {du:+.2f}pp (<−0.5)")
        L.append(f"**F9 RECURS** ({'; '.join(why)}) — even after R1/R3/v2, predictive under-defends l1 at this budget. "
                 f"STOP predictive; do NOT run β/frontier. Kiet decision: alternative lane (e.g. early-stop attack) or "
                 f"reactive-only. Reactive stands as the method row.")
    L.append("")
    L.append(f"<!-- du={du:.3f} dl1={dl1:.3f} flops={flops} l1_holds={l1_holds} -->")
    _w(L)
    print(f"DERISK du={du:.2f} dl1={dl1:.2f} flops={flops} l1_holds={l1_holds}")


def _w(lines):
    open(os.path.join(ROOT, "results/phase1_derisk.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

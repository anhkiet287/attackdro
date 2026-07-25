"""T3 — threat dominance from existing masks. NO GPU, NO model load, read-only.

For each arm: among examples that FAIL the union, which norm blocks are responsible?
  fails_p  = NOT (AND over that norm's attacks in the tier)
  union_fail = NOT (AND over all attacks in the tier)
Reported over union-failures: coverage% per norm (overlapping) and EXCLUSIVE% (that norm alone).
Tier is printed per arm and never mixed.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
MAIN = os.path.join(ROOT, "results/main")
NORMS = ("linf", "l2", "l1")
ARMS = [("msd", "MSD (public, Maini)"), ("M0", "MSD-10 control"), ("M1a", "CLAMP (MSD-10 base)"),
        ("M0_max", "MAX control"), ("M1a_max", "CLAMP (MAX base)"), ("M1a_avg", "CLAMP (AVG base)"),
        ("M0_full", "MSD-50 control"), ("M1a_full", "CLAMP-50"),
        ("Rprime", "RAMP (from-scratch)"), ("B1", "RAMP (full budget)")]


def load(arm, prefer=("10k", "1k", "10k_nosq", "1k_nosq")):
    for lbl in prefer:
        p = os.path.join(MAIN, arm, lbl, "masks_multinorm_v1.npz")
        if os.path.exists(p):
            d = np.load(p)
            m = {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}
            return lbl, m
    return None, None


def blocks(m):
    ks = list(m)
    n = len(m[ks[0]])
    per = {}
    for nm in NORMS:
        u = np.ones(n, bool)
        for k in ks:
            if k.endswith(nm):
                u &= m[k]
        per[nm] = u
    uni = np.ones(n, bool)
    for k in ks:
        uni &= m[k]
    return per, uni, len(ks)


def main():
    rows, detail = [], {}
    for arm, label in ARMS:
        lbl, m = load(arm)
        if m is None:
            print(f"  [skip] {arm}: no masks");  continue
        per, uni, natk = blocks(m)
        tier = "full" if natk == 12 else ("no_square" if natk == 9 else f"{natk}atk")
        fail = ~uni
        nf = int(fail.sum())
        f = {nm: ~per[nm] for nm in NORMS}
        cov = {nm: float(f[nm][fail].mean()) for nm in NORMS}                    # overlapping
        exc = {nm: float((f[nm] & ~f[[o for o in NORMS if o != nm][0]] &
                          ~f[[o for o in NORMS if o != nm][1]])[fail].mean()) for nm in NORMS}
        allthree = float((f["linf"] & f["l2"] & f["l1"])[fail].mean())
        rows.append(dict(arm=arm, label=label, scale=lbl, tier=tier, n=len(uni), n_attacks=natk,
                         union=float(uni.mean()), n_union_fail=nf,
                         cov=cov, exclusive=exc, all_three=allthree,
                         per_norm={nm: float(per[nm].mean()) for nm in NORMS}))
        detail[arm] = rows[-1]

    W = max(len(r["label"]) for r in rows)
    print("\n" + "=" * 118)
    print("T3 — THREAT DOMINANCE: among UNION FAILURES, which norm block is responsible?")
    print("=" * 118)
    print(f"{'model':<{W}} {'scale':>6} {'tier':>10} {'union':>7} {'#fail':>6} | "
          f"{'ℓ∞%':>7}{'ℓ2%':>7}{'ℓ1%':>7} | {'ℓ∞only':>7}{'ℓ2only':>7}{'ℓ1only':>7} | {'all3':>6}")
    print("-" * 118)
    for r in rows:
        c, e = r["cov"], r["exclusive"]
        print(f"{r['label']:<{W}} {r['scale']:>6} {r['tier']:>10} {r['union']:>7.4f} {r['n_union_fail']:>6} | "
              f"{c['linf']*100:>6.1f} {c['l2']*100:>6.1f} {c['l1']*100:>6.1f} | "
              f"{e['linf']*100:>6.1f} {e['l2']*100:>6.1f} {e['l1']*100:>6.1f} | {r['all_three']*100:>5.1f}")
    print("-" * 118)
    print("  ℓp%     = share of union-failures where the ℓp block also fails (OVERLAPPING, sums > 100)")
    print("  ℓp only = share where ONLY that norm fails -> the norm that is solely responsible")
    print("  all3    = share where all three blocks fail (example is broadly non-robust)")
    print("  NOTE: rows at different tiers are NOT comparable to each other.")

    out = os.path.join(ROOT, "results/analysis/T3_threat_dominance/threat_dominance.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n  saved -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()

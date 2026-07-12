#!/usr/bin/env python
"""RE-P0/P1/P2 (+descriptive RE-P3) paired-bootstrap analysis for the endogenous
static-counterfactual pre-registration.

FROZEN pre-analysis lock. Implements EXACTLY the locked contract in
docs/preregistrations/preregistration_endogenous_static_counterfactual.md §5-§6.
This script must NOT change after real canonical masks are opened, except
documented correctness-only fixes applied WITHOUT inspecting/interpreting results.

Inputs are exported canonical UNION-survival masks (boolean, True=robust, same
test_final examples in the same order) for the three arms:
    --b4              B4-adaptive (fixed reference; val_best)
    --linf-weighted   B3-static-L∞-weighted   (b3_mis)
    --l1-weighted     B3-static-L1-weighted   (b3_bottleneck_informed)
Optional per-norm canonical masks (per arm × {linf,l2,l1}) enable descriptive
RE-P3a/b. Masks may be .npy (1-D array), .json (list), or .npz (--mask-key, or a
lone array / a canonical-union key).

Statistics (LOCKED): paired sample-level bootstrap; unit = example; B = 10000;
seed = 0; one-sided 95% LCB = 5th percentile of resampled paired mean-differences.
All contrasts share the same resample draws (fully paired).

    RE-P0:  LCB(U_linf_weighted − U_l1_weighted) > 0
    RE-P1:  LCB(U_B4            − U_l1_weighted) > 0
    RE-P2:  LCB(U_B4            − U_linf_weighted) > −0.02
    Verdict: CONJUNCTIVE  RE-P0 ∧ RE-P1 ∧ RE-P2.
RE-P3 is DESCRIPTIVE, NON-gating (never rescues a failed conjunction).

Self-test:  python scripts/dev/analyze_endogenous_statics.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- LOCKED constants (do not change — pre-reg §5) -------------------------
B_BOOTSTRAP = 10000
SEED = 0
RE_P2_MARGIN = -0.02      # B4 non-inferiority margin to the L∞-weighted static
RE_P3_M = 0.03            # descriptive uniqueness / tradeoff threshold (m_b, δ)
LCB_PCT = 5.0             # one-sided 95% lower confidence bound


# ---- mask IO ---------------------------------------------------------------
def load_mask(path: str, key: str | None = None) -> np.ndarray:
    """Load a boolean survival mask as a 0/1 int8 vector (True/1 = robust)."""
    p = Path(path)
    if p.suffix == ".npy":
        a = np.load(p, allow_pickle=False)
    elif p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        arrays = [k for k in z.files if k != "metadata_json"]
        if key is not None:
            a = z[key]
        elif len(arrays) == 1:
            a = z[arrays[0]]
        else:
            for cand in ("canonical_union", "full_audit_union", "union", "full_union"):
                if cand in z.files:
                    a = z[cand]
                    break
            else:
                raise SystemExit(f"{path}: multiple arrays {arrays}; pass --mask-key")
    elif p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            a = np.asarray(json.load(f))
    else:
        raise SystemExit(f"{path}: unsupported mask format {p.suffix!r} (use .npy/.npz/.json)")
    a = np.asarray(a).ravel()
    uniq = set(np.unique(a).tolist())
    if not uniq.issubset({0, 1, True, False}):
        raise SystemExit(f"{path}: mask is not boolean/0-1 (values {sorted(uniq)[:6]}…)")
    return a.astype(np.int8)


# ---- paired bootstrap ------------------------------------------------------
def shared_bootstrap(arms: dict[str, np.ndarray], B: int = B_BOOTSTRAP,
                     seed: int = SEED) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (names, point_means, boot_means[B, n_arms]) with all arms resampled
    on the SAME example indices per replicate (fully paired)."""
    names = list(arms)
    lengths = {len(arms[k]) for k in names}
    if len(lengths) != 1:
        raise SystemExit(f"masks differ in length: "
                         f"{ {k: len(arms[k]) for k in names} }")
    n = lengths.pop()
    if n == 0:
        raise SystemExit("empty masks")
    mat = np.stack([arms[k].astype(np.float64) for k in names])  # (A, n)
    point = mat.mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty((B, len(names)), dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = mat[:, idx].mean(axis=1)
    return names, point, boot


def contrast(names, point, boot, x, y):
    ix, iy = names.index(x), names.index(y)
    pt = float(point[ix] - point[iy])
    diffs = boot[:, ix] - boot[:, iy]
    lcb = float(np.percentile(diffs, LCB_PCT))
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    return pt, lcb, (ci_lo, ci_hi)


# ---- interpretation matrix (pre-reg §6) ------------------------------------
def matrix_row(p0, p1, p2, u_linfw, u_l1w, rep3b_supports, bottlenecks_differ):
    if not p0:
        if u_l1w > u_linfw:
            concl = ("No evidence L1 weighting is worse; observed ordering FAVORS L1 "
                     "(U_L1w > U_L∞w) → harmful-calibration hypothesis REJECTED.")
        else:
            concl = ("No evidence L1 weighting is worse → harmful-calibration claim "
                     "NOT supported.")
        row = "RE-P0 fail"
    elif p0 and p1 and p2:
        tail = ("RE-P3b supports the per-norm tradeoff" if rep3b_supports
                else "RE-P3b does not clearly support the tradeoff")
        concl = ("Full conditional static-counterfactual pattern; CONSISTENT WITH "
                 "(not proof of) the post-hoc endogenous hypothesis "
                 f"[{tail}].")
        row = "RE-P0 pass / RE-P1 pass / RE-P2 pass"
    elif p0 and p1 and not p2:
        concl = ("L∞ weighting beats L1 weighting AND B4 beats L1 weighting, but B4 "
                 "does NOT match L∞ weighting (fails non-inferiority).")
        row = "RE-P0 pass / RE-P1 pass / RE-P2 fail"
    else:  # p0 pass, p1 fail (any p2)
        concl = ("Static weighting contrast exists, but B4 NOT shown superior to the "
                 "L1-weighted static.")
        row = "RE-P0 pass / RE-P1 fail"
    caveat = ("  ⚠ RE-P3a: the two static arms have DIFFERENT terminal bottlenecks "
              "→ NO shared regime-level bottleneck statement permitted."
              if bottlenecks_differ else "")
    return row, concl, caveat


# ---- reporting -------------------------------------------------------------
def _fmt(pt, lcb, thr, passed, sign):
    return (f"point={pt:+.4f}  LCB(one-sided95)={lcb:+.4f}  "
            f"{'PASS' if passed else 'FAIL'} (need LCB {sign} {thr:+.2f})")


def run_analysis(masks: dict[str, np.ndarray], pernorm: dict | None = None) -> dict:
    names, point, boot = shared_bootstrap(masks)
    U = {k: float(point[names.index(k)]) for k in masks}

    p0_pt, p0_lcb, _ = contrast(names, point, boot, "linf_weighted", "l1_weighted")
    p1_pt, p1_lcb, _ = contrast(names, point, boot, "b4", "l1_weighted")
    p2_pt, p2_lcb, _ = contrast(names, point, boot, "b4", "linf_weighted")
    p0 = p0_lcb > 0.0
    p1 = p1_lcb > 0.0
    p2 = p2_lcb > RE_P2_MARGIN
    conj = p0 and p1 and p2

    print("=== Union robust accuracy (canonical, test_final) ===")
    for k in ("b4", "linf_weighted", "l1_weighted"):
        print(f"  U[{k:14s}] = {U[k]:.4f}")
    print("\n=== Confirmatory (paired bootstrap: unit=example, B=%d, seed=%d, one-sided 95%%) ==="
          % (B_BOOTSTRAP, SEED))
    print("  RE-P0  U_L∞w − U_L1w :", _fmt(p0_pt, p0_lcb, 0.0, p0, ">"))
    print("  RE-P1  U_B4  − U_L1w :", _fmt(p1_pt, p1_lcb, 0.0, p1, ">"))
    print("  RE-P2  U_B4  − U_L∞w :", _fmt(p2_pt, p2_lcb, RE_P2_MARGIN, p2, ">"))
    print(f"\n  CONJUNCTIVE VERDICT RE-P0 ∧ RE-P1 ∧ RE-P2: "
          f"{'PASS' if conj else 'FAIL'}")

    rep3b_supports = False
    bottlenecks_differ = False
    if pernorm:
        print("\n=== RE-P3 (DESCRIPTIVE, non-gating) ===")
        # RE-P3a: per-arm terminal ordering
        bottleneck = {}
        for arm in ("b4", "linf_weighted", "l1_weighted"):
            if all((arm, nk) in pernorm for nk in ("linf", "l2", "l1")):
                a = {nk: float(pernorm[(arm, nk)].mean()) for nk in ("linf", "l2", "l1")}
                bmin = min(a, key=a.get)
                uniq_linf = a["linf"] <= min(a["l2"], a["l1"]) - RE_P3_M
                bottleneck[arm] = bmin
                print(f"  RE-P3a[{arm:14s}] a_∞={a['linf']:.4f} a_2={a['l2']:.4f} "
                      f"a_1={a['l1']:.4f} → argmin={bmin}"
                      f"{'  (ℓ∞ UNIQUELY limiting)' if uniq_linf else ''}")
        if {"linf_weighted", "l1_weighted"} <= set(bottleneck):
            bottlenecks_differ = bottleneck["linf_weighted"] != bottleneck["l1_weighted"]
        # RE-P3b: per-norm tradeoff (L1w − L∞w) on L1 and on ℓ∞, paired CIs
        need = [("l1_weighted", "l1"), ("linf_weighted", "l1"),
                ("l1_weighted", "linf"), ("linf_weighted", "linf")]
        if all(k in pernorm for k in need):
            pn = {f"{arm}:{nk}": pernorm[(arm, nk)] for (arm, nk) in need}
            nn, pnt, bt = shared_bootstrap(pn)
            d_l1_pt, _, d_l1_ci = contrast(nn, pnt, bt, "l1_weighted:l1", "linf_weighted:l1")
            d_inf_pt, _, d_inf_ci = contrast(nn, pnt, bt, "l1_weighted:linf", "linf_weighted:linf")
            print(f"  RE-P3b  a_L1w,L1 − a_L∞w,L1 = {d_l1_pt:+.4f} "
                  f"CI95[{d_l1_ci[0]:+.4f},{d_l1_ci[1]:+.4f}] (expect ≥ +{RE_P3_M})")
            print(f"  RE-P3b  a_L1w,∞  − a_L∞w,∞  = {d_inf_pt:+.4f} "
                  f"CI95[{d_inf_ci[0]:+.4f},{d_inf_ci[1]:+.4f}] (expect ≤ −{RE_P3_M})")
            rep3b_supports = (d_l1_pt >= RE_P3_M) and (d_inf_pt <= -RE_P3_M)

    row, concl, caveat = matrix_row(p0, p1, p2, U["linf_weighted"], U["l1_weighted"],
                                    rep3b_supports, bottlenecks_differ)
    print("\n=== Interpretation matrix (pre-reg §6) ===")
    print(f"  matched row: [{row}]")
    print(f"  conclusion : {concl}")
    if caveat:
        print(caveat)
    return {"RE-P0": p0, "RE-P1": p1, "RE-P2": p2, "conjunction": conj,
            "U": U, "matrix_row": row}


# ---- CLI -------------------------------------------------------------------
def _pernorm_from_args(args) -> dict:
    out = {}
    for arm, prefix in (("b4", "b4"), ("linf_weighted", "linf_weighted"),
                        ("l1_weighted", "l1_weighted")):
        for nk in ("linf", "l2", "l1"):
            v = getattr(args, f"{prefix}_{nk}".replace("-", "_"), None)
            if v:
                out[(arm, nk)] = load_mask(v, key=args.mask_key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b4")
    ap.add_argument("--linf-weighted")
    ap.add_argument("--l1-weighted")
    ap.add_argument("--mask-key", default=None)
    for arm in ("b4", "linf-weighted", "l1-weighted"):
        for nk in ("linf", "l2", "l1"):
            ap.add_argument(f"--{arm}-{nk}")
    ap.add_argument("--selftest", action="store_true",
                    help="Run synthetic constructed-case tests and exit (no real data).")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.b4 and args.linf_weighted and args.l1_weighted):
        ap.error("need --b4, --linf-weighted, --l1-weighted (or --selftest)")
    masks = {"b4": load_mask(args.b4, args.mask_key),
             "linf_weighted": load_mask(args.linf_weighted, args.mask_key),
             "l1_weighted": load_mask(args.l1_weighted, args.mask_key)}
    run_analysis(masks, _pernorm_from_args(args) or None)


# ---- synthetic self-test (constructed cases; NO real audit data) -----------
def _union(n: int, robust_upto: int) -> np.ndarray:
    """Mask with examples [0, robust_upto) robust (=1)."""
    m = np.zeros(n, dtype=np.int8)
    m[:robust_upto] = 1
    return m


def selftest() -> None:
    n = 8000
    print("### SELF-TEST (synthetic constructed masks; no real audit output) ###\n")

    # Nested construction so contrasts have controlled point + tight bootstrap:
    #   l1w robust = [0,3000)         U=0.3750
    #   linfw robust= [0,3400)        U=0.4250   (RE-P0 diff = +0.0500)
    l1w = _union(n, 3000)
    linfw = _union(n, 3400)

    # Case A — all pass: b4 = [0,3320) -> U=0.4150; b4-l1w=+0.040(>0), b4-linfw=-0.010(>-0.02)
    b4_A = _union(n, 3320)
    print("[Case A] expect RE-P0 pass, RE-P1 pass, RE-P2 pass, CONJUNCTION PASS")
    rA = run_analysis({"b4": b4_A, "linf_weighted": linfw, "l1_weighted": l1w})
    assert rA["RE-P0"] and rA["RE-P1"] and rA["RE-P2"] and rA["conjunction"], rA
    assert rA["matrix_row"].endswith("RE-P2 pass"), rA["matrix_row"]

    # Case B — RE-P2 just FAILS at the −0.02 margin: b4 = [0,3160) -> U=0.3950;
    #   b4-linfw = -0.030 (< -0.02 -> FAIL); b4-l1w = +0.020 (>0 -> RE-P1 pass)
    b4_B = _union(n, 3160)
    print("\n[Case B] expect RE-P0 pass, RE-P1 pass, RE-P2 FAIL (@ −0.02), CONJUNCTION FAIL")
    rB = run_analysis({"b4": b4_B, "linf_weighted": linfw, "l1_weighted": l1w})
    assert rB["RE-P0"] and rB["RE-P1"] and (not rB["RE-P2"]) and (not rB["conjunction"]), rB
    assert rB["matrix_row"].endswith("RE-P2 fail"), rB["matrix_row"]

    # Case C — RE-P2 just PASSES near margin: b4 = [0,3300) -> U=0.4125;
    #   b4-linfw = -0.0125 (> -0.02 -> PASS)
    b4_C = _union(n, 3300)
    print("\n[Case C] expect RE-P2 PASS just inside −0.02 margin (point −0.0125)")
    rC = run_analysis({"b4": b4_C, "linf_weighted": linfw, "l1_weighted": l1w})
    assert rC["RE-P2"], rC

    # Case D — RE-P0 FAIL, ordering favors L1 (hypothesis rejected):
    #   swap so l1w robust ⊃ linfw robust
    print("\n[Case D] expect RE-P0 FAIL with U_L1w > U_L∞w → hypothesis rejected")
    rD = run_analysis({"b4": b4_A, "linf_weighted": _union(n, 3000),
                       "l1_weighted": _union(n, 3400)})
    assert not rD["RE-P0"], rD
    assert rD["matrix_row"] == "RE-P0 fail", rD["matrix_row"]

    # Case E — RE-P3 descriptive: ℓ∞ uniquely limiting for both statics (shared
    #   bottleneck) + tradeoff signs; then a differing-bottleneck variant.
    print("\n[Case E] RE-P3a/b descriptive (ℓ∞ uniquely limiting; tradeoff signs)")
    pernorm = {
        # linf-weighted: defends ℓ∞ more (a_∞ higher), L1 lower
        ("linf_weighted", "linf"): _union(n, 3600),
        ("linf_weighted", "l2"):   _union(n, 5400),
        ("linf_weighted", "l1"):   _union(n, 3900),
        # l1-weighted: defends L1 more (a_1 higher, +0.05), ℓ∞ lower (−0.05)
        ("l1_weighted", "linf"):   _union(n, 3200),
        ("l1_weighted", "l2"):     _union(n, 5400),
        ("l1_weighted", "l1"):     _union(n, 4300),
        # b4
        ("b4", "linf"): _union(n, 3400),
        ("b4", "l2"):   _union(n, 5400),
        ("b4", "l1"):   _union(n, 3950),
    }
    rE = run_analysis({"b4": b4_A, "linf_weighted": linfw, "l1_weighted": l1w}, pernorm)
    # a_{L1w,L1}-a_{L∞w,L1} = (4300-3900)/8000 = +0.05 >= +0.03 ; a_{L1w,∞}-a_{L∞w,∞}=(3200-3600)/8000=-0.05 <= -0.03
    # both static argmin = linf here -> no differ note

    # Case F — differing terminal bottlenecks -> caveat must fire
    print("\n[Case F] RE-P3a differing bottlenecks → shared-bottleneck caveat")
    pernorm_F = dict(pernorm)
    pernorm_F[("l1_weighted", "l1")] = _union(n, 2600)   # now L1 is L1w's argmin
    pernorm_F[("l1_weighted", "linf")] = _union(n, 3500)
    pernorm_F[("l1_weighted", "l2")] = _union(n, 5400)
    _ = run_analysis({"b4": b4_A, "linf_weighted": linfw, "l1_weighted": l1w}, pernorm_F)

    print("\n### SELF-TEST PASSED — all constructed RE-P thresholds triggered as expected ###")


if __name__ == "__main__":
    main()

"""RQ4 — union-failure anatomy, recomputed from the frozen @10k per-example masks.

Every number the paper's RQ4 section cites must regenerate from the published artifact. This
script recomputes the anatomy for EVERY arm that has a complete 12-key @10k mask set, plus the
two geometry radii, and reports an explicit drop list of quantities that do NOT regenerate.

DEFINITIONS (used exactly as specified, mask convention True = robust):
  norm n BLOCKS example i   <=>  at least one of n's four attacks defeats i
                            <=>  NOT (AND of n's four masks)
  i is a UNION FAILURE      <=>  at least one norm blocks it  <=>  NOT (AND of all 12 masks)
  BLOCKING SET of a union failure = the set of norms that block it (size 1, 2 or 3)
  "X-sole"        = blocking set is exactly {X}
  "X participates"= X is in the blocking set
All shares are over UNION FAILURES only, never over all 10000 examples.

Mask loading and the arm loader are reused from scripts/dev/make_table1.py.

Read-only on results/main/ and on every checkpoint. No attack, training, or evaluation.

  python scripts/dev/rq4_anatomy.py
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, ROOT)
_s = importlib.util.spec_from_file_location("t1", os.path.join(ROOT, "scripts/dev/make_table1.py"))
t1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t1)

OUT = os.path.join(ROOT, "results/analysis/RQ4_anatomy")
MAIN = os.path.join(ROOT, "results/main")
NORMS = ("linf", "l2", "l1")
SETS = [("linf",), ("l2",), ("l1",), ("linf", "l2"), ("linf", "l1"), ("l2", "l1"),
        ("linf", "l2", "l1")]
SETKEY = ["{" + ",".join(s) + "}" for s in SETS]
EPS_INF, EPS_1, EPS_2, DIM = 8 / 255, 12.0, 0.5, 3 * 32 * 32

# draft values to be checked (paper's current RQ4 text)
DRAFT = {"M0_full": {"label": "MSD-50", "single": 0.146, "multi": 0.854,
                     "sole": {"linf": 0.132, "l2": 0.000, "l1": 0.014}},
         "Rprime":  {"label": "RAMP",   "single": 0.104, "multi": 0.896,
                     "sole": {"linf": 0.082, "l2": 0.000, "l1": 0.022}}}
DRAFT_RADII = {"r_union": 0.218, "r_hull": 0.614}


# --------------------------------------------------------------------------- anatomy
def anatomy(A):
    """Per-arm blocking-set anatomy from the 12 masks."""
    m = A["masks"]
    blocked = {}
    for nm in NORMS:
        bl, ks = t1.block(m, nm)                 # AND of that norm's 4 masks (True = survives norm)
        assert len(ks) == 4, f"{A['arm']}/{nm}: {len(ks)} attacks in block"
        blocked[nm] = ~bl                        # norm BLOCKS the example
    uf = blocked["linf"] | blocked["l2"] | blocked["l1"]
    assert np.array_equal(uf, ~t1.union(m)), "union failure != NOT(AND of all 12)"
    n_uf = int(uf.sum())

    counts, shares = {}, {}
    for key, s in zip(SETKEY, SETS):
        sel = uf.copy()
        for nm in NORMS:
            sel &= blocked[nm] if nm in s else ~blocked[nm]
        counts[key] = int(sel.sum())
        shares[key] = (counts[key] / n_uf) if n_uf else float("nan")
    assert sum(counts.values()) == n_uf, "blocking sets do not partition the union failures"

    k = blocked["linf"].astype(int) + blocked["l2"].astype(int) + blocked["l1"].astype(int)
    single = int(((k == 1) & uf).sum())
    multi = int(((k >= 2) & uf).sum())
    assert single + multi == n_uf
    return {
        "n": int(len(uf)), "n_union_failures": n_uf,
        "union_robust_acc": float(t1.union(m).mean()),
        "blocking_set_counts": counts,
        "blocking_set_shares": shares,
        "participation_count": {nm: int((blocked[nm] & uf).sum()) for nm in NORMS},
        "participation_share": {nm: float((blocked[nm] & uf).sum() / n_uf) if n_uf else float("nan")
                                for nm in NORMS},
        "single_norm_blocked_count": single, "multi_norm_blocked_count": multi,
        "single_norm_blocked_share": single / n_uf if n_uf else float("nan"),
        "multi_norm_blocked_share": multi / n_uf if n_uf else float("nan"),
    }


# --------------------------------------------------------------------------- geometry
def r_union_closed(eps_inf, eps_1, d):
    """Largest r with  B2(r) subset of  Binf(eps_inf) UNION B1(eps_1).

    B2(r) is contained in the union iff no point of norm <= r lies OUTSIDE both, so
        r_union = min ||x||_2  s.t.  ||x||_inf >= eps_inf  AND  ||x||_1 >= eps_1.
    By symmetry take x >= 0. One coordinate must carry a >= eps_inf; for a fixed L1 budget the
    L2 norm of the remainder is minimised by spreading it equally, and over as many coordinates
    as possible, i.e. all d-1 of them. So
        f(a) = a^2 + (eps_1 - a)^2 / (d-1),   a in [eps_inf, eps_1].
    df/da = 0 gives a* = eps_1/d. Here eps_1/d < eps_inf, so f is increasing on the feasible
    interval and the optimum sits on the boundary a = eps_inf:
        r_union = sqrt( eps_inf^2 + (eps_1 - eps_inf)^2 / (d-1) ).
    """
    a_star = eps_1 / d
    a = max(eps_inf, a_star)
    return float(np.sqrt(a ** 2 + (eps_1 - a) ** 2 / (d - 1))), float(a_star), float(a)


def r_union_numeric(eps_inf, eps_1, d, n=200001):
    """Same quantity by 1-D scan over the free coordinate a, as an independent check."""
    a = np.linspace(eps_inf, eps_1, n)
    f = a ** 2 + (eps_1 - a) ** 2 / (d - 1)
    i = int(np.argmin(f))
    return float(np.sqrt(f[i])), float(a[i]), float((eps_1 - eps_inf) / (n - 1))


def r_hull_bounds(eps_inf, eps_1, d, n=2000001):
    """Largest r with  B2(r) subset of conv( Binf(eps_inf) UNION B1(eps_1) ).

    The support function of a convex hull is the max of the support functions:
        h(u) = max( eps_inf * ||u||_1 , eps_1 * ||u||_inf ).
    B2(r) subset of K iff r <= h(u) for every unit u, so r_hull = min_{||u||_2=1} h(u).

    LOWER BOUND (closed form). For unit u with ||u||_inf = t,
        1 = sum u_i^2 <= ||u||_inf * ||u||_1 = t ||u||_1   =>   ||u||_1 >= 1/t,
    hence h(u) >= max(eps_inf/t, eps_1 t) >= min_t max(eps_inf/t, eps_1 t) = sqrt(eps_inf*eps_1),
    the two branches crossing at t = sqrt(eps_inf/eps_1).

    UPPER BOUND (attained). Taking k equal non-zero coordinates, u = (k^-1/2, ..., 0), gives
    ||u||_1 = sqrt(k) and ||u||_inf = 1/sqrt(k), so
        r_hull <= min_{k integer} max( eps_inf*sqrt(k), eps_1/sqrt(k) ),
    the balance point being k = eps_1/eps_inf (non-integer in general, hence the gap).

    EXACT (numeric). For a fixed t the smallest achievable ||u||_1 is k*t + sqrt(1 - k t^2) with
    k = floor(1/t^2) coordinates at t and one remainder coordinate, so
        r_hull = min_t max( eps_inf * (k t + sqrt(1 - k t^2)) , eps_1 * t )
    is a 1-D problem, scanned below. This is exact up to the scan resolution.
    """
    lower = float(np.sqrt(eps_inf * eps_1))
    t_star = float(np.sqrt(eps_inf / eps_1))
    ks = np.arange(1, d + 1, dtype=float)
    fk = np.maximum(eps_inf * np.sqrt(ks), eps_1 / np.sqrt(ks))
    ik = int(np.argmin(fk))
    upper, k_best = float(fk[ik]), int(ks[ik])

    t = np.linspace(1.0 / np.sqrt(d), 1.0, n)          # ||u||_inf ranges over [d^-1/2, 1]
    k = np.floor(1.0 / t ** 2)
    k = np.minimum(k, d)
    rem = np.clip(1.0 - k * t ** 2, 0.0, None)
    l1min = k * t + np.sqrt(rem)
    h = np.maximum(eps_inf * l1min, eps_1 * t)
    i = int(np.argmin(h))
    return {"lower_bound_closed_form": lower, "t_star_closed_form": t_star,
            "equal_coordinate_upper_bound": upper, "k_balance_exact": float(eps_1 / eps_inf),
            "k_best_integer": k_best,
            "exact_numeric": float(h[i]), "t_at_min": float(t[i]),
            "scan_points": n, "scan_step_t": float((1.0 - 1.0 / np.sqrt(d)) / (n - 1))}


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="10k")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- arm discovery: every complete 12-key @10k mask set, no substitution ----
    included, excluded = [], []
    for arm in sorted(os.listdir(MAIN)):
        pth = os.path.join(MAIN, arm, a.scale, "masks_multinorm_v1.npz")
        if not os.path.isdir(os.path.join(MAIN, arm)):
            continue
        if not os.path.exists(pth):
            excluded.append({"arm": arm, "reason": f"no {a.scale} mask file"})
            continue
        d = np.load(pth, allow_pickle=True)
        ks = [k for k in d.files if k != "metadata_json"]
        n = len(d[ks[0]])
        if len(ks) != 12:
            excluded.append({"arm": arm, "reason": f"{len(ks)} mask keys, need 12"})
        elif n != 10000:
            excluded.append({"arm": arm, "reason": f"n={n}, need 10000"})
        else:
            included.append(arm)

    fail = []
    arms = {arm: t1.load_arm(arm, fail) for arm in included}
    assert not fail, fail

    subs = {v["meta"].get("audit_subset_sha256") for v in arms.values()}
    assert len(subs) == 1, f"arms do not share one audit subset: {subs}"
    sub_sha = subs.pop()
    sub_path = next(iter(arms.values()))["meta"].get("audit_subset_path")

    print("=" * 122)
    print(f"RQ4 ANATOMY — recomputed from the frozen @{a.scale} per-example masks")
    print(f"  arms with a COMPLETE 12-key @{a.scale} mask set: {len(included)}")
    print(f"  excluded: {len(excluded)}  ({', '.join(e['arm'] for e in excluded) or 'none'})")
    print(f"  audit subset {sub_path}  sha {sub_sha[:16]}")
    print("=" * 122)

    R = {arm: anatomy(arms[arm]) for arm in included}

    # ---- per-arm table ----
    hdr = (f"{'arm':<16}{'n_UF':>7} | " + "".join(f"{k:>13}" for k in SETKEY)
           + " | " + "".join(f"{'part.' + n:>10}" for n in NORMS) + f"{'1-norm':>9}{'multi':>8}")
    print(hdr)
    print("-" * len(hdr))
    for arm in included:
        r = R[arm]
        print(f"{arm:<16}{r['n_union_failures']:>7} | "
              + "".join(f"{r['blocking_set_shares'][k]:>13.4f}" for k in SETKEY)
              + " | " + "".join(f"{r['participation_share'][n]:>10.4f}" for n in NORMS)
              + f"{r['single_norm_blocked_share']:>9.4f}{r['multi_norm_blocked_share']:>8.4f}")
    print("\n  (shares are of UNION FAILURES; counts are in the JSON)")

    # ---- cross-arm summary ----
    def rng(f):
        vals = {arm: f(R[arm]) for arm in included}
        lo = min(vals, key=vals.get); hi = max(vals, key=vals.get)
        return {"min": vals[lo], "min_arm": lo, "max": vals[hi], "max_arm": hi, "per_arm": vals}

    l2_sole = rng(lambda r: r["blocking_set_shares"]["{l2}"])
    l2_pair = rng(lambda r: r["blocking_set_shares"]["{linf,l2}"] + r["blocking_set_shares"]["{l2,l1}"])
    linf_part = rng(lambda r: r["participation_share"]["linf"])
    l2_sole_zero = [arm for arm in included if R[arm]["blocking_set_counts"]["{l2}"] == 0]
    l2_pair_zero = [arm for arm in included if R[arm]["blocking_set_counts"]["{linf,l2}"]
                    + R[arm]["blocking_set_counts"]["{l2,l1}"] == 0]
    l2_sole_nonzero = [arm for arm in included if R[arm]["blocking_set_counts"]["{l2}"] > 0]
    l2_pair_nonzero = [arm for arm in included if R[arm]["blocking_set_counts"]["{linf,l2}"]
                       + R[arm]["blocking_set_counts"]["{l2,l1}"] > 0]

    print("\n" + "=" * 122)
    print("CROSS-ARM SUMMARY")
    print("=" * 122)
    print(f"  l2-sole share            min {l2_sole['min']:.6f} ({l2_sole['min_arm']})   "
          f"max {l2_sole['max']:.6f} ({l2_sole['max_arm']})")
    print(f"     arms with l2-sole EXACTLY zero: {len(l2_sole_zero)}/{len(included)}")
    if l2_sole_nonzero:
        print("     EXCEPTIONS (l2-sole > 0): " + ", ".join(
            f"{x} n={R[x]['blocking_set_counts']['{l2}']}" for x in l2_sole_nonzero))
    print(f"  l2 in a TWO-norm set     min {l2_pair['min']:.6f} ({l2_pair['min_arm']})   "
          f"max {l2_pair['max']:.6f} ({l2_pair['max_arm']})")
    print(f"     arms with it EXACTLY zero: {len(l2_pair_zero)}/{len(included)}")
    if l2_pair_nonzero:
        print("     EXCEPTIONS (l2 in a 2-norm set): " + ", ".join(
            f"{x} n={R[x]['blocking_set_counts']['{linf,l2}'] + R[x]['blocking_set_counts']['{l2,l1}']}"
            for x in l2_pair_nonzero))
    print(f"  linf participation share min {linf_part['min']:.6f} ({linf_part['min_arm']})   "
          f"max {linf_part['max']:.6f} ({linf_part['max_arm']})")

    claims = {
        "l2 is never the sole blocker": {
            "holds_for_all_arms": not l2_sole_nonzero, "n_arms_holding": len(l2_sole_zero),
            "n_arms_total": len(included), "exceptions": l2_sole_nonzero},
        "l2 never appears in a two-norm blocking set": {
            "holds_for_all_arms": not l2_pair_nonzero, "n_arms_holding": len(l2_pair_zero),
            "n_arms_total": len(included), "exceptions": l2_pair_nonzero},
    }
    for k, v in claims.items():
        print(f"\n  CLAIM \"{k}\": "
              + ("HOLDS for ALL " if v["holds_for_all_arms"] else "does NOT hold for all ")
              + f"{v['n_arms_total']} arms"
              + ("" if v["holds_for_all_arms"] else f" — exceptions: {v['exceptions']}"))

    # ---- Phase 2: cross-check against the draft and against the prior analysis file ----
    print("\n" + "=" * 122)
    print("PHASE 2 — CROSS-CHECK")
    print("=" * 122)
    xd = {}
    for arm, want in DRAFT.items():
        if arm not in R:
            xd[arm] = {"status": "arm not available at this scale/tier"}
            print(f"  {arm}: NOT AVAILABLE"); continue
        r = R[arm]
        got = {"single": r["single_norm_blocked_share"], "multi": r["multi_norm_blocked_share"],
               "sole": {n: r["blocking_set_shares"]["{" + n + "}"] for n in NORMS}}
        rows = [("single-norm-blocked", want["single"], got["single"]),
                ("multi-norm-blocked", want["multi"], got["multi"])]
        rows += [(f"{n}-sole", want["sole"][n], got["sole"][n]) for n in NORMS]
        print(f"  {arm}  (draft label '{want['label']}')")
        ok = True
        for lbl, w, g in rows:
            d_ = abs(w - g)
            hit = d_ <= 0.0005 + 1e-12                     # draft is quoted to 0.1 pp
            ok &= hit
            print(f"    {lbl:<22} draft {100*w:>6.1f}%   recomputed {100*g:>7.3f}%   "
                  f"|Δ| {100*d_:>5.3f} pp  {'match' if hit else 'MISMATCH'}")
        xd[arm] = {"draft": want, "recomputed": got, "all_match_to_0.1pp": bool(ok)}

    # prior anatomy file
    prior_p = os.path.join(ROOT, "results/analysis/T3_threat_dominance/threat_dominance.json")
    prior_cmp = {"file": os.path.relpath(prior_p, ROOT), "exists": os.path.exists(prior_p), "rows": []}
    if os.path.exists(prior_p):
        prior = json.load(open(prior_p))
        print(f"\n  prior file {prior_cmp['file']} — {len(prior)} rows")
        for row in prior:
            arm, sc, tier = row.get("arm"), row.get("scale"), row.get("tier")
            comparable = (sc == a.scale and tier == "full" and arm in R)
            e = {"arm": arm, "scale": sc, "tier": tier, "n_attacks": row.get("n_attacks"),
                 "comparable_to_this_run": bool(comparable)}
            if comparable:
                r = R[arm]
                e["prior"] = {"n_union_fail": row.get("n_union_fail"), "cov": row.get("cov"),
                              "exclusive": row.get("exclusive"), "all_three": row.get("all_three")}
                e["recomputed"] = {"n_union_fail": r["n_union_failures"],
                                   "cov": r["participation_share"],
                                   "exclusive": {n: r["blocking_set_shares"]["{" + n + "}"] for n in NORMS},
                                   "all_three": r["blocking_set_shares"]["{linf,l2,l1}"]}
                dmax = max([abs(row["cov"][n] - r["participation_share"][n]) for n in NORMS]
                           + [abs(row["exclusive"][n] - r["blocking_set_shares"]["{" + n + "}"]) for n in NORMS]
                           + [abs(row["all_three"] - r["blocking_set_shares"]["{linf,l2,l1}"]),
                              abs(row["n_union_fail"] - r["n_union_failures"])])
                e["max_abs_diff"] = float(dmax)
                e["agrees"] = bool(dmax < 1e-9)
                print(f"    {arm:<14} {sc:<8} {tier:<10} comparable -> "
                      f"{'AGREES exactly' if e['agrees'] else f'DIFFERS max|Δ|={dmax:.3e}'}")
            else:
                e["reason_not_comparable"] = (
                    f"prior row is scale={sc} tier={tier} ({row.get('n_attacks')} attacks); this run is "
                    f"scale={a.scale} tier=full (12 attacks)" if arm in R or True else "")
                print(f"    {arm:<14} {sc:<8} {tier:<10} NOT comparable "
                      f"({row.get('n_attacks')} attacks) — excluded from the cross-check")
            prior_cmp["rows"].append(e)

    # ---- Phase 3: geometry ----
    ru, a_star, a_used = r_union_closed(EPS_INF, EPS_1, DIM)
    ru_num, a_num, a_step = r_union_numeric(EPS_INF, EPS_1, DIM)
    rh = r_hull_bounds(EPS_INF, EPS_1, DIM)
    print("\n" + "=" * 122)
    print("PHASE 3 — GEOMETRY RADII   (d = 3*32*32 = 3072, eps_inf = 8/255, eps_1 = 12, eps_2 = 0.5)")
    print("=" * 122)
    print(f"  r_union  closed form = sqrt(eps_inf^2 + (eps_1-eps_inf)^2/(d-1)) = {ru:.6f}")
    print(f"           unconstrained optimum a* = eps_1/d = {a_star:.6f} < eps_inf = {EPS_INF:.6f},")
    print(f"           so the optimum is on the boundary a = {a_used:.6f}")
    print(f"           1-D numeric scan = {ru_num:.6f}  (a at min {a_num:.6f}, step {a_step:.2e})")
    print(f"           draft says {DRAFT_RADII['r_union']}  -> "
          f"{'MATCHES to 3 dp' if abs(ru - DRAFT_RADII['r_union']) < 5e-4 else 'DIFFERS'}")
    print(f"  r_hull   closed-form LOWER bound sqrt(eps_inf*eps_1) = {rh['lower_bound_closed_form']:.6f}")
    print(f"           equal-coordinate UPPER bound (k={rh['k_best_integer']}) = "
          f"{rh['equal_coordinate_upper_bound']:.6f}   (exact balance k = {rh['k_balance_exact']:.1f})")
    print(f"           exact 1-D numeric        = {rh['exact_numeric']:.6f}  "
          f"(t at min {rh['t_at_min']:.6f}, {rh['scan_points']} points)")
    print(f"           draft says {DRAFT_RADII['r_hull']}  -> "
          f"{'MATCHES to 3 dp' if abs(rh['exact_numeric'] - DRAFT_RADII['r_hull']) < 5e-4 else 'DIFFERS'}")
    print(f"  eps_2 = {EPS_2}:  r_union {ru:.6f} < eps_2 < r_hull {rh['exact_numeric']:.6f}"
          if ru < EPS_2 < rh["exact_numeric"] else f"  eps_2 = {EPS_2}: OUTSIDE the bracket")

    geom = {
        "dim": DIM, "eps_inf": EPS_INF, "eps_1": EPS_1, "eps_2": EPS_2,
        "r_union": {"value": ru, "closed_form": "sqrt(eps_inf^2 + (eps_1-eps_inf)^2/(d-1))",
                    "derivation": r_union_closed.__doc__.strip(),
                    "numeric_check": ru_num, "numeric_scan_step_a": a_step,
                    "draft_value": DRAFT_RADII["r_union"],
                    "matches_draft_3dp": bool(abs(ru - DRAFT_RADII["r_union"]) < 5e-4)},
        "r_hull": {**rh, "value": rh["exact_numeric"],
                   "derivation": r_hull_bounds.__doc__.strip(),
                   "draft_value": DRAFT_RADII["r_hull"],
                   "matches_draft_3dp": bool(abs(rh["exact_numeric"] - DRAFT_RADII["r_hull"]) < 5e-4)},
        "eps2_position": ("r_union < eps_2 < r_hull" if ru < EPS_2 < rh["exact_numeric"]
                          else "outside the bracket"),
    }

    # ---- Phase 4: drop list ----
    margin_p = os.path.join(ROOT, "results/analysis/margin_analysis_2026-07-18.json")
    margin_exists = os.path.exists(margin_p)
    margin_arms = []
    if margin_exists:
        margin_arms = [r.get("arm") for r in json.load(open(margin_p)).get("results", [])]
    drops = [
        {"quantity": "blocking-set anatomy (7 sets, participation, single/multi)",
         "regenerates": True,
         "how": "recomputed here from the 12 per-example masks of each arm"},
        {"quantity": "arm count for the anatomy figure", "regenerates": True,
         "how": f"{len(included)} arms have a complete 12-key @{a.scale} mask set",
         "note": f"excluded: {[e['arm'] for e in excluded]}"},
        {"quantity": "r_union and r_hull", "regenerates": True,
         "how": "closed form plus an independent 1-D numeric scan; needs no data at all"},
        {"quantity": "clean-margin-by-blocking-count table", "regenerates": False,
         "why": "clean logit margins are NOT in the frozen artifact. The mask sidecars store only "
                "the 12 attack outcomes; no clean per-example margin, logit, or even a clean "
                "correctness mask is written (scripts/eval_multinorm_audit.py computes clean_acc "
                "and reduces it to a scalar). Regenerating margins requires loading each "
                "checkpoint and running a forward pass, which is an evaluation, not a read of the "
                "artifact.",
         "prior_output_exists": margin_exists,
         "prior_output": os.path.relpath(margin_p, ROOT) if margin_exists else None,
         "prior_output_arms": margin_arms,
         "status": "DROP from the paper unless the section is allowed to cite a "
                   "forward-pass-derived quantity; the per-example margins behind the prior file "
                   "were never persisted, so even that file cannot be re-derived from what is stored"},
        {"quantity": "'margin predicts multi-blocked' AUC", "regenerates": False,
         "why": "same missing input as the margin table — the AUC is computed from per-example "
                "clean margins joined to the blocking count. The blocking count regenerates; the "
                "margin does not.",
         "prior_output_exists": margin_exists,
         "prior_output": os.path.relpath(margin_p, ROOT) if margin_exists else None,
         "prior_output_arms": margin_arms,
         "status": "DROP from the paper on the same grounds"},
    ]
    print("\n" + "=" * 122)
    print("PHASE 4 — WHAT REGENERATES FROM THE FROZEN ARTIFACT, AND WHAT DOES NOT")
    print("=" * 122)
    for d_ in drops:
        tag = "REGENERATES" if d_["regenerates"] else "DOES NOT REGENERATE -> DROP"
        print(f"  [{tag}] {d_['quantity']}")
        print(f"      {d_.get('how') or d_.get('why')}")
        if d_.get("prior_output_exists"):
            print(f"      a prior output file exists ({d_['prior_output']}) covering "
                  f"{len(d_.get('prior_output_arms') or [])} arm(s); it is NOT re-derivable from the masks")

    # ---- output ----
    payload = {
        "what": "RQ4 union-failure anatomy recomputed from the frozen per-example masks",
        "definitions": {
            "norm_blocks": "at least one of that norm's four attacks defeats the example "
                           "= NOT(AND of the norm's four masks)",
            "union_failure": "at least one norm blocks the example = NOT(AND of all 12 masks)",
            "blocking_set": "the set of norms that block a union failure (size 1, 2 or 3)",
            "X_sole": "blocking set is exactly {X}",
            "X_participates": "X is in the blocking set",
            "shares": "ALL shares are over union failures only, never over all 10000 examples",
            "mask_convention": "true_means_robust_false_means_failed",
        },
        "scope": {"scale": a.scale, "tier": "full (12 attacks)",
                  "n_arms_included": len(included), "arms_included": included,
                  "n_arms_excluded": len(excluded), "arms_excluded": excluded,
                  "audit_subset_path": sub_path, "audit_subset_sha256": sub_sha},
        "per_arm": R,
        "cross_arm": {"l2_sole_share": l2_sole, "l2_in_two_norm_set_share": l2_pair,
                      "linf_participation_share": linf_part,
                      "arms_with_l2_sole_exactly_zero": l2_sole_zero,
                      "arms_with_l2_in_two_norm_set_exactly_zero": l2_pair_zero,
                      "claims": claims},
        "cross_check_draft": xd,
        "cross_check_prior_analysis": prior_cmp,
        "geometry": geom,
        "drop_list": drops,
        "provenance": {arm: {"mask_file": arms[arm]["path"],
                             "checkpoint_path": arms[arm]["meta"].get("checkpoint_path"),
                             "checkpoint_sha256": arms[arm]["meta"].get("checkpoint_sha256"),
                             "checkpoint_role": arms[arm]["meta"].get("checkpoint_role"),
                             "audit_subset_path": arms[arm]["meta"].get("audit_subset_path"),
                             "audit_subset_sha256": arms[arm]["meta"].get("audit_subset_sha256")}
                       for arm in included},
    }
    json.dump(payload, open(os.path.join(OUT, "rq4_anatomy.json"), "w"), indent=2)
    print(f"\n  saved -> results/analysis/RQ4_anatomy/rq4_anatomy.json")


if __name__ == "__main__":
    main()

"""JOB A — GATE + mask analytics (read-only; NO GPU, NO training).

Consumes per-example AutoAttack masks in <base>/<arm>/[tier/]masks_multinorm_v1.npz produced by
union_bench_eval.py --export-masks. Mask convention (from eval_multinorm_audit): a mask entry is
True = example ROBUST to that attack, False = FOOLED. 4 attacks per norm (apgd-ce, apgd-dlr, fab-t,
square); the @1k no-Square tier has only 3 (square_* absent) — the tier is detected and reported.

Definitions (mirror the audit):
  example FAILS norm n  ⇔  ANY attack of norm n succeeds  ⇔  per-norm robust (AND of that norm's
                            attacks) is False.
  blocking set  B_i = { n : example i fails n }.   union-fail ⇔ B_i ≠ ∅.

Outputs (console + results/analysis/mask_analytics_<date>.{json,*.csv}):
  1. Per-norm Δ table for every requested pair: Δunion, Δℓ∞, Δℓ₂, Δℓ₁, Δclean (if available).
  2. Blocker census (M0, M1a, M0_full, R′): |B_i| distribution + exact-composition % on the failing set.
  3. Flip analysis (M0→M1a, M0_full→M1a_full): fail→pass B_i BEFORE; pass→fail B_i AFTER.
  4. GATE metric: % single-norm-blocked on the failing set of R′ and M0_full (+ per-norm breakdown),
     GATE {OPEN|CLOSED} vs threshold.

Read-only: never writes into any <arm> dir; only into results/analysis/.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

NORMS = ["linf", "l2", "l1"]
NORM_LABEL = {"linf": "ℓ∞", "l2": "ℓ₂", "l1": "ℓ₁"}
ATTACKS_BY_NORM = {n: [f"apgd_ce_{n}", f"apgd_dlr_{n}", f"fab_t_{n}", f"square_{n}"] for n in NORMS}
MASK_FILE = "masks_multinorm_v1.npz"

# logical arm -> ordered candidate relative paths (first existing wins). Tier noted per arm below.
ARM_SPECS = {
    "M0":          ["M0/10k", "M0"],
    "M1a":         ["M1a/10k", "M1a"],
    "M0_full":     ["M0_full/10k", "M0_full"],
    "M1a_full":    ["M1a_full/10k", "M1a_full"],
    "Rprime":      ["Rprime/10k", "Rprime", "R_prime/10k", "R_prime"],
    "B1":          ["B1/10k", "B1"],
    "M1a_msdglue": ["M1a_msdglue/10k", "M1a_msdglue"],
    # @1k no-Square tier (finetune arms). union_bench naming has varied → try several.
    "M0_ft":       ["ft_none/1k", "ft_none", "ft_none_nosq/1k", "ft_none_nosq", "M0_ft/1k", "M0_ft"],
    "M1_ft":       ["ft_clamp/1k", "ft_clamp", "ft_clamp_nosq/1k", "ft_clamp_nosq", "M1_ft/1k", "M1_ft"],
}
PAIRS = [                                     # (A, B) → Δ = A − B
    ("M1a", "M0"), ("M1a_full", "M0_full"), ("B1", "Rprime"),
    ("M1a_msdglue", "M0"), ("M1a", "M1a_msdglue"), ("M1_ft", "M0_ft"),
]
CENSUS_ARMS = ["M0", "M1a", "M0_full", "Rprime"]
FLIP_PAIRS = [("M0", "M1a"), ("M0_full", "M1a_full")]   # (before, after)
GATE_ARMS = ["Rprime", "M0_full"]


def subset_labels():
    """Exact-composition subset keys, ordered singletons→pairs→triple."""
    keys = []
    for r in (1, 2, 3):
        for combo in combinations(NORMS, r):
            keys.append("+".join(combo))
    return keys


class Arm:
    def __init__(self, name, npz_path, masks, meta, clean_acc):
        self.name = name
        self.path = npz_path
        self.masks = masks
        self.n = len(next(iter(masks.values())))
        self.meta = meta
        self.clean_acc = clean_acc
        self.present = {n: [a for a in ATTACKS_BY_NORM[n] if a in masks] for n in NORMS}
        self.per_norm_robust = {n: np.logical_and.reduce([masks[a] for a in self.present[n]]) for n in NORMS}
        self.per_norm_fail = {n: ~self.per_norm_robust[n] for n in NORMS}
        self.union_robust = np.logical_and.reduce([self.per_norm_robust[n] for n in NORMS])
        self.union_fail = ~self.union_robust
        self.tier = "full" if all(f"square_{n}" in masks for n in NORMS) else "no_square"
        self.subset_sha = (meta or {}).get("audit_subset_sha256")

    def block_count(self):
        return sum(self.per_norm_fail[n].astype(int) for n in NORMS)   # |B_i| per example

    def composition_mask(self, key):
        """Boolean mask: examples whose EXACT blocking set == the norms in `key`."""
        want = set(key.split("+"))
        m = np.ones(self.n, dtype=bool)
        for n in NORMS:
            m &= (self.per_norm_fail[n] if n in want else self.per_norm_robust[n])
        return m


def find_arm(base: Path, name: str) -> Arm | None:
    for rel in ARM_SPECS[name]:
        npz = base / rel / MASK_FILE
        if npz.exists():
            d = np.load(npz, allow_pickle=True)
            masks = {k: np.asarray(d[k]).astype(bool) for k in d.files if k != "metadata_json"}
            meta = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}
            evalj = npz.parent / "eval.json"
            clean = None
            if evalj.exists():
                try:
                    clean = json.load(open(evalj)).get("clean_acc")
                except Exception:
                    clean = None
            return Arm(name, npz, masks, meta, clean)
    return None


def load_all(base: Path):
    arms, missing = {}, []
    wanted = set(ARM_SPECS)
    for name in ARM_SPECS:
        a = find_arm(base, name)
        if a is None:
            missing.append(name)
        else:
            arms[name] = a
    return arms, missing


def aligned(a: Arm, b: Arm, warn):
    if a.n != b.n:
        warn.append(f"{a.name}(n={a.n}) vs {b.name}(n={b.n}): n mismatch — pair SKIPPED (not comparable)")
        return False
    if a.subset_sha and b.subset_sha and a.subset_sha != b.subset_sha:
        warn.append(f"{a.name} vs {b.name}: audit_subset_sha256 differ — examples may not align 1:1")
    if a.tier != b.tier:
        warn.append(f"{a.name}(tier={a.tier}) vs {b.name}(tier={b.tier}): TIER mismatch — Δ compares unlike audits")
    return True


# ----------------------------- tables -----------------------------
def per_norm_delta(arms, warn):
    rows = []
    for an, bn in PAIRS:
        a, b = arms.get(an), arms.get(bn)
        if a is None or b is None:
            rows.append({"pair": f"{an}−{bn}", "status": f"MISSING ({', '.join(x for x,o in [(an,a),(bn,b)] if o is None)})"})
            continue
        if not aligned(a, b, warn):
            rows.append({"pair": f"{an}−{bn}", "status": "n-mismatch (skipped)"})
            continue
        row = {"pair": f"{an}−{bn}", "status": "ok", "tier": a.tier, "n": a.n,
               "d_union": float(a.union_robust.mean() - b.union_robust.mean())}
        for nm in NORMS:
            row[f"d_{nm}"] = float(a.per_norm_robust[nm].mean() - b.per_norm_robust[nm].mean())
        row["d_clean"] = (float(a.clean_acc - b.clean_acc)
                          if (a.clean_acc is not None and b.clean_acc is not None) else None)
        rows.append(row)
    return rows


def blocker_census(arms):
    keys = subset_labels()
    out = {}
    for name in CENSUS_ARMS:
        a = arms.get(name)
        if a is None:
            out[name] = {"status": "MISSING"}
            continue
        fail = a.union_fail
        nfail = int(fail.sum())
        bc = a.block_count()
        size_dist = {str(k): (float((bc[fail] == k).mean()) if nfail else 0.0) for k in (1, 2, 3)}
        comp = {k: (float((a.composition_mask(k) & fail).sum() / nfail) if nfail else 0.0) for k in keys}
        out[name] = {"status": "ok", "tier": a.tier, "n": a.n,
                     "n_fail": nfail, "fail_rate": float(fail.mean()),
                     "size_dist_pct": {k: v * 100 for k, v in size_dist.items()},
                     "composition_pct": {k: v * 100 for k, v in comp.items()}}
    return out


def flip_analysis(arms, warn):
    keys = subset_labels()
    out = {}
    for bn, an in FLIP_PAIRS:                     # before, after
        b, a = arms.get(bn), arms.get(an)
        tag = f"{bn}→{an}"
        if a is None or b is None:
            out[tag] = {"status": f"MISSING ({', '.join(x for x,o in [(bn,b),(an,a)] if o is None)})"}
            continue
        if not aligned(a, b, warn):
            out[tag] = {"status": "n-mismatch (skipped)"}
            continue
        fp = b.union_fail & a.union_robust        # fail(before) → pass(after)
        pf = b.union_robust & a.union_fail        # pass(before) → fail(after)
        # fail→pass: blocking set BEFORE (from b); pass→fail: blocking set AFTER (from a)
        def comp_on(arm, sel):
            s = int(sel.sum())
            return {k: (float((arm.composition_mask(k) & sel).sum() / s) * 100 if s else 0.0) for k in keys}
        out[tag] = {"status": "ok", "tier": a.tier, "n": a.n,
                    "n_fail_to_pass": int(fp.sum()), "n_pass_to_fail": int(pf.sum()),
                    "fail_to_pass_Bbefore_pct": comp_on(b, fp),
                    "pass_to_fail_Bafter_pct": comp_on(a, pf)}
    return out


def gate(arms, threshold):
    out = {}
    for name in GATE_ARMS:
        a = arms.get(name)
        if a is None:
            out[name] = {"status": "MISSING"}
            continue
        fail = a.union_fail
        nfail = int(fail.sum())
        bc = a.block_count()
        single = fail & (bc == 1)
        single_pct = float(single.sum() / nfail) * 100 if nfail else 0.0
        by_norm = {}
        for nm in NORMS:                          # single-blocked BY this norm (only nm fails)
            m = single.copy()
            for other in NORMS:
                m &= (a.per_norm_fail[other] if other == nm else a.per_norm_robust[other])
            by_norm[nm] = float(m.sum() / nfail) * 100 if nfail else 0.0
        verdict = "OPEN" if single_pct >= threshold else "CLOSED"
        band = "borderline(15–20%)" if 15.0 <= single_pct < 20.0 else ""
        out[name] = {"status": "ok", "tier": a.tier, "n_fail": nfail,
                     "single_norm_blocked_pct": single_pct, "by_norm_pct": by_norm,
                     "verdict": verdict, "band": band, "threshold": threshold}
    open_any = any(v.get("verdict") == "OPEN" for v in out.values() if v.get("status") == "ok")
    out["_overall"] = "OPEN" if open_any else ("CLOSED" if any(v.get("status") == "ok" for v in out.values()) else "N/A")
    return out


# ----------------------------- rendering -----------------------------
def fmt_pct(x):
    return "  -  " if x is None else f"{x*100:+6.2f}" if isinstance(x, float) and abs(x) < 1 else f"{x:6.2f}"


def print_report(res):
    P = print
    P("\n" + "=" * 78)
    P("MASK ANALYTICS — per-norm Δ · blocker census · flips · GATE")
    P("=" * 78)
    P(f"base: {res['base_dir']}   date: {res['date']}   arms loaded: {sorted(res['arms_loaded'])}")
    if res["arms_missing"]:
        P(f"arms MISSING (skipped): {res['arms_missing']}")
    for w in res["warnings"]:
        P(f"  ⚠ {w}")

    P("\n--- 1. PER-NORM Δ TABLE (A−B, robust-acc points; +=A better) ---")
    P(f"{'pair':<20}{'tier':<9}{'Δunion':>9}{'Δℓ∞':>9}{'Δℓ₂':>9}{'Δℓ₁':>9}{'Δclean':>9}")
    for r in res["per_norm_delta"]:
        if r.get("status") != "ok":
            P(f"{r['pair']:<20}{r['status']}")
            continue
        dc = "  -  " if r["d_clean"] is None else f"{r['d_clean']*100:+8.2f}"
        P(f"{r['pair']:<20}{r['tier']:<9}{r['d_union']*100:+8.2f} {r['d_linf']*100:+8.2f} "
          f"{r['d_l2']*100:+8.2f} {r['d_l1']*100:+8.2f} {dc:>8}")

    P("\n--- 2. BLOCKER CENSUS (on union-failing set; % of failing examples) ---")
    for name, c in res["blocker_census"].items():
        if c.get("status") != "ok":
            P(f"{name}: {c['status']}"); continue
        P(f"\n{name} [tier={c['tier']}]  n={c['n']}  fail={c['n_fail']} ({c['fail_rate']*100:.1f}%)")
        sd = c["size_dist_pct"]
        P(f"  |B| distribution:  1-norm {sd['1']:5.1f}%   2-norm {sd['2']:5.1f}%   3-norm {sd['3']:5.1f}%")
        comp = c["composition_pct"]
        P("  exact blocker composition:")
        for k in subset_labels():
            lab = "+".join(NORM_LABEL[x] for x in k.split("+"))
            P(f"    {lab:<12} {comp[k]:6.2f}%")

    P("\n--- 3. FLIP ANALYSIS ---")
    for tag, f in res["flip_analysis"].items():
        if f.get("status") != "ok":
            P(f"{tag}: {f['status']}"); continue
        P(f"\n{tag} [tier={f['tier']}]  fail→pass={f['n_fail_to_pass']}  pass→fail={f['n_pass_to_fail']}")
        P("  fail→pass: blocking set BEFORE the flip (want: mostly ℓ₁-only):")
        for k in subset_labels():
            v = f["fail_to_pass_Bbefore_pct"][k]
            if v > 0:
                P(f"    {'+'.join(NORM_LABEL[x] for x in k.split('+')):<12} {v:6.2f}%")
        P("  pass→fail: blocking norm AFTER the flip (want: mostly ℓ∞):")
        for k in subset_labels():
            v = f["pass_to_fail_Bafter_pct"][k]
            if v > 0:
                P(f"    {'+'.join(NORM_LABEL[x] for x in k.split('+')):<12} {v:6.2f}%")

    P("\n--- 4. GATE METRIC (% single-norm-blocked on failing set) ---")
    g = res["gate"]
    for name in GATE_ARMS:
        c = g.get(name, {})
        if c.get("status") != "ok":
            P(f"{name}: {c.get('status','MISSING')}"); continue
        bn = c["by_norm_pct"]
        P(f"  {name} [tier={c['tier']}] fail={c['n_fail']}: single-norm-blocked = {c['single_norm_blocked_pct']:.2f}% "
          f"[ℓ∞ {bn['linf']:.1f}% · ℓ₂ {bn['l2']:.1f}% · ℓ₁ {bn['l1']:.1f}%]  → {c['verdict']} {c['band']}")
    P(f"\n  >>> GATE {g['_overall']}  (threshold {res['gate_threshold']}% single-norm-blocked; "
      f"OPEN on any baseline ⇒ per-sample binding-norm glue is motivated)")
    P("=" * 78 + "\n")


def write_csvs(res, out_dir: Path, date: str):
    paths = []
    # per-norm delta
    p = out_dir / f"mask_analytics_{date}_per_norm_delta.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "status", "tier", "n", "d_union", "d_linf", "d_l2", "d_l1", "d_clean"])
        for r in res["per_norm_delta"]:
            if r.get("status") == "ok":
                w.writerow([r["pair"], r["status"], r["tier"], r["n"], r["d_union"],
                            r["d_linf"], r["d_l2"], r["d_l1"], r["d_clean"]])
            else:
                w.writerow([r["pair"], r["status"]])
    paths.append(p)
    # blocker census
    p = out_dir / f"mask_analytics_{date}_blocker_census.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "tier", "n", "n_fail", "fail_rate", "pct_1norm", "pct_2norm", "pct_3norm"]
                   + [f"comp_{k}" for k in subset_labels()])
        for name, c in res["blocker_census"].items():
            if c.get("status") != "ok":
                w.writerow([name, c["status"]]); continue
            sd = c["size_dist_pct"]
            w.writerow([name, c["tier"], c["n"], c["n_fail"], c["fail_rate"],
                        sd["1"], sd["2"], sd["3"]] + [c["composition_pct"][k] for k in subset_labels()])
    paths.append(p)
    # gate
    p = out_dir / f"mask_analytics_{date}_gate.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "status", "tier", "n_fail", "single_norm_blocked_pct",
                    "pct_linf", "pct_l2", "pct_l1", "verdict"])
        for name in GATE_ARMS:
            c = res["gate"].get(name, {})
            if c.get("status") != "ok":
                w.writerow([name, c.get("status", "MISSING")]); continue
            bn = c["by_norm_pct"]
            w.writerow([name, "ok", c["tier"], c["n_fail"], c["single_norm_blocked_pct"],
                        bn["linf"], bn["l2"], bn["l1"], c["verdict"]])
    paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="results/eval/union_bench",
                    help="dir holding <arm>/[tier/]masks_multinorm_v1.npz (point at Drive union_bench on Colab)")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--gate-threshold", type=float, default=15.0,
                    help="single-norm-blocked %% for GATE OPEN (15–20 band); default 15")
    a = ap.parse_args()

    base = Path(a.base_dir)
    if not base.exists():
        sys.exit(f"base-dir not found: {base}")
    arms, missing = load_all(base)
    warn = []
    res = {
        "date": a.date, "base_dir": str(base), "gate_threshold": a.gate_threshold,
        "mask_convention": "true_means_robust_false_means_failed",
        "arms_loaded": {n: {"path": str(x.path), "tier": x.tier, "n": x.n,
                            "clean_acc": x.clean_acc, "union_robust": float(x.union_robust.mean())}
                        for n, x in arms.items()},
        "arms_missing": missing, "warnings": warn,
        "per_norm_delta": per_norm_delta(arms, warn),
        "blocker_census": blocker_census(arms),
        "flip_analysis": flip_analysis(arms, warn),
        "gate": gate(arms, a.gate_threshold),
    }
    # (warn is mutated in-place by the table builders above)
    res["warnings"] = warn
    print_report(res)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"mask_analytics_{a.date}.json"
    jpath.write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    csvs = write_csvs(res, out_dir, a.date)
    print(f"saved: {jpath}")
    for c in csvs:
        print(f"saved: {c}")


if __name__ == "__main__":
    main()

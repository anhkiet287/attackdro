"""Table 2 — budget / staleness decomposition, recomputed from the frozen @10k masks.

Reuses scripts/dev/make_table1.py wholesale: the same arm loader, the same union / per-norm
block definitions, the same paired bootstrap estimator (itself reused verbatim from
scripts/dev/pernorm_ci.py:boot == scripts/dev/c5_attribution.py:paired), the same two labelled
CI conventions (two_sided_95 = [q025, q975]; lcb_95 = q05), the same significance rule, and the
same LaTeX formatting. Nothing about the estimator or the conventions is redefined here.

Contrasts are A - B, each a separate row (no pooling across seeds). Contrast order is the
decomposition order given in the task and is preserved in every output — rows are NOT reordered
by effect size, unlike table1.tex.

CLEAN CAVEAT (identical to Table 1): the frozen mask sets hold only the 12 attack masks, so
clean is an UNPAIRED scalar difference of eval.json clean_acc with no bootstrap CI and verdict
"n/a", never "ns". No CI is synthesised for clean.

Read-only: no mask, eval.json, checkpoint, or training artifact is written or regenerated, and
no attack / training / evaluation is run.

  python scripts/dev/make_table2.py [--b 10000]
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
_s = importlib.util.spec_from_file_location("t1", os.path.join(ROOT, "scripts/dev/make_table1.py"))
t1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t1)

OUT = os.path.join(ROOT, "results/analysis/table2")
SCALE, TIER, N_ATTACKS, N_EXAMPLES = t1.SCALE, t1.TIER, t1.N_ATTACKS, t1.N_EXAMPLES
NORMS, CELLS, SEED = t1.NORMS, t1.CELLS, t1.SEED

# (contrast label, arm_A, arm_B, seed label)   delta = A - B
CONTRASTS = [
    ("views-10 $-$ MSD-50",   "M1a_full",       "M0_full",       "seed0"),
    ("views-10 $-$ MSD-50",   "M1a_full_seed1", "M0_full_seed1", "seed1"),
    ("views-50 $-$ MSD-50",   "M1a_full_v50",   "M0_full",       "seed0"),
    ("views-50 $-$ views-10", "M1a_full_v50",   "M1a_full",      "seed0"),
    ("MSD-10 gain (ref)",     "M1a",            "M0",            "seed0"),
]
PLAIN = {"views-10 $-$ MSD-50": "views-10 - MSD-50", "views-50 $-$ MSD-50": "views-50 - MSD-50",
         "views-50 $-$ views-10": "views-50 - views-10", "MSD-10 gain (ref)": "MSD-10 gain (ref)"}
# the two boundary cells the paper's wording turns on — printed unrounded
BOUNDARY = [(3, "union", "contrast 4  views-50 - views-10, UNION"),
            (2, "linf",  "contrast 3  views-50 - MSD-50, LINF")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=t1.B)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- load arms; stop rather than substitute a scale/tier ----
    fail, arms = [], {}
    for _, A, Bm, _ in CONTRASTS:
        for arm in (A, Bm):
            if arm not in arms:
                arms[arm] = t1.load_arm(arm, fail)
    if fail:
        print("STOP — missing @10k mask set(s); no @1k and no tier=no_square fallback is used:")
        for f in fail:
            print("  " + f)
        raise SystemExit(1)

    print("=" * 118)
    print(f"SANITY  scale={SCALE} tier={TIER} ({N_ATTACKS} attacks)   arms={len(arms)}")
    print("=" * 118)
    xcheck = []
    print(f"  {'arm':<16}{'keys':>5}{'n':>7}  {'subset_sha':<14}{'union(masks)':>13}{'union(eval)':>12}"
          f"{'clean(eval)':>12}  checkpoint_sha256")
    for arm, A in sorted(arms.items()):
        m = A["masks"]
        assert len(m) == N_ATTACKS, f"{arm}: {len(m)} mask keys, expected {N_ATTACKS}"
        n = len(next(iter(m.values())))
        assert n == N_EXAMPLES, f"{arm}: n={n}, expected {N_EXAMPLES}"
        u = t1.union(m)
        blocks = {}
        for nm in NORMS:
            bl, ks = t1.block(m, nm)
            assert len(ks) == 4, f"{arm}/{nm}: {len(ks)} attacks in block, expected 4"
            assert not (u & ~bl).any(), f"{arm}: union mask not a subset of the {nm} block"
            assert u.mean() <= bl.mean() + 1e-12, f"{arm}: union acc > {nm} block acc"
            blocks[nm] = bl
        A["u"], A["blocks"] = u, blocks
        d = abs(float(u.mean()) - A["union_eval"])
        if d > 5e-5:
            xcheck.append(f"{arm}: union(masks)={u.mean():.6f} vs eval.json={A['union_eval']:.6f} (|d|={d:.2e})")
        print(f"  {arm:<16}{len(m):>5}{n:>7}  {A['meta'].get('audit_subset_sha256','')[:12]:<14}"
              f"{u.mean():>13.4f}{A['union_eval']:>12.4f}{A['clean']:>12.4f}  "
              f"{A['meta'].get('checkpoint_sha256','')[:16]}")
    print(f"  [ok] every arm: {N_ATTACKS} mask keys, n={N_EXAMPLES}, 4 attacks per norm block,")
    print(f"       union mask ⊆ every per-norm block mask (so union acc ≤ every per-norm acc)")

    v50 = arms["M1a_full_v50"]
    print(f"\n  [ok] M1a_full_v50 HAS a 12-key @10k mask set: {v50['path']}")
    print(f"       n={N_EXAMPLES}  checkpoint_sha256 = {v50['meta'].get('checkpoint_sha256')}")
    print(f"       checkpoint_path  = {v50['meta'].get('checkpoint_path')}")

    subs = {A["meta"].get("audit_subset_sha256") for A in arms.values()}
    cfgs = {A["meta"].get("config_path") for A in arms.values()}
    assert len(subs) == 1, f"arms do not share one audit subset: {subs}"
    assert len(cfgs) == 1, f"arms do not share one audit config: {cfgs}"
    cfg_path = cfgs.pop()
    cfg_abs = os.path.join(ROOT, cfg_path)
    cfg_sha = hashlib.sha256(open(cfg_abs, "rb").read()).hexdigest() if os.path.exists(cfg_abs) else None
    print(f"  [ok] all {len(arms)} arms share audit subset sha {list(subs)[0][:16]} and config {cfg_path}")
    print(f"       -> identical example ordering across every contrast; config sha {str(cfg_sha)[:16]}")
    if xcheck:
        print("\n  [CROSS-CHECK FAILURES] union(masks) != eval.json union — reported, neither preferred:")
        for x in xcheck:
            print("    " + x)
    else:
        print(f"  [ok] union recomputed from masks == eval.json union for all {len(arms)} arms (tol 5e-5)")

    # ---- contrasts ----
    rows = []
    for lbl, An, Bn, sd in CONTRASTS:
        A, Bo = arms[An], arms[Bn]
        assert A["meta"].get("audit_subset_sha256") == Bo["meta"].get("audit_subset_sha256"), \
            f"{An}/{Bn}: different audit subsets, examples are not paired"
        r = {"contrast": PLAIN[lbl], "contrast_tex": lbl, "seed": sd, "arm_a": An, "arm_b": Bn,
             "n": N_EXAMPLES, "same_example_ordering": True,
             "acc_a": {}, "acc_b": {}, "delta": {}, "significance": {}}
        for c in CELLS:
            va = A["u"] if c == "union" else A["blocks"][c]
            vb = Bo["u"] if c == "union" else Bo["blocks"][c]
            r["acc_a"][c] = float(va.mean())
            r["acc_b"][c] = float(vb.mean())
            e = t1.boot(va, vb, a.b)
            r["delta"][c] = e
            # single-contrast row: the Table 1 rule reduces to this one interval
            r["significance"][c] = t1.verdict([tuple(e["two_sided_95"])], [e["delta_plugin"]])
        r["acc_a"]["clean"] = A["clean"]
        r["acc_b"]["clean"] = Bo["clean"]
        r["delta"]["clean"] = {"delta_plugin": A["clean"] - Bo["clean"], "delta_boot_mean": None,
                               "two_sided_95": None, "lcb_95": None,
                               "note": "UNPAIRED scalar difference of eval.json clean_acc; the frozen "
                                       "mask set has no clean per-example mask, so no paired bootstrap "
                                       "is possible and no CI is synthesised"}
        r["significance"]["clean"] = "n/a"
        rows.append(r)
        print(f"  bootstrapped {PLAIN[lbl]:<21} {sd:<6} {An} - {Bn}")

    # ---- boundary cells, printed unrounded ----
    print("\n" + "=" * 118)
    print("BOUNDARY CELLS — exact two-sided 95% endpoints, unrounded")
    print("=" * 118)
    bnd = {}
    for i, c, desc in BOUNDARY:
        r = rows[i]
        e = r["delta"][c]
        lo, hi = e["two_sided_95"]
        ex = not (lo <= 0.0 <= hi)
        bnd[f"{r['contrast']}|{c}"] = {"contrast": r["contrast"], "cell": c,
                                       "arm_a": r["arm_a"], "arm_b": r["arm_b"],
                                       "delta_plugin": e["delta_plugin"],
                                       "two_sided_95": [lo, hi], "excludes_zero": ex,
                                       "lcb_95": e["lcb_95"], "verdict": r["significance"][c]}
        print(f"  {desc}   [{r['arm_a']} - {r['arm_b']}]")
        print(f"    delta          = {e['delta_plugin']:+.4f}  (fraction)   {100*e['delta_plugin']:+.4f} pp")
        print(f"    two_sided_95   = [{lo:+.4f}, {hi:+.4f}]  (fraction, 4 dp)")
        print(f"                   = [{100*lo:+.4f}, {100*hi:+.4f}] pp")
        print(f"    full precision = [{lo!r}, {hi!r}]")
        print(f"    excludes zero  = {'YES' if ex else 'NO'}        verdict = {r['significance'][c]}")
        print(f"    lcb_95 (q05)   = {e['lcb_95']:+.4f}  (fraction)   {100*e['lcb_95']:+.4f} pp")

    # ---- console table ----
    print("\n" + "=" * 118)
    print(f"TABLE 2 — budget / staleness decomposition. @{SCALE}, tier={TIER} ({N_ATTACKS} attacks).")
    print(f"          Accuracies in %, deltas in pp, delta = A - B. Order = decomposition order, not effect size.")
    print("=" * 118)
    print(f"{'contrast':<22}{'seed':<7}{'UnionB':>8}{'UnionA':>8}{'dUnion':>9}{'dLinf':>9}{'dL2':>9}"
          f"{'dL1':>9}{'dClean':>9}")
    for r in rows:
        def cell(c):
            s = r["significance"][c]
            return t1.pp(r["delta"][c]["delta_plugin"]) + ("*" if s == "sig" else "" if s == "n/a" else "n")
        print(f"{r['contrast']:<22}{r['seed']:<7}{t1.pct(r['acc_b']['union']):>8}"
              f"{t1.pct(r['acc_a']['union']):>8}{cell('union'):>9}{cell('linf'):>9}{cell('l2'):>9}"
              f"{cell('l1'):>9}{cell('clean'):>9}")
    print("  * = two-sided 95% CI excludes 0 (Table 1 rule, single-seed form);  n = ns;")
    print("      dClean carries no marker — UNPAIRED scalar difference, no CI, verdict n/a.")

    print(f"\n  lcb_95 (q05) per contrast, pp:")
    for r in rows:
        print(f"    {r['contrast']:<22}{r['seed']:<7}" + "  ".join(
            f"{c}: {100 * r['delta'][c]['lcb_95']:+.2f}" for c in CELLS))

    # ---- json ----
    payload = {
        "what": "Paper Table 2 — budget/staleness decomposition, recomputed from the frozen @10k "
                "mask sets under results/main/.",
        "config": {
            "scale": SCALE, "tier": TIER, "n_attacks": N_ATTACKS, "n_examples": N_EXAMPLES,
            "delta": "A - B", "row_order": "decomposition order as specified; NOT sorted by effect size",
            "estimator": "reused from scripts/dev/make_table1.py:boot, itself verbatim from "
                         "scripts/dev/pernorm_ci.py:boot (== scripts/dev/c5_attribution.py:paired): "
                         "paired sample-level bootstrap, one resampled index vector indexes BOTH arms",
            "bootstrap": f"B={a.b}, fresh np.random.default_rng({SEED}) per call, percentile",
            "ci_conventions": {"two_sided_95": "[q025, q975] — used by the significance rule",
                               "lcb_95": "q05 — 1-sided lower confidence bound, reported alongside"},
            "significance_rule": "same as Table 1; every contrast here is a single row, so the rule "
                                 "reduces to: two_sided_95 excludes zero",
            "union": "AND over all 12 attack masks",
            "per_norm_block": "AND over that norm's 4 attack masks",
            "clean": "UNPAIRED scalar difference of eval.json clean_acc — no clean per-example mask "
                     "exists in the frozen artifact, so clean has NO CI and its verdict is 'n/a'",
            "audit_config": cfg_path, "audit_config_sha256": cfg_sha,
        },
        "m1a_full_v50_availability": {
            "mask_set_present": True, "mask_keys": N_ATTACKS, "n_examples": N_EXAMPLES,
            "mask_file": v50["path"], "checkpoint_path": v50["meta"].get("checkpoint_path"),
            "checkpoint_role": v50["meta"].get("checkpoint_role"),
            "checkpoint_sha256": v50["meta"].get("checkpoint_sha256"),
            "fallback_used": None,
        },
        "sanity": {"mask_keys_per_arm": N_ATTACKS, "n_per_arm": N_EXAMPLES,
                   "attacks_per_norm_block": 4,
                   "one_audit_subset_across_all_arms": list(subs)[0] if subs else None,
                   "union_subset_of_every_per_norm_block": True},
        "cross_check_union_masks_vs_eval_json": {"tolerance": 5e-5, "failures": xcheck,
                                                 "all_match": not xcheck},
        "boundary_cells": bnd,
        "rows": rows,
        "provenance": {arm: {"mask_file": A["path"],
                             "checkpoint_path": A["meta"].get("checkpoint_path"),
                             "checkpoint_role": A["meta"].get("checkpoint_role"),
                             "checkpoint_sha256": A["meta"].get("checkpoint_sha256"),
                             "audit_subset_path": A["meta"].get("audit_subset_path"),
                             "audit_subset_sha256": A["meta"].get("audit_subset_sha256"),
                             "n_examples": A["meta"].get("n_examples"),
                             "mask_convention": A["meta"].get("mask_convention")}
                       for arm, A in sorted(arms.items())},
    }
    json.dump(payload, open(os.path.join(OUT, "table2.json"), "w"), indent=2)

    # ---- table2.tex : same style as table1.tex, \multirow on the contrast column ----
    L, i = [], 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]["contrast_tex"] == rows[i]["contrast_tex"]:
            j += 1
        grp = rows[i:j + 1]
        for k, r in enumerate(grp):
            first = (f"\\multirow{{{len(grp)}}}{{*}}{{{r['contrast_tex']}}}" if k == 0 else "")
            d = " & ".join(t1.tex_delta(r["delta"][c]["delta_plugin"], r["significance"][c]) for c in CELLS)
            L.append(f"{first} & {r['seed']} & {t1.pct(r['acc_b']['union'])} & "
                     f"{t1.pct(r['acc_a']['union'])} & {d} & {t1.pp(r['delta']['clean']['delta_plugin'])} \\\\")
        if j + 1 < len(rows):
            L.append("\\midrule")
        i = j + 1
    hdr = [
        "% Table 2 body — generated by scripts/dev/make_table2.py from the frozen @10k masks.",
        "% Drop into a table* environment with:",
        "%   \\begin{tabular}{llrrrrrrr}  and header:",
        "%   Contrast & Seed & Union B & Union A & $\\Delta$Union & $\\Delta\\ell_\\infty$ &",
        "%   $\\Delta\\ell_2$ & $\\Delta\\ell_1$ & $\\Delta$Clean \\\\",
        "% Requires \\usepackage{multirow}. Accuracies in %, deltas in percentage points, "
        "$\\Delta$ = A $-$ B.",
        f"% Same estimator and conventions as Table 1 (paired bootstrap, B={a.b}, rng({SEED})).",
        "% Bold = two-sided 95% CI excludes 0; ns superscript = not significant.",
        "% Row order is the decomposition order, NOT sorted by effect size.",
        "% Delta-Clean is an UNPAIRED scalar difference with NO CI (the frozen mask sets contain no",
        "% clean per-example mask), so it is never bolded and carries no ns marker.",
    ]
    tex = "\n".join(hdr + L) + "\n"
    open(os.path.join(OUT, "table2.tex"), "w").write(tex)

    print(f"\n  saved -> results/analysis/table2/table2.json")
    print(f"           results/analysis/table2/table2.tex")


if __name__ == "__main__":
    main()

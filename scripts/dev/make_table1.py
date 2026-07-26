"""Table 1 — recompute the paper's main results table from the frozen @10k masks.

Every number in the table is recomputed from results/main/<arm>/10k/masks_multinorm_v1.npz, so
the table is reproducible from the published artifact alone. eval.json is read ONLY for the
scalar clean accuracy and as a cross-check on union.

Estimator: reused unchanged from scripts/dev/pernorm_ci.py:boot (identical to
scripts/dev/c5_attribution.py:paired) — paired sample-level bootstrap, B=10000, a fresh
np.random.default_rng(0) per call, one resampled index vector indexing BOTH arms. The two
existing scripts differ only in which quantiles they read off that distribution
(c5_attribution: q05 as a 1-sided LCB; pernorm_ci: [q05, q95], a central 90% interval). This
script emits BOTH conventions explicitly labelled — two_sided_95 = [q025, q975] and
lcb_95 = q05 — because the repo currently mixes them and the paper must not.

CLEAN CAVEAT: the frozen mask sets contain only the 12 attack masks. clean_mask() is computed
during the audit (eval_multinorm_audit.py:1086) but reduced to the scalar clean_acc (line 957)
and never written to the npz (savez payload, line 857). Clean deltas are therefore an UNPAIRED
difference of scalar accuracies with NO bootstrap CI and NO significance verdict ("n/a", not
"ns"). No CI is synthesised for clean.

Read-only: no mask, eval.json, checkpoint, or training artifact is written or regenerated, and
no attack / training / evaluation is run.

  python scripts/dev/make_table1.py [--b 10000] [--check-pernorm]
"""
from __future__ import annotations
import argparse, hashlib, json, os
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
MAIN = os.path.join(ROOT, "results/main")
OUT = os.path.join(ROOT, "results/analysis/table1")
SCALE, TIER, N_ATTACKS, N_EXAMPLES = "10k", "full", 12, 10000
NORMS = ("linf", "l2", "l1")
CELLS = ("union",) + NORMS                       # cells with paired per-example masks
B, SEED = 10000, 0

# (budget label, base label, [(arm_ON, arm_OFF, seed label), ...])   delta = ON - OFF
ROWS = [
    ("80 ep, from scratch", "MSD-10", [("M1a", "M0", "seed0"),
                                       ("M1a_seed2", "M0_seed2", "seed2"),
                                       ("M1a_seed3", "M0_seed3", "seed3")]),
    ("80 ep, from scratch", "AVG",    [("M1a_avg", "M0_avg", "seed0")]),
    ("80 ep, from scratch", "RAMP",   [("B1", "Rprime", "seed0")]),
    ("80 ep, from scratch", "MAX",    [("M1a_max", "M0_max", "seed0")]),
    ("50 ep, full budget",  "MSD-50", [("M1a_full", "M0_full", "seed0"),
                                       ("M1a_full_seed1", "M0_full_seed1", "seed1")]),
    ("fine-tune",           "FT",     [("ft_clamp", "ft_none", "n/a")]),
]
BUDGETS = ["80 ep, from scratch", "50 ep, full budget", "fine-tune"]


# ----------------------------------------------------------------------------- artifact I/O
def mask_path(arm):
    return os.path.join(MAIN, arm, SCALE, "masks_multinorm_v1.npz")


def load_arm(arm, fail):
    """12 attack masks + metadata + scalar clean_acc. Never falls back to another scale/tier."""
    p = mask_path(arm)
    if not os.path.exists(p):
        fail.append(f"{arm}: no mask set at {os.path.relpath(p, ROOT)}")
        return None
    d = np.load(p, allow_pickle=True)
    m = {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}
    md = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}
    e = json.load(open(os.path.join(MAIN, arm, SCALE, "eval.json")))
    return {"arm": arm, "masks": m, "meta": md, "path": os.path.relpath(p, ROOT),
            "clean": float(e["clean_acc"]), "union_eval": float(e["full_audit_union"])}


def block(m, nm):
    """Per-norm block = AND over that norm's attacks in the tier (4 at tier=full)."""
    ks = [k for k in m if k.endswith(nm)]
    u = np.ones(len(m[ks[0]]), bool)
    for k in ks:
        u &= m[k]
    return u, sorted(ks)


def union(m):
    ks = sorted(m)
    u = np.ones(len(m[ks[0]]), bool)
    for k in ks:
        u &= m[k]
    return u


# ----------------------------------------------------------------------------- estimator
def boot(a, b, nboot=B):
    """VERBATIM the estimator of pernorm_ci.py:boot / c5_attribution.py:paired.

    Paired: one resampled index vector indexes BOTH arms. Fresh default_rng(SEED) per call, so
    the (q05, q95) pair reproduces pernorm_ci.json exactly. Returns every quantile the two
    conventions need, read off the SAME bootstrap distribution.
    """
    assert len(a) == len(b)
    rng = np.random.default_rng(SEED)
    n = len(a)
    D = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        D[i] = a[idx].mean() - b[idx].mean()
    return {"delta_plugin": float(a.mean() - b.mean()),
            "delta_boot_mean": float(D.mean()),
            "two_sided_95": [float(np.quantile(D, .025)), float(np.quantile(D, .975))],
            "lcb_95": float(np.quantile(D, .05)),
            "q95": float(np.quantile(D, .95))}


def verdict(seed_cis, seed_deltas):
    """PHASE 2 rule, applied exactly: SIGNIFICANT iff every seed's two_sided_95 excludes zero
    AND all seeds share the same sign. Single-seed rows reduce to the single interval."""
    excl = all(not (lo <= 0.0 <= hi) for lo, hi in seed_cis)
    signs = {int(np.sign(d)) for d in seed_deltas}
    return "sig" if (excl and len(signs) == 1 and 0 not in signs) else "ns"


# ----------------------------------------------------------------------------- formatting
def pct(x):
    return f"{100 * x:.2f}"


def pp(x):
    return f"{100 * x:+.2f}"


TEX_NORM = {"linf": r"\ell_\infty", "l2": r"\ell_2", "l1": r"\ell_1"}


def tex_esc(s):
    return s.replace("_", r"\_")


def tex_delta(v, sig):
    s = pp(v)
    return f"\\textbf{{{s}}}" if sig == "sig" else f"{s}\\textsuperscript{{ns}}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=B)
    ap.add_argument("--check-pernorm", action="store_true", default=True,
                    help="assert the [q05,q95] reproduce results/analysis/pernorm_ci/pernorm_ci.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- load every arm, stop if any mask set is missing (no substitution, no fallback) ----
    fail = []
    arms = {}
    for _, _, pairs in ROWS:
        for on, off, _ in pairs:
            for arm in (on, off):
                if arm not in arms:
                    arms[arm] = load_arm(arm, fail)
    if fail:
        print("STOP — missing mask sets, no substitute arm and no @1k / no_square fallback is used:")
        for f in fail:
            print("  " + f)
        raise SystemExit(1)

    # ---- sanity assertions (must hold, and are printed) ----
    print("=" * 112)
    print(f"SANITY  scale={SCALE} tier={TIER} ({N_ATTACKS} attacks)   arms={len(arms)}")
    print("=" * 112)
    checks, xcheck = [], []
    for arm, A in sorted(arms.items()):
        m = A["masks"]
        assert len(m) == N_ATTACKS, f"{arm}: {len(m)} mask keys, expected {N_ATTACKS}"
        n = len(next(iter(m.values())))
        assert n == N_EXAMPLES, f"{arm}: n={n}, expected {N_EXAMPLES}"
        u = union(m)
        blocks = {}
        for nm in NORMS:
            bl, ks = block(m, nm)
            assert len(ks) == 4, f"{arm}/{nm}: {len(ks)} attacks in block, expected 4"
            # union <= per-norm block, as masks (implication) and therefore as accuracies
            assert not (u & ~bl).any(), f"{arm}: union mask not a subset of the {nm} block"
            assert u.mean() <= bl.mean() + 1e-12, f"{arm}: union acc > {nm} block acc"
            blocks[nm] = bl
        A["u"], A["blocks"] = u, blocks
        # cross-check: union recomputed from masks vs the union field in eval.json
        d = abs(float(u.mean()) - A["union_eval"])
        if d > 5e-5:
            xcheck.append(f"{arm}: union(masks)={u.mean():.6f} vs eval.json={A['union_eval']:.6f} (|d|={d:.2e})")
        checks.append((arm, len(m), n, A["meta"].get("audit_subset_sha256", "")[:12],
                       float(u.mean()), A["union_eval"], A["clean"]))
    print(f"  {'arm':<16}{'keys':>5}{'n':>7}  {'subset_sha':<14}{'union(masks)':>13}{'union(eval)':>12}{'clean(eval)':>12}")
    for arm, k, n, sha, um, ue, cl in checks:
        print(f"  {arm:<16}{k:>5}{n:>7}  {sha:<14}{um:>13.4f}{ue:>12.4f}{cl:>12.4f}")
    print(f"  [ok] every arm: {N_ATTACKS} mask keys, n={N_EXAMPLES}, 4 attacks per norm block,")
    print(f"       union mask ⊆ every per-norm block mask (so union acc ≤ every per-norm acc)")

    subs = {A["meta"].get("audit_subset_sha256") for A in arms.values()}
    cfgs = {A["meta"].get("config_path") for A in arms.values()}
    assert len(subs) == 1, f"arms do not share one audit subset: {subs}"
    assert len(cfgs) == 1, f"arms do not share one audit config: {cfgs}"
    cfg_path = cfgs.pop()
    cfg_abs = os.path.join(ROOT, cfg_path)
    cfg_sha = hashlib.sha256(open(cfg_abs, "rb").read()).hexdigest() if os.path.exists(cfg_abs) else None
    print(f"  [ok] all {len(arms)} arms share audit subset sha {list(subs)[0][:16]} and config {cfg_path}")
    print(f"       -> identical example ordering across every pair; config sha {str(cfg_sha)[:16]}")

    if xcheck:
        print("\n  [CROSS-CHECK FAILURES] union(masks) != eval.json union — reported, neither preferred:")
        for x in xcheck:
            print("    " + x)
    else:
        print(f"  [ok] union recomputed from masks == eval.json union for all {len(arms)} arms (tol 5e-5)")

    # ---- per pair: accuracies, paired deltas, both CI conventions ----
    table = []
    for budget, base, pairs in ROWS:
        seeds = []
        for on, off, sd in pairs:
            A, Bo = arms[on], arms[off]
            same_order = A["meta"].get("audit_subset_sha256") == Bo["meta"].get("audit_subset_sha256")
            assert same_order, f"{on}/{off}: different audit subsets, examples are not paired"
            rec = {"seed": sd, "arm_on": on, "arm_off": off, "n": N_EXAMPLES,
                   "same_example_ordering": True, "acc_on": {}, "acc_off": {}, "delta": {}}
            for c in CELLS:
                va = A["u"] if c == "union" else A["blocks"][c]
                vb = Bo["u"] if c == "union" else Bo["blocks"][c]
                rec["acc_on"][c] = float(va.mean())
                rec["acc_off"][c] = float(vb.mean())
                rec["delta"][c] = boot(va, vb, a.b)
            # clean: scalar only — no per-example clean mask exists in the frozen artifact
            rec["acc_on"]["clean"] = A["clean"]
            rec["acc_off"]["clean"] = Bo["clean"]
            rec["delta"]["clean"] = {"delta_plugin": A["clean"] - Bo["clean"],
                                     "delta_boot_mean": None, "two_sided_95": None, "lcb_95": None,
                                     "note": "UNPAIRED scalar difference of eval.json clean_acc; the "
                                             "frozen mask set has no clean per-example mask, so no "
                                             "paired bootstrap is possible and no CI is synthesised"}
            seeds.append(rec)
            print(f"  bootstrapped {base:<7} {sd:<6} {on} - {off}")

        row = {"budget": budget, "base": base, "n_seeds": len(seeds), "seeds": seeds,
               "mean_acc_off": {}, "mean_acc_on": {}, "mean_delta": {},
               "significance": {}, "seed_min_lcb_95": {}}
        for c in CELLS + ("clean",):
            row["mean_acc_off"][c] = float(np.mean([s["acc_off"][c] for s in seeds]))
            row["mean_acc_on"][c] = float(np.mean([s["acc_on"][c] for s in seeds]))
            ds = [s["delta"][c]["delta_plugin"] for s in seeds]
            row["mean_delta"][c] = float(np.mean(ds))
            if c == "clean":
                row["significance"][c] = "n/a"
                row["seed_min_lcb_95"][c] = None
            else:
                cis = [tuple(s["delta"][c]["two_sided_95"]) for s in seeds]
                row["significance"][c] = verdict(cis, ds)
                row["seed_min_lcb_95"][c] = float(min(s["delta"][c]["lcb_95"] for s in seeds))
        table.append(row)

    # ---- optional reconciliation with the existing pernorm_ci.json (same estimator) ----
    recon = None
    pc = os.path.join(ROOT, "results/analysis/pernorm_ci/pernorm_ci.json")
    if a.check_pernorm and os.path.exists(pc):
        ref = json.load(open(pc))
        keymap = {("MSD-10", "seed0"): "RQ1/MSD-10 seed0", ("MSD-10", "seed2"): "MSD-10 seed2",
                  ("MSD-10", "seed3"): "MSD-10 seed3", ("AVG", "seed0"): "AVG",
                  ("MAX", "seed0"): "MAX", ("MSD-50", "seed0"): "MSD-50 (views-10) s0",
                  ("MSD-50", "seed1"): "MSD-50 (views-10) s1", ("RAMP", "seed0"): "RAMP (B1-Rprime)",
                  ("FT", "n/a"): "FT-linf"}
        bad = []
        for r in table:
            for s in r["seeds"]:
                k = keymap.get((r["base"], s["seed"]))
                if k is None or k not in ref:
                    continue
                for nm in NORMS:
                    got = (s["delta"][nm]["lcb_95"], s["delta"][nm]["q95"])
                    exp = (ref[k][nm]["lo"], ref[k][nm]["hi"])
                    if max(abs(got[0] - exp[0]), abs(got[1] - exp[1])) > 1e-12:
                        bad.append(f"{k}/{nm}: got {got} vs pernorm_ci.json {exp}")
        recon = {"reference": os.path.relpath(pc, ROOT), "convention": "[q05, q95]",
                 "matches": not bad, "mismatches": bad}
        print(f"\n  [{'ok' if not bad else 'MISMATCH'}] estimator reuse: [q05,q95] "
              f"{'reproduces' if not bad else 'DIFFERS from'} pernorm_ci.json"
              + ("" if not bad else "\n    " + "\n    ".join(bad)))

    # ---- provenance ----
    prov = {arm: {"mask_file": A["path"],
                  "checkpoint_path": A["meta"].get("checkpoint_path"),
                  "checkpoint_role": A["meta"].get("checkpoint_role"),
                  "checkpoint_sha256": A["meta"].get("checkpoint_sha256"),
                  "audit_subset_path": A["meta"].get("audit_subset_path"),
                  "audit_subset_sha256": A["meta"].get("audit_subset_sha256"),
                  "n_examples": A["meta"].get("n_examples"),
                  "mask_convention": A["meta"].get("mask_convention")}
            for arm, A in sorted(arms.items())}

    payload = {
        "what": "Paper Table 1, recomputed from the frozen @10k mask sets under results/main/.",
        "config": {
            "scale": SCALE, "tier": TIER, "n_attacks": N_ATTACKS, "n_examples": N_EXAMPLES,
            "delta": "ON - OFF",
            "estimator": "paired sample-level bootstrap, reused verbatim from "
                         "scripts/dev/pernorm_ci.py:boot (== scripts/dev/c5_attribution.py:paired): "
                         "one resampled index vector indexes BOTH arms",
            "bootstrap": f"B={a.b}, fresh np.random.default_rng({SEED}) per call, percentile",
            "ci_conventions": {"two_sided_95": "[q025, q975] — used by the significance rule",
                               "lcb_95": "q05 — 1-sided lower confidence bound, reported alongside"},
            "significance_rule": "SIGNIFICANT iff, for every seed of the row, two_sided_95 excludes "
                                 "zero AND all seeds share the same sign; otherwise ns. Single-seed "
                                 "rows reduce to the single interval.",
            "union": "AND over all 12 attack masks",
            "per_norm_block": "AND over that norm's 4 attack masks",
            "clean": "UNPAIRED scalar difference of eval.json clean_acc — the frozen mask set holds "
                     "only the 12 attack masks (eval_multinorm_audit.py writes clean_acc as a scalar "
                     "and never stores a clean per-example mask), so clean has NO CI and its "
                     "significance is 'n/a', not 'ns'",
            "audit_config": cfg_path, "audit_config_sha256": cfg_sha,
        },
        "provenance_untraced": {
            "FT (ft_clamp / ft_none)": "UNTRACED — results/fromscratch/explore3_ftinf/{clamp,none}/ "
                                       "hold only val_best.pt; no train.json or equivalent run record "
                                       "exists, so epochs / lr / schedule are not stated here",
            "RAMP (B1 / Rprime)": "UNTRACED — checkpoints were downloaded from Drive (Colab paths in "
                                  "results/main/<arm>/meta.json); no local training record exists, so "
                                  "the trainer, epochs, lr and schedule are not stated and it CANNOT "
                                  "be confirmed that both arms used identical settings",
        },
        "sanity": {"mask_keys_per_arm": N_ATTACKS, "n_per_arm": N_EXAMPLES,
                   "attacks_per_norm_block": 4,
                   "one_audit_subset_across_all_arms": list(subs)[0] if subs else None,
                   "union_subset_of_every_per_norm_block": True},
        "cross_check_union_masks_vs_eval_json": {"tolerance": 5e-5, "failures": xcheck,
                                                 "all_match": not xcheck},
        "reconciliation_pernorm_ci": recon,
        "rows": table,
        "provenance": prov,
    }
    json.dump(payload, open(os.path.join(OUT, "table1.json"), "w"), indent=2)

    # ---- console table ----
    print("\n" + "=" * 112)
    print(f"TABLE 1 — @{SCALE}, tier={TIER} ({N_ATTACKS} attacks). Accuracies in %, deltas in pp. Delta = ON - OFF.")
    print("=" * 112)
    print(f"{'budget':<21}{'base':<8}{'sd':>3}{'UnionOFF':>10}{'UnionON':>9}{'dUnion':>9}"
          f"{'dLinf':>9}{'dL2':>9}{'dL1':>9}{'dClean':>9}")
    for budget in BUDGETS:
        grp = sorted([r for r in table if r["budget"] == budget],
                     key=lambda r: -r["mean_delta"]["union"])
        for r in grp:
            def cell(c):
                return pp(r["mean_delta"][c]) + ("*" if r["significance"][c] == "sig" else
                                                 "" if r["significance"][c] == "n/a" else "n")
            print(f"{budget:<21}{r['base']:<8}{r['n_seeds']:>3}"
                  f"{pct(r['mean_acc_off']['union']):>10}{pct(r['mean_acc_on']['union']):>9}"
                  f"{cell('union'):>9}{cell('linf'):>9}{cell('l2'):>9}{cell('l1'):>9}{cell('clean'):>9}")
    print("  * = significant by the Phase-2 rule (every seed's two-sided 95% CI excludes 0 and all")
    print("      seeds share a sign);  n = ns;  dClean carries no marker — it is an UNPAIRED scalar")
    print("      difference with no CI (no clean per-example mask exists in the frozen artifact).")

    print(f"\n  seed-minimum lcb_95 (q05), multi-seed rows:")
    for r in table:
        if r["n_seeds"] > 1:
            print(f"    {r['base']:<8} " + "  ".join(
                f"{c}: {100 * r['seed_min_lcb_95'][c]:+.2f}pp" for c in CELLS))

    # ---- table1.tex : 9-column body, \multirow budget, rows desc by dUnion ----
    L = []
    for budget in BUDGETS:
        grp = sorted([r for r in table if r["budget"] == budget],
                     key=lambda r: -r["mean_delta"]["union"])
        for i, r in enumerate(grp):
            first = (f"\\multirow{{{len(grp)}}}{{*}}{{{budget}}}" if i == 0 else "")
            d = " & ".join(tex_delta(r["mean_delta"][c], r["significance"][c]) for c in CELLS)
            L.append(f"{first} & {r['base']} & {pct(r['mean_acc_off']['union'])} & "
                     f"{pct(r['mean_acc_on']['union'])} & {d} & {pp(r['mean_delta']['clean'])} \\\\")
        L.append("\\midrule" if budget != BUDGETS[-1] else "")
    tex = ("% Table 1 body — generated by scripts/dev/make_table1.py from the frozen @10k masks.\n"
           "% Drop into a table* environment with:\n"
           "%   \\begin{tabular}{llrrrrrrr}  and header:\n"
           "%   Training budget & Base & Union OFF & Union ON & $\\Delta$Union & $\\Delta\\ell_\\infty$ &\n"
           "%   $\\Delta\\ell_2$ & $\\Delta\\ell_1$ & $\\Delta$Clean \\\\\n"
           "% Requires \\usepackage{multirow}. Accuracies in %, deltas in percentage points.\n"
           "% Bold = significant (every seed's two-sided 95% CI excludes 0 and all seeds share a sign).\n"
           "% ns superscript = not significant. Rows within a budget are ordered by descending Delta-Union.\n"
           "% Delta-Clean is an UNPAIRED scalar difference with NO CI (the frozen mask sets contain no\n"
           "% clean per-example mask), so it is never bolded and carries no ns marker.\n"
           + "\n".join(x for x in L if x != "") + "\n")
    open(os.path.join(OUT, "table1.tex"), "w").write(tex)

    # ---- pernorm_appendix.tex : per-seed per-norm, BOTH CI conventions ----
    A = ["% Appendix — per-seed per-norm deltas, generated by scripts/dev/make_table1.py.",
         "% \\begin{tabular}{llllrrr} ; requires nothing beyond booktabs.",
         "% BOTH interval conventions are shown and are read off the SAME bootstrap distribution",
         f"% (paired, B={a.b}, np.random.default_rng({SEED})):",
         "%   two-sided 95\\% = [q025, q975]  (the interval the significance rule uses)",
         "%   LCB 95\\%       = q05           (1-sided lower bound; the convention pernorm_ci.py reports)",
         f"% Delta = ON - OFF, in percentage points, at @{SCALE} tier={TIER}.",
         "Base & Seed & Arms (ON $-$ OFF) & Norm & $\\Delta$ (pp) & two-sided 95\\% (pp) & LCB 95\\% (pp) \\\\",
         "\\midrule"]
    for r in table:
        for s in r["seeds"]:
            for nm in NORMS:
                e = s["delta"][nm]
                lo, hi = e["two_sided_95"]
                A.append(f"{r['base']} & {s['seed']} & {tex_esc(s['arm_on'])} $-$ "
                         f"{tex_esc(s['arm_off'])} & ${TEX_NORM[nm]}$ & {pp(e['delta_plugin'])} & "
                         f"[{pp(lo)}, {pp(hi)}] & {pp(e['lcb_95'])} \\\\")
        A.append("\\midrule")
    open(os.path.join(OUT, "pernorm_appendix.tex"), "w").write("\n".join(A[:-1]) + "\n")

    print(f"\n  saved -> results/analysis/table1/table1.json")
    print(f"           results/analysis/table1/table1.tex")
    print(f"           results/analysis/table1/pernorm_appendix.tex")


if __name__ == "__main__":
    main()

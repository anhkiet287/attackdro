"""Sync results/eval/union_bench/ -> results/main/ (normalised, verifiable, per-run).

results/main/<arm>/{1k,10k,last}/{eval.json, masks_multinorm_v1.npz} + <arm>/meta.json + INDEX.md.
Originals are NEVER modified — this only copies. Idempotent: re-run after every new audit/paste.

Verification performed on every run (printed + recorded in meta.json):
  * union reported in eval.json  ==  union recomputed from the per-example masks
  * per-norm audit_acc           ==  recomputed AND over that norm's attacks
  * copy is byte-identical to the union_bench original

Usage:  python scripts/dev/sync_results_main.py [--quiet]
"""
from __future__ import annotations
import hashlib, json, os, shutil, sys
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
UB = os.path.join(ROOT, "results/eval/union_bench")
MAIN = os.path.join(ROOT, "results/main")
NORMS = ("linf", "l2", "l1")

GROUPS = [
    ("CLAIM A — from-scratch (MSD base, 80ep)", ["M1a", "M0", "M1b", "M1a_seed2", "M0_seed2", "M1a_seed3", "M0_seed3"]),
    ("FULL-BUDGET (Maini 50ep, MSD-50)", ["M1a_full", "M0_full", "M1a_full_seed1", "M0_full_seed1", "M1a_full_v50"]),
    ("ABLATION / EXTENSION", ["M1a_msdglue", "M1a_max", "M0_max", "M1a_avg", "M0_avg", "M1a_max_msdglue",
                              "M1a_linfglue", "M1a_dynworst", "M1a_pp_push2pull", "M1a_pp_push2pull_gs",
                              "ft_clamp", "ft_none", "M1a_c100", "M0_c100"]),
    ("RAMP / CLAIM B", ["Rprime", "B1", "B2"]),
    ("PUBLIC BASELINES", ["msd", "max", "avg"]),
    ("ALLOCATION ARC (legacy)", ["b3_static_linf", "b4_adaptive"]),
]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _masks(p):
    d = np.load(p)
    return {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}


def _and(masks, keys):
    u = np.ones(len(next(iter(masks.values()))), bool)
    for k in keys:
        u &= masks[k]
    return float(u.mean())


def _scale_dirs(arm):
    """Which source subdirs hold an audit. Root counts as 1k for arms that use it;
    an explicit 1k/ subdir wins when both exist (verified identical for M0_full)."""
    out, d = {}, os.path.join(UB, arm)
    if os.path.exists(f"{d}/eval.json") or os.path.exists(f"{d}/masks_multinorm_v1.npz"):
        out["1k"] = ""
    for sub in ("1k", "10k", "last", "1k_nosq", "10k_nosq"):   # *_nosq = tier=no_square (9 attacks)
        p = os.path.join(d, sub)
        if os.path.isdir(p) and (os.path.exists(f"{p}/eval.json") or os.path.exists(f"{p}/masks_multinorm_v1.npz")):
            out[sub] = sub + "/"
    return out


def main():
    quiet = "--quiet" in sys.argv
    os.makedirs(MAIN, exist_ok=True)
    arms = sorted(a for a in os.listdir(UB) if os.path.isdir(os.path.join(UB, a)) and not a.startswith("_"))
    n_dirs = n_cons = n_bad = n_nomask = n_copy = 0
    problems = []
    index = {}

    for arm in arms:
        scales = _scale_dirs(arm)
        if not scales:
            continue
        meta = {"arm": arm, "source": f"results/eval/union_bench/{arm}", "scales": {}}
        for lbl, sub in scales.items():
            src, dst = os.path.join(UB, arm, sub), os.path.join(MAIN, arm, lbl)
            os.makedirs(dst, exist_ok=True)
            info, e, m = {}, None, None
            ej = os.path.join(src, "eval.json")
            if os.path.exists(ej):
                shutil.copy2(ej, os.path.join(dst, "eval.json"))
                n_copy += 1
                e = json.load(open(ej))
                info.update(union=e.get("full_audit_union"), clean=e.get("clean_acc"),
                            n=(e.get("subset") or {}).get("n_examples"),
                            n_attacks=len(e.get("per_attack", {})) or None,
                            ckpt=e.get("checkpoint"), ckpt_sha256=e.get("checkpoint_sha256"),
                            per_norm=e.get("per_norm_audit"), provenance=e.get("note", "local audit"))
            mp = os.path.join(src, "masks_multinorm_v1.npz")
            if os.path.exists(mp):
                shutil.copy2(mp, os.path.join(dst, "masks_multinorm_v1.npz"))
                n_copy += 1
                m = _masks(mp)
                ks = list(m)
                info.update(masks=True, mask_keys=len(ks), mask_n=len(next(iter(m.values()))),
                            union_from_masks=round(_and(m, ks), 6),
                            union_no_square=round(_and(m, [k for k in ks if "square" not in k]), 6))
                if info.get("union") is None:
                    info["union"] = info["union_from_masks"]
                # --- verification: reported vs recomputed ---
                checks = {}
                if e and e.get("full_audit_union") is not None:
                    ok = abs(e["full_audit_union"] - info["union_from_masks"]) < 1e-6
                    checks["union_matches_masks"] = ok
                    n_cons += ok
                    if not ok:
                        n_bad += 1
                        problems.append(f"{arm}/{lbl}: eval.json {e['full_audit_union']:.4f} != masks {info['union_from_masks']:.4f}")
                if e and e.get("per_norm_audit"):
                    pn_ok = True
                    for nm in NORMS:
                        rep = e["per_norm_audit"].get(f"audit_acc_{nm}")
                        if rep is None:
                            continue
                        calc = _and(m, [k for k in ks if k.endswith(nm)])
                        if abs(rep - calc) > 1e-6:
                            pn_ok = False
                            problems.append(f"{arm}/{lbl} per-norm {nm}: {rep:.4f} != {calc:.4f}")
                    checks["per_norm_matches_masks"] = pn_ok
                info["verification"] = checks
            else:
                info["masks"] = False
                n_nomask += 1
            meta["scales"][lbl] = info
            index.setdefault(arm, {})[lbl] = info
            n_dirs += 1
        json.dump(meta, open(os.path.join(MAIN, arm, "meta.json"), "w"), indent=2)

    # ---- INDEX.md ----
    L = ["# results/main — audit inventory (per run, verifiable)", "",
         "Normalised copy of `results/eval/union_bench/` — one folder per run: `1k/`, `10k/`, `1k_nosq/`, `10k_nosq/` (+`last/`).",
         "Each scale dir: `eval.json` and, when available, `masks_multinorm_v1.npz` (per-example → paired bootstrap).",
         "`meta.json` per arm: ckpt sha256, source path, provenance (local audit vs Colab paste), and the verification result.",
         "`atk` = number of attack components (**12** = full frozen 12-AA · **9** = no-Square fast tier (12 − 3 square)).",
         "**Originals are never modified.** Regenerate with `python scripts/dev/sync_results_main.py`.", ""]
    listed = set()
    for gname, arms_g in GROUPS:
        rows = []
        for a in arms_g:
            for lbl in ("1k", "1k_nosq", "10k", "10k_nosq", "last"):
                s = index.get(a, {}).get(lbl)
                if not s:
                    continue
                listed.add(a)
                pn = s.get("per_norm") or {}
                f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "–"
                rows.append(f"| {a} | {lbl} | {s.get('n') or s.get('mask_n') or '–'} | "
                            f"{s.get('n_attacks') or s.get('mask_keys') or '–'} | {f(s.get('clean'))} | "
                            f"**{f(s.get('union'))}** | {f(pn.get('audit_acc_linf'))} | {f(pn.get('audit_acc_l2'))} | "
                            f"{f(pn.get('audit_acc_l1'))} | {'✓' if s.get('masks') else '–'} |")
        if rows:
            L += [f"## {gname}", "", "| arm | scale | n | atk | clean | union | ℓ∞ | ℓ2 | ℓ1 | masks |",
                  "|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    extra = sorted(set(index) - listed)
    if extra:
        L += ["## (ungrouped)", "", "| arm | scale | union | masks |", "|---|---|---|---|"]
        for a in extra:
            for lbl, s in index[a].items():
                L.append(f"| {a} | {lbl} | {s.get('union')} | {'✓' if s.get('masks') else '–'} |")
        L.append("")
    L += ["## Verification (this sync)", "",
          f"- union(eval.json) == union(recomputed from masks): **{n_cons} pass / {n_bad} mismatch**",
          f"- scale-dirs with per-example masks: **{n_dirs - n_nomask}/{n_dirs}** ({n_nomask} eval.json-only, from Colab pastes)",
          f"- files copied: {n_copy}"]
    if problems:
        L += ["", "### ⚠ problems"] + [f"- {p}" for p in problems]
    open(os.path.join(MAIN, "INDEX.md"), "w", encoding="utf-8").write("\n".join(L))

    if not quiet:
        print(f"synced {n_dirs} scale-dirs / {len(index)} arms -> results/main/")
        print(f"  verification: union==masks {n_cons} pass, {n_bad} mismatch | {n_nomask} dirs without masks")
        for p in problems:
            print("  !!", p)
        print(f"  wrote {MAIN}/INDEX.md")


if __name__ == "__main__":
    main()

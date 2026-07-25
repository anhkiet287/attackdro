"""P1 LOCUS pilot — REDUCED-harness evaluation of arm E against M0 and M1a.

Pre-registered in docs/preregistrations/preregistration_P1_locus.md §3 and §6.

REDUCED HARNESS: APGD-CE only, three norms (ℓ∞ 8/255, ℓ₂ 0.5, ℓ₁ 12), on the frozen 1k audit
subset, val_best weights, steps=100 restarts=1 seed=20260709 — exactly the apgd_ce parameters of
the frozen 12-attack audit config. FAB-T and Square are ABSENT: these unions are OPTIMISTIC
relative to the 12-attack harness and carry NO black-box gradient-masking check. No number here
may be placed beside a 12-attack number from the paper.

Why M0/M1a are not re-attacked: their frozen 1k mask sets already contain apgd_ce_linf/l2/l1
computed with exactly these parameters on exactly this subset. Restricting those masks to the
three columns IS this harness — read-only, and identical to re-running it. Only arm E is attacked.

Attack implementation is the harness's own: this script imports scripts/eval_multinorm_audit.py
and calls its run_attack_mask/load_subset/clean_mask, so E's APGD-CE is the same code path that
produced the frozen masks. The audit config is loaded UNMODIFIED (its validator hard-requires all
12 canonical attack names); the restriction to apgd_ce happens after validation, in memory.

Writes only under results/analysis/P1_locus/. results/main/ and all existing checkpoints are
read-only. No training, no re-attack of any existing arm.

  python scripts/dev/p1_locus_eval.py --e-ckpt results/fromscratch/P1_locus/E/ckpt/val_best.pt
  python scripts/dev/p1_locus_eval.py            # existing arms only (E omitted)
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


t1 = _load("t1", "scripts/dev/make_table1.py")          # reuse the estimator, do not rewrite it

OUT = os.path.join(ROOT, "results/analysis/P1_locus")
CFG = "configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml"
NORMS = ("linf", "l2", "l1")
PRIMARY = [f"apgd_ce_{n}" for n in NORMS]
EXISTING = {"M0": "results/main/M0/1k", "M1a": "results/main/M1a/1k"}
CONTRASTS = [("E", "M0", "PRIMARY   E - M0   (does a locus-corrected term help at all?)"),
             ("E", "M1a", "SECONDARY E - M1a  (does the locus matter? READ ASYMMETRICALLY, §4)")]


def block_from(masks):
    """union3 = AND over the three APGD-CE masks; each per-norm block is the single apgd_ce mask."""
    u = np.ones(len(masks[PRIMARY[0]]), bool)
    for k in PRIMARY:
        u &= masks[k]
    return {"union": u, **{n: masks[f"apgd_ce_{n}"] for n in NORMS}}


def from_frozen(arm, rel):
    p = os.path.join(ROOT, rel, "masks_multinorm_v1.npz")
    assert os.path.exists(p), f"{arm}: frozen 1k masks missing at {rel}"
    d = np.load(p, allow_pickle=True)
    md = json.loads(str(d["metadata_json"]))
    m = {k: d[k].astype(bool) for k in PRIMARY}
    e = json.load(open(os.path.join(ROOT, rel, "eval.json")))
    return {"arm": arm, "source": "frozen 12-attack mask set restricted to the 3 apgd_ce columns "
                                  "(read-only; not re-attacked)",
            "mask_file": os.path.join(rel, "masks_multinorm_v1.npz"),
            "checkpoint_path": md.get("checkpoint_path"), "checkpoint_sha256": md.get("checkpoint_sha256"),
            "checkpoint_role": md.get("checkpoint_role"),
            "audit_subset_sha256": md.get("audit_subset_sha256"),
            "clean_acc": float(e["clean_acc"]), "val_best_epoch": None,
            "blocks": block_from(m)}


def attack_E(ckpt, bs, device):
    """Run the 3 APGD-CE attacks on arm E through the audit harness's own code path."""
    A = _load("A", "scripts/eval_multinorm_audit.py")
    cfg = A.load_audit_config(os.path.join(ROOT, CFG))
    A.validate_config(cfg)                                   # unmodified config: validator passes
    attacks = [a for a in A.iter_attacks(cfg) if a["name"] in PRIMARY]
    assert len(attacks) == 3, f"expected 3 apgd_ce attacks, got {[a['name'] for a in attacks]}"
    model, _ = A.load_eval_checkpoint(ckpt, A.to_eval_cfg(cfg), model_family="robustdro", device=device)
    x, y, _, idx_sha = A.load_subset(cfg)
    clean = A.clean_mask(model, x, y, device, bs)
    masks, per = {}, {}
    for at in attacks:
        m, _, _ = A.run_attack_mask(model, x, y, at, float(cfg["eps"][at["eps_key"]]), device, bs)
        masks[at["name"]] = m.numpy().astype(bool)
        per[at["name"]] = {"robust_acc": float(m.float().mean()), "params": dict(at)}
        print(f"    {at['name']:<14} robust_acc {per[at['name']]['robust_acc']:.4f}")
    return {"arm": "E", "source": "attacked by this script (reduced harness)",
            "mask_file": "results/analysis/P1_locus/E_masks_apgd_ce_1k.npz",
            "checkpoint_path": os.path.relpath(ckpt, ROOT) if ckpt.startswith(ROOT) else ckpt,
            "checkpoint_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
            "checkpoint_role": "val_best", "audit_subset_sha256": idx_sha,
            "clean_acc": float(np.asarray(clean.cpu() if hasattr(clean, "cpu") else clean).mean()),
            "per_attack": per, "clean_mask": np.asarray(
                clean.cpu() if hasattr(clean, "cpu") else clean).astype(bool),
            "blocks": block_from(masks), "_raw": masks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e-ckpt", default=None, help="arm E val_best.pt; omit to report existing arms only")
    p.add_argument("--e-train-json", default=None, help="arm E train.json, for val_best epoch + curves")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = a.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    print("=" * 100)
    print("P1 LOCUS — REDUCED HARNESS: APGD-CE only (3 attacks), 1k frozen subset, val_best weights")
    print("  FAB-T and Square ABSENT -> unions are OPTIMISTIC; no black-box gradient-masking check.")
    print("  These numbers match NO arm reported in the paper and must not be tabled beside one.")
    print("=" * 100)

    R = {k: from_frozen(k, v) for k, v in EXISTING.items()}
    for k, v in R.items():
        print(f"  {k:<4} from frozen masks: {v['mask_file']}")
    if a.e_ckpt:
        ck = a.e_ckpt if os.path.isabs(a.e_ckpt) else os.path.join(ROOT, a.e_ckpt)
        assert os.path.exists(ck), f"arm E checkpoint not found: {ck}"
        print(f"  E    attacking {ck}")
        R["E"] = attack_E(ck, a.bs, device)
        np.savez_compressed(os.path.join(OUT, "E_masks_apgd_ce_1k.npz"),
                            **{k: v for k, v in R["E"]["_raw"].items()},
                            clean=R["E"]["clean_mask"])
    else:
        print("  E    SKIPPED (--e-ckpt not given) — contrasts are not computed")

    subs = {v["audit_subset_sha256"] for v in R.values()}
    assert len(subs) == 1, f"arms are not on the same audit subset: {subs}"
    n = len(R["M0"]["blocks"]["union"])
    for k, v in R.items():
        assert len(v["blocks"]["union"]) == n, f"{k}: n mismatch"
        for nm in NORMS:
            assert not (v["blocks"]["union"] & ~v["blocks"][nm]).any(), f"{k}: union not subset of {nm}"
    print(f"  [ok] all arms on subset {list(subs)[0][:16]}, n={n}, union ⊆ every per-norm block")

    if a.e_train_json and os.path.exists(a.e_train_json):
        tj = json.load(open(a.e_train_json))
        h = tj.get("history") or []
        if h:
            R["E"]["val_best_epoch"] = 1 + max(range(len(h)), key=lambda i: h[i]["val_worst_union"])
            R["E"]["curves"] = [{k: r.get(k) for k in
                                 ("epoch", "loss", "ce", "glue", "scaffold", "val_worst_union",
                                  "val_p1_cos_f_anchor", "val_p1_cos_f_negpair", "lr", "sec")} for r in h]

    CELLS = ("union",) + NORMS
    contrasts = {}
    if "E" in R:
        for A_, B_, lbl in CONTRASTS:
            c = {"label": lbl, "arm_a": A_, "arm_b": B_, "cells": {}}
            for cell in CELLS:
                e = t1.boot(R[A_]["blocks"][cell], R[B_]["blocks"][cell], a.boot)
                lo, hi = e["two_sided_95"]
                c["cells"][cell] = {**e, "significant": not (lo <= 0.0 <= hi)}
            c["cells"]["clean"] = {"delta_plugin": R[A_]["clean_acc"] - R[B_]["clean_acc"],
                                   "two_sided_95": None, "lcb_95": None, "significant": None,
                                   "note": "UNPAIRED scalar difference; the frozen mask sets carry no "
                                           "clean per-example mask, so no paired bootstrap is possible"}
            contrasts[f"{A_}-{B_}"] = c

    payload = {
        "status": "PILOT — pre-registered in docs/preregistrations/preregistration_P1_locus.md",
        "harness": {
            "name": "REDUCED: APGD-CE only", "n_attacks": 3, "attacks": PRIMARY,
            "scale": "1k", "weights": "val_best", "n_examples": n,
            "subset": "results/audit/subsets/cifar10_testfinal_1000_seed20260709_v3A.json",
            "subset_sha256": list(subs)[0], "config_source": CFG,
            "apgd_params": "steps=100, restarts=1, seed=20260709 (identical to the frozen audit)",
            "ABSENT": ["fab_t_*", "square_*"],
            "caveat": "FAB-T and Square are absent, so these unions are OPTIMISTIC relative to the "
                      "12-attack harness and carry NO black-box gradient-masking check",
            "comparability": "This configuration and harness match NO arm reported in the paper. "
                             "No number here may be placed beside a 12-attack number.",
            "existing_arms": "M0 and M1a are the frozen 12-attack 1k mask sets restricted to the "
                             "three apgd_ce columns — read-only, not re-attacked",
        },
        "estimator": {"source": "scripts/dev/make_table1.py:boot (verbatim from pernorm_ci.py:boot)",
                      "bootstrap": f"paired per example, B={a.boot}, fresh np.random.default_rng(0) per call",
                      "conventions": {"two_sided_95": "[q025,q975] — the reading rule uses this",
                                      "lcb_95": "q05"}},
        "reading_rule": "Pre-registration §4: E-M1a is confounded (locus AND push range change "
                        "together). Positive+significant = evidence for locus; anything else = "
                        "INCONCLUSIVE. E-M0 is the unambiguous primary contrast.",
        "arms": {k: {"source": v["source"], "checkpoint_path": v["checkpoint_path"],
                     "checkpoint_sha256": v["checkpoint_sha256"], "checkpoint_role": v["checkpoint_role"],
                     "mask_file": v["mask_file"], "val_best_epoch": v.get("val_best_epoch"),
                     "clean_acc": v["clean_acc"],
                     "acc": {c: float(v["blocks"][c].mean()) for c in CELLS}} for k, v in R.items()},
        "contrasts": contrasts,
        "training_curves_E": R.get("E", {}).get("curves"),
    }
    json.dump(payload, open(os.path.join(OUT, "p1_locus.json"), "w"), indent=2)

    print("\n" + "=" * 100)
    print("P1 — accuracies under the REDUCED harness (APGD-CE ×3), 1k, val_best")
    print("=" * 100)
    print(f"  {'arm':<5}{'clean':>9}{'union3':>9}{'linf':>9}{'l2':>9}{'l1':>9}   source")
    for k in ("M0", "M1a", "E"):
        if k not in R:
            continue
        v = R[k]
        print(f"  {k:<5}{v['clean_acc']:>9.4f}" + "".join(f"{float(v['blocks'][c].mean()):>9.4f}" for c in CELLS)
              + f"   {'attacked' if k == 'E' else 'frozen masks'}")
    if contrasts:
        print("\n" + "=" * 100)
        print("P1 — contrasts, paired per example, two-sided 95% CI")
        print("=" * 100)
        for key, c in contrasts.items():
            print(f"  {c['label']}")
            for cell in CELLS:
                e = c["cells"][cell]
                lo, hi = e["two_sided_95"]
                print(f"    {cell:<7} Δ={e['delta_plugin']:+.4f}  95%[{lo:+.4f},{hi:+.4f}]"
                      f"  lcb95={e['lcb_95']:+.4f}  {'SIG' if e['significant'] else 'ns'}")
            print(f"    clean   Δ={c['cells']['clean']['delta_plugin']:+.4f}  (unpaired scalar, no CI)")
        print("\n  Reading rule (pre-registration §4): E-M0 is primary and unambiguous. E-M1a is")
        print("  confounded — positive+significant = evidence for locus; ns or negative = INCONCLUSIVE.")
    print(f"\n  saved -> results/analysis/P1_locus/p1_locus.json")


if __name__ == "__main__":
    main()

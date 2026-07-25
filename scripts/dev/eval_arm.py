"""Single-arm audit CLI with EXPLICIT scale + tier. Frozen 12-AA harness (config sha 9162ce44).

Both flags are REQUIRED — there is no default scale and no default tier, so a 10k run can never
happen by accident. 1k and 10k are separate invocations; this script never chains them.

  --tier full        12 components (apgd_ce/apgd_dlr/fab_t/square × ℓ∞/ℓ2/ℓ1) via the native harness
  --tier no_square    9 components (square_* dropped)          via union_bench_eval --skip-square

TIER IS NOT INTERCHANGEABLE: union(no_square) >= union(full) always (dropping an attack can only
add survivors); measured drift 0…+0.0010 over 29 audits (non-zero in 7). The CIFAR-10 CLAIM A @1k
table was produced with **tier=full**, so quote 2×2 numbers at tier=full to compare against it.

Usage:
  python scripts/dev/eval_arm.py --arm M1a_max --ckpt <path> --scale 1k  --tier no_square
  python scripts/dev/eval_arm.py --arm M1a_max --ckpt <path> --scale 10k --tier full
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
import numpy as np

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
UB = os.path.join(ROOT, "results/eval/union_bench")
CFG = {
    ("1k", "cifar10"):  "configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml",
    ("10k", "cifar10"): "results/eval/union_bench/_config/audit_v3A_test_10k.yaml",
    ("1k", "cifar100"):  "configs/eval/audit_cifar100_preactrn18_multinorm_v3A_test1k.yaml",
    ("10k", "cifar100"): "configs/eval/audit_cifar100_preactrn18_multinorm_v3A_test10k.yaml",
}
NORMS = ("linf", "l2", "l1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="output name under results/eval/union_bench/")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scale", required=True, choices=["1k", "10k"], help="REQUIRED — no default")
    p.add_argument("--tier", required=True, choices=["full", "no_square"], help="REQUIRED — full=12, no_square=9")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--family", default="robustdro", help="robustdro | robust_union_preact | robustbench | ramp")
    p.add_argument("--rb-name", default=None, help="robustbench arch when --ckpt is a fine-tuned state_dict")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--force", action="store_true", help="re-audit even if the output exists")
    a = p.parse_args()

    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(ROOT, a.ckpt)
    if not os.path.exists(ck):
        sys.exit(f"ckpt not found: {ck}")
    cfg = CFG[(a.scale, a.dataset)]
    # preflight: the frozen harness reads its config (and the subset indices it names) from paths
    # under results/. Those are data, not code — if the deployment dropped them the harness would
    # either die on the config or, worse, silently REGENERATE the subset and audit different
    # examples, which quietly invalidates any paired comparison against a run made elsewhere.
    if not os.path.exists(os.path.join(ROOT, cfg)):
        # Distinguish the two causes instead of blaming packaging for both: if the file IS present
        # next to this script, the deployment is fine and ROOT is simply pointing somewhere else
        # (typically: launched without ATTACKDRO_ROOT and without cd-ing into the repo).
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.path.exists(os.path.join(here, cfg)):
            sys.exit(f"[eval] ATTACKDRO_ROOT is wrong.\n"
                     f"       ROOT resolved to : {ROOT}\n"
                     f"       config not there : {os.path.join(ROOT, cfg)}\n"
                     f"       but IS here      : {os.path.join(here, cfg)}\n"
                     f"       Fix: run with  ATTACKDRO_ROOT={here}  (or cd {here} first).")
        sys.exit(f"[eval] audit config missing: {cfg}\n"
                 f"       ROOT = {ROOT}; not found there and not found beside the script either.\n"
                 f"       (packaging bug — ship it with the code, do not hand-write a replacement)")
    try:
        import yaml
        ip = (yaml.safe_load(open(os.path.join(ROOT, cfg))).get("subset") or {}).get("indices_path")
        if ip and not os.path.exists(os.path.join(ROOT, ip)):
            print(f"[eval] WARNING: subset indices absent ({ip}); the harness will REGENERATE them.\n"
                  f"       Only safe if every run being compared regenerates identically.", flush=True)
    except Exception:
        pass
    # TIER-AWARE OUTPUT — a no_square run must NEVER clobber a full-12 record (e.g. M1a/1k = the
    # CLAIM A number). full keeps the canonical path; no_square gets its own <scale>_nosq/ dir.
    if a.tier == "full":
        sub = "" if a.scale == "1k" else "10k/"
    else:
        sub = f"{a.scale}_nosq/"
    out = os.path.join(UB, a.arm, sub, "eval.json")
    masks = os.path.join(UB, a.arm, sub, "masks_multinorm_v1.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()

    # clobber guard: refuse to overwrite an existing audit recorded at a DIFFERENT tier
    if os.path.exists(out):
        try:
            prev = len(json.load(open(out)).get("per_attack", {}))
            want = 12 if a.tier == "full" else 9
            if prev and prev != want and not a.force:
                sys.exit(f"[eval] REFUSING to overwrite {out}: it holds a {prev}-attack audit but "
                         f"--tier {a.tier} would write {want}. Use --force only if you mean it.")
        except Exception:
            pass

    print(f"[eval] arm={a.arm} scale={a.scale} tier={a.tier} dataset={a.dataset} family={a.family}")
    print(f"[eval] ckpt={ck}\n[eval] ckpt_sha256={sha}")
    print(f"[eval] config={cfg}  (frozen 12-AA schema)")

    if os.path.exists(out) and not a.force:
        print(f"[eval] SKIP — {out} exists (use --force to redo)")
    else:
        if a.tier == "full":
            if a.family in ("robustdro", "ramp"):
                cmd = [sys.executable, "scripts/eval_multinorm_audit.py", "--config", cfg, "--checkpoint", ck,
                       "--model-family", a.family, "--run-id", f"{a.arm}_{a.scale}",
                       "--checkpoint-role", "val_best", "--out", out, "--export-masks", "--bs", str(a.bs)]
            else:
                cmd = [sys.executable, "scripts/dev/union_bench_eval.py", "--config", cfg, "--checkpoint", ck,
                       "--arch", a.family, "--run-id", f"{a.arm}_{a.scale}", "--checkpoint-role", "val_best",
                       "--out", out, "--export-masks", "--bs", str(a.bs)]
                if a.rb_name: cmd += ["--rb-name", a.rb_name]
        else:                                            # no_square -> always the union_bench driver
            cmd = [sys.executable, "scripts/dev/union_bench_eval.py", "--config", cfg, "--checkpoint", ck,
                   "--arch", a.family, "--skip-square", "--run-id", f"{a.arm}_{a.scale}_nosq",
                   "--checkpoint-role", "val_best", "--out", out, "--export-masks", "--bs", str(a.bs)]
            if a.rb_name: cmd += ["--rb-name", a.rb_name]
        print("[eval] $ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, ATTACKDRO_ROOT=ROOT))
        if r.returncode != 0 or not os.path.exists(out):
            sys.exit(f"[eval] FAILED (rc={r.returncode})")

    # ---- report ----
    e = json.load(open(out))
    natt = len(e.get("per_attack", {}))
    expect = 12 if a.tier == "full" else 9
    pn = e.get("per_norm_audit") or {}
    if not pn and os.path.exists(masks):
        d = np.load(masks); mm = {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}
        for nm in NORMS:
            ks = [k for k in mm if k.endswith(nm)]
            u = np.ones(len(mm[ks[0]]), bool)
            for k in ks: u &= mm[k]
            pn[f"audit_acc_{nm}"] = float(u.mean())
    print("\n" + "=" * 64)
    print(f"  arm        : {a.arm}   ({a.scale}, tier={a.tier})")
    print(f"  n_attacks  : {natt}   (expected {expect} for tier={a.tier}) "
          f"{'OK' if natt == expect else '<-- MISMATCH'}")
    print(f"  n_examples : {(e.get('subset') or {}).get('n_examples')}")
    print(f"  clean      : {e.get('clean_acc'):.4f}" if e.get("clean_acc") is not None else "  clean      : –")
    print(f"  union      : {e.get('full_audit_union'):.4f}")
    print(f"  per-norm   : ℓ∞ {pn.get('audit_acc_linf', float('nan')):.4f}  "
          f"ℓ2 {pn.get('audit_acc_l2', float('nan')):.4f}  ℓ1 {pn.get('audit_acc_l1', float('nan')):.4f}")
    print(f"  ckpt_sha   : {sha}")
    print(f"  eval.json  : {os.path.relpath(out, ROOT)}")
    print(f"  masks      : {os.path.relpath(masks, ROOT) if os.path.exists(masks) else '(none)'}")
    print("=" * 64)
    if a.tier == "no_square":
        print("  NOTE: tier=no_square is an UPPER BOUND on the full-12 union (0…+0.0010 drift).")
        print("        CLAIM A @1k was measured at tier=full — quote tier=full for that comparison.")


if __name__ == "__main__":
    main()

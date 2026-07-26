"""BACKFILL CAMPAIGN (Part B) — re-run every completed arm WITH standard per-norm train logging,
into an ARCHIVE dir, WITHOUT touching originals. Paper numbers = the ORIGINAL runs (inviolable);
backfill = archive-with-logging only. If a backfill's @1k union drifts > noise (0.003) from the
original -> log INVESTIGATE, do NOT touch paper numbers.

HARD GATES (this script refuses to run otherwise):
  * deadline-critical runs done: seed-3 + seed-1 + v50 (unless --force).
  * explicit --go (never auto-runs).
Priority: background, GPU-idle, many days OK. No deadline.

Per arm: read ORIGINAL config from ckpt cfg / train.json (NOT hardcoded) -> relaunch the SAME
trainer + config + logging ON, outdir results/backfill_logged/<arm>/ -> eval @1k -> compare union
to the original -> row into backfill_report.csv + print backfill_check.

Trainer wiring status (logging must be wired in the trainer before its arms can backfill):
  c5_fromscratch  : WIRED  -> M0, M1a, M1b, M1a/M0 seed2/seed3, M0_c100/M1a_c100,
                             M1a_dynworst, M1a_msdglue, M1a_pp_push2pull(+gs)
  train_full_msd  : PENDING (deferred to protect the in-flight seed-1/v50 deadline-critical runs)
                             -> M0_full, M1a_full
  finetune_msd_clamp: PENDING -> M0_ft, M1_ft
  RAMP_claimB     : PENDING (external) -> R', B1
Run `--list` to see per-arm readiness; arms whose trainer is PENDING are skipped with a note.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))
BF = ROOT / "results" / "backfill_logged"
UB = ROOT / "results" / "eval" / "union_bench"
C1K = "configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml"
C1K_C100 = "configs/eval/audit_cifar100_preactrn18_multinorm_v3A_test1k.yaml"
NOISE = 0.003

# arm -> where the ORIGINAL train.json + a canonical eval.json (for the orig union) live.
# trainer='c5' arms are WIRED; others carry trainer='train_full'/'finetune'/'ramp' = PENDING wiring.
ARMS = {
    "M0":   dict(trainer="c5", train_json="results/fromscratch/C5/M0_matched/seed0/train.json",  orig_eval="M0/eval.json"),
    "M1a":  dict(trainer="c5", train_json="results/fromscratch/C5/M1a_advneg/seed0/train.json",  orig_eval="M1a/eval.json"),
    "M1b":  dict(trainer="c5", train_json="results/fromscratch/C5/M1b_cleanneg/seed0/train.json", orig_eval="M1b/eval.json"),
    "M1a_seed2": dict(trainer="c5", train_json="results/fromscratch/C5/M1a_advneg/seed2/train.json", orig_eval="M1a_seed2/10k/eval.json"),
    "M0_seed2":  dict(trainer="c5", train_json="results/fromscratch/C5/M0_matched/seed2/train.json", orig_eval="M0_seed2/10k/eval.json"),
    "M1a_seed3": dict(trainer="c5", train_json="results/fromscratch/C5/M1a_advneg/seed3/train.json", orig_eval="M1a_seed3/10k/eval.json"),
    "M0_seed3":  dict(trainer="c5", train_json="results/fromscratch/C5/M0_matched/seed3/train.json", orig_eval="M0_seed3/10k/eval.json"),
    "M0_c100":   dict(trainer="c5", train_json="results/fromscratch/C5_c100/M0_c100/train.json",  orig_eval="M0_c100/eval.json"),
    "M1a_c100":  dict(trainer="c5", train_json="results/fromscratch/C5_c100/M1a_c100/train.json", orig_eval="M1a_c100/eval.json"),
    "M1a_dynworst":     dict(trainer="c5", train_json="results/fromscratch/C5/M1a_dynworst/seed0/train.json",     orig_eval="M1a_dynworst/eval.json"),
    "M1a_msdglue":      dict(trainer="c5", train_json="results/fromscratch/C5/M1a_msdglue/seed0/train.json",      orig_eval="M1a_msdglue/eval.json"),
    "M1a_pp_push2pull":    dict(trainer="c5", train_json="results/fromscratch/C5_fromscratch/M1a_pp_push2pull/train.json",    orig_eval="M1a_pp_push2pull/eval.json"),
    "M1a_pp_push2pull_gs": dict(trainer="c5", train_json="results/fromscratch/C5_fromscratch/M1a_pp_push2pull_gs/train.json", orig_eval="M1a_pp_push2pull_gs/eval.json"),
    # PENDING trainer wiring (skipped until their trainer logs per-norm train metrics):
    "M0_full":  dict(trainer="train_full", train_json="results/fromscratch/C5_full/M0_full/train.json",  orig_eval="M0_full/10k/eval.json"),
    "M1a_full": dict(trainer="train_full", train_json="results/fromscratch/C5_full/M1a_full/train.json", orig_eval="M1a_full/10k/eval.json"),
    "M0_ft":    dict(trainer="finetune", train_json="results/fromscratch/explore3_ftinf/none/train.json",  orig_eval="ft_none/10k/eval.json"),
    "M1_ft":    dict(trainer="finetune", train_json="results/fromscratch/explore3_ftinf/clamp/train.json", orig_eval="ft_clamp/10k/eval.json"),
    "Rprime":   dict(trainer="ramp", train_json=None, orig_eval="Rprime/10k/eval.json"),
    "B1":       dict(trainer="ramp", train_json=None, orig_eval="B1/10k/eval.json"),
}
WIRED = {"c5"}     # trainers that already log per-norm train metrics


def c5_cmd(cfg, outdir):
    """Reconstruct the c5_fromscratch launch from the ORIGINAL train.json (read, not guessed).
    Only ADD logging (default ON) — every hyperparam comes from the original."""
    f = ["python", "scripts/dev/c5_fromscratch.py",
         "--variant", cfg["variant"], "--seed", str(cfg["seed"]),
         "--epochs", str(cfg["epochs"]), "--outdir", outdir, "--wandb-mode", "offline"]
    for k, flag in [("dataset", "--dataset"), ("glue_view", "--glue-view"), ("base", "--base"),
                    ("neg_source", "--neg"), ("pp_schedule", "--pp-schedule")]:
        if cfg.get(k) not in (None, "cifar10" if k == "dataset" else None):
            f += [flag, str(cfg[k])]
    if cfg.get("pp_switch"): f += ["--pp-switch", str(cfg["pp_switch"])]
    if cfg.get("grad_surgery"): f += ["--grad-surgery"]
    return f


def orig_union(orig_eval):
    p = UB / orig_eval
    return json.load(open(p)).get("full_audit_union") if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually run (else dry preview only)")
    ap.add_argument("--force", action="store_true", help="skip the deadline-critical gate")
    ap.add_argument("--list", action="store_true", help="print per-arm readiness and exit")
    ap.add_argument("--only", nargs="*", help="restrict to these arm names")
    a = ap.parse_args()

    if a.list:
        for name, m in ARMS.items():
            tj = ROOT / m["train_json"] if m["train_json"] else None
            print(f"  {name:22} trainer={m['trainer']:11} "
                  f"{'WIRED' if m['trainer'] in WIRED else 'PENDING-wiring':16} "
                  f"orig_union={orig_union(m['orig_eval'])} "
                  f"train_json={'ok' if (tj and tj.exists()) else 'MISSING'}")
        return

    # GATE: deadline-critical done?
    dc = [UB / "M0_seed3/10k/eval.json", UB / "M0_full_seed1/eval.json", UB / "M1a_full_v50/eval.json"]
    if not a.force and not all(p.exists() for p in dc):
        print("GATED: deadline-critical runs not all done (seed-3 / seed-1 / v50). "
              "Re-run with --force to override.\n  missing: "
              + ", ".join(str(p.relative_to(UB)) for p in dc if not p.exists()))
        return

    BF.mkdir(parents=True, exist_ok=True)
    report = BF / "backfill_report.csv"
    rows = []
    names = a.only or list(ARMS)
    for name in names:
        m = ARMS[name]
        if m["trainer"] not in WIRED:
            print(f"[skip] {name}: trainer '{m['trainer']}' PENDING logging wiring"); continue
        tj = ROOT / m["train_json"]
        if not tj.exists():
            print(f"[skip] {name}: original train.json missing ({tj})"); continue
        cfg = json.load(open(tj))
        outdir = str(BF / name)
        cmd = c5_cmd(cfg, outdir)
        print(f"\n[{name}] {'RUN' if a.go else 'DRY'}: {' '.join(cmd)}")
        if not a.go:
            continue
        env = dict(os.environ, ATTACKDRO_ROOT=str(ROOT), C5_NUM_WORKERS="4")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        # eval @1k (full-12, robustdro) -> compare union to original
        ck = f"{outdir}/ckpt/val_best.pt"
        out1k = f"{outdir}/eval_1k.json"
        cfg1k = C1K_C100 if cfg.get("dataset") == "cifar100" else C1K
        subprocess.run(["python", "scripts/eval_multinorm_audit.py", "--config", cfg1k,
                        "--checkpoint", ck, "--model-family", "robustdro", "--run-id", f"{name}_bf_1k",
                        "--checkpoint-role", "backfill", "--out", out1k, "--export-masks", "--bs", "128"],
                       cwd=ROOT, env=env, check=True)
        new = json.load(open(out1k)).get("full_audit_union")
        orig = orig_union(m["orig_eval"])
        d = abs(new - orig) if (new is not None and orig is not None) else None
        verdict = "OK" if (d is not None and d < NOISE) else "INVESTIGATE"
        print(f"  backfill_check: {name} orig={orig} new={new:.4f} |Δ|={d:.4f} {verdict}")
        rows.append(dict(arm=name, orig_union=orig, new_union=round(new, 4),
                         abs_delta=round(d, 4) if d is not None else None, verdict=verdict,
                         note="paper numbers unchanged; backfill is archive-with-logging"))
    if rows:
        new_file = not report.exists()
        with open(report, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new_file: w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {report}")


if __name__ == "__main__":
    main()

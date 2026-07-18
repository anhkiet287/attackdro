"""Single-attack PROBE: run ONE named AutoAttack component and print robust accuracy.

This is the fast first step of the standing 3-step audit path (PROBE -> 1k full 12-AA ->
STOP for decision). It reuses the FROZEN harness verbatim -- same config load/validate,
same subset, same `run_attack_mask` -- and only restricts the attack set to a single
component (default apgd_ce_linf), so the number is directly comparable to that component
in a full audit's eval.json. It writes nothing by default; it is a read.

Usage (10k, robustdro family):
  ATTACKDRO_ROOT=/content/attackdro python scripts/dev/probe_attack.py \
      --config results/eval/union_bench/_config/audit_v3A_test_10k.yaml \
      --checkpoint /content/drive/MyDrive/attackdro/C5_full/M0_full/val_best.pt \
      --model-family robustdro --attack apgd_ce_linf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import eval_multinorm_audit as H  # noqa: E402
from robustdro.eval.eval_union import load_eval_checkpoint  # noqa: E402  (not re-exported by H)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-family", required=True, choices=["robustdro", "ramp"])
    p.add_argument("--attack", default="apgd_ce_linf")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--out", default=None, help="optional JSON to record the probe number")
    a = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = H.load_audit_config(a.config)
    H.validate_config(cfg)
    eval_cfg = H.to_eval_cfg(cfg)
    model, _ = load_eval_checkpoint(H.repo_path(a.checkpoint), eval_cfg,
                                    model_family=a.model_family, device=device)
    x, y, _, indices_sha = H.load_subset(cfg)

    att = next((at for at in H.iter_attacks(cfg) if at.get("name") == a.attack), None)
    if att is None:
        raise SystemExit(f"attack {a.attack!r} not in config; "
                         f"available: {[at['name'] for at in H.iter_attacks(cfg)]}")

    clean = H.clean_mask(model, x, y, device, a.bs)
    eps = float(cfg["eps"][att["eps_key"]])
    mask, qm, qx = H.run_attack_mask(model, x, y, att, eps, device, a.bs)
    clean_acc, robust = H.mean_mask(clean), H.mean_mask(mask)
    print(f"PROBE  attack={a.attack}  n={len(y)}  clean={clean_acc:.4f}  "
          f"robust_acc={robust:.4f}  eps={eps:g}", flush=True)

    if a.out:
        outp = H.repo_path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "attack": a.attack, "n": int(len(y)), "clean_acc": clean_acc,
            "robust_acc": robust, "eps": eps, "checkpoint": a.checkpoint,
            "model_family": a.model_family, "config": a.config,
            "indices_sha256": indices_sha, "note": "single-attack probe, not a full audit",
        }, indent=2))
        print(f"  -> {outp}", flush=True)


if __name__ == "__main__":
    main()

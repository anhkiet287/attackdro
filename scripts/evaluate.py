#!/usr/bin/env python
"""Evaluate a checkpoint's worst-case UNION robustness (linf, l2, l1).

The evaluation linchpin — works on both self-trained checkpoints and downloaded
baselines, always under the shared protocol (`configs/paper/base_ramp_apgd_8255.yaml`). Uses AutoAttack
per norm and the union rule.

Examples
--------
    # Quick eval on 1000 test images (APGD-CE + APGD-T), our PGD-AT checkpoint:
    python scripts/evaluate.py --checkpoint checkpoints/pgd_at_linf_best.pt -n 1000

    # Develop eval on the selection split:
    python scripts/evaluate.py --config configs/paper/base_ramp_apgd_8255.yaml \
        --checkpoint results/run/s0/ckpt/val_best.pt --eval-split val_select \
        --checkpoint-role val_best --version apgd -n 1000

    # Final numbers: full AutoAttack on the whole test set, separate W&B final run:
    python scripts/evaluate.py --checkpoint checkpoints/pgd_at_linf_best.pt \
        --eval-split test_final --version standard --n-examples 10000

    # Just linf (e.g. sanity vs the in-training PGD probe):
    python scripts/evaluate.py --checkpoint ckpt.pt --norms linf

    # Upstream RAMP checkpoint under our independent union harness:
    python scripts/evaluate.py --model_family ramp --config configs/paper/base_ramp_apgd_8255.yaml \
        --checkpoint external/RAMP/models/pretr_Linf.pth -n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robustdro.eval import evaluate_union, load_eval_checkpoint, load_eval_split  # noqa: E402
from robustdro.utils.io import apply_overrides, load_config, save_json            # noqa: E402
from robustdro.utils.seed import set_seed                     # noqa: E402
from robustdro.utils.wandb_log import log_eval_summary        # noqa: E402


def _checkpoint_label(path: str) -> str:
    return os.path.basename(path)


def _checkpoint_role(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in {"val_best", "last"}:
        return stem
    if stem == "best":
        return "val_best"
    if re.fullmatch(r"ep\d+", stem):
        return stem
    return stem or "checkpoint"


def _final_run_name(run_name: str, seed: int) -> str:
    if re.search(r"_s\d+$", run_name):
        return f"{run_name}_final_fullAA"
    return f"{run_name}_s{seed}_final_fullAA"


def _maybe_load_efficiency(out: str) -> float | None:
    """Attach the canonical train-time attack-FLOPs ratio to eval summaries when
    evaluating the standard per-run layout. Uses the final epoch value, falling
    back to method-specific legacy keys for older files."""
    train_json = os.path.join(os.path.dirname(out), "train.json")
    if not os.path.exists(train_json):
        return None
    try:
        with open(train_json, encoding="utf-8") as f:
            hist = (json.load(f).get("history") or [])
    except Exception:
        return None
    for row in reversed(hist):
        for key in ("efficiency/attack_flops_ratio", "pb/attack_flops_ratio", "exp/attack_flops_ratio"):
            if key in row and row[key] is not None:
                return float(row[key])
    return None


def _add_canonical_eval_keys(metrics: dict, cfg: dict, *, checkpoint: str, seed: int,
                             split: str, checkpoint_role: str,
                             used_for_selection: bool, final: bool,
                             n_examples: int, efficiency: float | None = None) -> dict:
    per_norm = metrics.get("per_norm_robust_acc", {}) or {}
    eval_attack = cfg.get("eval_attack", {}) or {}
    eval_grade = "full_autoattack_standard" if final and metrics.get("version") == "standard" else metrics.get("version")
    aliases = {
        "eval/checkpoint": _checkpoint_label(checkpoint),
        "eval/checkpoint_role": checkpoint_role,
        "eval/split": split,
        "eval/clean_acc": metrics.get("clean_acc"),
        "eval/worst_union": metrics.get("worst_union_acc"),
        "eval/selected_by_test": False,
        "eval/used_for_selection": bool(used_for_selection),
        "eval/attack_linf_steps": (eval_attack.get("linf", {}) or {}).get("steps"),
        "eval/attack_l2_steps": (eval_attack.get("l2", {}) or {}).get("steps"),
        "eval/attack_l1_steps": (eval_attack.get("l1", {}) or {}).get("steps"),
        "eval/seed": seed,
    }
    for norm in ("linf", "l2", "l1"):
        aliases[f"eval/robust_{norm}"] = per_norm.get(norm)
    if split in {"val_select", "test_monitor"} and checkpoint_role:
        prefix = f"eval/{split}/{checkpoint_role}"
        aliases.update({
            f"{prefix}/clean_acc": metrics.get("clean_acc"),
            f"{prefix}/worst_union": metrics.get("worst_union_acc"),
            f"{prefix}/n_examples": n_examples,
            f"{prefix}/used_for_selection": bool(used_for_selection),
        })
        for norm in ("linf", "l2", "l1"):
            aliases[f"{prefix}/robust_{norm}"] = per_norm.get(norm)
    if final:
        prefix = "final/test_final"
        aliases.update({
            f"{prefix}/clean_acc": metrics.get("clean_acc"),
            f"{prefix}/worst_union": metrics.get("worst_union_acc"),
            f"{prefix}/n_examples": n_examples,
            f"{prefix}/eval_grade": eval_grade,
            f"{prefix}/used_for_selection": False,
        })
        for norm in ("linf", "l2", "l1"):
            aliases[f"{prefix}/robust_{norm}"] = per_norm.get(norm)
    if efficiency is not None:
        aliases["efficiency/attack_flops_ratio"] = efficiency
    out = dict(metrics)
    out.update({k: v for k, v in aliases.items() if v is not None})
    return out


def _update_run_meta(out: str, cfg: dict, args, checkpoint_label: str,
                     checkpoint_role: str, used_for_selection: bool) -> None:
    seed_dir = os.path.dirname(out)
    if not os.path.basename(seed_dir).startswith("s"):
        return
    meta_path = os.path.join(os.path.dirname(seed_dir), "run_meta.json")
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    eval_attack = cfg.get("eval_attack", {}) or {}
    meta.update({
        "checkpoint_evaluated": checkpoint_label,
        "eval_checkpoint": checkpoint_label,
        "eval_checkpoint_role": checkpoint_role,
        "eval_split": args.eval_split,
        "eval_used_for_selection": bool(used_for_selection),
        "eval_selected_by_test": False,
        "eval_attack": {
            "version": args.version,
            "linf_steps": (eval_attack.get("linf", {}) or {}).get("steps"),
            "l2_steps": (eval_attack.get("l2", {}) or {}).get("steps"),
            "l1_steps": (eval_attack.get("l1", {}) or {}).get("steps"),
            "n_examples": args.n_examples,
        },
        "eval_seed": args.seed,
        "eval_out": out,
    })
    train_cfg = cfg.get("train", {}) or {}
    meta.setdefault("train_attack", {
        "attack": train_cfg.get("attack"),
        "steps": [{"norm": a.get("norm"), "steps": a.get("steps")}
                  for a in train_cfg.get("attacks", [])],
    })
    save_json(meta, meta_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper/base_ramp_apgd_8255.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-family", "--model_family", dest="model_family",
                   default="robustdro", choices=["robustdro", "ramp"],
                   help="Checkpoint/model format. 'robustdro' preserves existing behavior; "
                        "'ramp' loads external/RAMP/model_zoo.fast_models.PreActResNet18.")
    p.add_argument("-n", "--n-examples", type=int, default=None,
                   help="Number of images. Defaults: 1000 for val_select/test_monitor, "
                        "10000/full test for test_final.")
    p.add_argument("--norms", nargs="+", default=["linf", "l2", "l1"],
                   choices=["linf", "l2", "l1"])
    p.add_argument("--version", default="apgd", choices=["apgd", "standard"],
                   help="apgd = APGD-CE+APGD-T (fast); standard = full AutoAttack (final).")
    p.add_argument("--eval-split", choices=["val_select", "cal", "test_monitor", "test_final"],
                   default="test_monitor",
                   help="Eval role. val_select is held-out train selection split; "
                        "cal is the held-out calibration split (test 9000-9999, pre-reg "
                        "v3-A) — calibration/E_id only, never selection; test_monitor is "
                        "fixed 1k test monitor; test_final is test 1000-8999 (8k).")
    p.add_argument("--checkpoint-role", choices=["val_best", "last", "best", "external"], default=None,
                   help="Role label for nested W&B keys. Defaults from checkpoint filename.")
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSON output path. Default per-run outputs are split-specific, "
                        "e.g. eval_val_select_val_best.json; test_final uses eval_fullAA.json.")
    p.add_argument("--run-name", default=None,
                   help="Training W&B run to attach develop evals to. test_final derives a separate "
                        "<run>_s<seed>_final_fullAA run from this base name.")
    p.add_argument("--tier", default=None, choices=["in-house", "repro", "official"],
                   help="Comparison-lane tag for W&B (default: cfg wandb.tier; RAMP evals should pass 'repro').")
    p.add_argument("--no-wandb", action="store_true", help="Skip the W&B eval-summary log (JSON still written).")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Generic dotted-key config override, repeatable. "
             "e.g. --set threat_model.linf.eps=0.03137254901960784 "
             "--set eval_attack.linf.restarts=10",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    import yaml
    overrides = {}
    for kv in args.set:
        key, _, raw = kv.partition("=")
        if not _:
            raise ValueError(f"--set expects KEY=VALUE, got {kv!r}")
        overrides[key.strip()] = yaml.safe_load(raw)
    cfg = apply_overrides(cfg, overrides)
    if args.n_examples is None:
        args.n_examples = 10000 if args.eval_split == "test_final" else 1000
    checkpoint_role = args.checkpoint_role or _checkpoint_role(args.checkpoint)
    if checkpoint_role == "best":
        checkpoint_role = "val_best"
    final_eval = args.eval_split == "test_final"
    used_for_selection = args.eval_split == "val_select" and checkpoint_role == "val_best"
    device = args.device if torch.cuda.is_available() else "cpu"

    model, ckpt = load_eval_checkpoint(
        args.checkpoint, cfg, model_family=args.model_family, device=device
    )
    epoch = ckpt.get("epoch", "?") if ckpt is not None else "external"
    print(f"[eval] model_family={args.model_family}  checkpoint={args.checkpoint} "
          f"(epoch {epoch})  split={args.eval_split}  role={checkpoint_role}  device={device}")

    x, y = load_eval_split(cfg, split=args.eval_split, n_examples=args.n_examples)
    metrics = evaluate_union(model, x, y, cfg, norms=tuple(args.norms),
                             version=args.version, device=device, bs=args.bs, seed=args.seed)

    print("\n===== UNION ROBUSTNESS =====")
    print(f"clean          : {metrics['clean_acc']:.4f}")
    for k, v in metrics["per_norm_robust_acc"].items():
        print(f"robust {k:>4s}   : {v:.4f}")
    print(f"average-case   : {metrics['avg_robust_acc']:.4f}")
    print(f"WORST-UNION    : {metrics['worst_union_acc']:.4f}   <-- primary metric")

    if args.out:
        out = args.out
    else:
        # Default into the per-run layout: results/<run>/s<seed>/eval[_fullAA].json
        # (screening APGD -> eval.json; full AutoAttack -> eval_fullAA.json). Explicit
        # --out still wins (e.g. flat RAMP-reference evals).
        from robustdro.utils.io import run_paths as _rp
        _rn = args.run_name or re.sub(r"_(best|val_best|last|ep\d+)$", "",
                                      os.path.splitext(os.path.basename(args.checkpoint))[0])
        _p = _rp({"run_name": _rn, "seed": args.seed, "results_dir": "results/"})
        if final_eval:
            out = _p["eval_fullAA"]
        else:
            out = os.path.join(_p["seed_dir"], f"eval_{args.eval_split}_{checkpoint_role}.json")
    # Training protocol recorded FROM THE CHECKPOINT'S OWN cfg (if present), so
    # the views can auto-flag a train/eval eps MISMATCH (e.g. a 0.03-trained ckpt
    # evaluated at 8/255 is a lower bound, not a paper number). None for external
    # ckpts (ramp/official) that carry no cfg.
    train_tm = None
    if isinstance(ckpt, dict):
        train_tm = (ckpt.get("cfg") or {}).get("threat_model")
    train_eps = (train_tm or {}).get("linf", {}).get("eps") if train_tm else None
    eval_eps = cfg["threat_model"]["linf"]["eps"]
    mismatch = train_eps is not None and abs(float(train_eps) - float(eval_eps)) > 1e-6

    efficiency = _maybe_load_efficiency(out)
    metrics = _add_canonical_eval_keys(
        metrics, cfg, checkpoint=args.checkpoint, seed=args.seed, split=args.eval_split,
        checkpoint_role=checkpoint_role, used_for_selection=used_for_selection,
        final=final_eval, n_examples=args.n_examples, efficiency=efficiency
    )
    checkpoint_label = _checkpoint_label(args.checkpoint)

    save_json({
        "checkpoint": args.checkpoint,
        "checkpoint_evaluated": checkpoint_label,
        "checkpoint_role": checkpoint_role,
        "eval_split": args.eval_split,
        "used_for_selection": bool(used_for_selection),
        "selected_by_test": False,
        "model_family": args.model_family,
        "epoch": ckpt.get("epoch") if ckpt is not None else None,
        "n_examples": args.n_examples,
        "protocol": {"threat_model": cfg["threat_model"], "eval_attack": cfg.get("eval_attack")},
        "train_protocol": {"threat_model": train_tm} if train_tm else None,
        "train_eval_eps_mismatch": mismatch,
        "metrics": metrics,
    }, out)
    _update_run_meta(out, cfg, args, checkpoint_label, checkpoint_role, used_for_selection)
    print(f"\n[eval] saved -> {out}")
    if mismatch:
        print(f"[eval] ⚠ train/eval eps MISMATCH: trained@{float(train_eps):g}, "
              f"eval@{float(eval_eps):g} — this row is a LOWER BOUND, not a paper number.")

    # W&B view: attach union + per-norm to the run (JSON above is source of truth).
    if not args.no_wandb:
        base_run_name = args.run_name or re.sub(r"_(best|val_best|last)$", "",
                                           os.path.splitext(os.path.basename(args.checkpoint))[0])
        run_name = _final_run_name(base_run_name, args.seed) if final_eval else base_run_name
        tier = args.tier or ("repro" if args.model_family == "ramp" else None)
        log_eval_summary(cfg, run_name, metrics, version=args.version,
                         tier=tier, n_examples=args.n_examples,
                         checkpoint=checkpoint_label, seed=args.seed,
                         split=args.eval_split, checkpoint_role=checkpoint_role,
                         used_for_selection=used_for_selection, final=final_eval)


if __name__ == "__main__":
    main()

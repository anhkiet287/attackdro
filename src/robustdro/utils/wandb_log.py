"""Reusable Weights & Biases logging.

Design goals (per the author's request — "set it up so I can reuse it simply"):
  * One class, `WandbLogger`, used by every script the same way.
  * Fully config-driven: project / entity / mode come from cfg['wandb'].
  * For ordinary dev use, explicit `mode: disabled` still keeps training local.
    For controlled runs with `wandb.required: true`, online/offline init failure
    is fatal so a run cannot silently lose W&B metrics.
  * Three modes, controlled by cfg['wandb']['mode']:
      - "online"   : log to wandb.ai (requires `wandb login` once, see SETUP).
      - "offline"  : log locally to wandb/ ; sync later with `wandb sync`.
      - "disabled" : no wandb at all, metrics still print to stdout.
  * Env override: set WANDB_MODE=offline (or disabled) to force a mode without
    editing configs — handy for smoke tests and CI.

Usage
-----
    from robustdro.utils.wandb_log import WandbLogger
    logger = WandbLogger(cfg, run_name="pgd_at_linf")
    logger.log({"train/loss": 0.3, "val/robust_acc": 0.21}, step=epoch)
    logger.summary({"best/robust_acc": 0.47})
    logger.finish()

One-time login (persists to ~/.netrc, reused by every future run):
    .venv/bin/wandb login
"""

from __future__ import annotations

import os
import re
from typing import Any

try:
    import wandb

    _HAS_WANDB = True
except Exception:  # pragma: no cover - wandb optional
    wandb = None
    _HAS_WANDB = False

_PROJECT_DEFAULT = "attackdro-union"
_FINAL_GROUP = "ramp80_t49k_v1k_final"
_FINAL_TAGS = ["final", "test-final", "full-autoattack", "selected-winner"]


def _wandb_id(run_name: str) -> str:
    """Deterministic W&B run id from a run name (so a later eval can resume the
    SAME run and attach final-eval summary metrics next to the training curves)."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", run_name)[:64] or "run"


def _auto_tags(cfg: dict, run_name: str | None) -> list[str]:
    """Standardized tags for every run: method/recipe, seed, tier. Merged with
    any cfg['wandb']['tags']. Kept string-typed and de-duplicated."""
    wcfg = (cfg or {}).get("wandb", {}) or {}
    tags = list(wcfg.get("tags") or [])
    method = cfg.get("method")
    if method:
        tags.append(f"method:{method}")
    if run_name:
        tags.append(f"run:{run_name}")
    if "seed" in (cfg or {}):
        tags.append(f"seed:{cfg['seed']}")
    tier = wcfg.get("tier")
    if tier:
        tags.append(f"tier:{tier}")
    seen, out = set(), []
    for t in tags:
        t = str(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class WandbLogger:
    def __init__(self, cfg: dict, run_name: str | None = None):
        wcfg = (cfg or {}).get("wandb", {}) or {}
        # Env var wins over config so you can force a mode per-invocation.
        self.mode = os.environ.get("WANDB_MODE", wcfg.get("mode", "online"))
        self.required = bool(wcfg.get("required", False))
        self.run = None
        self._printed_header = False

        if not _HAS_WANDB:
            msg = "[wandb] package not installed"
            if self.required:
                raise RuntimeError(f"{msg}; wandb.required=true so this run must stop.")
            print(f"{msg} — logging to stdout only.")
            self.mode = "disabled"
            return
        if self.mode == "disabled":
            if self.required:
                raise RuntimeError("[wandb] mode=disabled but wandb.required=true; refusing to run.")
            print("[wandb] mode=disabled — logging to stdout only.")
            return

        name = run_name or cfg.get("run_name")
        try:
            self.run = wandb.init(
                project=wcfg.get("project", _PROJECT_DEFAULT),
                entity=wcfg.get("entity") or None,
                group=wcfg.get("group") or None,
                tags=_auto_tags(cfg, name) or None,
                name=name,
                id=_wandb_id(name) if name else None,   # deterministic -> eval can resume
                resume="allow",
                mode=self.mode,          # online | offline
                config=cfg,
            )
            print(f"[wandb] initialized (mode={self.mode}) project={wcfg.get('project', _PROJECT_DEFAULT)} "
                  f"run={self.run.name} tags={_auto_tags(cfg, name)}")
        except Exception as e:  # not logged in, offline network, etc.
            if self.required:
                raise RuntimeError(
                    f"[wandb] init failed in mode={self.mode}: {e}. "
                    "Use online W&B, or set wandb.mode=offline and sync later."
                ) from e
            print(f"[wandb] init failed ({e}); falling back to stdout only.")
            print("[wandb] tip: run `.venv/bin/wandb login` once to enable online logging.")
            self.run = None
            self.mode = "disabled"

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        # Always echo to stdout so tmux logs are self-contained.
        pretty = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in metrics.items())
        prefix = f"[step {step}] " if step is not None else ""
        print(prefix + pretty, flush=True)
        if self.run is not None:
            self.run.log(metrics, step=step)

    def summary(self, summary: dict[str, Any]) -> None:
        for k, v in summary.items():
            print(f"[summary] {k}={v}", flush=True)
        if self.run is not None:
            self.run.summary.update(summary)

    def watch(self, model) -> None:
        if self.run is not None:
            wandb.watch(model, log=None)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def log_eval_summary(cfg: dict, run_name: str, metrics: dict, *, version: str,
                     tier: str | None = None, n_examples: int | None = None,
                     checkpoint: str | None = None, seed: int | None = None,
                     split: str = "test_monitor", checkpoint_role: str | None = None,
                     used_for_selection: bool = False, final: bool = False) -> bool:
    """Attach a harness eval_union result to W&B as SUMMARY metrics on the run
    named `run_name` (resumes the training run via its deterministic id, so
    training curves AND final eval live in one place). Returns True iff it logged.
    For cfg['wandb']['required']=true, any online/offline init failure raises.
    Honors WANDB_MODE / cfg['wandb']['mode'].

    metrics is the eval_union dict: clean_acc, per_norm_robust_acc{linf,l2,l1},
    avg_robust_acc, worst_union_acc. JSON on disk stays the source of truth; this
    is only a view.
    """
    wcfg = (cfg or {}).get("wandb", {}) or {}
    mode = os.environ.get("WANDB_MODE", wcfg.get("mode", "online"))
    required = bool(wcfg.get("required", False))
    if not _HAS_WANDB:
        if required:
            raise RuntimeError("[wandb] package not installed; eval W&B logging is required.")
        return False
    if mode == "disabled":
        if required:
            raise RuntimeError("[wandb] mode=disabled but eval W&B logging is required.")
        return False
    tier = tier or wcfg.get("tier")
    tags = _auto_tags(cfg, run_name) + [f"eval:{version}"]
    if final:
        tags = tags + _FINAL_TAGS
    per_norm = metrics.get("per_norm_robust_acc", {}) or {}
    eval_grade = "full_autoattack_standard" if final and version == "standard" else version
    summary = {}
    if final:
        prefix = "final/test_final"
        summary.update({
            f"{prefix}/clean_acc": metrics.get("clean_acc"),
            f"{prefix}/worst_union": metrics.get("worst_union_acc"),
            f"{prefix}/n_examples": n_examples,
            f"{prefix}/eval_grade": eval_grade,
            f"{prefix}/used_for_selection": False,
        })
        for norm in ("linf", "l2", "l1"):
            if norm in per_norm:
                summary[f"{prefix}/robust_{norm}"] = per_norm[norm]
    else:
        summary.update({
            # Canonical eval summary keys.
            "eval/split": split,
            "eval/checkpoint": checkpoint,
            "eval/checkpoint_role": checkpoint_role,
            "eval/clean_acc": metrics.get("clean_acc"),
            "eval/worst_union": metrics.get("worst_union_acc"),
            "eval/selected_by_test": False,
            "eval/used_for_selection": bool(used_for_selection),
            "eval/attack_linf_steps": (cfg.get("eval_attack", {}).get("linf", {}) or {}).get("steps"),
            "eval/attack_l2_steps": (cfg.get("eval_attack", {}).get("l2", {}) or {}).get("steps"),
            "eval/attack_l1_steps": (cfg.get("eval_attack", {}).get("l1", {}) or {}).get("steps"),
            "eval/seed": seed,
            # Backward-compatible aliases used by older views.
            "eval/union": metrics.get("worst_union_acc"),
            "eval/clean": metrics.get("clean_acc"),
            "eval/avg": metrics.get("avg_robust_acc"),
            "eval/version": version,
            "eval/n": n_examples,
            "eval/tier": tier,
        })
        for norm in ("linf", "l2", "l1"):
            if norm in per_norm:
                summary[f"eval/robust_{norm}"] = per_norm[norm]
                summary[f"eval/{norm}"] = per_norm[norm]
        if split and checkpoint_role:
            prefix = f"eval/{split}/{checkpoint_role}"
            summary.update({
                f"{prefix}/clean_acc": metrics.get("clean_acc"),
                f"{prefix}/worst_union": metrics.get("worst_union_acc"),
                f"{prefix}/n_examples": n_examples,
                f"{prefix}/used_for_selection": bool(used_for_selection),
            })
            for norm in ("linf", "l2", "l1"):
                if norm in per_norm:
                    summary[f"{prefix}/robust_{norm}"] = per_norm[norm]
    if "efficiency/attack_flops_ratio" in metrics:
        summary["efficiency/attack_flops_ratio"] = metrics["efficiency/attack_flops_ratio"]
    try:
        group = _FINAL_GROUP if final else (wcfg.get("group") or None)
        run = wandb.init(project=wcfg.get("project", _PROJECT_DEFAULT),
                         entity=wcfg.get("entity") or None,
                         name=run_name, id=_wandb_id(run_name), resume="allow",
                         group=group, tags=tags, mode=mode,
                         config={"eval_only": True, "eval_split": split,
                                 "checkpoint_role": checkpoint_role, "final": final})
        run.summary.update({k: v for k, v in summary.items() if v is not None})
        run.finish()
        union_key = "final/test_final/worst_union" if final else "eval/worst_union"
        print(f"[wandb] eval summary logged to run={run_name} "
              f"(union={summary.get(union_key)}, version={version}, tier={tier}, split={split})")
        return True
    except Exception as e:  # pragma: no cover
        if required:
            raise RuntimeError(f"[wandb] eval summary log failed: {e}") from e
        print(f"[wandb] eval summary log failed ({e}); JSON on disk is unaffected.")
        return False

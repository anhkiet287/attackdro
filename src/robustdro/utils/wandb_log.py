"""Reusable Weights & Biases logging.

Design goals (per the author's request — "set it up so I can reuse it simply"):
  * One class, `WandbLogger`, used by every script the same way.
  * Fully config-driven: project / entity / mode come from cfg['wandb'].
  * Never crashes a run: if wandb is missing or you are not logged in, it
    degrades to stdout-only instead of raising.
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
        self.run = None
        self._printed_header = False

        if not _HAS_WANDB:
            print("[wandb] package not installed — logging to stdout only.")
            self.mode = "disabled"
            return
        if self.mode == "disabled":
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
                     tier: str | None = None, n_examples: int | None = None) -> bool:
    """Attach a harness eval_union result to W&B as SUMMARY metrics on the run
    named `run_name` (resumes the training run via its deterministic id, so
    training curves AND final eval live in one place). Best-effort: never raises,
    returns True iff it logged. Honors WANDB_MODE / cfg['wandb']['mode'].

    metrics is the eval_union dict: clean_acc, per_norm_robust_acc{linf,l2,l1},
    avg_robust_acc, worst_union_acc. JSON on disk stays the source of truth; this
    is only a view.
    """
    wcfg = (cfg or {}).get("wandb", {}) or {}
    mode = os.environ.get("WANDB_MODE", wcfg.get("mode", "online"))
    if not _HAS_WANDB or mode == "disabled":
        return False
    tier = tier or wcfg.get("tier")
    tags = _auto_tags(cfg, run_name) + [f"eval:{version}"]
    per_norm = metrics.get("per_norm_robust_acc", {}) or {}
    summary = {
        "eval/union": metrics.get("worst_union_acc"),
        "eval/clean": metrics.get("clean_acc"),
        "eval/avg": metrics.get("avg_robust_acc"),
        "eval/version": version,
        "eval/n": n_examples,
        "eval/tier": tier,
    }
    for norm in ("linf", "l2", "l1"):
        if norm in per_norm:
            summary[f"eval/{norm}"] = per_norm[norm]
    try:
        run = wandb.init(project=wcfg.get("project", _PROJECT_DEFAULT),
                         entity=wcfg.get("entity") or None,
                         name=run_name, id=_wandb_id(run_name), resume="allow",
                         tags=tags, mode=mode, config={"eval_only": True})
        run.summary.update({k: v for k, v in summary.items() if v is not None})
        run.finish()
        print(f"[wandb] eval summary logged to run={run_name} "
              f"(union={summary['eval/union']}, version={version}, tier={tier})")
        return True
    except Exception as e:  # pragma: no cover
        print(f"[wandb] eval summary log failed ({e}); JSON on disk is unaffected.")
        return False

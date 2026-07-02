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
from typing import Any

try:
    import wandb

    _HAS_WANDB = True
except Exception:  # pragma: no cover - wandb optional
    wandb = None
    _HAS_WANDB = False


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

        try:
            self.run = wandb.init(
                project=wcfg.get("project", "union-robustness-dro"),
                entity=wcfg.get("entity") or None,
                group=wcfg.get("group") or None,
                tags=wcfg.get("tags") or None,
                name=run_name or cfg.get("run_name"),
                mode=self.mode,          # online | offline
                config=cfg,
            )
            print(f"[wandb] initialized (mode={self.mode}) run={self.run.name}")
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

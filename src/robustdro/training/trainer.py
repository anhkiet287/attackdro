"""PGD-AT trainer — the minimal runnable training path for P0.

Method = adversarial training with PGD-linf (Madry et al.). The clean/robust
accuracies reported here are cheap in-training probes; the authoritative numbers
come later from eval_union with a strong attack (golden rule #4).

The loop is intentionally small and config-driven. Later methods (AVG/MAX/MSD,
Group-DRO, AttackDRO++) will subclass or reuse this trainer with a different
adversary / loss.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from ..attacks import pgd_linf
from ..data import build_loaders
from ..models import build_model
from ..utils.io import save_json
from ..utils.wandb_log import WandbLogger


class Trainer:
    def __init__(self, cfg: dict, download_data: bool = True, loaders=None):
        self.cfg = cfg
        self.device = cfg["device"] if torch.cuda.is_available() else "cpu"
        if self.device != cfg["device"]:
            print(f"[trainer] CUDA unavailable — falling back to {self.device}")

        # `loaders` lets callers (e.g. a synthetic smoke test) inject their own
        # (train_loader, test_loader) instead of the real dataset.
        if loaders is not None:
            self.train_loader, self.test_loader = loaders
        else:
            self.train_loader, self.test_loader = build_loaders(cfg, download=download_data)
        self.model = build_model(cfg).to(self.device)

        tcfg = cfg["train"]
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=tcfg["lr"],
            momentum=tcfg.get("momentum", 0.9),
            weight_decay=tcfg.get("weight_decay", 5e-4),
        )
        self.epochs = tcfg["epochs"]
        if tcfg.get("lr_schedule") == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=tcfg.get("milestones", [50, 75]), gamma=0.1
            )
        else:
            self.scheduler = None

        # Threat model eps comes from the shared protocol (base.yaml), NOT the
        # method config — golden rule #1.
        self.eps = cfg["threat_model"]["linf"]["eps"]
        atk = tcfg["attack"]
        self.train_steps = atk["steps"]
        self.train_step_size = atk["step_size"]
        self.random_start = atk.get("random_start", True)
        self.eval_pgd_steps = tcfg.get("eval_pgd_steps", 20)

        self.logger = WandbLogger(cfg, run_name=cfg.get("run_name"))
        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_dir = cfg.get("checkpoints_dir", "checkpoints/")
        self.results_dir = cfg.get("results_dir", "results/")
        self.best_robust = -1.0

    # --------------------------------------------------------------------- #
    def _attack(self, x, y, steps):
        return pgd_linf(
            self.model, x, y, eps=self.eps, step_size=self.train_step_size,
            steps=steps, random_start=self.random_start,
        )

    def train_epoch(self, epoch: int, max_steps: int | None = None) -> dict:
        self.model.train()
        t0 = time.time()
        running_loss = correct = total = 0
        for i, (x, y) in enumerate(self.train_loader):
            if max_steps is not None and i >= max_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            x_adv = self._attack(x, y, self.train_steps)

            self.model.train()
            logits = self.model(x_adv)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return {
            "train/loss": running_loss / max(total, 1),
            "train/adv_acc": correct / max(total, 1),
            "train/epoch_time_s": time.time() - t0,
        }

    @torch.no_grad()
    def _clean_acc(self, max_batches: int | None = None) -> float:
        self.model.eval()
        correct = total = 0
        for i, (x, y) in enumerate(self.test_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            correct += (self.model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    def _robust_acc(self, max_batches: int | None = None) -> float:
        """Cheap PGD-linf robustness probe on the test set (NOT final eval)."""
        self.model.eval()
        correct = total = 0
        for i, (x, y) in enumerate(self.test_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            x_adv = self._attack(x, y, self.eval_pgd_steps)
            with torch.no_grad():
                correct += (self.model(x_adv).argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    def evaluate(self, max_batches: int | None = None) -> dict:
        return {
            "val/clean_acc": self._clean_acc(max_batches),
            "val/robust_acc_pgd": self._robust_acc(max_batches),
        }

    # --------------------------------------------------------------------- #
    def save_checkpoint(self, tag: str, extra: dict | None = None) -> str:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"{self.cfg.get('run_name', 'run')}_{tag}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "cfg": self.cfg,
            **(extra or {}),
        }, path)
        return path

    def fit(self, max_steps_per_epoch: int | None = None,
            eval_max_batches: int | None = None) -> dict:
        history = []
        for epoch in range(self.epochs):
            tr = self.train_epoch(epoch, max_steps=max_steps_per_epoch)
            ev = self.evaluate(max_batches=eval_max_batches)
            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            metrics = {**tr, **ev, "lr": lr, "epoch": epoch}
            self.logger.log(metrics, step=epoch)
            history.append(metrics)

            rob = ev["val/robust_acc_pgd"]
            if rob > self.best_robust:
                self.best_robust = rob
                self.save_checkpoint("best", {"epoch": epoch, "robust_acc_pgd": rob})

        self.save_checkpoint("last", {"epoch": self.epochs - 1})
        self.logger.summary({"best/robust_acc_pgd": self.best_robust})

        # Persist metrics as JSON for the Research Hub exporter (see CLAUDE.md).
        os.makedirs(self.results_dir, exist_ok=True)
        out = os.path.join(self.results_dir, f"{self.cfg.get('run_name', 'run')}.json")
        save_json({"cfg": self.cfg, "history": history,
                   "best_robust_acc_pgd": self.best_robust}, out)
        self.logger.finish()
        return {"best_robust_acc_pgd": self.best_robust, "results_json": out}

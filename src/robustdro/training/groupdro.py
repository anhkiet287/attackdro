"""AttackDRO++ union trainer — GroupDRO over (linf, l2, l1) source attacks.

Each optimizer step applies all K source attacks to the batch, computes a
per-group CE loss, updates the difficulty-aware weights q (GroupDRO), and steps
on the q-weighted sum. This is the P1 probe: does difficulty-aware DRO over the
union (now WITH l1, which the thesis omitted) move worst-case union robustness?

The eps for every attack comes from the locked protocol (base.threat_model), so
train and eval share one threat model.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from ..attacks import pgd_l1_topk, pgd_l2, pgd_linf
from ..attacks.sources import build_source_attack
from ..data import build_loaders
from ..models import build_model
from ..utils.io import save_json
from ..utils.wandb_log import WandbLogger
from .losses import GroupDRO

# Cheap in-training robustness probes (NOT the final eval — that is eval_union).
_PROBE = {"linf": pgd_linf, "l2": pgd_l2, "l1": pgd_l1_topk}


class GroupDROTrainer:
    def __init__(self, cfg: dict, download_data: bool = True, loaders=None):
        self.cfg = cfg
        self.device = cfg["device"] if torch.cuda.is_available() else "cpu"
        if self.device != cfg["device"]:
            print(f"[trainer] CUDA unavailable — falling back to {self.device}")

        if loaders is not None:
            self.train_loader, self.test_loader = loaders
        else:
            self.train_loader, self.test_loader = build_loaders(cfg, download=download_data)
        self.model = build_model(cfg).to(self.device)

        tcfg = cfg["train"]
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=tcfg["lr"],
            momentum=tcfg.get("momentum", 0.9), weight_decay=tcfg.get("weight_decay", 5e-4),
        )
        self.epochs = tcfg["epochs"]
        if tcfg.get("lr_schedule") == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=tcfg.get("milestones", [25, 40]), gamma=0.1)
        else:
            self.scheduler = None

        # Source attacks (groups). eps from the locked protocol per norm.
        tm = cfg["threat_model"]
        self.attack_specs = tcfg["attacks"]
        self.group_norms = [s["norm"].lower() for s in self.attack_specs]
        self.attacks = [build_source_attack(s, eps=tm[s["norm"].lower()]["eps"])
                        for s in self.attack_specs]
        self.num_groups = len(self.attacks)

        gcfg = tcfg.get("groupdro", {})
        self.warmup_epochs = gcfg.get("warmup_epochs", 0)
        self.dro = GroupDRO(self.num_groups, eta_q=gcfg.get("eta_q", 0.02), device=self.device)
        # Weighting variants (each is a 1-component change vs the P1 AttackDRO++):
        #   freeze_q=True            -> CARD-1 avg_frozen: q fixed uniform (DRO off).
        #   weight_signal=robust_acc -> CARD-3a bindaware: q = softmax(-robust_acc_g / tau)
        #                               set once per epoch from the cheap probe (weakest
        #                               norm gets most weight), REPLACING the per-step loss update.
        #   weight_signal=loss (default) -> P1 GroupDRO: per-step multiplicative loss update.
        self.freeze_q = gcfg.get("freeze_q", False)
        self.weight_signal = gcfg.get("weight_signal", "loss")
        self.tau = gcfg.get("tau", 0.1)
        # CARD-3b: objective = per_sample_soft -> per-sample temperature-softmax
        # over per-norm losses (T->inf = AVG, T->0 = hard per-sample MAX; keep soft).
        # normalize_losses = zscore is the DECLARED-EXCEPTION variant (amendment #1),
        # only to be used if the pilot's loss distributions show cross-norm scale bias.
        self.objective = gcfg.get("objective", "group_dro")
        self.temperature = gcfg.get("temperature", 1.0)
        self.normalize_losses = gcfg.get("normalize_losses", "none")  # none | zscore
        # CARD-3a-v2: weight_signal = val_apgd -> calibrate q with a reduced
        # APGD-CE on a held-out val split every val_every epochs (EMA-smoothed).
        # Fixes the probe's ranking inversion (probe overstates l1 by ~12pp).
        self.val_every = gcfg.get("val_every", 5)
        self.val_iters = gcfg.get("val_iters", 30)
        self.ema_beta = gcfg.get("ema_beta", 0.5)
        self._val_data = None
        self._val_acc_ema = None
        if self.weight_signal == "val_apgd":
            from ..data.datasets import get_train_holdout
            self._val_data = get_train_holdout(cfg)
            print(f"[groupdro] val_apgd calibration: {self._val_data[0].shape[0]} held-out "
                  f"train imgs, APGD-CE {self.val_iters} it every {self.val_every} epochs, "
                  f"ema_beta={self.ema_beta}")

        self.criterion = nn.CrossEntropyLoss()
        self.logger = WandbLogger(cfg, run_name=cfg.get("run_name"))
        self.ckpt_dir = cfg.get("checkpoints_dir", "checkpoints/")
        self.results_dir = cfg.get("results_dir", "results/")
        self.probe_steps = tcfg.get("eval_pgd_steps", 20)
        self.probe_batches = tcfg.get("eval_probe_batches", 8)  # cheap subset
        self.best_union = -1.0
        eps_str = ", ".join(f"{n}:{tm[n]['eps']:g}" for n in self.group_norms)
        print(f"[groupdro] groups={self.group_norms}  eta_q={self.dro.eta_q}  eps=({eps_str})  "
              f"objective={self.objective}  weight_signal={self.weight_signal}  "
              f"freeze_q={self.freeze_q}  tau={self.tau}  T={self.temperature}  "
              f"norm_losses={self.normalize_losses}")

    # --------------------------------------------------------------------- #
    def train_epoch(self, epoch: int, max_steps=None) -> dict:
        self.model.train()
        t0 = time.time()
        n = 0
        group_loss_sum = torch.zeros(self.num_groups)
        correct = torch.zeros(self.num_groups)
        q_sum = torch.zeros(self.num_groups)
        n_batches = 0
        self._epoch_loss_samples = []
        do_update = epoch >= self.warmup_epochs
        for i, (x, y) in enumerate(self.train_loader):
            if max_steps is not None and i >= max_steps:
                break
            x, y = x.to(self.device), y.to(self.device)

            per_sample = []                            # list of [B] loss vectors
            for g, atk in enumerate(self.attacks):
                x_adv = atk(self.model, x, y)          # crafted in eval mode internally
                self.model.train()
                logits = self.model(x_adv)             # train-mode forward (BN updates)
                loss_vec = nn.functional.cross_entropy(logits, y, reduction="none")
                per_sample.append(loss_vec)
                group_loss_sum[g] += loss_vec.sum().item()
                correct[g] += (logits.argmax(1) == y).sum().item()

            L = torch.stack(per_sample)                 # [G, B], with grad
            # Amendment #1 (CARD-3b): keep per-sample losses for the epoch-level
            # distribution log (scale-bias check across norms).
            self._epoch_loss_samples.append(L.detach().cpu())

            if self.objective == "per_sample_soft":
                # CARD-3b: per-sample temperature-softmax over norms.
                Lw = L.detach()
                if self.normalize_losses == "zscore":   # declared-exception variant
                    Lw = (Lw - Lw.mean(dim=1, keepdim=True)) / Lw.std(dim=1, keepdim=True).clamp(min=1e-6)
                w = torch.softmax(Lw / self.temperature, dim=0)   # [G,B], detached
                total_loss = (w * L).sum(dim=0).mean()
                q = w.mean(dim=1)                       # avg weight per norm (for logging)
            else:
                losses = L.mean(dim=1)                  # [G] group means, with grad
                # Per-step loss update ONLY in the default GroupDRO mode. freeze_q and
                # the robust_acc signal set q elsewhere (uniform / once-per-epoch).
                if self.weight_signal == "loss" and not self.freeze_q and do_update:
                    q = self.dro.update(losses)
                else:
                    q = self.dro.weights()
                total_loss = (q.detach() * losses).sum()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()
            n += y.size(0)
            q_sum += q.detach().cpu()
            n_batches += 1

        if self.objective == "per_sample_soft":
            q = q_sum / max(n_batches, 1)     # epoch-mean per-norm softmax weight
        else:
            q = self.dro.weights().detach().cpu()
        metrics = {
            "train/loss": (group_loss_sum.sum().item() / max(self.num_groups * n, 1)),
            "train/epoch_time_s": time.time() - t0,
        }
        # Amendment #1 (CARD-3b): per-norm per-sample loss distribution -> scale-bias check.
        all_L = torch.cat(self._epoch_loss_samples, dim=1) if self._epoch_loss_samples else None
        self._epoch_loss_samples = []
        for g, norm in enumerate(self.group_norms):
            metrics[f"train/adv_acc_{norm}"] = correct[g].item() / max(n, 1)
            metrics[f"train/loss_{norm}"] = group_loss_sum[g].item() / max(n, 1)
            metrics[f"q/{norm}"] = q[g].item()
            if all_L is not None:
                v = all_L[g]
                p10, p50, p90 = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9])).tolist()
                metrics[f"dist/{norm}/std"] = v.std().item()
                metrics[f"dist/{norm}/p10"] = p10
                metrics[f"dist/{norm}/p50"] = p50
                metrics[f"dist/{norm}/p90"] = p90
        return metrics

    def _probe_norm(self, norm: str) -> torch.Tensor:
        """Per-sample robust mask under a cheap PGD probe on a test subset."""
        eps = self.cfg["threat_model"][norm]["eps"]
        atk = _PROBE[norm]
        masks = []
        self.model.eval()
        for i, (x, y) in enumerate(self.test_loader):
            if i >= self.probe_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            if norm == "linf":
                x_adv = atk(self.model, x, y, eps=eps, step_size=eps / 4, steps=self.probe_steps)
            elif norm == "l2":
                x_adv = atk(self.model, x, y, eps=eps, step_size=eps / 4, steps=self.probe_steps)
            else:
                x_adv = atk(self.model, x, y, eps=eps, step_size=0.05, steps=self.probe_steps)
            with torch.no_grad():
                masks.append((self.model(x_adv).argmax(1) == y).cpu())
        return torch.cat(masks)

    def evaluate(self) -> dict:
        # Cheap union probe over all three norms on the same subset.
        per_norm = {n: self._probe_norm(n) for n in ("linf", "l2", "l1")}
        union = per_norm["linf"] & per_norm["l2"] & per_norm["l1"]
        out = {f"probe/robust_{n}": m.float().mean().item() for n, m in per_norm.items()}
        out["probe/worst_union"] = union.float().mean().item()
        return out

    def _val_calibrate(self, epoch: int) -> dict:
        """CARD-3a-v2: per-norm robust acc on the held-out val split via reduced
        APGD-CE, EMA-smoothed, -> q = softmax(-acc_ema / tau)."""
        from ..eval import robust_mask  # lazy: pulls in autoattack
        xv, yv = self._val_data
        accs = []
        for norm in self.group_norms:
            eps = self.cfg["threat_model"][norm]["eps"]
            mask, _ = robust_mask(self.model, xv, yv, norm, eps,
                                  steps=self.val_iters, restarts=1,
                                  device=self.device, version="apgd-ce",
                                  seed=self.cfg.get("seed", 0), bs=500)
            accs.append(mask.float().mean().item())
        acc = torch.tensor(accs, dtype=torch.float32)
        if self._val_acc_ema is None:
            self._val_acc_ema = acc
        else:
            self._val_acc_ema = self.ema_beta * acc + (1 - self.ema_beta) * self._val_acc_ema
        self.dro.set_scores(-self._val_acc_ema / self.tau)
        self.model.train()
        out = {}
        for g, norm in enumerate(self.group_norms):
            out[f"val_calib/robust_{norm}"] = accs[g]
            out[f"val_calib/ema_{norm}"] = self._val_acc_ema[g].item()
        return out

    def save_checkpoint(self, tag: str, extra=None) -> str:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"{self.cfg.get('run_name', 'run')}_{tag}.pt")
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg,
                    "dro": self.dro.state_dict(), **(extra or {})}, path)
        return path

    def fit(self, max_steps_per_epoch=None, **_) -> dict:
        history = []
        for epoch in range(self.epochs):
            tr = self.train_epoch(epoch, max_steps=max_steps_per_epoch)
            ev = self.evaluate()
            # CARD-3a: set next epoch's q from THIS epoch's per-norm probe robust-acc
            # (binding-aware: weakest norm -> most weight). Epoch 0 trains uniform.
            if self.weight_signal == "robust_acc" and not self.freeze_q:
                racc = torch.tensor([ev[f"probe/robust_{n}"] for n in self.group_norms],
                                    dtype=torch.float32)
                self.dro.set_scores(-racc / self.tau)
            # CARD-3a-v2: q from reduced APGD-CE on the held-out val split (EMA).
            elif (self.weight_signal == "val_apgd" and not self.freeze_q
                  and epoch % self.val_every == 0):
                ev.update(self._val_calibrate(epoch))
            if self.scheduler is not None:
                self.scheduler.step()
            metrics = {**tr, **ev, "lr": self.optimizer.param_groups[0]["lr"], "epoch": epoch}
            self.logger.log(metrics, step=epoch)
            history.append(metrics)

            if ev["probe/worst_union"] > self.best_union:
                self.best_union = ev["probe/worst_union"]
                self.save_checkpoint("best", {"epoch": epoch, **ev})

        self.save_checkpoint("last", {"epoch": self.epochs - 1})
        self.logger.summary({"best/probe_worst_union": self.best_union})
        os.makedirs(self.results_dir, exist_ok=True)
        out = os.path.join(self.results_dir, f"{self.cfg.get('run_name', 'run')}.json")
        save_json({"cfg": self.cfg, "history": history,
                   "best_probe_worst_union": self.best_union}, out)
        self.logger.finish()
        return {"best_probe_worst_union": self.best_union, "results_json": out}

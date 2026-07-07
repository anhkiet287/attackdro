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
from ..attacks.norms import msd_v0
from ..attacks.sources import build_source_attack
from ..data import build_loaders
from ..models import build_model
from ..utils.io import save_json, run_paths, write_run_meta
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
        # Training inner-max attack class: 'apgd' (RAMP-matched: momentum+adaptive+best-iterate,
        # incl. APGD-l1) or 'pgd' (plain fixed-step). Reactive AND predictive craft with this;
        # for predictive the floor attacks use it too (see _rebuild_floor).
        self.train_attack = tcfg.get("attack", "pgd")
        self.group_norms = [s["norm"].lower() for s in self.attack_specs]
        self.attacks = [build_source_attack(s, eps=tm[s["norm"].lower()]["eps"], attack=self.train_attack)
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

        # Gate-alpha instrumentation: a fixed, indexed train probe batch for
        # per-sample binding persistence. Off by default; unlike loss_mats/, this
        # uses the same sample indices every dump and no train augmentation.
        self.probe_binding = bool(gcfg.get("probe_binding", False))
        self.probe_binding_size = int(gcfg.get("probe_binding_size", 512))
        self.probe_binding_every = max(1, int(gcfg.get("probe_binding_every", 1)))
        self.probe_binding_batch_size = int(
            gcfg.get("probe_binding_batch_size", cfg["train"].get("batch_size", 128))
        )
        self.probe_binding_dir = gcfg.get(
            "probe_binding_dir",
            os.path.join("dumps", "probe_binding", cfg.get("run_name", "run")),
        )
        self._probe_binding_data = None
        if self.probe_binding:
            from ..data.datasets import get_train_probe_batch
            probe_seed = int(gcfg.get("probe_binding_seed", cfg.get("seed", 0) + 1729))
            self._probe_binding_data = get_train_probe_batch(
                cfg, size=self.probe_binding_size, seed=probe_seed, download=False,
            )
            print(f"[groupdro] fixed probe_binding: n={self._probe_binding_data[0].shape[0]} "
                  f"every={self.probe_binding_every} dir={self.probe_binding_dir}")

        # CARD-PB predictive-binding: EMA predictor φ of per-sample binding, budget
        # allocation (full attack on predicted norm, floor on the rest), 4-layer
        # safety floor, and MEASURED attack-FLOPs (forward+backward passes counted
        # from the real per-batch partitions, not a fixed K-step estimate).
        if self.objective == "predictive_binding":
            self.pb_beta = float(gcfg.get("pb_beta", 0.5))
            self.pb_k_floor = int(gcfg.get("pb_k_floor", 3))
            self.pb_recal_rho = float(gcfg.get("pb_recal_rho", 0.1))
            self.pb_cold_epochs = int(gcfg.get("pb_cold_epochs", 5))
            self.pb_recal_every = int(gcfg.get("pb_recal_every", 0))  # >0 => full-reactive epoch every N
            self.pb_guard_pp = float(gcfg.get("pb_guard_pp", 3.0))
            # CARD-PB v2 (Option B): predictability-aware floor. In "confidence" mode the
            # per-norm floor is floor_g = clamp(k_min + round(kspan*miss_ema_g), k_min,
            # full_g), miss_g = 1 - recall_g measured UNBIASED on the recal subset (NOT the
            # probe — the l1 probe is the weakest attack, F1). "fixed" = the v1 constant.
            self.pb_floor_mode = gcfg.get("pb_floor_mode", "fixed")   # fixed | confidence
            self.pb_floor_kspan = int(gcfg.get("pb_floor_kspan", 12))
            self.pb_floor_ema = float(gcfg.get("pb_floor_ema", 0.5))  # smooth the ~12-sample/batch miss (R5)
            # Adaptive probe-driven guard: kept for fixed mode (v1) but OFF in confidence
            # mode by default — it double-counts with the confidence floor and is probe-blind
            # on l1 (F1), which inflated FLOPs in sweep-1 (linf pushed to full). Overridable.
            self.pb_adaptive = bool(gcfg.get("pb_adaptive_floor", self.pb_floor_mode != "confidence"))
            self._full_steps = [int(s.get("steps", 10)) for s in self.attack_specs]
            self._pb_extra_floor = [0 for _ in self.attack_specs]   # adaptive-guard bumps, ON TOP of the base floor
            self._pb_conf_floor = [self.pb_k_floor for _ in self.attack_specs]  # confidence base (>= k_min)
            self._pb_miss_ema = [None for _ in self.attack_specs]   # per-norm 1-recall EMA
            # R1: floor attacks are (re)built from the CURRENT effective step count so a
            # raised floor runs MORE steps, not just logs a higher number.
            self.floor_attacks = [None] * len(self.attack_specs)
            for _g in range(len(self.attack_specs)):
                self._rebuild_floor(_g)
            self._pb_norm_max = [0.0 for _ in self.attack_specs]    # running max probe acc per norm
            self._pb_wrap_train_loader()
            N = len(self.train_loader.dataset)
            self.pb_Lbar = torch.zeros(N, self.num_groups, device=self.device)
            self.pb_seen = torch.zeros(N, dtype=torch.bool, device=self.device)
            print(f"[cardpb] predictive-binding: beta={self.pb_beta} k_floor={self.pb_k_floor} "
                  f"recal_rho={self.pb_recal_rho} cold_epochs={self.pb_cold_epochs} "
                  f"T={self.temperature} N={N} full_steps={self._full_steps}")

        self.criterion = nn.CrossEntropyLoss()
        self.logger = WandbLogger(cfg, run_name=cfg.get("run_name"))
        # Per-run results layout: results/<run>/s<seed>/{train,smoke}.json + ckpt/ .
        # Root = cfg.results_dir (default results/; Colab overrides to Drive). Checkpoints
        # nest under the run (checkpoints_dir is no longer used).
        self.paths = run_paths(cfg)
        self.ckpt_dir = self.paths["ckpt_dir"]
        self.results_dir = cfg.get("results_dir", "results/")
        self.is_smoke = str(cfg.get("run_name", "")).endswith("_smoke")
        self.save_freq = int(tcfg.get("save_freq", 0))   # >0 => checkpoint every N epochs (curves)
        self.start_epoch = 0
        _resume = tcfg.get("resume")                     # ckpt path, or "auto" (=<ckpt_dir>/last.pt)
        if _resume:
            self._resume_from(_resume)
        self.probe_steps = tcfg.get("eval_pgd_steps", 20)
        self.probe_batches = tcfg.get("eval_probe_batches", 8)  # cheap subset
        self.best_union = -1.0
        eps_str = ", ".join(f"{n}:{tm[n]['eps']:g}" for n in self.group_norms)
        print(f"[groupdro] groups={self.group_norms}  eta_q={self.dro.eta_q}  eps=({eps_str})  "
              f"objective={self.objective}  weight_signal={self.weight_signal}  "
              f"freeze_q={self.freeze_q}  tau={self.tau}  T={self.temperature}  "
              f"norm_losses={self.normalize_losses}")

    # ------------------------- CARD-PB predictive-binding ----------------- #
    def _floor_steps(self, g: int) -> int:
        """Effective floor steps for norm g = base (confidence floor in v2, else k_min)
        + adaptive-guard bump, capped at that norm's full attack. SINGLE source of truth
        for the actual attack (R1), the FLOPs count, and the logged floor."""
        base = self._pb_conf_floor[g] if self.pb_floor_mode == "confidence" else self.pb_k_floor
        return int(min(base + self._pb_extra_floor[g], self._full_steps[g]))

    def _rebuild_floor(self, g: int) -> None:
        """(Re)build norm g's floor attack so it runs _floor_steps(g) steps. R1 fix: the
        floor must change the ACTUAL attack, not just the logged step count — otherwise a
        'raised' floor never attacks harder and the guarantee is vacuous. Called at init
        and whenever the floor changes (confidence recompute or adaptive guard)."""
        spec = {**self.attack_specs[g], "steps": max(1, self._floor_steps(g))}
        self.floor_attacks[g] = build_source_attack(
            spec, eps=self.cfg["threat_model"][spec["norm"].lower()]["eps"], attack=self.train_attack)

    def _recompute_conf_floor(self) -> None:
        """v2 Option B: set each norm's confidence floor from its measured miss rate
        (1 - recall on the recal subset): floor_g = clamp(k_min + round(kspan*miss_ema_g),
        k_min, full_g). Norms φ predicts poorly (l1) get a higher floor automatically;
        norms it nails (linf) or that few samples bind (l2) stay near k_min. Rebuilds the
        floor attack so the change is physical (R1). No-op outside confidence mode."""
        if self.pb_floor_mode != "confidence":
            return
        for g in range(len(self.attack_specs)):
            m = self._pb_miss_ema[g]
            if m is None:
                continue
            cf = self.pb_k_floor + int(round(self.pb_floor_kspan * m))
            cf = max(self.pb_k_floor, min(cf, self._full_steps[g]))
            if cf != self._pb_conf_floor[g]:
                self._pb_conf_floor[g] = cf
                self._rebuild_floor(g)

    def _pb_wrap_train_loader(self) -> None:
        """Rebuild the train loader to yield (x, y, idx) so φ can keep a per-sample
        EMA of binding across epochs (idx = position into the train dataset)."""
        base = self.train_loader.dataset

        class _Indexed(torch.utils.data.Dataset):
            def __init__(s, d):
                s.d = d

            def __len__(s):
                return len(s.d)

            def __getitem__(s, i):
                xy = s.d[i]
                return xy[0], xy[1], i

        tcfg = self.cfg["train"]
        dcfg = self.cfg["dataset"]
        g = torch.Generator().manual_seed(self.cfg.get("seed", 0))
        self.train_loader = torch.utils.data.DataLoader(
            _Indexed(base), batch_size=tcfg["batch_size"], shuffle=True,
            num_workers=dcfg.get("num_workers", 4), pin_memory=True,
            drop_last=True, generator=g,
        )

    def _pb_step(self, x, y, idx, epoch):
        """One predictive-binding optimizer step. Returns (group_loss[G], correct[G],
        w_mean[G]) and accumulates pb metrics on self._pb_*."""
        B, G = y.size(0), self.num_groups
        dev = self.device
        Lbar_b = self.pb_Lbar[idx]                       # [B,G]
        seen = self.pb_seen[idx]                         # [B]
        b_hat = Lbar_b.argmax(dim=1)                     # [B] predicted binding norm
        # Cold start (early epochs or unseen sample) + recalibration subset get the
        # FULL reactive attack in every norm (unbiased signal; seeds/corrects φ).
        cold = (epoch < self.pb_cold_epochs) | (~seen)
        if self.pb_recal_every and (epoch % self.pb_recal_every == 0):
            recal = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            recal = torch.rand(B, device=dev) < self.pb_recal_rho
        full_all = cold | recal                          # [B] full attack in ALL norms

        per_sample, correct = [], torch.zeros(G)
        attack_passes = 0
        for g, (atk_full, atk_floor) in enumerate(zip(self.attacks, self.floor_attacks)):
            full_m = (b_hat == g) | full_all             # full budget in norm g
            floor_m = ~full_m                            # floor budget in norm g
            xg = x.clone()
            if full_m.any():
                xg[full_m] = atk_full(self.model, x[full_m], y[full_m])
                attack_passes += int(full_m.sum()) * self._full_steps[g]
            if floor_m.any():
                xg[floor_m] = atk_floor(self.model, x[floor_m], y[floor_m])
                attack_passes += int(floor_m.sum()) * self._floor_steps(g)
            self.model.train()
            logits = self.model(xg)                      # train-mode forward (BN)
            per_sample.append(nn.functional.cross_entropy(logits, y, reduction="none"))
            correct[g] = (logits.argmax(1) == y).sum().item()

        L = torch.stack(per_sample)                      # [G,B] with grad
        w = torch.softmax(L.detach() / self.temperature, dim=0)   # [G,B] detached
        total_loss = (w * L).sum(dim=0).mean()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        # φ EMA update from the crafted per-norm losses (first sighting sets directly).
        # R3: only norms that received FULL budget for a sample carry an unbiased loss;
        # floored norms are under-attacked (loss artificially low) and would drag φ's
        # estimate away from them → self-reinforcing misprediction (a likely F9 driver).
        # Update ONLY full-budget entries; leave floored entries at their last unbiased
        # (cold/recal) value. First sighting is always cold (full in all norms, since
        # cold = epoch<cold | ~seen), so every sample is seeded unbiased across all norms.
        newL = L.detach().t()                            # [B,G]
        full_mask = full_all.unsqueeze(1) | (
            b_hat.unsqueeze(1) == torch.arange(self.num_groups, device=dev).unsqueeze(0))  # [B,G]
        ema = self.pb_beta * newL + (1 - self.pb_beta) * Lbar_b
        updated = torch.where(seen.unsqueeze(1), ema, newL)
        self.pb_Lbar[idx] = torch.where(full_mask, updated, Lbar_b)
        self.pb_seen[idx] = True

        # Misprediction on the recalibration subset (unbiased: full in all norms). Aggregate
        # rate p AND per-norm miss = 1 - recall_g = P(b_hat != g | true == g), which drives
        # the v2 confidence floor (norms φ misses most get more floor).
        if recal.any():
            true_bind = newL[recal].argmax(dim=1)
            bh = b_hat[recal]
            self._pb_mis_num += int((true_bind != bh).sum())
            self._pb_mis_den += int(recal.sum())
            for g in range(G):
                is_g = (true_bind == g)
                self._pb_missd[g] += int(is_g.sum())
                self._pb_missn[g] += int((is_g & (bh != g)).sum())
        # MEASURED FLOPs: attack forward+backward passes from the REAL partitions, vs
        # the reactive-equivalent (all norms full every step).
        self._pb_attack_passes += attack_passes
        self._pb_reactive_passes += sum(self._full_steps) * B
        return L.detach().sum(dim=1).cpu(), correct, w.mean(dim=1).detach().cpu()

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
        self._pb_attack_passes = self._pb_reactive_passes = 0
        self._pb_mis_num = self._pb_mis_den = 0
        self._pb_missn = [0 for _ in range(self.num_groups)]
        self._pb_missd = [0 for _ in range(self.num_groups)]
        for i, batch in enumerate(self.train_loader):
            if max_steps is not None and i >= max_steps:
                break

            if self.objective == "predictive_binding":
                x, y, idx = batch
                x, y, idx = x.to(self.device), y.to(self.device), idx.to(self.device)
                gl, cor, wm = self._pb_step(x, y, idx, epoch)
                group_loss_sum += gl
                correct += cor
                q_sum += wm
                n += y.size(0)
                n_batches += 1
                continue

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            if self.objective == "msd":
                # CARD-7: MSD in-house — faithful msd_v0 port (robust_union
                # commit ef34194), OUR recipe/protocol. Single adversarial
                # example per sample; standard CE training on it.
                tm = self.cfg["threat_model"]
                x_adv = msd_v0(self.model, x, y, eps_linf=tm["linf"]["eps"],
                               eps_l2=tm["l2"]["eps"], eps_l1=tm["l1"]["eps"],
                               steps=self.cfg["train"]["groupdro"].get("msd_steps", 50))
                self.model.train()
                logits = self.model(x_adv)
                loss_vec = nn.functional.cross_entropy(logits, y, reduction="none")
                total_loss = loss_vec.mean()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()
                group_loss_sum += loss_vec.sum().item() / self.num_groups  # shared log
                correct += (logits.argmax(1) == y).sum().item() / self.num_groups
                q_sum += torch.full((self.num_groups,), 1 / self.num_groups)
                n += y.size(0); n_batches += 1
                continue

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
            if i == 0:
                # Kiet 2026-07-03: dump one real [G,B] loss matrix per epoch so the
                # per-sample temperature-dial figure needs no "illustrative" label.
                self._first_batch_L = L.detach().cpu()

            if self.objective == "per_sample_max":
                # MAX in-house anchor (Tramèr-style hard per-sample worst norm),
                # under OUR recipe/attacks — the clean T->0 anchor for the T-axis.
                # NOTE vs pure Tramèr MAX: all 3 attack forwards still update BN
                # (identical pipeline to per_sample_soft; ONLY aggregation differs).
                total_loss = L.max(dim=0).values.mean()
                q = torch.zeros(self.num_groups, device=L.device).scatter_add_(
                    0, L.argmax(dim=0), torch.full((L.shape[1],), 1.0 / L.shape[1],
                                                   device=L.device))  # selection freq, logging
            elif self.objective == "per_sample_soft":
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

        if self.objective.startswith("per_sample") or self.objective == "predictive_binding":
            q = q_sum / max(n_batches, 1)     # epoch-mean weight / selection freq
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
        # CARD-PB: MEASURED attack-FLOPs ratio (vs reactive all-norms-full) + φ error.
        if self.objective == "predictive_binding":
            metrics["pb/attack_flops_ratio"] = (
                self._pb_attack_passes / max(self._pb_reactive_passes, 1))
            metrics["pb/attack_passes"] = float(self._pb_attack_passes)
            if self._pb_mis_den:
                metrics["phi/misprediction_rate"] = self._pb_mis_num / self._pb_mis_den
            metrics["pb/beta"] = self.pb_beta        # EMA horizon (comparability across β-ablation)
            metrics["pb/floor_mode"] = self.pb_floor_mode
            metrics["pb/floor_kspan"] = self.pb_floor_kspan
            for g, norm in enumerate(self.group_norms):
                # v2 FIX (sweep-1 was confounded): drive the floor by miss VOLUME =
                # P(true=g AND b_hat!=g) — the joint, POPULATION-aware probability over ALL
                # recal samples — not 1-recall (conditional). A rarely-bound norm (l2, free
                # per F3) then can't get a high floor just because its few true samples are
                # mispredicted; only norms that are BOTH frequently bound AND often missed
                # (l1) earn a high floor. EMA-smoothed (R5). 1-recall still logged for diag.
                if self._pb_mis_den:
                    vol = self._pb_missn[g] / self._pb_mis_den
                    self._pb_miss_ema[g] = (vol if self._pb_miss_ema[g] is None
                                            else self.pb_floor_ema * vol
                                            + (1 - self.pb_floor_ema) * self._pb_miss_ema[g])
                if self._pb_miss_ema[g] is not None:
                    metrics[f"phi/miss_vol_{norm}"] = self._pb_miss_ema[g]     # the floor DRIVER
                if self._pb_missd[g]:
                    metrics[f"phi/miss_rate_{norm}"] = self._pb_missn[g] / self._pb_missd[g]  # 1-recall (diag)
                metrics[f"floor/{norm}"] = self._floor_steps(g)
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

    def _clean_acc(self) -> float:
        """Clean accuracy on the same cheap test subset (standardized W&B key)."""
        self.model.eval()
        correct = total = 0
        for i, (x, y) in enumerate(self.test_loader):
            if i >= self.probe_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                correct += (self.model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    def evaluate(self) -> dict:
        # Cheap union probe over all three norms on the same subset.
        per_norm = {n: self._probe_norm(n) for n in ("linf", "l2", "l1")}
        union = per_norm["linf"] & per_norm["l2"] & per_norm["l1"]
        out = {f"probe/robust_{n}": m.float().mean().item() for n, m in per_norm.items()}
        out["probe/worst_union"] = union.float().mean().item()
        out["probe/clean_acc"] = self._clean_acc()
        return out

    def _dump_probe_binding(self, epoch: int) -> dict:
        if not self.probe_binding or epoch % self.probe_binding_every != 0:
            return {}
        if self._probe_binding_data is None:
            return {}

        xs, ys, indices = self._probe_binding_data
        was_training = self.model.training
        self.model.eval()
        chunks = [[] for _ in self.group_norms]
        bs = max(1, self.probe_binding_batch_size)
        fork_devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        for start in range(0, xs.shape[0], bs):
            xb = xs[start:start + bs].to(self.device)
            yb = ys[start:start + bs].to(self.device)
            for g, atk in enumerate(self.attacks):
                # Fix stochastic random starts/top-k draws across epochs for the
                # same sample chunk, so persistence reflects model state changes.
                seed = int(self.cfg.get("seed", 0)) + 100_003 + 997 * g + start
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(seed)
                    if self.device.startswith("cuda"):
                        torch.cuda.manual_seed_all(seed)
                    x_adv = atk(self.model, xb, yb)
                with torch.no_grad():
                    logits = self.model(x_adv)
                    loss = nn.functional.cross_entropy(logits, yb, reduction="none")
                chunks[g].append(loss.detach().cpu())
        if was_training:
            self.model.train()

        L = torch.stack([torch.cat(v, dim=0) for v in chunks], dim=0)
        bind = L.argmax(dim=0)
        freq = torch.bincount(bind, minlength=len(self.group_norms)).float() / bind.numel()
        top2 = L.topk(2, dim=0).values
        margin = (top2[0] - top2[1]).mean().item()

        os.makedirs(self.probe_binding_dir, exist_ok=True)
        path = os.path.join(self.probe_binding_dir, f"probe_binding_ep{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "run_name": self.cfg.get("run_name", "run"),
            "norms": self.group_norms,
            "indices": indices.cpu(),
            "labels": ys.cpu(),
            "L": L,
            "meta": {
                "source": "cifar10_train_totensor_no_aug",
                "probe_binding_size": xs.shape[0],
                "probe_binding_every": self.probe_binding_every,
                "probe_binding_batch_size": bs,
                "attack_specs": self.attack_specs,
            },
        }, path)

        out = {"probe_binding/mean_margin": margin}
        for g, norm in enumerate(self.group_norms):
            out[f"probe_binding/freq_{norm}"] = freq[g].item()
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
        """Resume-capable checkpoint into results/<run>/s<seed>/ckpt/<tag>.pt — carries
        optimizer + scheduler + epoch + RNG so PHASE-2 can continue the winner on the same
        trajectory (item 1)."""
        import numpy as _np
        import random as _rnd
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        rng = {"torch": torch.get_rng_state(),
               "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
               "numpy": _np.random.get_state(), "python": _rnd.getstate()}
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg,
                    "dro": self.dro.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
                    "rng": rng, **(extra or {})}, path)
        return path

    def _resume_from(self, resume) -> None:
        """Restore model + optimizer + scheduler + epoch + RNG for a valid continuation
        (item 1). resume='auto' -> <ckpt_dir>/last.pt. Missing ckpt -> start fresh."""
        path = os.path.join(self.ckpt_dir, "last.pt") if resume == "auto" else resume
        if not os.path.exists(path):
            print(f"[resume] no checkpoint at {path}; starting fresh")
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler is not None and ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        if "dro" in ck:
            try:
                self.dro.load_state_dict(ck["dro"])
            except Exception:
                pass
        self.start_epoch = int(ck.get("epoch", -1)) + 1
        rng = ck.get("rng")
        if rng:
            try:
                import numpy as _np
                import random as _rnd
                torch.set_rng_state(rng["torch"].to("cpu", torch.uint8))   # map_location may move it to CUDA
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([s.to("cpu", torch.uint8) for s in rng["cuda"]])
                _np.random.set_state(rng["numpy"])
                _rnd.setstate(rng["python"])
            except Exception as e:
                print(f"[resume] RNG restore partial ({e})")
        print(f"[resume] {path} -> start_epoch={self.start_epoch} "
              f"lr={self.optimizer.param_groups[0]['lr']:.5f}")

    def _pb_adaptive_floor(self, ev: dict) -> None:
        """Safety-valve: if any norm's probe robust acc falls > pb_guard_pp below its
        own running max, raise that norm's floor budget (so a wrong φ can't quietly
        erode that norm). Self-referential guard; the exact >2pp-vs-reactive-control
        check is done post-hoc against the paired reactive run."""
        for g, norm in enumerate(self.group_norms):
            acc = ev.get(f"probe/robust_{norm}")
            if acc is None:
                continue
            self._pb_norm_max[g] = max(self._pb_norm_max[g], acc)
            if acc < self._pb_norm_max[g] - self.pb_guard_pp / 100.0:
                bump = min(self._pb_extra_floor[g] + 2, self._full_steps[g])
                if bump != self._pb_extra_floor[g]:
                    self._pb_extra_floor[g] = bump
                    self._rebuild_floor(g)   # R1: the raised floor now actually attacks more
                    print(f"[cardpb] adaptive floor: {norm} probe {acc:.3f} dropped "
                          f">{self.pb_guard_pp}pp vs max {self._pb_norm_max[g]:.3f} "
                          f"-> floor = {self._floor_steps(g)} steps")

    def fit(self, max_steps_per_epoch=None, **_) -> dict:
        # On resume, continue the epoch curve from the existing train.json (item 1).
        history = []
        if self.start_epoch > 0 and os.path.exists(self.paths["train"]):
            try:
                from ..utils.io import load_json as _lj
                history = _lj(self.paths["train"]).get("history", [])
            except Exception:
                history = []
        if not self.is_smoke:
            write_run_meta(self.cfg, self.paths,
                           extra={"resumed_from_epoch": self.start_epoch or None})
        # Provenance: the eps this model is TRAINED at + whether that differs from
        # the canonical eval protocol (8/255). Logged once as run summary so every
        # row in the W&B table self-declares train==eval vs a mismatch (a 0.03-ckpt
        # re-eval'd at 8/255 is a lower-bound estimate, not paper-grade).
        _eps_inf = float(self.cfg["threat_model"]["linf"]["eps"])
        _mismatch = abs(_eps_inf - 8 / 255) > 1e-6
        self.logger.summary({
            "train/protocol": ("8/255" if not _mismatch else f"eps_inf={_eps_inf:.5f}"),
            "train/eps_inf": _eps_inf,
            "train/eps_mismatch": int(_mismatch),
        })
        for epoch in range(self.start_epoch, self.epochs):
            tr = self.train_epoch(epoch, max_steps=max_steps_per_epoch)
            ev = self.evaluate()
            if self.objective == "predictive_binding":
                self._recompute_conf_floor()       # v2: set next epoch's floor from measured miss
                if self.pb_adaptive:
                    self._pb_adaptive_floor(ev)     # probe-driven bump (fixed mode only by default)
            pb = self._dump_probe_binding(epoch)
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
            metrics = {**tr, **ev, **pb, "lr": self.optimizer.param_groups[0]["lr"], "epoch": epoch}
            self.logger.log(metrics, step=epoch)
            history.append(metrics)
            # Per-sample loss-matrix dump (first batch of the epoch), ~1.5 KB each.
            if getattr(self, "_first_batch_L", None) is not None and not self.is_smoke:
                d = os.path.join(self.paths["seed_dir"], "loss_mats")
                os.makedirs(d, exist_ok=True)
                torch.save({"epoch": epoch, "norms": self.group_norms,
                            "L": self._first_batch_L}, os.path.join(d, f"ep{epoch:03d}.pt"))

            if not self.is_smoke and ev["probe/worst_union"] > self.best_union:
                self.best_union = ev["probe/worst_union"]
                self.save_checkpoint("best", {"epoch": epoch, **ev})
            # save_freq: periodic checkpoints (ep010.pt ... ep080.pt) for epoch curves +
            # PHASE-2 continuation from a fixed epoch (item 1). 1-indexed to match RAMP.
            if (not self.is_smoke and self.save_freq
                    and (epoch + 1) % self.save_freq == 0):
                self.save_checkpoint(f"ep{epoch + 1:03d}", {"epoch": epoch})
            # Incrementally persist the curve so a disconnect keeps completed epochs.
            if not self.is_smoke:
                save_json({"cfg": self.cfg, "history": history,
                           "best_probe_worst_union": self.best_union}, self.paths["train"])

        if not self.is_smoke:
            self.save_checkpoint("last", {"epoch": self.epochs - 1})
        self.logger.summary({"best/probe_worst_union": self.best_union})
        out = self.paths["smoke"] if self.is_smoke else self.paths["train"]
        save_json({"cfg": self.cfg, "history": history,
                   "best_probe_worst_union": self.best_union}, out)
        self.logger.finish()
        return {"best_probe_worst_union": self.best_union, "results_json": out}

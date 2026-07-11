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
from ..data import NON_SELECTION_SPLITS, build_loaders, get_eval_split
from ..models import build_model
from ..utils.io import save_json, run_paths, write_run_meta
from ..utils.wandb_log import WandbLogger
from .losses import GroupDRO

# Cheap in-training robustness probes (NOT the final eval — that is eval_union).
_PROBE = {"linf": pgd_linf, "l2": pgd_l2, "l1": pgd_l1_topk}

_FULL_ALLOCATION_ALIASES = {"full", "reactive_full", "all_source", "all_sources"}
_ALLOCATION_MODES = {*_FULL_ALLOCATION_ALIASES, "static_cycle", "predictive_refresh"}

# Pre-reg v3-A gatekeeper: checkpoint selection uses this metric ONLY.
# cal / test_monitor / test_final must never drive ckpt/val_best.pt.
SELECTION_METRIC_KEY = "val_select/worst_union"


def resolve_train_allocation_mode(train_cfg: dict) -> str:
    """Canonicalize the train-time source allocation mode.

    Existing configs do not set this key; they keep the original all-source
    behavior. ``static_cycle`` is the B2 one-source-per-batch baseline.
    """
    group_cfg = train_cfg.get("groupdro", {}) or {}
    raw = train_cfg.get("allocation_mode", group_cfg.get("allocation_mode", "full"))
    mode = str(raw).lower()
    if mode not in _ALLOCATION_MODES:
        raise ValueError(
            f"Unknown train allocation_mode {raw!r}. "
            "Expected one of: full, reactive_full, all_source, static_cycle, predictive_refresh"
        )
    return "full" if mode in _FULL_ALLOCATION_ALIASES else mode


def resolve_static_cycle_order(train_cfg: dict, group_norms: list[str]) -> list[str]:
    order = train_cfg.get("static_cycle_order", train_cfg.get("source_cycle_order", group_norms))
    order = [str(n).lower() for n in order]
    missing = sorted(set(group_norms) - set(order))
    extra = sorted(set(order) - set(group_norms))
    if missing or extra:
        raise ValueError(
            "static_cycle_order must contain only configured sources and cover every source; "
            f"group_norms={group_norms}, order={order}, missing={missing}, extra={extra}"
        )
    return order


def resolve_static_extra_config(train_cfg: dict, group_norms: list[str]) -> dict:
    if bool(train_cfg.get("static_extra_update", False)):
        raise ValueError(
            "static_extra_update is invalid for B3 equal-budget semantics; "
            "use static_extra_mode: aggregate_loss instead"
        )
    source = train_cfg.get("static_extra_source")
    every_raw = train_cfg.get("static_extra_every_n_batches", 0)
    mode = train_cfg.get("static_extra_mode")
    primary_weight_raw = train_cfg.get("static_extra_primary_weight", 1.0)
    source_weight_raw = train_cfg.get("static_extra_source_weight", 0.0)
    every = int(every_raw or 0)
    primary_weight = float(primary_weight_raw)
    source_weight = float(source_weight_raw)

    if source is None and every == 0 and mode is None:
        return {
            "enabled": False,
            "source": None,
            "every_n_batches": 0,
            "mode": None,
            "primary_weight": 1.0,
            "source_weight": 0.0,
        }
    if mode != "aggregate_loss":
        raise ValueError("static_extra_mode must be 'aggregate_loss' when static extra fields are set")
    if source is None:
        raise ValueError("static_extra_source is required when static extra aggregate-loss fields are set")
    source = str(source).lower()
    if source not in group_norms:
        raise ValueError(f"static_extra_source must be one of {group_norms}, got {source!r}")
    if every <= 0:
        raise ValueError("static_extra_every_n_batches must be positive when static extra aggregate-loss is set")
    if abs((primary_weight + source_weight) - 1.0) > 1e-12:
        raise ValueError("static_extra_primary_weight + static_extra_source_weight must equal 1.0")
    if primary_weight < 0.0 or source_weight < 0.0:
        raise ValueError("static extra aggregate-loss weights must be nonnegative")
    return {
        "enabled": True,
        "source": source,
        "every_n_batches": every,
        "mode": mode,
        "primary_weight": primary_weight,
        "source_weight": source_weight,
    }


def resolve_predictive_refresh_config(train_cfg: dict, group_norms: list[str]) -> dict:
    """Validate B4 Route A-Diagnostic predictive-refresh settings.

    This is intentionally a separate allocation mode from legacy CARD-PB
    ``predictive_binding`` so historical predictive runs remain untouched.
    """
    cfg = {
        "refresh_every_n_batches": int(train_cfg.get("predictive_refresh_every_n_batches", 10)),
        "starvation_floor_batches": int(train_cfg.get("predictive_starvation_floor_batches", 30)),
        "ema_alpha": float(train_cfg.get("predictive_ema_alpha", 0.9)),
        "use_cheap_probe": bool(train_cfg.get("predictive_use_cheap_probe", False)),
        "score_rule": str(train_cfg.get("predictive_score_rule", "loss_plus_bind")),
        "log_batch_trace": bool(train_cfg.get("predictive_log_batch_trace", True)),
    }
    if cfg["refresh_every_n_batches"] <= 0:
        raise ValueError("predictive_refresh_every_n_batches must be positive")
    if cfg["starvation_floor_batches"] <= 0:
        raise ValueError("predictive_starvation_floor_batches must be positive")
    if not 0.0 <= cfg["ema_alpha"] < 1.0:
        raise ValueError("predictive_ema_alpha must be in [0, 1)")
    if cfg["use_cheap_probe"]:
        raise ValueError("Route A-Diagnostic requires predictive_use_cheap_probe: false")
    if cfg["score_rule"] != "loss_plus_bind":
        raise ValueError("Route A-Diagnostic requires predictive_score_rule: loss_plus_bind")
    if len(group_norms) != 3 or sorted(group_norms) != ["l1", "l2", "linf"]:
        raise ValueError(f"predictive_refresh expects linf/l2/l1 sources, got {group_norms}")
    return cfg


def _normalized_scores(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.float().std(unbiased=False)
    if float(std) < 1e-12:
        return torch.zeros_like(values)
    return (values.float() - values.float().mean()) / std.clamp(min=1e-12)


def _predictive_score_from_state(ema_loss: torch.Tensor, ema_bind: torch.Tensor) -> torch.Tensor:
    return _normalized_scores(ema_loss) + ema_bind.float()


def predictive_refresh_dry_run(
    group_norms: list[str],
    full_attack_steps: list[int],
    num_batches: int,
    *,
    refresh_every_n_batches: int = 10,
    starvation_floor_batches: int = 30,
    ema_alpha: float = 0.9,
) -> dict:
    """Deterministic, no-attack dry run for B4 Route A accounting.

    The fake refresh losses deliberately make linf the dominant source after the
    first refresh, mirroring the diagnostic's bottleneck setting while exercising
    score, trace, refresh, and checkpoint-state accounting without GPU work.
    """
    if num_batches < 0:
        raise ValueError("num_batches must be nonnegative")
    if refresh_every_n_batches <= 0:
        raise ValueError("refresh_every_n_batches must be positive")
    if starvation_floor_batches <= 0:
        raise ValueError("starvation_floor_batches must be positive")
    G = len(group_norms)
    if G != len(full_attack_steps):
        raise ValueError("group_norms and full_attack_steps length mismatch")
    ema_loss = torch.zeros(G, dtype=torch.float32)
    ema_bind = torch.zeros(G, dtype=torch.float32)
    initialized = False
    last_selected = [-10**9 for _ in range(G)]
    source_counts = {n: 0 for n in group_norms}
    selected_counts = {n: 0 for n in group_norms}
    attack_step_units_by_source = {n: 0 for n in group_norms}
    refresh_indices: list[int] = []
    trace = []
    prediction_correct_worst = 0
    prediction_correct_binding = 0
    prediction_den = 0
    floor_overrides = 0

    for b in range(num_batches):
        scores_before = (
            _predictive_score_from_state(ema_loss, ema_bind)
            if initialized else torch.zeros(G)
        )
        pred_before = int(scores_before.argmax().item()) if initialized else None
        is_refresh = (not initialized) or (b % refresh_every_n_batches == 0)
        if is_refresh:
            refresh_indices.append(b)
            # Fixed fake refresh losses: linf is worst, l1 second, l2 easiest.
            obs_loss = torch.tensor([1.30, 0.70, 1.00], dtype=torch.float32)[:G]
            bind_freq = torch.tensor([0.60, 0.15, 0.25], dtype=torch.float32)[:G]
            worst_g = int(obs_loss.argmax().item())
            bind_g = int(bind_freq.argmax().item())
            if pred_before is not None:
                prediction_den += 1
                prediction_correct_worst += int(pred_before == worst_g)
                prediction_correct_binding += int(pred_before == bind_g)
            if initialized:
                ema_loss = ema_alpha * ema_loss + (1.0 - ema_alpha) * obs_loss
                ema_bind = ema_alpha * ema_bind + (1.0 - ema_alpha) * bind_freq
            else:
                ema_loss = obs_loss.clone()
                ema_bind = bind_freq.clone()
                initialized = True
            selected_g = worst_g
            for g, norm in enumerate(group_norms):
                source_counts[norm] += 1
                attack_step_units_by_source[norm] += int(full_attack_steps[g])
                last_selected[g] = b
            selected_counts[group_norms[selected_g]] += 1
            floor_override = False
        else:
            overdue = [
                (b - last_selected[g], g)
                for g in range(G)
                if b - last_selected[g] >= starvation_floor_batches
            ]
            if overdue:
                age, selected_g = max(overdue, key=lambda t: (t[0], -t[1]))
                floor_override = selected_g != pred_before
                floor_overrides += int(floor_override)
            else:
                selected_g = int(scores_before.argmax().item())
                floor_override = False
            norm = group_norms[selected_g]
            source_counts[norm] += 1
            selected_counts[norm] += 1
            attack_step_units_by_source[norm] += int(full_attack_steps[selected_g])
            last_selected[selected_g] = b

        scores_after = _predictive_score_from_state(ema_loss, ema_bind)
        trace.append({
            "batch_index": b,
            "refresh": bool(is_refresh),
            "selected_source": group_norms[selected_g],
            "predicted_source": (group_norms[pred_before] if pred_before is not None else None),
            "scores": {n: float(scores_after[g]) for g, n in enumerate(group_norms)},
            "ema_loss": {n: float(ema_loss[g]) for g, n in enumerate(group_norms)},
            "ema_bind": {n: float(ema_bind[g]) for g, n in enumerate(group_norms)},
            "source_age": {n: int(b - last_selected[g]) for g, n in enumerate(group_norms)},
            "floor_override": bool(floor_override),
        })

    total_units = sum(attack_step_units_by_source.values())
    b1_reference = int(num_batches) * sum(int(s) for s in full_attack_steps)
    return {
        "refresh_indices": refresh_indices,
        "refresh_count": len(refresh_indices),
        "non_refresh_count": int(num_batches) - len(refresh_indices),
        "source_counts": source_counts,
        "selected_source_counts": selected_counts,
        "attack_step_units_by_source": attack_step_units_by_source,
        "attack_step_units": total_units,
        "b1_all_source_reference_units": b1_reference,
        "efficiency_ratio": total_units / max(b1_reference, 1),
        "training_batches": int(num_batches),
        "optimizer_step_calls": int(num_batches),
        "backward_calls": int(num_batches),
        "floor_override_count": floor_overrides,
        "prediction_accuracy_worst": prediction_correct_worst / max(prediction_den, 1),
        "prediction_accuracy_binding": prediction_correct_binding / max(prediction_den, 1),
        "prediction_denominator": prediction_den,
        "batch_trace": trace,
        "predictor_state": {
            "ema_loss": ema_loss.tolist(),
            "ema_bind": ema_bind.tolist(),
            "last_selected_batch": list(last_selected),
            "initialized": bool(initialized),
            "global_batch_index": int(num_batches - 1) if num_batches else -1,
        },
        "cheap_probe_used": False,
        "validation_or_test_state_accessed": False,
        "extra_optimizer_step_from_starvation_floor": False,
    }


def static_cycle_dry_run(
    group_norms: list[str],
    full_attack_steps: list[int],
    cycle_order: list[str],
    num_batches: int,
    *,
    start_batch: int = 0,
    extra_source: str | None = None,
    extra_every_n_batches: int = 0,
    extra_mode: str | None = None,
    extra_primary_weight: float = 1.0,
    extra_source_weight: float = 0.0,
) -> dict:
    """Count the calls/step units made by the static-cycle branch.

    This is intentionally dry-run only: it uses the same selected-source rule as
    training and invokes one fake source attack per batch to prove call counts
    without running GPU attacks.
    """
    if num_batches < 0:
        raise ValueError("num_batches must be nonnegative")
    norm_to_i = {n: i for i, n in enumerate(group_norms)}
    schedule = []
    extra_schedule = []
    extra_batch_indices = []
    call_counts = {n: 0 for n in group_norms}
    primary_call_counts = {n: 0 for n in group_norms}
    source_counts = {n: 0 for n in group_norms}
    units_by_source = {n: 0 for n in group_norms}
    extra_call_counts = {n: 0 for n in group_norms}
    extra_enabled = (
        extra_source is not None
        and extra_every_n_batches > 0
        and extra_mode == "aggregate_loss"
    )
    if extra_source is not None:
        extra_source = str(extra_source).lower()
        if extra_source not in norm_to_i:
            raise ValueError(f"extra_source must be one of {group_norms}, got {extra_source!r}")

    def fake_source_attack(norm: str) -> None:
        call_counts[norm] += 1

    for b in range(num_batches):
        global_batch = start_batch + b
        norm = cycle_order[global_batch % len(cycle_order)]
        schedule.append(norm)
        fake_source_attack(norm)
        primary_call_counts[norm] += 1
        source_counts[norm] += 1
        units_by_source[norm] += int(full_attack_steps[norm_to_i[norm]])
        if extra_enabled and global_batch % int(extra_every_n_batches) == 0:
            assert extra_source is not None
            fake_source_attack(extra_source)
            source_counts[extra_source] += 1
            extra_call_counts[extra_source] += 1
            units_by_source[extra_source] += int(full_attack_steps[norm_to_i[extra_source]])
            extra_schedule.append(extra_source)
            extra_batch_indices.append(b)

    static_units = sum(units_by_source.values())
    b1_reference_units = int(num_batches) * sum(int(s) for s in full_attack_steps)
    ratio = static_units / max(b1_reference_units, 1)
    return {
        "schedule": schedule,
        "source_counts": source_counts,
        "attack_call_counts": call_counts,
        "primary_attack_call_counts": primary_call_counts,
        "attack_step_units_by_source": units_by_source,
        "attack_step_units": static_units,
        "b1_all_source_reference_units": b1_reference_units,
        "efficiency_ratio": ratio,
        "training_batches": num_batches,
        "optimizer_step_calls": num_batches,
        "backward_calls": num_batches,
        "scheduler_step_calls_in_train_epoch": 0,
        "scheduler_step_count_matches_b2": True,
        "extra_source": extra_source,
        "extra_every_n_batches": int(extra_every_n_batches or 0),
        "extra_mode": extra_mode,
        "extra_primary_weight": float(extra_primary_weight),
        "extra_source_weight": float(extra_source_weight),
        "extra_batch_indices": extra_batch_indices,
        "extra_schedule": extra_schedule,
        "extra_call_counts": extra_call_counts,
        "validation_or_test_state_accessed": False,
        "adaptive_logic_used": False,
        "all_source_loop_bypassed": True,
        "proof": (
            "static_cycle training invokes exactly one selected source attack per batch "
            "plus any configured deterministic extra aggregate-loss attack, then continues before the all-source loop"
        ),
    }


class GroupDROTrainer:
    def __init__(self, cfg: dict, download_data: bool = True, loaders=None):
        self.cfg = cfg
        self.device = cfg["device"] if torch.cuda.is_available() else "cpu"
        if self.device != cfg["device"]:
            print(f"[trainer] CUDA unavailable — falling back to {self.device}")
        self.logger = WandbLogger(cfg, run_name=cfg.get("run_name"))

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
        self.full_attack_steps = [int(s.get("steps", 10)) for s in self.attack_specs]
        self.attacks = [build_source_attack(s, eps=tm[s["norm"].lower()]["eps"], attack=self.train_attack)
                        for s in self.attack_specs]
        self.num_groups = len(self.attacks)
        self.allocation_mode = resolve_train_allocation_mode(tcfg)
        self.static_cycle_order = resolve_static_cycle_order(tcfg, self.group_norms)
        self.static_extra = resolve_static_extra_config(tcfg, self.group_norms)
        self.static_extra_group = (
            self.group_norms.index(self.static_extra["source"])
            if self.static_extra["enabled"] else None
        )
        self.predictive_refresh = (
            resolve_predictive_refresh_config(tcfg, self.group_norms)
            if self.allocation_mode == "predictive_refresh" else None
        )

        gcfg = tcfg.get("groupdro", {})
        self.warmup_epochs = gcfg.get("warmup_epochs", 0)
        self.dro = GroupDRO(self.num_groups, eta_q=gcfg.get("eta_q", 0.02), device=self.device)
        # Weighting variants (each is a 1-component change vs the P1 AttackDRO++):
        #   freeze_q=True            -> CARD-1 avg_frozen: q fixed uniform (DRO off).
        #   weight_signal=robust_acc -> CARD-3a bindaware: q = softmax(-robust_acc_g / tau)
        #                               set once per epoch from val_select (weakest
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
            # CARD-PB v2: predictability-aware floor. In "confidence" mode the
            # per-norm floor is driven by miss volume, P(true_bind=g and b_hat!=g),
            # measured unbiased on the recal subset. "fixed" = the v1 constant.
            self.pb_floor_mode = gcfg.get("pb_floor_mode", "fixed")   # fixed | confidence
            self.pb_floor_kspan = int(gcfg.get("pb_floor_kspan", 12))
            self.pb_floor_ema = float(gcfg.get("pb_floor_ema", 0.5))  # smooth the ~12-sample/batch miss (R5)
            # Adaptive val_select-driven guard: kept for fixed mode (v1) but OFF in confidence
            # mode by default — it double-counts with the confidence floor and is probe-blind
            # on l1 (F1), which inflated FLOPs in sweep-1 (linf pushed to full). Overridable.
            self.pb_adaptive = bool(gcfg.get("pb_adaptive_floor", self.pb_floor_mode != "confidence"))
            self._full_steps = list(self.full_attack_steps)
            self._pb_extra_floor = [0 for _ in self.attack_specs]   # adaptive-guard bumps, ON TOP of the base floor
            self._pb_conf_floor = [self.pb_k_floor for _ in self.attack_specs]  # confidence base (>= k_min)
            self._pb_miss_ema = [None for _ in self.attack_specs]   # per-norm miss-volume EMA
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

        # --- EXPLORATION lanes (Colab idea-testing; OFF by default → paper runs unaffected). ---
        # Idea 1: fail-rate threshold allocation. Idea 3: curriculum step ramp. Both act only on
        # the reactive (non-predictive) path and only when their knob is set. Idea 2 needs no code
        # (it is CARD-PB with kspan/k_min config). See docs — results go to results/exploration/.
        self.exp_failrate = gcfg.get("failrate_threshold")          # Idea 1: None = off
        self.exp_failrate_max = int(gcfg.get("failrate_max_steps", 10))
        self.exp_curriculum = gcfg.get("curriculum")                # Idea 3: [[ep_lt, steps], ...] or None
        self._exp_cur_steps = None
        if self.allocation_mode == "static_cycle":
            if self.objective == "predictive_binding":
                raise ValueError("static_cycle is a non-predictive B2 mode; predictive_binding is not allowed")
            if self.exp_failrate is not None or self.exp_curriculum is not None:
                raise ValueError("static_cycle cannot be combined with exploration failrate/curriculum knobs")
        if self.allocation_mode == "predictive_refresh":
            if self.objective == "predictive_binding":
                raise ValueError(
                    "predictive_refresh is separate from legacy predictive_binding; "
                    "use objective=per_sample_soft for Route A-Diagnostic"
                )
            if self.exp_failrate is not None or self.exp_curriculum is not None:
                raise ValueError("predictive_refresh cannot be combined with exploration failrate/curriculum knobs")
            self._pred_ema_loss = torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
            self._pred_ema_bind = torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
            self._pred_initialized = False
            self._pred_last_selected_batch = [-10**9 for _ in range(self.num_groups)]
            self._pred_last_global_batch = -1

        self.criterion = nn.CrossEntropyLoss()
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
        self.val_select_n = int(tcfg.get("val_select_n_examples", cfg["dataset"].get("val_holdout", 0)))
        self.val_select_frequency = max(1, int(tcfg.get("val_select_frequency", 1)))
        self.test_monitor_n = int(tcfg.get("test_monitor_n_examples", 1000))
        self.test_monitor_frequency = max(1, int(tcfg.get("test_monitor_frequency", 10)))
        self._val_select_data = (
            get_eval_split(cfg, "val_select", n_examples=self.val_select_n)
            if cfg["dataset"].get("val_holdout", 0) else None
        )
        self._test_monitor_data = get_eval_split(cfg, "test_monitor", n_examples=self.test_monitor_n)
        self.best_union = -1.0
        eps_str = ", ".join(f"{n}:{tm[n]['eps']:g}" for n in self.group_norms)
        print(f"[groupdro] groups={self.group_norms}  eta_q={self.dro.eta_q}  eps=({eps_str})  "
              f"objective={self.objective}  weight_signal={self.weight_signal}  "
              f"freeze_q={self.freeze_q}  tau={self.tau}  T={self.temperature}  "
              f"allocation_mode={self.allocation_mode}  static_cycle_order={self.static_cycle_order}  "
              f"static_extra={self.static_extra}  "
              f"norm_losses={self.normalize_losses}")
        val_n = self._val_select_data[0].shape[0] if self._val_select_data is not None else 0
        print(f"[groupdro] eval roles: val_select n={val_n} "
              f"freq={self.val_select_frequency}; test_monitor n={self._test_monitor_data[0].shape[0]} "
              f"freq={self.test_monitor_frequency}")

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
        """Set each norm's confidence floor from its measured miss volume:
        floor_g = clamp(k_min + round(kspan*miss_vol_ema_g), k_min, full_g).
        Norms that are both frequently true-binding and often missed get a higher floor;
        rare/free norms stay near k_min. Rebuilds the
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

        # Misprediction on the recalibration subset (unbiased: full in all norms).
        # Aggregate rate p plus per-norm counts. The floor uses population-aware
        # miss volume P(true_bind=g and b_hat!=g); conditional miss rate is logged
        # only as a diagnostic.
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
    # ------------------------- EXPLORATION helpers (off by default) ------- #
    def _apply_curriculum(self, epoch: int) -> None:
        """Idea 3: rebuild the source attacks with a per-epoch ramped step count.
        curriculum = [[ep_lt, steps], ...]; first entry with epoch < ep_lt wins, else last."""
        if not self.exp_curriculum:
            return
        k = None
        for ep_lt, steps in self.exp_curriculum:
            if epoch < ep_lt:
                k = int(steps)
                break
        if k is None:
            k = int(self.exp_curriculum[-1][1])
        tm = self.cfg["threat_model"]
        self.attacks = [build_source_attack({**s, "steps": k},
                                            eps=tm[s["norm"].lower()]["eps"], attack=self.train_attack)
                        for s in self.attack_specs]
        self._exp_cur_steps = k

    def _failrate_craft(self, g: int, x, y):
        """Idea 1: craft norm g with a batch-level fail-rate early-stop (APGD, ported). Returns
        (x_adv, steps_used). Stops the norm's attack once >= threshold of the batch has failed."""
        from ..attacks.apgd_train import apgd_train
        norm = {"linf": "Linf", "l2": "L2", "l1": "L1"}[self.group_norms[g]]
        eps = self.cfg["threat_model"][self.group_norms[g]]["eps"]
        was = self.model.training
        self.model.eval()
        x_adv, steps = apgd_train(self.model, x, y, norm=norm, eps=eps,
                                  n_iter=self.exp_failrate_max, is_train=True,
                                  stop_frac=self.exp_failrate)
        if was:
            self.model.train()
        return x_adv, steps

    def _static_cycle_group(self, epoch: int, batch_idx: int) -> int:
        global_batch = int(epoch) * len(self.train_loader) + int(batch_idx)
        norm = self.static_cycle_order[global_batch % len(self.static_cycle_order)]
        return self.group_norms.index(norm)

    def _static_extra_fires(self, epoch: int, batch_idx: int) -> bool:
        if not self.static_extra["enabled"]:
            return False
        global_batch = int(epoch) * len(self.train_loader) + int(batch_idx)
        return global_batch % int(self.static_extra["every_n_batches"]) == 0

    def _predictive_refresh_scores(self) -> torch.Tensor:
        if not getattr(self, "_pred_initialized", False):
            return torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
        return _predictive_score_from_state(
            self._pred_ema_loss.detach().float(),
            self._pred_ema_bind.detach().float(),
        ).to(self.device)

    def _predictive_refresh_update_ema(self, L: torch.Tensor) -> dict:
        """Update scalar per-source EMA state from a full all-source refresh."""
        loss_means = L.detach().mean(dim=1).float()
        bind = L.detach().argmax(dim=0)
        bind_freq = torch.bincount(bind, minlength=self.num_groups).float() / max(bind.numel(), 1)
        bind_freq = bind_freq.to(loss_means.device)
        if self._pred_initialized:
            alpha = float(self.predictive_refresh["ema_alpha"])
            self._pred_ema_loss = alpha * self._pred_ema_loss + (1.0 - alpha) * loss_means
            self._pred_ema_bind = alpha * self._pred_ema_bind + (1.0 - alpha) * bind_freq
        else:
            self._pred_ema_loss = loss_means.clone()
            self._pred_ema_bind = bind_freq.clone()
            self._pred_initialized = True
        worst_group = int(loss_means.argmax().item())
        binding_group = int(bind_freq.argmax().item())
        return {
            "loss_means": loss_means,
            "bind_freq": bind_freq,
            "worst_group": worst_group,
            "binding_group": binding_group,
        }

    def _predictive_refresh_loss(self, L: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh batches use the inherited reactive aggregation rule.

        For the B4 Route A draft config this is B1's per-sample soft objective
        with T=0.25. Non-refresh batches train on one selected source.
        """
        if self.objective == "per_sample_soft":
            Lw = L.detach()
            if self.normalize_losses == "zscore":
                Lw = (Lw - Lw.mean(dim=1, keepdim=True)) / Lw.std(dim=1, keepdim=True).clamp(min=1e-6)
            w = torch.softmax(Lw / self.temperature, dim=0)
            return (w * L).sum(dim=0).mean(), w.mean(dim=1)
        if self.objective == "per_sample_max":
            selected = L.argmax(dim=0)
            q = torch.zeros(self.num_groups, device=L.device).scatter_add_(
                0, selected, torch.full((L.shape[1],), 1.0 / L.shape[1], device=L.device)
            )
            return L.max(dim=0).values.mean(), q
        losses = L.mean(dim=1)
        q = torch.zeros(self.num_groups, device=L.device)
        q[int(losses.argmax().item())] = 1.0
        return losses.max(), q

    def _predictive_refresh_select(self, scores: torch.Tensor, global_batch: int) -> tuple[int, bool, int | None]:
        predicted = int(scores.argmax().item())
        S = int(self.predictive_refresh["starvation_floor_batches"])
        overdue = [
            (int(global_batch) - int(self._pred_last_selected_batch[g]), g)
            for g in range(self.num_groups)
            if int(global_batch) - int(self._pred_last_selected_batch[g]) >= S
        ]
        if not overdue:
            return predicted, False, None
        _, selected = max(overdue, key=lambda t: (t[0], -t[1]))
        return int(selected), int(selected) != predicted, predicted

    def _predictive_refresh_state_dict(self) -> dict:
        return {
            "ema_loss": self._pred_ema_loss.detach().cpu(),
            "ema_bind": self._pred_ema_bind.detach().cpu(),
            "initialized": bool(self._pred_initialized),
            "last_selected_batch": list(self._pred_last_selected_batch),
            "last_global_batch": int(self._pred_last_global_batch),
            "config": dict(self.predictive_refresh or {}),
            "group_norms": list(self.group_norms),
        }

    def _load_predictive_refresh_state(self, state: dict) -> None:
        if not state:
            raise RuntimeError("predictive_refresh checkpoint is missing predictor state")
        if list(state.get("group_norms", self.group_norms)) != list(self.group_norms):
            raise RuntimeError(
                f"predictive_refresh checkpoint group_norms mismatch: "
                f"{state.get('group_norms')} vs {self.group_norms}"
            )
        self._pred_ema_loss = state["ema_loss"].to(self.device).float()
        self._pred_ema_bind = state["ema_bind"].to(self.device).float()
        self._pred_initialized = bool(state.get("initialized", False))
        self._pred_last_selected_batch = [int(v) for v in state["last_selected_batch"]]
        self._pred_last_global_batch = int(state.get("last_global_batch", -1))

    def train_epoch(self, epoch: int, max_steps=None) -> dict:
        self.model.train()
        self._apply_curriculum(epoch)                 # Idea 3 (no-op unless configured)
        self._exp_steps_sum = [0.0 for _ in range(self.num_groups)]   # Idea 1 step accounting
        self._exp_steps_cnt = 0
        t0 = time.time()
        n = 0
        group_loss_sum = torch.zeros(self.num_groups)
        correct = torch.zeros(self.num_groups)
        q_sum = torch.zeros(self.num_groups)
        n_batches = 0
        self._first_batch_L = None
        self._epoch_loss_samples = []
        source_counts = [0 for _ in range(self.num_groups)]
        source_sample_counts = [0 for _ in range(self.num_groups)]
        attack_call_counts = [0 for _ in range(self.num_groups)]
        attack_step_units = [0.0 for _ in range(self.num_groups)]
        selected_source_counts = [0 for _ in range(self.num_groups)]
        pred_refresh_indices = []
        pred_trace = []
        pred_refresh_count = 0
        pred_floor_override_count = 0
        pred_correct_worst = 0
        pred_correct_binding = 0
        pred_accuracy_den = 0
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

            if self.allocation_mode == "predictive_refresh":
                global_batch = int(epoch) * len(self.train_loader) + int(i)
                R = int(self.predictive_refresh["refresh_every_n_batches"])
                scores_before = self._predictive_refresh_scores()
                pred_before = int(scores_before.argmax().item()) if self._pred_initialized else None
                is_refresh = (not self._pred_initialized) or (global_batch % R == 0)
                floor_override = False
                binding_group = None
                worst_group = None
                q = torch.zeros(self.num_groups, device=x.device)

                if is_refresh:
                    pred_refresh_count += 1
                    pred_refresh_indices.append(i)
                    per_sample = []
                    for g, atk in enumerate(self.attacks):
                        x_adv = atk(self.model, x, y)
                        source_counts[g] += 1
                        source_sample_counts[g] += int(y.size(0))
                        attack_call_counts[g] += 1
                        attack_step_units[g] += float(self.full_attack_steps[g])
                        self.model.train()
                        logits = self.model(x_adv)
                        loss_vec = nn.functional.cross_entropy(logits, y, reduction="none")
                        per_sample.append(loss_vec)
                        group_loss_sum[g] += loss_vec.sum().item()
                        correct[g] += (logits.argmax(1) == y).sum().item()

                    L = torch.stack(per_sample)
                    total_loss, q = self._predictive_refresh_loss(L)
                    refresh_info = self._predictive_refresh_update_ema(L)
                    worst_group = int(refresh_info["worst_group"])
                    binding_group = int(refresh_info["binding_group"])
                    selected_group = worst_group
                    selected_source_counts[selected_group] += 1
                    for g in range(self.num_groups):
                        self._pred_last_selected_batch[g] = global_batch
                    if pred_before is not None:
                        pred_accuracy_den += 1
                        pred_correct_worst += int(pred_before == worst_group)
                        pred_correct_binding += int(pred_before == binding_group)
                else:
                    selected_group, floor_override, raw_pred = self._predictive_refresh_select(
                        scores_before, global_batch
                    )
                    pred_floor_override_count += int(floor_override)
                    x_adv = self.attacks[selected_group](self.model, x, y)
                    source_counts[selected_group] += 1
                    source_sample_counts[selected_group] += int(y.size(0))
                    selected_source_counts[selected_group] += 1
                    attack_call_counts[selected_group] += 1
                    attack_step_units[selected_group] += float(self.full_attack_steps[selected_group])
                    self.model.train()
                    logits = self.model(x_adv)
                    loss_vec = nn.functional.cross_entropy(logits, y, reduction="none")
                    total_loss = loss_vec.mean()
                    group_loss_sum[selected_group] += loss_vec.sum().item()
                    correct[selected_group] += (logits.argmax(1) == y).sum().item()
                    q[selected_group] = 1.0
                    self._pred_last_selected_batch[selected_group] = global_batch

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()
                self._pred_last_global_batch = global_batch

                q_sum += q.detach().cpu()
                n += y.size(0)
                n_batches += 1
                if self.predictive_refresh["log_batch_trace"]:
                    scores_after = self._predictive_refresh_scores().detach().cpu()
                    pred_trace.append({
                        "batch_index": int(i),
                        "global_batch_index": int(global_batch),
                        "refresh": bool(is_refresh),
                        "selected_source": self.group_norms[selected_group],
                        "predicted_source": (
                            self.group_norms[pred_before] if pred_before is not None else None
                        ),
                        "binding_source": (
                            self.group_norms[binding_group] if binding_group is not None else None
                        ),
                        "worst_source": (
                            self.group_norms[worst_group] if worst_group is not None else None
                        ),
                        "prediction_correct_worst": (
                            bool(pred_before == worst_group)
                            if pred_before is not None and worst_group is not None else None
                        ),
                        "prediction_correct_binding": (
                            bool(pred_before == binding_group)
                            if pred_before is not None and binding_group is not None else None
                        ),
                        "scores": {
                            norm: float(scores_after[g]) for g, norm in enumerate(self.group_norms)
                        },
                        "ema_loss": {
                            norm: float(self._pred_ema_loss.detach().cpu()[g])
                            for g, norm in enumerate(self.group_norms)
                        },
                        "ema_bind": {
                            norm: float(self._pred_ema_bind.detach().cpu()[g])
                            for g, norm in enumerate(self.group_norms)
                        },
                        "source_age": {
                            norm: int(global_batch - self._pred_last_selected_batch[g])
                            for g, norm in enumerate(self.group_norms)
                        },
                        "floor_override": bool(floor_override),
                    })
                continue

            if self.allocation_mode == "static_cycle":
                primary_group = self._static_cycle_group(epoch, i)
                weighted_losses = []
                selected_groups = [(primary_group, 1.0)]
                if self._static_extra_fires(epoch, i):
                    assert self.static_extra_group is not None
                    selected_groups = [
                        (primary_group, float(self.static_extra["primary_weight"])),
                        (self.static_extra_group, float(self.static_extra["source_weight"])),
                    ]
                q = torch.zeros(self.num_groups, device=x.device)
                for g, loss_weight in selected_groups:
                    x_adv = self.attacks[g](self.model, x, y)
                    source_counts[g] += 1
                    source_sample_counts[g] += int(y.size(0))
                    attack_call_counts[g] += 1
                    attack_step_units[g] += float(self.full_attack_steps[g])

                    self.model.train()
                    logits = self.model(x_adv)
                    loss_vec = nn.functional.cross_entropy(logits, y, reduction="none")
                    weighted_losses.append(float(loss_weight) * loss_vec.mean())

                    group_loss_sum[g] += loss_vec.sum().item()
                    correct[g] += (logits.argmax(1) == y).sum().item()
                    q[g] += float(loss_weight)
                total_loss = sum(weighted_losses)
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()
                q_sum += q.detach().cpu()
                n += y.size(0)
                n_batches += 1
                continue

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
                if self.exp_failrate is not None:      # Idea 1: fail-rate early-stop craft
                    x_adv, _steps = self._failrate_craft(g, x, y)
                    self._exp_steps_sum[g] += _steps
                    steps_used = float(_steps)
                else:
                    x_adv = atk(self.model, x, y)      # crafted in eval mode internally
                    steps_used = float(
                        self._exp_cur_steps
                        if self.exp_curriculum and self._exp_cur_steps is not None
                        else self.full_attack_steps[g]
                    )
                source_counts[g] += 1
                source_sample_counts[g] += int(y.size(0))
                attack_call_counts[g] += 1
                attack_step_units[g] += steps_used
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

        if (self.allocation_mode in {"static_cycle", "predictive_refresh"}
                or self.objective.startswith("per_sample")
                or self.objective == "predictive_binding"):
            q = q_sum / max(n_batches, 1)     # epoch-mean weight / selection freq
        else:
            q = self.dro.weights().detach().cpu()
        total_attack_step_units = float(sum(attack_step_units))
        reference_attack_step_units = float(n_batches * sum(self.full_attack_steps))
        log_source_accounting = self.objective != "predictive_binding"
        metrics = {
            "train/loss": (
                group_loss_sum.sum().item()
                / max((
                    sum(source_sample_counts)
                    if self.allocation_mode in {"static_cycle", "predictive_refresh"}
                    else self.num_groups * n
                ), 1)
            ),
            "train/epoch_time_s": time.time() - t0,
        }
        # Amendment #1 (CARD-3b): per-norm per-sample loss distribution -> scale-bias check.
        all_L = torch.cat(self._epoch_loss_samples, dim=1) if self._epoch_loss_samples else None
        self._epoch_loss_samples = []
        for g, norm in enumerate(self.group_norms):
            if self.allocation_mode == "static_cycle":
                den = max(source_sample_counts[g], 1)
            else:
                den = max(source_sample_counts[g] or n, 1)
            metrics[f"train/adv_acc_{norm}"] = correct[g].item() / den
            metrics[f"train/loss_{norm}"] = group_loss_sum[g].item() / den
            metrics[f"q/{norm}"] = q[g].item()
            if log_source_accounting:
                metrics[f"train/source_count_{norm}"] = float(source_counts[g])
                metrics[f"train/attack_call_count_{norm}"] = float(attack_call_counts[g])
                metrics[f"train/attack_step_units_{norm}"] = float(attack_step_units[g])
                metrics[f"train/attack_step_fraction_{norm}"] = (
                    attack_step_units[g] / max(total_attack_step_units, 1.0)
                )
                if self.allocation_mode == "predictive_refresh":
                    metrics[f"predictive_refresh/selected_count_{norm}"] = float(
                        selected_source_counts[g]
                    )
                    metrics[f"predictive_refresh/selected_fraction_{norm}"] = (
                        selected_source_counts[g] / max(sum(selected_source_counts), 1)
                    )
                    metrics[f"predictive_refresh/ema_loss_{norm}"] = float(
                        self._pred_ema_loss.detach().cpu()[g]
                    )
                    metrics[f"predictive_refresh/ema_bind_{norm}"] = float(
                        self._pred_ema_bind.detach().cpu()[g]
                    )
                    metrics[f"predictive_refresh/final_score_{norm}"] = float(
                        self._predictive_refresh_scores().detach().cpu()[g]
                    )
                    metrics[f"predictive_refresh/source_age_{norm}"] = float(
                        self._pred_last_global_batch - self._pred_last_selected_batch[g]
                    )
            if all_L is not None:
                v = all_L[g]
                p10, p50, p90 = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9])).tolist()
                metrics[f"dist/{norm}/std"] = v.std().item()
                metrics[f"dist/{norm}/p10"] = p10
                metrics[f"dist/{norm}/p50"] = p50
                metrics[f"dist/{norm}/p90"] = p90
        # EXPLORATION metrics (off by default). Idea 1: per-norm steps the fail-rate threshold
        # actually used + attack-FLOPs vs full 10/10/10 (denom = max_steps × G = 30). Idea 3: the
        # epoch's curriculum step count.
        if self.exp_failrate is not None:
            fr = [self._exp_steps_sum[g] / max(n_batches, 1) for g in range(self.num_groups)]
            for g, norm in enumerate(self.group_norms):
                metrics[f"exp/steps_{norm}"] = fr[g]
            metrics["exp/attack_flops_ratio"] = sum(fr) / max(sum(self.full_attack_steps), 1)
            metrics["exp/failrate_threshold"] = float(self.exp_failrate)
        if self.exp_curriculum and self._exp_cur_steps is not None:
            metrics["exp/curriculum_steps"] = float(self._exp_cur_steps)
        # CARD-PB: MEASURED attack-FLOPs ratio (vs reactive all-norms-full) + φ error.
        if self.objective == "predictive_binding":
            metrics["pb/attack_flops_ratio"] = (
                self._pb_attack_passes / max(self._pb_reactive_passes, 1))
            metrics["pb/attack_passes"] = float(self._pb_attack_passes)
            if self._pb_mis_den:
                metrics["phi/misprediction_rate"] = self._pb_mis_num / self._pb_mis_den
            else:
                metrics["phi/misprediction_rate"] = 0.0
            metrics["pb/beta"] = self.pb_beta        # EMA horizon (comparability across β-ablation)
            metrics["pb/floor_mode"] = self.pb_floor_mode
            metrics["pb/floor_kspan"] = self.pb_floor_kspan
            for g, norm in enumerate(self.group_norms):
                # v2 FIX (sweep-1 was confounded): drive the floor by miss VOLUME =
                # P(true=g AND b_hat!=g) — the joint, POPULATION-aware probability over ALL
                # recal samples, not the conditional miss rate. A rarely-bound norm (l2, free
                # per F3) then can't get a high floor just because its few true samples are
                # mispredicted; only norms that are BOTH frequently bound AND often missed
                # (l1) earn a high floor. EMA-smoothed (R5). Conditional miss rate is diagnostic.
                if self._pb_mis_den:
                    vol = self._pb_missn[g] / self._pb_mis_den
                    self._pb_miss_ema[g] = (vol if self._pb_miss_ema[g] is None
                                            else self.pb_floor_ema * vol
                                            + (1 - self.pb_floor_ema) * self._pb_miss_ema[g])
                if self._pb_miss_ema[g] is not None:
                    metrics[f"phi/miss_vol_{norm}"] = self._pb_miss_ema[g]     # the floor DRIVER
                elif self._pb_mis_den:
                    metrics[f"phi/miss_vol_{norm}"] = 0.0
                if self._pb_mis_den:
                    metrics[f"phi/miss_rate_{norm}"] = (
                        self._pb_missn[g] / self._pb_missd[g] if self._pb_missd[g] else 0.0
                    )  # conditional miss diagnostic
                metrics[f"floor/{norm}"] = self._floor_steps(g)
        if log_source_accounting:
            metrics["train/attack_step_units"] = total_attack_step_units
            metrics["train/reactive_attack_step_units_reference"] = reference_attack_step_units
            if self.allocation_mode == "static_cycle":
                metrics["train/static_cycle_length"] = float(len(self.static_cycle_order))
                metrics["train/static_extra_enabled"] = float(bool(self.static_extra["enabled"]))
                metrics["train/static_extra_every_n_batches"] = float(
                    self.static_extra["every_n_batches"]
                )
                for g, norm in enumerate(self.group_norms):
                    metrics[f"train/static_cycle_weight_{norm}"] = float(
                        self.static_cycle_order.count(norm)
                    )
                    metrics[f"train/static_cycle_fraction_{norm}"] = (
                        self.static_cycle_order.count(norm) / max(len(self.static_cycle_order), 1)
                    )
                    metrics[f"train/source_update_fraction_{norm}"] = (
                        source_counts[g] / max(sum(source_counts), 1)
                    )
            if self.allocation_mode == "predictive_refresh":
                metrics["predictive_refresh/refresh_count"] = float(pred_refresh_count)
                metrics["predictive_refresh/non_refresh_count"] = float(n_batches - pred_refresh_count)
                metrics["predictive_refresh/refresh_indices"] = pred_refresh_indices
                metrics["predictive_refresh/prediction_accuracy_worst"] = (
                    pred_correct_worst / max(pred_accuracy_den, 1)
                )
                metrics["predictive_refresh/prediction_accuracy_binding"] = (
                    pred_correct_binding / max(pred_accuracy_den, 1)
                )
                metrics["predictive_refresh/prediction_denominator"] = float(pred_accuracy_den)
                metrics["predictive_refresh/floor_override_count"] = float(pred_floor_override_count)
                metrics["predictive_refresh/refresh_every_n_batches"] = float(
                    self.predictive_refresh["refresh_every_n_batches"]
                )
                metrics["predictive_refresh/starvation_floor_batches"] = float(
                    self.predictive_refresh["starvation_floor_batches"]
                )
                metrics["predictive_refresh/ema_alpha"] = float(self.predictive_refresh["ema_alpha"])
                metrics["predictive_refresh/uses_cheap_probe"] = 0.0
                if self.predictive_refresh["log_batch_trace"]:
                    trace_dir = os.path.join(self.paths["seed_dir"], "predictive_refresh_traces")
                    os.makedirs(trace_dir, exist_ok=True)
                    trace_path = os.path.join(trace_dir, f"ep{epoch:03d}.json")
                    save_json({
                        "epoch": int(epoch),
                        "run_name": self.cfg.get("run_name"),
                        "allocation_mode": self.allocation_mode,
                        "group_norms": self.group_norms,
                        "predictive_refresh": dict(self.predictive_refresh),
                        "trace": pred_trace,
                    }, trace_path)
                    metrics["predictive_refresh/batch_trace_path"] = trace_path
                    metrics["predictive_refresh/batch_trace_length"] = float(len(pred_trace))
                    metrics["predictive_refresh/batch_trace_fields"] = (
                        "batch_index,global_batch_index,refresh,selected_source,"
                        "predicted_source,binding_source,worst_source,"
                        "prediction_correct_worst,prediction_correct_binding,"
                        "scores,ema_loss,ema_bind,source_age,floor_override"
                    )
        metrics["train/allocation_mode"] = self.allocation_mode
        # Canonical comparable attack-FLOPs key for every method. Method-specific keys are
        # retained above for old parsers, but dashboards/tables should read this alias.
        if "pb/attack_flops_ratio" in metrics:
            metrics["efficiency/attack_flops_ratio"] = metrics["pb/attack_flops_ratio"]
        elif "exp/attack_flops_ratio" in metrics:
            metrics["efficiency/attack_flops_ratio"] = metrics["exp/attack_flops_ratio"]
        elif self.allocation_mode == "static_cycle":
            metrics["efficiency/attack_flops_ratio"] = (
                total_attack_step_units / max(reference_attack_step_units, 1.0)
            )
        elif self.allocation_mode == "predictive_refresh":
            metrics["efficiency/attack_flops_ratio"] = (
                total_attack_step_units / max(reference_attack_step_units, 1.0)
            )
        elif self.exp_curriculum and self._exp_cur_steps is not None:
            metrics["efficiency/attack_flops_ratio"] = (
                self._exp_cur_steps * self.num_groups / max(sum(self.full_attack_steps), 1)
            )
        else:
            metrics["efficiency/attack_flops_ratio"] = 1.0
        return metrics

    def _probe_tensor_norm(self, norm: str, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Per-sample robust mask under the cheap in-training probe attack."""
        eps = self.cfg["threat_model"][norm]["eps"]
        atk = _PROBE[norm]
        masks = []
        bs = int(self.cfg["train"].get("batch_size", 128))
        was_training = self.model.training
        self.model.eval()
        for start in range(0, xs.shape[0], bs):
            x = xs[start:start + bs]
            y = ys[start:start + bs]
            x, y = x.to(self.device), y.to(self.device)
            if norm == "linf":
                x_adv = atk(self.model, x, y, eps=eps, step_size=eps / 4, steps=self.probe_steps)
            elif norm == "l2":
                x_adv = atk(self.model, x, y, eps=eps, step_size=eps / 4, steps=self.probe_steps)
            else:
                x_adv = atk(self.model, x, y, eps=eps, step_size=0.05, steps=self.probe_steps)
            with torch.no_grad():
                masks.append((self.model(x_adv).argmax(1) == y).cpu())
        if was_training:
            self.model.train()
        return torch.cat(masks)

    def _clean_acc_tensors(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        """Clean accuracy on the same split tensors used by the in-training probe."""
        self.model.eval()
        correct = total = 0
        bs = int(self.cfg["train"].get("batch_size", 128))
        for start in range(0, xs.shape[0], bs):
            x = xs[start:start + bs]
            y = ys[start:start + bs]
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                correct += (self.model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    def _evaluate_role(self, prefix: str, data: tuple[torch.Tensor, torch.Tensor],
                       *, frequency: int, used_for_selection: bool | None = None) -> dict:
        xs, ys = data
        per_norm = {n: self._probe_tensor_norm(n, xs, ys) for n in ("linf", "l2", "l1")}
        union = per_norm["linf"] & per_norm["l2"] & per_norm["l1"]
        out = {f"{prefix}/robust_{n}": m.float().mean().item() for n, m in per_norm.items()}
        out[f"{prefix}/worst_union"] = union.float().mean().item()
        out[f"{prefix}/clean_acc"] = self._clean_acc_tensors(xs, ys)
        out[f"{prefix}/n_examples"] = int(xs.shape[0])
        out[f"{prefix}/frequency"] = int(frequency)
        if used_for_selection is not None:
            out[f"{prefix}/used_for_selection"] = bool(used_for_selection)
        return out

    def evaluate(self, epoch: int) -> dict:
        """Training-run split probes.

        val_select is selection-grade for choosing ckpt/val_best.pt. test_monitor
        is logged only every N epochs and must never drive selection.
        """
        out = {}
        epoch_one_indexed = epoch + 1
        if (self._val_select_data is not None
                and epoch_one_indexed % self.val_select_frequency == 0):
            out.update(self._evaluate_role(
                "val_select", self._val_select_data,
                frequency=self.val_select_frequency,
            ))
        if epoch_one_indexed % self.test_monitor_frequency == 0:
            out.update(self._evaluate_role(
                "test_monitor", self._test_monitor_data,
                frequency=self.test_monitor_frequency,
                used_for_selection=False,
            ))
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
        payload = {"model": self.model.state_dict(), "cfg": self.cfg,
                   "dro": self.dro.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
                   "rng": rng, **(extra or {})}
        if self.allocation_mode == "predictive_refresh":
            payload["predictive_refresh"] = self._predictive_refresh_state_dict()
        torch.save(payload, path)
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
        if self.allocation_mode == "predictive_refresh":
            if "predictive_refresh" not in ck:
                raise RuntimeError(
                    "predictive_refresh resume requested, but checkpoint lacks predictor state"
                )
            self._load_predictive_refresh_state(ck["predictive_refresh"])
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
        """Safety-valve: if any norm's val_select robust acc falls > pb_guard_pp below its
        own running max, raise that norm's floor budget (so a wrong φ can't quietly
        erode that norm). Self-referential guard; the exact >2pp-vs-reactive-control
        check is done post-hoc against the paired reactive run."""
        for g, norm in enumerate(self.group_norms):
            acc = ev.get(f"val_select/robust_{norm}", ev.get(f"probe/robust_{norm}"))
            if acc is None:
                continue
            self._pb_norm_max[g] = max(self._pb_norm_max[g], acc)
            if acc < self._pb_norm_max[g] - self.pb_guard_pp / 100.0:
                bump = min(self._pb_extra_floor[g] + 2, self._full_steps[g])
                if bump != self._pb_extra_floor[g]:
                    self._pb_extra_floor[g] = bump
                    self._rebuild_floor(g)   # R1: the raised floor now actually attacks more
                    print(f"[cardpb] adaptive floor: {norm} val_select {acc:.3f} dropped "
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
            ev = self.evaluate(epoch)
            if self.objective == "predictive_binding":
                self._recompute_conf_floor()       # v2: set next epoch's floor from measured miss
                if self.pb_adaptive:
                    self._pb_adaptive_floor(ev)     # val_select-driven bump (fixed mode only by default)
            pb = self._dump_probe_binding(epoch)
            # CARD-3a: set next epoch's q from THIS epoch's per-norm val_select robust-acc
            # (binding-aware: weakest norm -> most weight). Epoch 0 trains uniform.
            if self.weight_signal == "robust_acc" and not self.freeze_q:
                vals = [ev.get(f"val_select/robust_{n}", ev.get(f"probe/robust_{n}"))
                        for n in self.group_norms]
                if all(v is not None for v in vals):
                    racc = torch.tensor(vals, dtype=torch.float32)
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

            # Selection gatekeeper (pre-reg v3-A): only val_select/worst_union
            # may drive ckpt/val_best.pt. Fail closed if any held-out split
            # (cal / test_monitor / test_final) is ever marked selection-driving.
            assert SELECTION_METRIC_KEY.startswith("val_select/")
            for _held in NON_SELECTION_SPLITS:
                if ev.get(f"{_held}/used_for_selection"):
                    raise RuntimeError(
                        f"Selection gatekeeper: split {_held!r} marked "
                        f"used_for_selection; only val_select/worst_union may "
                        f"select ckpt/val_best.pt."
                    )
            val_union = ev.get(SELECTION_METRIC_KEY)
            if not self.is_smoke and val_union is not None and val_union > self.best_union:
                self.best_union = val_union
                extra = {
                    "epoch": epoch,
                    "selection_source": "val_select",
                    "selection_metric": "val_select/worst_union",
                    "used_for_selection": True,
                    **ev,
                }
                self.save_checkpoint("val_best", extra)
                self.save_checkpoint("best", extra)  # legacy alias; new scripts use val_best.pt.
            # save_freq: periodic checkpoints (ep010.pt ... ep080.pt) for epoch curves +
            # PHASE-2 continuation from a fixed epoch (item 1). 1-indexed to match RAMP.
            if (not self.is_smoke and self.save_freq
                    and (epoch + 1) % self.save_freq == 0):
                self.save_checkpoint(f"ep{epoch + 1:03d}", {"epoch": epoch})
            # Incrementally persist the curve so a disconnect keeps completed epochs.
            if not self.is_smoke:
                save_json({"cfg": self.cfg, "history": history,
                           "best_val_select_worst_union": self.best_union}, self.paths["train"])

        if not self.is_smoke:
            self.save_checkpoint("last", {"epoch": self.epochs - 1})
        self.logger.summary({
            "best/val_select_worst_union": self.best_union,
            "best/selection_source": "val_select",
            "best/checkpoint": "val_best.pt",
        })
        out = self.paths["smoke"] if self.is_smoke else self.paths["train"]
        save_json({"cfg": self.cfg, "history": history,
                   "best_val_select_worst_union": self.best_union}, out)
        self.logger.finish()
        return {"best_val_select_worst_union": self.best_union, "results_json": out}

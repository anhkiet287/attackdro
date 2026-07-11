#!/usr/bin/env python
"""Measure small timing units and project full AttackDRO/RAMP runs.

This script is intentionally a measuring instrument, not a launcher for full
experiments. It times tiny units on the current machine, then linearly projects
larger runs and labels them as projections.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
RAMP = REPO / "external" / "RAMP"
sys.path.insert(0, str(SRC))

from robustdro.attacks import pgd_l1_topk, pgd_l2, pgd_linf  # noqa: E402
from robustdro.eval.attacks_aa import robust_mask  # noqa: E402
from robustdro.eval.eval_union import load_eval_checkpoint, load_test_subset  # noqa: E402
from robustdro.training.groupdro import GroupDROTrainer  # noqa: E402
from robustdro.utils.io import apply_overrides, load_config  # noqa: E402
from robustdro.utils.seed import set_seed  # noqa: E402


@dataclass
class Row:
    task: str
    per_unit: str
    projected_full: str
    device: str
    status: str
    notes: str = ""
    seconds: float | None = None
    extra: dict = field(default_factory=dict)


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=10)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 127, str(e)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return time.perf_counter() - t0, out


def fmt_s(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 120:
        return f"{minutes:.1f}min"
    return f"{minutes / 60:.2f}h"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def gpu_snapshot() -> dict:
    snap = {
        "cuda": torch.cuda.is_available(),
        "device": "cpu",
        "free_vram_gb": None,
        "total_vram_gb": None,
        "tmux": "",
        "nvidia_smi": "",
    }
    code, out = run_cmd(["tmux", "ls"])
    snap["tmux"] = out if code == 0 else f"unavailable: {out}"
    code, out = run_cmd([
        "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    snap["nvidia_smi"] = out if code == 0 else f"unavailable: {out}"
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        snap.update({
            "device": torch.cuda.get_device_name(0),
            "free_vram_gb": free_b / 1024**3,
            "total_vram_gb": total_b / 1024**3,
        })
    return snap


def should_run_gpu(args, snap: dict) -> tuple[bool, str]:
    if args.cpu_only:
        return False, "cpu-only requested"
    if not snap["cuda"]:
        return False, "CUDA unavailable"
    if snap["free_vram_gb"] is not None and snap["free_vram_gb"] < args.min_free_gb:
        return False, f"free VRAM < {args.min_free_gb:g}GB"
    tmux = snap.get("tmux", "").lower()
    if not args.force_gpu and ("finalpipe" in tmux or "ramp" in tmux):
        return False, "finalpipe/RAMP tmux session present"
    return True, "measured"


def trainer_cfg() -> dict:
    cfg = load_config("configs/paper/reactive_ramprecipe.yaml")
    return apply_overrides(cfg, {
        "seed": 0,
        "run_name": "timing_reactive_ramprecipe_T025",
        "wandb.mode": "disabled",
        "train.groupdro.temperature": 0.25,
        "train.groupdro.probe_binding": False,
        "dataset.num_workers": 2,
    })


def time_our_trainer(rows: list[Row]) -> tuple[float | None, float | None]:
    cfg = trainer_cfg()
    set_seed(0)
    trainer = GroupDROTrainer(cfg, download_data=False)
    train_steps_per_epoch = len(trainer.train_loader)

    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train_epoch(0, max_steps=1)  # CUDA/kernel/data-loader warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    elapsed, metrics = timed(lambda: trainer.train_epoch(1, max_steps=20))
    sec_per_step = elapsed / 20.0
    sec_per_epoch = sec_per_step * train_steps_per_epoch
    full_seed = sec_per_epoch * 80
    peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    rows.append(Row(
        "our per-sample-soft trainer T=0.25",
        f"20 steps = {fmt_s(elapsed)} ({sec_per_step:.2f}s/step)",
        f"{fmt_s(sec_per_epoch)}/epoch x80 = {fmt_s(full_seed)}/seed",
        trainer.device,
        "measured",
        f"{train_steps_per_epoch} steps/epoch; train loss {metrics.get('train/loss', float('nan')):.4f}",
        seconds=elapsed,
        extra={"sec_per_step": sec_per_step, "peak_reserved_gb": peak_reserved_gb},
    ))

    # Attack-generation-only timing on one real minibatch.
    x, y = next(iter(trainer.train_loader))
    x, y = x.to(trainer.device), y.to(trainer.device)
    attacks = [
        ("linf10", trainer.attacks[0]),
        ("l2_10", trainer.attacks[1]),
        ("l1_10", trainer.attacks[2]),
    ]
    per_attack: list[str] = []

    def run_attacks():
        outs = []
        for name, atk in attacks:
            dt, _ = timed(lambda atk=atk: atk(trainer.model, x, y))
            per_attack.append(f"{name}={fmt_s(dt)}")
            outs.append(dt)
        return outs

    attack_total = sum(run_attacks())
    frac = attack_total / sec_per_step if sec_per_step > 0 else float("nan")
    rows.append(Row(
        "attack generation only, one minibatch",
        f"{fmt_s(attack_total)} total ({', '.join(per_attack)})",
        f"{frac:.2f}x one full train step ({pct(min(frac, 10.0))} of step time ratio)",
        trainer.device,
        "measured",
        "linf 10 + l2 10 + l1 20 source attacks; no optimizer step",
        seconds=attack_total,
        extra={"fraction_of_train_step": frac},
    ))
    return sec_per_step, peak_reserved_gb


def load_ramp_model_and_data(n: int):
    cfg = load_config("configs/paper/base_ramp_apgd_8255.yaml")
    model, _ = load_eval_checkpoint(
        "external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth",
        cfg, model_family="ramp", device="cuda",
    )
    x, y = load_test_subset(cfg, n_examples=n)
    return cfg, model, x, y


def time_eval(rows: list[Row], version: str, n: int, projections: list[int]) -> None:
    cfg, model, x, y = load_ramp_model_and_data(n)
    tm = cfg["threat_model"]
    eval_cfg = cfg.get("eval_attack", {})
    per_norm = {}
    total = 0.0
    for norm in ("linf", "l2", "l1"):
        eps = tm[norm]["eps"]
        steps = eval_cfg.get(norm, {}).get("steps", 100)
        restarts = eval_cfg.get(norm, {}).get("restarts", 1)
        dt, _ = timed(lambda norm=norm, eps=eps, steps=steps, restarts=restarts:
                      robust_mask(model, x, y, norm, eps, steps=steps,
                                  restarts=restarts, device="cuda",
                                  version=version, seed=0, bs=n))
        per_norm[norm] = dt
        total += dt

    proj_text = ", ".join(
        f"n={p}: {fmt_s(total * (p / n))}" for p in projections
    )
    rows.append(Row(
        f"eval_union {version}, RAMP repro",
        f"n={n}: {fmt_s(total)} total; "
        + ", ".join(f"{k}={fmt_s(v)}" for k, v in per_norm.items()),
        proj_text,
        "cuda",
        "measured",
        "linear projection; per-norm attacks run separately and union is AND masks",
        seconds=total,
        extra={f"{k}_seconds": v for k, v in per_norm.items()},
    ))


def time_ramp8b(rows: list[Row]) -> None:
    sys.path.insert(0, str(RAMP))
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as T

    from autopgd_train import apgd_train
    from model_zoo.fast_models import PreActResNet18
    from utils import gp

    device = "cuda"
    batch_size = 128
    steps_full_epoch = 391  # ceil(50000 / 128), matching RAMP logs.
    ds = torchvision.datasets.CIFAR10(
        root=str(REPO / "data"), train=True, download=False,
        transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]),
    )
    loader = DataLoader(Subset(ds, range(batch_size * 5)), batch_size=batch_size,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model = PreActResNet18(10, activation="softplus1").to(device).eval()
    model_nat = copy.deepcopy(model)
    model_old = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    opt_nat = torch.optim.SGD(model_nat.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="sum").to(device)

    def clean_step(x, y):
        model_nat.train()
        opt_nat.zero_grad(set_to_none=True)
        loss = ce(model_nat(x), y)
        loss.backward()
        opt_nat.step()

    def target_step(x, y):
        model.eval()
        x_s, _, _, loss_s, _ = apgd_train(model, x, y, norm="Linf", eps=8 / 255,
                                          n_iter=10, is_train=True)
        x_t, _, _, loss_t, _ = apgd_train(model, x, y, norm="L1", eps=12,
                                          n_iter=10, is_train=True)
        loss_arr = torch.stack((loss_t, loss_s))
        delta_arr = torch.stack((x_t.view(len(y), 1, -1), x_s.view(len(y), 1, -1)))
        max_loss = loss_arr.max(dim=0)
        x_best = delta_arr[max_loss.indices, torch.arange(len(y), device=device), 0].view_as(x_s)

        model.train()
        opt.zero_grad(set_to_none=True)
        out_t = model(x_t)
        out_s = model(x_s)
        out_best = model(x_best)
        loss_hard = ce(out_best, y)

        selected = (out_s.max(dim=-1)[1] == y).detach()
        loss_kl = torch.tensor(0.0, device=device)
        if selected.any():
            loss_kl = kl(F.log_softmax(out_t[selected] + 1e-12, dim=1),
                         F.softmax(out_s[selected], dim=1)) / selected.sum()

        loss_pair = torch.stack((
            F.cross_entropy(out_t, y, reduction="none"),
            F.cross_entropy(out_s, y, reduction="none"),
        ), dim=0)
        w = torch.softmax(loss_pair.detach() / 0.25, dim=0)
        loss_soft = (w * loss_pair).sum(dim=0).mean()
        loss = loss_hard + 0.3 * loss_soft + 5.0 * loss_kl
        loss.backward()
        opt.step()

    batches = [(x.to(device), y.to(device)) for x, y in loader]
    clean_step(*batches[0])
    target_step(*batches[0])
    sync()

    clean_elapsed, _ = timed(lambda: [clean_step(x, y) for x, y in batches])
    target_elapsed, _ = timed(lambda: [target_step(x, y) for x, y in batches])
    gp_elapsed, _ = timed(lambda: gp(0.5, [model_nat.state_dict()],
                                    model_old.state_dict(), model.state_dict(), [1.0]))
    step_total = clean_elapsed / 5 + target_elapsed / 5
    epoch_proj = step_total * steps_full_epoch + gp_elapsed
    full_proj = epoch_proj * 80
    rows.append(Row(
        "RAMP-8b v1 augment core timing",
        f"5 clean steps {fmt_s(clean_elapsed)} + 5 target steps {fmt_s(target_elapsed)}; "
        f"GP/copy fusion {fmt_s(gp_elapsed)}",
        f"{fmt_s(epoch_proj)}/epoch x80 = {fmt_s(full_proj)}/seed",
        "cuda",
        "measured",
        "in-process external/RAMP model + apgd_train with augment loss; no final_eval",
        seconds=clean_elapsed + target_elapsed + gp_elapsed,
        extra={"sec_per_combined_step": step_total, "gp_seconds": gp_elapsed},
    ))


def time_trait_state(rows: list[Row]) -> None:
    cmd = [sys.executable, "scripts/legacy/binding_trait_vs_state_v2.py",
           "--out", "results/reports/trait_state_v2.md"]
    dt, result = timed(lambda: subprocess.run(cmd, cwd=REPO, text=True,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT))
    rows.append(Row(
        "trait/state diagnostic on existing dumps",
        fmt_s(dt),
        fmt_s(dt),
        "cpu",
        "measured" if result.returncode == 0 else "failed",
        result.stdout.strip().replace("\n", " ")[:240],
        seconds=dt,
    ))


def pending_gpu_rows(rows: list[Row], reason: str) -> None:
    tasks = [
        "our per-sample-soft trainer T=0.25",
        "attack generation only, one minibatch",
        "eval_union apgd, RAMP repro",
        "eval_union standard, RAMP repro",
        "RAMP-8b v1 augment core timing",
    ]
    for task in tasks:
        rows.append(Row(task, "not measured", "pending", "cuda", "pending", reason))


def render(rows: list[Row], snap: dict, trainer_peak_gb: float | None) -> str:
    free = snap.get("free_vram_gb")
    total = snap.get("total_vram_gb")
    if free is not None and trainer_peak_gb and trainer_peak_gb > 0:
        fit = int(free // trainer_peak_gb)
        fit_text = f"about {fit} timing-sized trainers by VRAM; playbook cap remains <=2"
    elif free is not None:
        fit_text = "trainer fit not estimated because trainer GPU timing is pending"
    else:
        fit_text = "unknown"

    lines = [
        "# Timing Projections",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "All full-run values below are linear projections from the measured unit.",
        "",
        "## Hardware / Occupancy",
        f"- Device: {snap.get('device', 'unknown')}",
        f"- CUDA visible: {snap.get('cuda')}",
        f"- Free VRAM: {free:.2f} / {total:.2f} GB" if free is not None else "- Free VRAM: unavailable",
        f"- Trainer fit estimate: {fit_text}",
        f"- tmux: `{snap.get('tmux', '')}`",
        f"- nvidia-smi: `{snap.get('nvidia_smi', '')}`",
        "",
        "## Timing Table",
        "",
        "| task | per-unit measured | projected full | device | measured-or-pending |",
        "|---|---:|---:|---|---|",
    ]
    for r in rows:
        status = r.status if not r.notes else f"{r.status}; {r.notes}"
        lines.append(
            f"| {r.task} | {r.per_unit} | {r.projected_full} | {r.device} | {status} |"
        )
    lines.extend([
        "",
        "## Raw JSON",
        "",
        "```json",
        json.dumps({
            "hardware": snap,
            "rows": [r.__dict__ for r in rows],
        }, indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/reports/timing_projections.md")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--force-gpu", action="store_true",
                   help="Run GPU timings even if tmux sessions are present.")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    rows: list[Row] = []
    snap = gpu_snapshot()
    run_gpu, reason = should_run_gpu(args, snap)
    trainer_peak_gb = None

    time_trait_state(rows)

    if run_gpu:
        sec_per_step, trainer_peak_gb = time_our_trainer(rows)
        time_eval(rows, version="apgd", n=100, projections=[1000])
        time_eval(rows, version="standard", n=50, projections=[1000, 10000])
        time_ramp8b(rows)
    else:
        pending_gpu_rows(rows, reason)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, snap, trainer_peak_gb), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

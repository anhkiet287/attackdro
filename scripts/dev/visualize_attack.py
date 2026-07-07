#!/usr/bin/env python
"""Visualize an adversarial attack: clean | perturbation | adversarial.

A sanity-check tool for the attack implementations across threat models. For a
handful of CIFAR-10 test images it crafts adversarials under the chosen norm
(linf / l2 / l1) and plots three columns per image:

    clean image  |  perturbation (delta, per-image normalized)  |  adversarial

Each row is annotated with the true label, the model's clean / adversarial
predictions (green = still correct, red = fooled), and the measured linf, l2 and
l1 norms of delta so you can confirm the perturbation respects the active norm's
epsilon budget.

Attacks: linf uses the fast in-repo PGD by default; l2 / l1 use AutoAttack's
APGD (the eval attack). Pass --attack apgd to force APGD for linf too.

By default it runs on CPU so it does NOT compete with a training run on the GPU.

Examples
--------
    python scripts/visualize_attack.py --norm linf --only-correct -n 6
    python scripts/visualize_attack.py --norm l2   -n 6
    python scripts/visualize_attack.py --norm l1   -n 6 --attack apgd
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless / SSH — save to file, no display needed
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.attacks import pgd_linf            # noqa: E402
from robustdro.models import build_model           # noqa: E402
from robustdro.utils.io import load_config         # noqa: E402
from robustdro.utils.seed import set_seed          # noqa: E402

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pgd_at.yaml")
    p.add_argument("--checkpoint", default="checkpoints/pgd_at_linf_best.pt",
                   help="Model checkpoint. If missing, uses a random-init model.")
    p.add_argument("--norm", default="linf", choices=["linf", "l2", "l1"])
    p.add_argument("--attack", default="auto", choices=["auto", "pgd", "apgd"],
                   help="auto = PGD for linf, APGD for l2/l1. pgd only supports linf.")
    p.add_argument("-n", "--num-images", type=int, default=6)
    p.add_argument("--eps", type=float, default=None, help="Override the norm's eps.")
    p.add_argument("--steps", type=int, default=None, help="Override attack steps.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--only-correct", action="store_true",
                   help="Only visualize images classified correctly when clean.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output PNG (default results/figures/attack_viz_<norm>.png)")
    return p.parse_args()


def load_model(args, cfg, device):
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(ckpt.get("cfg", cfg))
        model.load_state_dict(ckpt["model"])
        info = f"checkpoint {args.checkpoint} (epoch {ckpt.get('epoch', '?')})"
    else:
        model = build_model(cfg)
        info = "RANDOM-INIT model (no checkpoint found)"
    model.to(device).eval()
    return model, info


def craft(model, x, y, args, cfg, eps, steps, device):
    """Dispatch to the right attack for the chosen norm."""
    use_apgd = args.attack == "apgd" or (args.attack == "auto" and args.norm != "linf")
    if use_apgd:
        from robustdro.eval import craft_adv  # imports autoattack lazily
        return craft_adv(model, x, y, args.norm, eps, steps=steps, restarts=1,
                         device=device, version="apgd", seed=args.seed, bs=len(x))
    if args.norm != "linf":
        raise ValueError("--attack pgd only supports --norm linf; use apgd for l2/l1.")
    step_size = cfg["train"]["attack"]["step_size"]
    return pgd_linf(model, x, y, eps=eps, step_size=step_size, steps=steps,
                    random_start=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    eps = args.eps if args.eps is not None else cfg["threat_model"][args.norm]["eps"]
    if args.steps is not None:
        steps = args.steps
    elif args.attack == "apgd" or (args.attack == "auto" and args.norm != "linf"):
        steps = cfg.get("eval_attack", {}).get(args.norm, {}).get("steps", 100)
    else:
        steps = cfg["train"]["attack"]["steps"]

    model, info = load_model(args, cfg, device)
    print(f"[viz] model: {info}")
    print(f"[viz] norm={args.norm}  eps={eps:g}  steps={steps}  attack={args.attack}  device={device}")

    test_ds = torchvision.datasets.CIFAR10(
        root=cfg["dataset"]["root"], train=False, download=False, transform=T.ToTensor()
    )

    xs, ys = [], []
    for i in range(len(test_ds)):
        if len(xs) >= args.num_images:
            break
        x_i, y_i = test_ds[i]
        if args.only_correct:
            with torch.no_grad():
                if model(x_i.unsqueeze(0).to(device)).argmax(1).item() != y_i:
                    continue
        xs.append(x_i); ys.append(y_i)

    x = torch.stack(xs).to(device)
    y = torch.tensor(ys, device=device)

    with torch.no_grad():
        clean_pred = model(x).argmax(1)
    x_adv = craft(model, x, y, args, cfg, eps, steps, device).to(device)
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(1)

    delta = (x_adv - x).cpu()
    x_c, x_a = x.cpu(), x_adv.cpu()
    clean_pred, adv_pred, y = clean_pred.cpu(), adv_pred.cpu(), y.cpu()

    n = len(xs)
    fig, axes = plt.subplots(n, 3, figsize=(7.8, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    col_titles = ["clean", "perturbation (normalized)", "adversarial"]

    n_fooled = 0
    for r in range(n):
        d = delta[r]
        linf = d.abs().max().item()
        l2 = d.flatten().norm(p=2).item()
        l1 = d.abs().sum().item()
        # Per-image min-max normalization so perturbations are visible in any norm.
        rng = d.max() - d.min()
        d_vis = (d - d.min()) / (rng + 1e-12)

        for c, img in enumerate([x_c[r], d_vis, x_a[r]]):
            ax = axes[r][c]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)

        fooled = adv_pred[r].item() != y[r].item()
        n_fooled += int(fooled)
        color = "red" if fooled else "green"
        axes[r][0].set_ylabel(f"true: {CIFAR10_CLASSES[y[r].item()]}", fontsize=9)
        axes[r][2].text(
            1.05, 0.5,
            f"clean→{CIFAR10_CLASSES[clean_pred[r].item()]}\n"
            f"adv→{CIFAR10_CLASSES[adv_pred[r].item()]}\n"
            f"linf={linf*255:.2f}/255\nl2={l2:.3f}\nl1={l1:.2f}",
            transform=axes[r][2].transAxes, va="center", ha="left",
            fontsize=8, color=color,
        )

    fig.suptitle(
        f"{args.norm.upper()} attack — {n_fooled}/{n} fooled  (eps={eps:g}, {steps} steps)\n{info}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    out = args.out or f"results/figures/attack_viz_{args.norm}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[viz] fooled {n_fooled}/{n}. Saved -> {out}")

    # Budget sanity check for the active norm.
    order = {"linf": float("inf"), "l2": 2, "l1": 1}[args.norm]
    batch_norm = delta.flatten(1).norm(p=order, dim=1).max().item()
    unit = "/255 (as linf*255)" if args.norm == "linf" else ""
    shown = batch_norm * 255 if args.norm == "linf" else batch_norm
    print(f"[viz] max {args.norm} over batch = {shown:.3f}{unit}  (eps={eps:g})")


if __name__ == "__main__":
    main()

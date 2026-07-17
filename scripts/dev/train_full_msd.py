"""M1a-FULL: full-scale MSD base (Maini appendix recipe) + the M1a pull-push term.

Why a separate file: `c5_fromscratch.py` is the live trainer used by the running seed
chain, so it is not touched. The pull-push term, head, and val-selection proxy are
IMPORTED from it verbatim, which guarantees the representation term is bit-identical to
M1a rather than a re-implementation.

BASE (locuslab/robust_union CIFAR10/train.py + cifar_funcs.py, verbatim values):
  epochs 50 | batch 128 | SGD lr peak 0.1, momentum 0.9, weight_decay 5e-4
  lr(t) = np.interp(t, [0, 20, 40, 50], [0, 0.1, 0.005, 0])     # their one-cycle
  MSD training attack: msd_v0, num_iter 50, alphas (0.003, 0.05, 0.05)

TERM (identical to M1a): 3 APGD views (10 steps each), alpha=beta=0.5, tau=0.1,
  warm-up 0->10 epochs, SimCLR head 512->512->128 (discarded at eval).

ONE THREAT TRIPLE EVERYWHERE: Linf 8/255, L2 0.5, L1 12.
  This is a DELIBERATE deviation from the upstream recipe, which trains at Linf 0.03.
  Everything else in the base is theirs; only the radius changes. Training, val-select
  and the audit therefore all use the same triple, which means:
    * M1a-full is directly comparable to M1a / M0, which also trained at 8/255;
    * there is no train/audit threat-model gap to caveat.
  The triple is imported from the M1a trainer (`EPS`) rather than restated, so it cannot
  drift from the other arms. `MAINI_LINF` below is kept only to document what upstream used.

RESUME SAFETY (Colab can drop at any moment; never lose more than one epoch):
  * `ckpt_latest.pt` is written to OUTDIR after EVERY epoch (model + optimizer +
    epoch counter + best + history).
  * The lr schedule is a pure function of the epoch, so there is no scheduler state to
    restore -- resuming just recomputes lr(epoch). This is why their np.interp schedule
    is used directly instead of torch's OneCycleLR.
  * `train.json` is rewritten with the full history after every epoch, so it doubles as
    a progress record and (at epoch 49, 0-indexed) the done-sentinel.
  * Re-running with the same --outdir resumes silently. No prompts.

Usage:
  ATTACKDRO_ROOT=/content/attackdro python scripts/dev/train_full_msd.py \
      --outdir /content/drive/MyDrive/attackdro/C5_full/M1a_full --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
sys.path.insert(0, str(ROOT / "src"))

from c5_fromscratch import (  # noqa: E402  -- the M1a term, imported verbatim
    EPS as PROXY_EPS,
    MODEL_CFG,
    VAL_SELECT,
    BackboneHead,
    ce_at_loss,
    decoupled_pullpush,
    load_eval_tensors,
    make_aug_loader,
    worst_union_acc,
)
from robustdro.attacks.apgd_train import apgd_train  # noqa: E402
from robustdro.attacks.norms import msd_v0  # noqa: E402

MAINI_LINF = 0.03          # what upstream trains at; recorded for provenance only
TRAIN_EPS = PROXY_EPS      # {"Linf": 8/255, "L2": 0.5, "L1": 12.0} -- imported, cannot drift
NORMS = ["Linf", "L2", "L1"]


def lr_at(t: float, epochs: int, peak: float) -> float:
    """Their schedule: np.interp(t, [0, 2/5, 4/5, 1] * epochs, [0, peak, 0.005, 0])."""
    return float(np.interp([t], [0, epochs * 2 // 5, epochs * 4 // 5, epochs],
                           [0, peak, 0.005, 0])[0])


def main():
    p = argparse.ArgumentParser(description="M1a-FULL: Maini MSD base + M1a pull-push term.")
    p.add_argument("--outdir", required=True, help="Drive-direct output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)          # Maini
    p.add_argument("--bs", type=int, default=128)             # Maini
    p.add_argument("--lr-peak", type=float, default=0.1)      # Maini
    p.add_argument("--wd", type=float, default=5e-4)          # Maini
    p.add_argument("--momentum", type=float, default=0.9)     # Maini
    p.add_argument("--msd-steps", type=int, default=50)       # Maini num_iter
    p.add_argument("--n-iter", type=int, default=10)          # APGD views (== M1a)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("C5_NUM_WORKERS", 2)))
    p.add_argument("--smoke", action="store_true", help="1 epoch, few batches; verifies wiring only")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model = BackboneHead().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr_peak, momentum=a.momentum,
                          weight_decay=a.wd)

    # ---- resume: ckpt_latest is the single source of truth -------------------
    start_ep, best, hist = 0, -1.0, []
    latest = out / "ckpt_latest.pt"
    if latest.exists():
        st = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_ep = int(st["epoch"]) + 1
        best = float(st["best"])
        hist = st["hist"]
        print(f"[M1a-full] RESUMED from epoch {st['epoch']} "
              f"(best_valWU={best:.4f}) -> starting at epoch {start_ep}", flush=True)
    else:
        print("[M1a-full] fresh start (no ckpt_latest.pt in outdir)", flush=True)

    if start_ep >= a.epochs:
        print(f"[M1a-full] already complete ({start_ep}/{a.epochs}). Nothing to do.", flush=True)
        return

    loader = make_aug_loader(a.bs, a.seed, a.num_workers)
    X, Y = load_eval_tensors()
    val_idx = list(range(*VAL_SELECT))

    print(f"[M1a-full] base=MSD steps={a.msd_steps} | views=APGD n_iter={a.n_iter} "
          f"| ONE triple everywhere (train == val-select == audit): {TRAIN_EPS} "
          f"| upstream trains Linf {MAINI_LINF} -- deliberately not used", flush=True)

    for ep in range(start_ep, a.epochs):
        model.train()
        t0 = time.time()
        lr = lr_at(ep + 1, a.epochs, a.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr
        w = min(1.0, (ep + 1) / max(1, a.warmup))     # term warm-up 0 -> 1 over `warmup` epochs
        rl = rc = rg = rs = 0.0
        n = 0
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # base: MSD adversarial drives the CE term
            x_msd = msd_v0(model, x, y, TRAIN_EPS["Linf"], TRAIN_EPS["L2"], TRAIN_EPS["L1"],
                           steps=a.msd_steps)
            lce = ce_at_loss(model, [x_msd], y)
            # term: 3 APGD views feed the pull-push positives (identical to M1a)
            xs_adv = [apgd_train(model, x, y, nm, TRAIN_EPS[nm], n_iter=a.n_iter, is_train=False)
                      for nm in NORMS]
            lpp, comp = decoupled_pullpush(model, x, xs_adv, y, w * a.alpha, w * a.beta,
                                           a.tau, "adv")
            loss = lce + lpp
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} batch {bi}: "
                                   f"ce={float(lce)} pullpush={float(lpp)}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            # .item() rather than float(): the tensors still require grad, and float() on
            # them raises a UserWarning. Values are logging-only either way.
            rl += loss.item(); rc += lce.item(); rg += comp["glue"]; rs += comp["scaffold"]; n += 1
            if a.smoke and bi >= 2:
                break

        wu = worst_union_acc(model, X, Y, val_idx, device, n_iter=20)   # standard-triple proxy
        hist.append({"epoch": ep, "loss": rl / n, "ce": rc / n, "glue": rg / n,
                     "scaffold": rs / n, "val_worst_union": wu, "lr": lr,
                     "sec": round(time.time() - t0)})
        print(f"[M1a-full] ep{ep + 1}/{a.epochs} loss={rl / n:.3f} ce={rc / n:.3f} "
              f"lr={lr:.4f} valWU={wu:.4f} ({hist[-1]['sec']}s)", flush=True)

        # ---- Drive-direct persistence, every epoch --------------------------
        if wu > best:
            best = wu
            torch.save({"cfg": MODEL_CFG, "model": model.b.state_dict()}, out / "val_best.pt")
            torch.save({"epoch": ep, "val_worst_union": wu}, out / "val_best_meta.pt")
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best": best, "hist": hist}, out / "ckpt_latest.pt")
        (out / "train.json").write_text(json.dumps({
            "arm": "M1a_full", "seed": a.seed, "epochs": a.epochs,
            "base": "msd_v0", "msd_steps": a.msd_steps, "train_eps": TRAIN_EPS,
            "view_eps": TRAIN_EPS, "upstream_linf_not_used": MAINI_LINF, "n_iter": a.n_iter, "alpha": a.alpha, "beta": a.beta,
            "tau": a.tau, "warmup": a.warmup, "lr_peak": a.lr_peak, "wd": a.wd,
            "bs": a.bs, "recipe": "locuslab/robust_union CIFAR10/train.py (50ep, np.interp lr); Linf radius changed 0.03 -> 8/255 for cross-arm comparability",
            "val_proxy_eps": PROXY_EPS, "best_val_worst_union": best,
            "epochs_completed": ep + 1, "history": hist,
        }, indent=2))

        if a.smoke:
            print("[M1a-full] SMOKE OK -- wiring verified, stopping after 1 epoch", flush=True)
            return

    print(f"[M1a-full] DONE best_valWU={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

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
  * After EVERY epoch: `ckpt_latest.pt` = model + optimizer (incl. momentum buffers) +
    epoch counter + best + history + RNG state.
  * Writes are ATOMIC (temp file + rename). A ~90 MB save onto Drive is not atomic, so a
    runtime death mid-write would otherwise leave a truncated checkpoint and cost the
    whole run rather than one epoch.
  * The last good checkpoint is rotated to `ckpt_prev.pt` before the new one is written,
    and resume tries latest -> prev. So a death anywhere in the write window still leaves
    one loadable checkpoint: worst case is one lost epoch, which is the stated budget.
  * RNG (torch/cuda/numpy/python) is saved and restored, so attack randomness continues
    on the same stream. The per-epoch data order is seeded as f(seed, epoch), so a resumed
    epoch sees exactly the shuffle an uninterrupted run would have seen.
  * The lr schedule is a pure function of the epoch, so there is no scheduler state to
    restore -- resuming just recomputes lr(epoch). This is why their np.interp schedule
    is used directly instead of torch's OneCycleLR.
  * `train.json` is rewritten (atomically) with the full history after every epoch, so it
    doubles as a progress record and (at epochs_completed == 50) the done-sentinel.
  * Re-running with the same --outdir resumes silently. No prompts.
  Verified by scratchpad/test_resume.py: clean kill, truncated latest, missing latest
  (mid-rotation), RNG continuity, deterministic epoch shuffle -- 9/9.

Usage:
  ATTACKDRO_ROOT=/content/attackdro python scripts/dev/train_full_msd.py \
      --outdir /content/drive/MyDrive/attackdro/C5_full/M1a_full --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _atomic_save(obj, path: Path) -> None:
    """Save via temp file + rename.

    A ~90 MB torch.save straight onto Drive is NOT atomic: if the runtime dies part-way
    through, the checkpoint is left truncated and resume fails -- losing the whole run
    rather than one epoch. Writing to .tmp and renaming means the visible file is only
    ever a complete one (rename is atomic within a filesystem).
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _load_resume(cands, device):
    """First readable checkpoint wins. `ckpt_prev.pt` is the fallback for the case where
    the runtime died during the rename window, so a corrupt latest costs one epoch, not all."""
    for p in cands:
        if not p.exists():
            continue
        try:
            return torch.load(p, map_location=device, weights_only=False), p
        except Exception as e:
            print(f"[M1a-full] {p.name} is unreadable ({type(e).__name__}: {e}); "
                  f"falling back to the previous checkpoint", flush=True)
    return None, None


def _rng_state(device):
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def _restore_rng(st, device):
    """Attack randomness (MSD's k ~ U{5..20}, APGD init) draws from the global RNGs, so
    restoring them keeps a resumed run on the same stream as an uninterrupted one."""
    if not st:
        return False
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if device == "cuda" and st.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception as e:
            print(f"[M1a-full] could not restore CUDA RNG ({e}); continuing", flush=True)
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    return True

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
from robustdro.utils.wandb_log import WandbLogger  # noqa: E402

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
    p.add_argument("--variant", choices=["M1a", "M0"], default="M1a",
                   help="M1a = MSD base + pull-push term; M0 = matched control, pure MSD-AT "
                        "(no head, no APGD views, no term -- the honest full-budget denominator)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)          # Maini
    p.add_argument("--bs", type=int, default=128)             # Maini
    p.add_argument("--lr-peak", type=float, default=0.1)      # Maini
    p.add_argument("--wd", type=float, default=5e-4)          # Maini
    p.add_argument("--momentum", type=float, default=0.9)     # Maini
    p.add_argument("--msd-steps", type=int, default=50)       # Maini num_iter
    p.add_argument("--n-iter", type=int, default=10)          # APGD views (== M1a)
    p.add_argument("--apgd-view-steps", type=int, default=None,
                   help="steps for the 3 APGD glue views ONLY (default = --n-iter = 10); "
                        "does NOT touch the base msd_steps. Staleness test: set 50 to matched-strength views.")
    p.add_argument("--arm", default=None,
                   help="alias for --variant: 'M1a_full'/'M0_full' (or 'M1a'/'M0'); overrides --variant if given")
    p.add_argument("--resume", default="auto",
                   help="no-op: resume is ALWAYS automatic from ckpt_latest.pt in --outdir (idempotent)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("C5_NUM_WORKERS", 2)))
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--smoke", action="store_true", help="1 epoch, few batches; verifies wiring only")
    a = p.parse_args()
    if a.arm:                                    # --arm M1a_full -> variant M1a (cell-compat)
        a.variant = a.arm.replace("_full", "")
    if a.variant not in ("M1a", "M0"):
        p.error(f"--variant/--arm must resolve to 'M1a' or 'M0', got {a.variant!r}")
    # view steps: explicit --apgd-view-steps wins, else falls back to --n-iter (=10) -> byte-identical
    # to every prior run (which never passed --apgd-view-steps). Base msd_steps is untouched.
    view_steps = a.apgd_view_steps if a.apgd_view_steps is not None else a.n_iter

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- W&B: deterministic run id + resume="allow" (inside WandbLogger) means a
    # Colab disconnect RESUMES the same run instead of spawning a new one, so the
    # curve stays continuous across re-attaches.
    # staleness-test runs (views != canonical 10-step) get a distinct W&B run id so they do NOT
    # resume/collide with the canonical M1a_full/M0_full seed runs. views==10 -> unchanged name.
    vtag = "" if view_steps == 10 else f"_v{view_steps}"
    run_name = f"C5_{a.variant}_full_seed{a.seed}{vtag}"
    wcfg = {
        "method": ("clamp_pullpush_full" if a.variant != "M0" else "msd_at_full"),
        "variant": a.variant, "seed": a.seed, "run_name": run_name,
        "wandb": {"project": "attackdro-union", "entity": None, "mode": a.wandb_mode,
                  "tags": ["cifar10", "prn18", "C5", "fromscratch", "pullpush",
                           f"{a.variant}_full", "full-scale", f"msd{a.msd_steps}",
                           f"views{view_steps}"]},
        "arm": f"{a.variant}_full", "epochs": a.epochs, "bs": a.bs, "lr_peak": a.lr_peak,
        "wd": a.wd, "momentum": a.momentum, "msd_steps": a.msd_steps, "n_iter": a.n_iter,
        "alpha": a.alpha, "beta": a.beta, "tau": a.tau, "warmup": a.warmup,
        "train_eps": TRAIN_EPS, "val_proxy_eps": PROXY_EPS,
        "upstream_linf_not_used": MAINI_LINF,
        "recipe": "robust_union CIFAR10/train.py (50ep, np.interp one-cycle peak 0.1); "
                  "Linf radius 0.03 -> 8/255 for cross-arm comparability",
    }
    wb = WandbLogger(wcfg, run_name=run_name)

    model = BackboneHead().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr_peak, momentum=a.momentum,
                          weight_decay=a.wd)

    # ---- resume: ckpt_latest is the single source of truth -------------------
    start_ep, best, hist = 0, -1.0, []
    latest, prev = out / "ckpt_latest.pt", out / "ckpt_prev.pt"
    st, src = _load_resume([latest, prev], device)
    if st is not None:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_ep = int(st["epoch"]) + 1
        best = float(st["best"])
        hist = st["hist"]
        rng_ok = _restore_rng(st.get("rng"), device)
        print(f"[{a.variant}-full] RESUMED from {src.name} @ epoch {st['epoch']} "
              f"(best_valWU={best:.4f}) -> starting at epoch {start_ep} "
              f"| RNG {'restored' if rng_ok else 'NOT in ckpt (older format)'}", flush=True)
    else:
        print(f"[{a.variant}-full] fresh start (no readable checkpoint in outdir)", flush=True)

    if start_ep >= a.epochs:
        print(f"[{a.variant}-full] already complete ({start_ep}/{a.epochs}). Nothing to do.", flush=True)
        return

    loader = make_aug_loader(a.bs, a.seed, a.num_workers)
    X, Y = load_eval_tensors()
    val_idx = list(range(*VAL_SELECT))

    pullpush = a.variant != "M0"     # M0 = pure MSD-AT (no head/views/term); the matched control
    print(f"[{a.variant}-full] variant={a.variant} pull-push={pullpush} | base=MSD steps={a.msd_steps} "
          f"| views=APGD steps={view_steps if pullpush else '(none)'} "
          f"| ONE triple everywhere (train == val-select == audit): {TRAIN_EPS} "
          f"| upstream trains Linf {MAINI_LINF} -- deliberately not used", flush=True)

    for ep in range(start_ep, a.epochs):
        model.train()
        t0 = time.time()
        # Data order = f(seed, epoch), NOT a continuation of a stream. This is what makes a
        # resumed epoch see exactly the shuffle an uninterrupted run would have seen.
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(a.seed * 10_000 + ep)
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
            if pullpush:
                # term: 3 APGD views feed the pull-push positives (identical to M1a)
                xs_adv = [apgd_train(model, x, y, nm, TRAIN_EPS[nm], n_iter=view_steps, is_train=False)
                          for nm in NORMS]
                lpp, comp = decoupled_pullpush(model, x, xs_adv, y, w * a.alpha, w * a.beta,
                                               a.tau, "adv")
            else:
                # M0: no head, no views, no term -- CE on the MSD adversarial only.
                lpp, comp = lce.new_zeros(()), {"glue": 0.0, "scaffold": 0.0}
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
        print(f"[{a.variant}-full] ep{ep + 1}/{a.epochs} loss={rl / n:.3f} ce={rc / n:.3f} "
              f"lr={lr:.4f} valWU={wu:.4f} ({hist[-1]['sec']}s)", flush=True)
        # NOTE: no history replay on resume -- resume="allow" continues the same
        # W&B run, so earlier epochs are already on the curve.
        wb.log({f"train/{k}": v for k, v in hist[-1].items() if k != "epoch"}, step=ep)

        # ---- Drive-direct persistence, every epoch --------------------------
        if wu > best:
            best = wu
            _atomic_save({"cfg": MODEL_CFG, "model": model.b.state_dict()}, out / "val_best.pt")
            _atomic_save({"epoch": ep, "val_worst_union": wu}, out / "val_best_meta.pt")
        # Rotate the last good checkpoint aside before writing the new one, so a death
        # anywhere in this window still leaves one loadable checkpoint on Drive.
        if latest.exists():
            os.replace(latest, prev)
        _atomic_save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                      "best": best, "hist": hist, "rng": _rng_state(device)}, latest)
        _atomic_write_text(json.dumps({
            "arm": f"{a.variant}_full", "variant": a.variant, "seed": a.seed, "epochs": a.epochs,
            "base": "msd_v0", "msd_steps": a.msd_steps, "train_eps": TRAIN_EPS,
            "view_eps": TRAIN_EPS, "upstream_linf_not_used": MAINI_LINF, "n_iter": a.n_iter,
            "apgd_view_steps": view_steps, "alpha": a.alpha, "beta": a.beta,
            "tau": a.tau, "warmup": a.warmup, "lr_peak": a.lr_peak, "wd": a.wd,
            "bs": a.bs, "recipe": "locuslab/robust_union CIFAR10/train.py (50ep, np.interp lr); Linf radius changed 0.03 -> 8/255 for cross-arm comparability",
            "val_proxy_eps": PROXY_EPS, "best_val_worst_union": best,
            "epochs_completed": ep + 1, "history": hist,
        }, indent=2), out / "train.json")

        if a.smoke:
            print(f"[{a.variant}-full] SMOKE OK -- wiring verified, stopping after 1 epoch", flush=True)
            wb.finish()
            return

    wb.summary({"best_val_worst_union": best, "epochs_completed": a.epochs})
    wb.finish()
    print(f"[{a.variant}-full] DONE best_valWU={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

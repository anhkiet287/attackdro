"""Explore #1 -- fine-tune a matched pair from a robust base (MSD.pt) with the CLAMP term.

Question: does the CLAMP representation term add on top of an ALREADY-strong robust base, or
does it saturate the way it does at full training budget? Fine-tune regime is where RAMP reports
~0.53, so it is the honest place to look.

Matched pair, identical except the term:
  --arm clamp : continued MSD-AT  +  decoupled CLAMP term (glue + scaffold), the paper's term
  --arm none  : continued MSD-AT only (no head, no views, no term) -- the matched control

Everything else is shared: same MSD.pt init, same fine-tune recipe, same val-select proxy
(standard-triple worst-union, so val_best is chosen identically), same seed. Backbone is the
public robust_union PreActResNet18 (MSD.pt's own arch), so val_best audits via
union_bench_eval --arch robust_union_preact under the frozen 12-AA. We report BOTH union AND
clean, because the fine-tune regime trades them off.

Pre-registered: report regardless of outcome. Extension only -- does NOT touch the paper.

Usage:
  python scripts/dev/finetune_msd_clamp.py --arm clamp --epochs 8 --seed 0 \
      --outdir results/fromscratch/explore1_ft/clamp
"""
from __future__ import annotations

import argparse
import hashlib
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

from c0_killtest import BackboneHead as MSDBackboneHead, load_msd_backbone, load_train_tensors  # noqa: E402
from c5_fromscratch import (  # noqa: E402
    EPS, NORMS, VAL_SELECT, MODEL_CFG, BackboneHead as C5BackboneHead,
    ce_at_loss, decoupled_pullpush, worst_union_acc, pernorm_train_metrics,
)
from robustdro.attacks.apgd_train import apgd_train, _l2_norm, _l1_projection  # noqa: E402
from robustdro.attacks.norms import msd_v0  # noqa: E402
from train_full_msd import _atomic_save, _load_resume, _rng_state, _restore_rng  # noqa: E402  resume helpers
from robustdro.utils.wandb_log import WandbLogger  # noqa: E402

TRAIN_CORE = (0, 49000)


import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class RBBackboneHead(nn.Module):
    """RobustBench encoder + SimCLR head (discarded at eval). Penultimate features are read via
    a forward hook on the classifier's input, so it works for ANY RobustBench CIFAR model that
    ends in a Linear (ResNet-18, WRN, ...). Head dim inferred from the classifier's in_features."""
    def __init__(self, rb_model, hid=512, proj=128):
        super().__init__()
        self.b = rb_model
        self._clf = [m for m in rb_model.modules() if isinstance(m, nn.Linear)][-1]
        feat = self._clf.in_features
        self._feat = {}
        self._clf.register_forward_hook(lambda mod, inp, out: self._feat.__setitem__("f", inp[0]))
        self.head = nn.Sequential(nn.Linear(feat, hid), nn.ReLU(inplace=True), nn.Linear(hid, proj))
        self.feat_dim = feat

    def logits(self, x):
        return self.b(x)

    def features(self, x):
        self.b(x)                       # hook captures the classifier's input (penultimate)
        return self._feat["f"]

    def embed(self, x):
        return F.normalize(self.head(self.features(x)), dim=1)

    def forward(self, x):
        return self.b(x)


def load_rb_backbone(name, device):
    """Load a RobustBench CIFAR-10 Linf model by name. Handles torch 2.x weights_only + the
    gdrive download (RobustBench's util does the fetch; gdown must be a version that clicks the
    virus-scan token). Input convention is [0,1] (verified by the B2 reproduce-check)."""
    import torch as _t
    _orig = _t.load
    _t.load = lambda *ar, **kw: _orig(*ar, **{**kw, "weights_only": False})
    try:
        from robustbench.utils import load_model
        m = load_model(model_name=name, dataset="cifar10", threat_model="Linf",
                       model_dir=str(ROOT / "models"))
    finally:
        _t.load = _orig
    return m.to(device)


def load_robustdro_backbone(ckpt_path, device):
    """Load an in-repo robustdro PreActResNet18 backbone from a {cfg, model} checkpoint
    (e.g. M0_full/val_best.pt) into a c5_fromscratch BackboneHead (robust backbone + FRESH SimCLR
    head). Construction of .b is identical to the frozen harness's loader, so fine-tuning starts
    from EXACTLY the M0_full weights + a random head (the head trains during fine-tune)."""
    import torch as _t
    ck = _t.load(ckpt_path, map_location=device, weights_only=False)
    bh = C5BackboneHead()                         # fresh robustdro backbone + SimCLR head
    bh.b.load_state_dict(ck["model"])             # overwrite backbone with the robust base weights
    return bh.to(device)


def build_model(base, device, base_ckpt=None):
    """base = 'msd' (robust_union MSD.pt) | 'robustbench:<name>' | 'robustdro' (+ base_ckpt path).
    Same CLAMP head + interface either way, so the fine-tune loop and the term are identical
    across bases (fairness)."""
    if base_ckpt:                                 # regime (b): fine-tune from an in-repo robustdro ckpt
        return load_robustdro_backbone(base_ckpt, device)
    if base == "msd":
        return MSDBackboneHead(load_msd_backbone(device)).to(device)
    if base.startswith("robustbench:"):
        return RBBackboneHead(load_rb_backbone(base.split(":", 1)[1], device)).to(device)
    raise ValueError(f"unknown --base {base!r} (use 'msd' | 'robustbench:<name>' | --base-ckpt <path>)")


def fat_view_attack(model, x, y, norm, eps, kappa=0.0, max_steps=50):
    """FAT-style FRIENDLY view (Zhang 2020, adapted per-norm). Per-sample per-norm APGD-family
    step (momentum a=0.75, per-norm projection), but each sample FREEZES the instant its margin
    (logit_y − max_other) ≤ kappa — its perturbation is kept at that step and it is not stepped
    again; capped at max_steps. Already-misclassified samples freeze at step 0 (no perturbation).
    Returns (x_fat, steps_per_sample[int], never_fooled[bool]) — never-fooled samples run the full
    max_steps and return their final iterate. Base msd_steps is untouched; this only crafts the
    3 glue views. Expected step-count ordering ℓ₁ ≫ ℓ∞ (Croce&Hein) is what the fat/* logs test."""
    device = x.device; B = x.shape[0]; ndims = len(x.shape) - 1
    alpha = 2.0 if norm in ("Linf", "L2") else 1.0
    step_size = alpha * eps
    n_fts = x[0].numel()
    x_adv = x.clone().clamp(0., 1.)
    x_fat = x_adv.clone()
    frozen = torch.zeros(B, dtype=torch.bool, device=device)
    steps = torch.full((B,), max_steps, dtype=torch.long, device=device)
    x_adv_old = x_adv.clone()
    for i in range(max_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="none").sum()
        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        x_adv = x_adv.detach()
        with torch.no_grad():                     # margin on the CURRENT iterate -> freeze newly-fooled
            lg = logits.detach()
            true = lg.gather(1, y[:, None]).squeeze(1)
            other = lg.clone().scatter_(1, y[:, None], float("-inf")).max(1).values
            newly = ((true - other) <= kappa) & (~frozen)
        if newly.any():
            x_fat[newly] = x_adv[newly].clone(); steps[newly] = i; frozen[newly] = True
        if frozen.all():
            break
        a = 0.75 if i > 0 else 1.0
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv.clone()
        if norm == "Linf":
            x1 = torch.clamp(torch.min(torch.max(x_adv + step_size * torch.sign(grad), x - eps), x + eps), 0., 1.)
            x1 = torch.clamp(torch.min(torch.max(x_adv + (x1 - x_adv) * a + grad2 * (1 - a), x - eps), x + eps), 0., 1.)
        elif norm == "L2":
            x1 = x_adv + step_size * grad / (_l2_norm(grad, keepdim=True) + 1e-12)
            x1 = torch.clamp(x + (x1 - x) / (_l2_norm(x1 - x, keepdim=True) + 1e-12)
                             * torch.min(eps * torch.ones_like(x), _l2_norm(x1 - x, keepdim=True)), 0., 1.)
            x1 = x_adv + (x1 - x_adv) * a + grad2 * (1 - a)
            x1 = torch.clamp(x + (x1 - x) / (_l2_norm(x1 - x, keepdim=True) + 1e-12)
                             * torch.min(eps * torch.ones_like(x), _l2_norm(x1 - x, keepdim=True)), 0., 1.)
        else:  # L1: fixed 5% top-k (== apgd_train step-0) + L1-ball projection
            u = torch.arange(B, device=device)
            gt = grad.abs().view(B, -1).sort(-1)[0]
            topk_curr = min(int(0.95 * n_fts), n_fts - 1)
            gt = gt[u, topk_curr].view(-1, *[1] * ndims)
            sg = grad * (grad.abs() >= gt).float()
            x1 = x_adv + step_size * sg.sign() / (sg.sign().abs().view(B, -1).sum(-1).view(-1, *[1] * ndims) + 1e-10)
            du = x1 - x; x1 = x + du + _l1_projection(x, du, eps)
        x_adv = torch.where(frozen.view(-1, *[1] * ndims), x_fat, x1)   # frozen samples stay put
    nf = ~frozen
    if nf.any():
        x_fat[nf] = x_adv[nf].clone()             # never-fooled -> final iterate
    return x_fat.detach(), steps, nf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["clamp", "none"], required=True)
    p.add_argument("--base", default="msd",
                   help="'msd' (robust_union MSD.pt) or 'robustbench:<model_name>'")
    p.add_argument("--base-ckpt", default=None,
                   help="regime (b): fine-tune from an in-repo robustdro {cfg,model} ckpt "
                        "(e.g. C5_full/M0_full/val_best.pt); overrides --base")
    p.add_argument("--view-attack", choices=["apgd", "fat"], default="apgd",
                   help="glue view attack: 'apgd' (fixed n_iter, DEFAULT — byte-identical) or "
                        "'fat' (friendly, per-sample early-stop at margin<=kappa)")
    p.add_argument("--fat-kappa", type=float, default=0.0,
                   help="FAT freeze threshold on margin(logit_y - max_other); freeze when <= kappa")
    p.add_argument("--fat-max-steps", type=int, default=50, help="FAT per-sample step cap K_max")
    p.add_argument("--dry-run", action="store_true",
                   help="build + print config + time ONE batch (s/epoch estimate) then EXIT; no training")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", "--finetune-epochs", type=int, default=15, dest="epochs")  # LOCKED: val-select picks best
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)        # LOCKED: gentle -> stays a fine-tune, not a retrain
    p.add_argument("--msd-steps", type=int, default=10)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2)          # LOCKED: term full-strength sooner (head needs >=2 ep)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("C5_NUM_WORKERS", 4)))
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--no-log-train-pernorm", dest="log_train_pernorm", action="store_false",
                   help="disable standard per-norm TRAIN logging (train/acc_*, loss_*, worst_union); default ON, RNG-isolated")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.outdir); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    clamp = a.arm == "clamp"

    base_desc = f"robustdro:{a.base_ckpt}" if a.base_ckpt else a.base
    model = build_model(a.base, device, a.base_ckpt)
    # FULL fine-tune: encoder is NOT frozen (CLAMP must be able to reshape features; freezing
    # would neuter the term). Every parameter is trainable and goes to the optimizer.
    for p_ in model.parameters():
        p_.requires_grad_(True)
    n_enc = sum(p_.numel() for p_ in model.b.parameters())
    n_head = sum(p_.numel() for p_ in model.head.parameters())
    print(f"[ft-{a.arm}] FULL fine-tune: encoder {n_enc/1e6:.2f}M (trainable) + head {n_head/1e3:.0f}K", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    full = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=True, download=False, transform=tf)
    core = torch.utils.data.Subset(full, list(range(*TRAIN_CORE)))
    g = torch.Generator().manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(core, batch_size=a.bs, shuffle=True,
                                         num_workers=a.num_workers, drop_last=True, generator=g)
    X, Y = load_train_tensors(device)
    val_idx = list(range(*VAL_SELECT))

    view_desc = (f"FAT (kappa={a.fat_kappa}, K_max={a.fat_max_steps})" if a.view_attack == "fat"
                 else f"APGD (n_iter={a.n_iter})") if clamp else "(no term)"
    print(f"[ft-{a.arm}] base={base_desc} + {'CLAMP term' if clamp else 'no term (matched control)'} "
          f"| views={view_desc} | feat_dim={getattr(model,'feat_dim',512)} epochs={a.epochs} lr={a.lr} "
          f"| val-select standard-triple", flush=True)

    def make_views(x, y):
        """3 glue views + per-norm FAT raw counts (for EXACT epoch-level means). stats {} for APGD.
        Per norm: [sum_steps_all, n, sum_steps_fooled, n_fooled] — so the epoch aggregates
        mean_steps (incl never-fooled=K_max → robustness) AND mean_steps_fooled (excl never-fooled
        → the Croce&Hein convergence-rate claim) without averaging batch-means."""
        if a.view_attack == "fat":
            xs, st = [], {}
            for nm in NORMS:
                xv, steps, nf = fat_view_attack(model, x, y, nm, EPS[nm], a.fat_kappa, a.fat_max_steps)
                xs.append(xv); f = ~nf
                st[nm] = [float(steps.sum()), int(steps.numel()),
                          float(steps[f].sum()) if f.any() else 0.0, int(f.sum())]
            return xs, st
        return [apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=False) for nm in NORMS], {}

    # ---- DRY-RUN: build + time ONE batch (msd base + views + loss + backward), estimate s/epoch, EXIT ----
    if a.dry_run:
        model.train()
        xb, yb = next(iter(loader)); xb, yb = xb.to(device), yb.to(device)
        if device == "cuda": torch.cuda.synchronize()
        tb = time.time()
        x_msd = msd_v0(model, xb, yb, EPS["Linf"], EPS["L2"], EPS["L1"], steps=a.msd_steps)
        lce = ce_at_loss(model, [x_msd], yb)
        if clamp:
            xs, st = make_views(xb, yb)
            lpp, _ = decoupled_pullpush(model, xb, xs, yb, a.alpha, a.beta, a.tau, "adv")
        else:
            lpp, st = lce.new_zeros(()), {}
        (lce + lpp).backward()
        if device == "cuda": torch.cuda.synchronize()
        per_batch = time.time() - tb
        nb = len(loader)
        print(f"[ft-{a.arm}] DRY-RUN: 1 batch = {per_batch:.2f}s  x {nb} batches -> "
              f"~{per_batch*nb:.0f}s/epoch (~{per_batch*nb*a.epochs/3600:.2f}h for {a.epochs} ep)", flush=True)
        if st:
            print("           FAT views (1 batch): " + " ".join(
                f"{nm} mean_steps={st[nm][0]/st[nm][1]:.1f} mean_steps_fooled="
                f"{(st[nm][2]/st[nm][3]) if st[nm][3] else float('nan'):.1f} "
                f"never_fooled={1-st[nm][3]/st[nm][1]:.3f}" for nm in NORMS), flush=True)
        print(f"[ft-{a.arm}] WIRED, READY (dry-run only, no training performed).", flush=True)
        return

    # ---- W&B (same project/logger as c5/train_full; resume="allow" -> curve continuous across reconnects) ----
    run_name = f"{out.name}_seed{a.seed}"                 # deterministic id from outdir -> resume-safe, distinct per arm
    wb = WandbLogger({"wandb": {"project": "attackdro-union", "entity": None, "mode": a.wandb_mode,
                                "required": False, "group": "fatclamp",
                                "tags": ["cifar10", "prn18", "finetune", f"arm-{a.arm}", f"view-{a.view_attack}",
                                         base_desc.split("/")[-1][:24], f"seed{a.seed}"]},
                      "arm": a.arm, "base": base_desc, "view_attack": a.view_attack, "fat_kappa": a.fat_kappa,
                      "fat_max_steps": a.fat_max_steps, "epochs": a.epochs, "lr": a.lr, "alpha": a.alpha,
                      "beta": a.beta, "tau": a.tau, "msd_steps": a.msd_steps, "n_iter": a.n_iter}, run_name=run_name)

    def wb_payload(rec):
        p = {}
        for k in ("loss", "ce", "pp", "lr", "sec"):
            if k in rec: p[f"train/{k}"] = rec[k]
        for k, v in rec.items():
            if k.startswith("train/"): p[k] = v          # train/acc_*, loss_*, worst_union (already prefixed)
        if "val_worst_union" in rec: p["val/worst_union"] = rec["val_worst_union"]
        for k, v in rec.get("fat", {}).items(): p[f"fat/{k}"] = v
        return p

    # ---- resume-safe (Colab): full-state ckpt_latest every epoch → a disconnect CONTINUES, never re-trains from 0 ----
    latest, prev = out / "ckpt" / "ckpt_latest.pt", out / "ckpt" / "ckpt_prev.pt"
    start_ep, best, hist = 0, -1.0, []
    _st, _src = _load_resume([latest, prev], device)
    if _st is not None:
        model.load_state_dict(_st["model"]); opt.load_state_dict(_st["opt"]); sched.load_state_dict(_st["sched"])
        start_ep = int(_st["epoch"]) + 1; best = float(_st["best"]); hist = _st["hist"]
        _restore_rng(_st.get("rng"), device)
        print(f"[ft-{a.arm}] RESUMED from {_src.name} @ epoch {_st['epoch']} (best={best:.4f}) -> start ep {start_ep}", flush=True)
    if start_ep >= a.epochs:
        print(f"[ft-{a.arm}] already complete ({start_ep}/{a.epochs}); nothing to do.", flush=True)
        wb.finish(); return

    def write_trainjson(done):
        d = {"arm": a.arm, "clamp": clamp, "base": base_desc, "base_ckpt": a.base_ckpt, "seed": a.seed,
             "epochs": a.epochs, "view_attack": a.view_attack, "fat_kappa": a.fat_kappa, "fat_max_steps": a.fat_max_steps,
             "lr": a.lr, "msd_steps": a.msd_steps, "n_iter": a.n_iter, "alpha": a.alpha, "beta": a.beta, "tau": a.tau,
             "best_val_worst_union": best, "epochs_completed": done, "history": hist}
        vb = out / "ckpt" / "val_best.pt"
        if vb.exists(): d["val_best_sha256"] = hashlib.sha256(vb.read_bytes()).hexdigest()
        (out / "train.json").write_text(json.dumps(d, indent=2))

    for ep in range(start_ep, a.epochs):
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(a.seed * 10_000 + ep)   # data order = f(seed,ep) → resumed epoch == uninterrupted
        model.train(); t0 = time.time()
        w = min(1.0, (ep + 1) / max(1, a.warmup))
        rl = rc = rp = 0.0; n = 0
        fat_ep = {nm: [0.0, 0, 0.0, 0] for nm in NORMS}    # [sum_steps_all, n, sum_steps_fooled, n_fooled]
        tl_pn = {}; tl_pn_n = 0                            # standard per-norm TRAIN logging accumulators
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_msd = msd_v0(model, x, y, EPS["Linf"], EPS["L2"], EPS["L1"], steps=a.msd_steps)
            lce = ce_at_loss(model, [x_msd], y)
            if clamp:
                xs, st = make_views(x, y)
                for nm in st:
                    for j in range(4): fat_ep[nm][j] += st[nm][j]
                lpp, _ = decoupled_pullpush(model, x, xs, y, w * a.alpha, w * a.beta, a.tau, "adv")
            else:
                xs, lpp = None, lce.new_zeros(())
            loss = lce + lpp
            if not torch.isfinite(loss):
                raise RuntimeError(f"[ft-{a.arm}] non-finite loss ep{ep}: ce={lce.item()} pp={float(lpp)}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            rl += loss.item(); rc += lce.item(); rp += float(lpp); n += 1
            if a.log_train_pernorm:                       # standard per-norm TRAIN proxy (RNG-isolated)
                avail = {NORMS[i]: xs[i] for i in range(len(xs))} if (clamp and xs is not None and len(xs) == 3) else {}
                tm = pernorm_train_metrics(model, x, y, avail, n_iter=a.n_iter)
                for kk, vv in tm.items(): tl_pn[kk] = tl_pn.get(kk, 0.0) + vv
                tl_pn_n += 1
        sched.step()
        wu = worst_union_acc(model, X, Y, val_idx, device, n_iter=20)
        rec = {"epoch": ep, "loss": rl / n, "ce": rc / n, "pp": rp / n,
               "val_worst_union": wu, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)}
        if a.view_attack == "fat" and clamp:                # exact epoch means from raw counts
            fx = {}
            for nm in NORMS:
                sa, N, sf, nf = fat_ep[nm]; lo = nm.lower()
                fx[f"mean_steps_{lo}"] = sa / N                                  # incl never-fooled -> robustness
                fx[f"mean_steps_fooled_{lo}"] = (sf / nf) if nf else float("nan")  # excl -> convergence rate (C&H)
                fx[f"frac_never_fooled_{lo}"] = 1 - nf / N
            fx["frac_never_fooled"] = 1 - sum(fat_ep[nm][3] for nm in NORMS) / sum(fat_ep[nm][1] for nm in NORMS)
            rec["fat"] = fx
        if a.log_train_pernorm and tl_pn_n > 0:            # train/acc_*, loss_*, worst_union
            for kk, vv in tl_pn.items(): rec[f"train/{kk}"] = vv / tl_pn_n
        hist.append(rec)
        _fat = (" | FAT steps_fooled " + " ".join(
                f"{nm.lower()}={rec['fat'][f'mean_steps_fooled_{nm.lower()}']:.1f}" for nm in NORMS)
                + f" nf={rec['fat']['frac_never_fooled']:.3f}") if "fat" in rec else ""
        print(f"[ft-{a.arm}] ep{ep+1}/{a.epochs} loss={rl/n:.3f} ce={rc/n:.3f} pp={rp/n:.3f} "
              f"valWU={wu:.4f} ({rec['sec']}s){_fat}", flush=True)
        wb.log(wb_payload(rec), step=ep)                 # W&B: train/*, fat/*, val/worst_union
        if wu > best:
            best = wu
            torch.save(model.b.state_dict(), out / "ckpt" / "val_best.pt")   # robust_union arch, plain sd
        if latest.exists():                                                  # rotate prev, then atomic full-state save
            os.replace(latest, prev)
        _atomic_save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "best": best, "hist": hist, "rng": _rng_state(device)}, latest)
        write_trainjson(ep + 1)                                             # per-epoch → disconnect keeps progress
    torch.save(model.b.state_dict(), out / "ckpt" / "last.pt")
    write_trainjson(a.epochs)
    wb.summary({"best_val_worst_union": best, "epochs_completed": a.epochs}); wb.finish()
    print(f"[ft-{a.arm}] DONE best_valWU={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

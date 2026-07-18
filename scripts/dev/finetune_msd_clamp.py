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
    EPS, NORMS, VAL_SELECT, ce_at_loss, decoupled_pullpush, worst_union_acc,
)
from robustdro.attacks.apgd_train import apgd_train  # noqa: E402
from robustdro.attacks.norms import msd_v0  # noqa: E402

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


def build_model(base, device):
    """base = 'msd' (robust_union MSD.pt) | 'robustbench:<name>'. Same CLAMP head + interface
    either way, so the fine-tune loop and the term are identical across bases (fairness)."""
    if base == "msd":
        return MSDBackboneHead(load_msd_backbone(device)).to(device)
    if base.startswith("robustbench:"):
        return RBBackboneHead(load_rb_backbone(base.split(":", 1)[1], device)).to(device)
    raise ValueError(f"unknown --base {base!r} (use 'msd' or 'robustbench:<model_name>')")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["clamp", "none"], required=True)
    p.add_argument("--base", default="msd",
                   help="'msd' (robust_union MSD.pt) or 'robustbench:<model_name>'")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)         # LOCKED: wide budget, val-select picks best
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)        # LOCKED: gentle -> stays a fine-tune, not a retrain
    p.add_argument("--msd-steps", type=int, default=10)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2)          # LOCKED: term full-strength sooner (head needs >=2 ep)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("C5_NUM_WORKERS", 4)))
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.outdir); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    clamp = a.arm == "clamp"

    model = build_model(a.base, device)
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

    print(f"[ft-{a.arm}] base={a.base} + {'CLAMP term' if clamp else 'no term (matched control)'} "
          f"| feat_dim={getattr(model,'feat_dim',512)} epochs={a.epochs} lr={a.lr} | val-select standard-triple", flush=True)

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        w = min(1.0, (ep + 1) / max(1, a.warmup))
        rl = rc = rp = 0.0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_msd = msd_v0(model, x, y, EPS["Linf"], EPS["L2"], EPS["L1"], steps=a.msd_steps)
            lce = ce_at_loss(model, [x_msd], y)
            if clamp:
                xs = [apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=False) for nm in NORMS]
                lpp, _ = decoupled_pullpush(model, x, xs, y, w * a.alpha, w * a.beta, a.tau, "adv")
            else:
                lpp = lce.new_zeros(())
            loss = lce + lpp
            if not torch.isfinite(loss):
                raise RuntimeError(f"[ft-{a.arm}] non-finite loss ep{ep}: ce={lce.item()} pp={float(lpp)}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            rl += loss.item(); rc += lce.item(); rp += float(lpp); n += 1
        sched.step()
        wu = worst_union_acc(model, X, Y, val_idx, device, n_iter=20)
        hist.append({"epoch": ep, "loss": rl / n, "ce": rc / n, "pp": rp / n,
                     "val_worst_union": wu, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)})
        print(f"[ft-{a.arm}] ep{ep+1}/{a.epochs} loss={rl/n:.3f} ce={rc/n:.3f} pp={rp/n:.3f} "
              f"valWU={wu:.4f} ({hist[-1]['sec']}s)", flush=True)
        if wu > best:
            best = wu
            torch.save(model.b.state_dict(), out / "ckpt" / "val_best.pt")   # robust_union arch, plain sd
    torch.save(model.b.state_dict(), out / "ckpt" / "last.pt")
    (out / "train.json").write_text(json.dumps({
        "arm": a.arm, "clamp": clamp, "base": a.base, "seed": a.seed, "epochs": a.epochs,
        "lr": a.lr, "msd_steps": a.msd_steps, "alpha": a.alpha, "beta": a.beta, "tau": a.tau,
        "best_val_worst_union": best, "epochs_completed": a.epochs, "history": hist,
        "val_best_sha256": hashlib.sha256((out / "ckpt" / "val_best.pt").read_bytes()).hexdigest(),
    }, indent=2))
    print(f"[ft-{a.arm}] DONE best_valWU={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

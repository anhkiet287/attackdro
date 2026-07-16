"""Paper C — Stage C5 FROM-SCRATCH pull-push (CONFIRMATORY-grade, NEW training).

Authorized 2026-07-14 after C1b showed the 3-epoch fine-tune probe is under-powered
(even the RAMP positive control ties A0). This trains the pull-push method FROM SCRATCH
under the RAMP-80 recipe, so a representation loss has room to reshape geometry.

TRAIN ONLY the pull-push method (M1a/M1b). Everything with a checkpoint (MSD/RAMP/MAX/AVG/
B3/B4) is EVAL-ONLY under the frozen 12-AA harness — this script does not touch them.

Variants (Stage-1 screen, 1 seed each, SAME seed):
  M1a  neg-source=adv    — push other-class ADVERSARIAL embeddings   (C0/C1 behaviour)
  M1b  neg-source=clean  — push other-class CLEAN embeddings         (transitive-separation)

Loss (decoupled)   L = L_CE-AT + α·L_scaffold + β·L_glue
  L_CE-AT   = mean_i max_{p∈{∞,2,1}} CE(logits(x_i^p), y_i)   (worst-case per-norm APGD; == C0 base)
  L_glue    = mean(-pos_sim/τ)                 pull own adv → own DETACHED clean anchor (anchor A)
  L_scaffold= mean(logsumexp_neg(neg_sim/τ))   push own adv away from other-class negatives
  (α=β=λ recovers the decoupled-contrastive loss, ≈ C0 NT-Xent up to the neg-source swap.)
  α,β linearly warm up 0→target over epochs [0, warmup]; CE-AT active from epoch 0.

Recipe (== B3/B4 native, for comparability): PreActResNet-18 (robustdro, ReLU), standard triple
ℓ∞8/255/ℓ₂0.5/ℓ₁12, 80 ep, bs128, SGD lr0.05 mom0.9 wd5e-4, MultiStepLR@70 γ0.1,
RandomCrop(32,pad4)+HFlip, train_core train[0:49000], val_select train[49000:50000] (worst-union
selection). SimCLR head 512→512→128 (L2-norm, DISCARDED at eval). Adversarials via apgd_train
n_iter=10 per norm (== C0). ⚠ REVISABLE config (probe couldn't pre-tune): base=worst-case (not
true-MSD); attacks=APGD (B3/B4 used pgd_*). val_best saved in robustdro format for harness audit.
"""
from __future__ import annotations
import argparse, json, sys, hashlib, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))  # override on Colab
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from robustdro.models import build_model                    # robustdro PreActResNet18 (== B3/B4)
from robustdro.attacks.apgd_train import apgd_train
from robustdro.attacks.norms import msd_v0                   # true MSD (multi-steepest-descent) base
from robustdro.utils.wandb_log import WandbLogger            # same W&B logger as B3/B4

EPS = {"Linf": 8/255, "L2": 0.5, "L1": 12.0}
NORMS = ["Linf", "L2", "L1"]
TRAIN_CORE = (0, 49000)
VAL_SELECT = (49000, 50000)   # 1k worst-union selection (== B3/B4 val_holdout)
MODEL_CFG = {"model": {"arch": "preact_resnet18", "normalize": False},
             "dataset": {"num_classes": 10},
             "threat_model": {n.lower(): {"eps": EPS[n]} for n in NORMS}}


class BackboneHead(nn.Module):
    """robustdro PreActResNet18 backbone + SimCLR head (discarded at eval)."""
    def __init__(self):
        super().__init__()
        self.b = build_model(MODEL_CFG)                      # fresh random init
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))

    def features(self, x):
        b = self.b
        out = b.conv1(x); out = b.layer1(out); out = b.layer2(out); out = b.layer3(out); out = b.layer4(out)
        out = F.relu(b.bn(out))                              # robustdro has a final BN+relu
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)   # 512-d penultimate
        return out

    def logits(self, x):
        return self.b(x)

    def embed(self, x):
        return F.normalize(self.head(self.features(x)), dim=1)

    def forward(self, x):
        return self.logits(x)


def ce_at_loss(model, xs_adv, y):
    ce = torch.stack([F.cross_entropy(model.logits(xa), y, reduction="none") for xa in xs_adv], 0)
    return ce.max(0).values.mean()


def decoupled_pullpush(model, x_clean, xs_adv, y, alpha, beta, tau=0.1, neg_source="adv"):
    """Decoupled pull-push. glue = pull own adv → own detached clean anchor;
    scaffold = push own adv away from other-class negatives (adv or clean embeddings)."""
    B = x_clean.shape[0]
    with torch.no_grad():
        z_anchor = model.embed(x_clean)                      # (B,d) stop-grad anchor (glue target)
    z_adv = torch.cat([model.embed(xa) for xa in xs_adv], 0)  # (3B,d)
    anchor_idx = torch.arange(3 * B, device=x_clean.device) % B
    y3 = y.repeat(3)

    pos_sim = (z_adv * z_anchor[anchor_idx]).sum(1)          # (3B,)
    glue = (-pos_sim / tau).mean()

    if neg_source == "adv":
        z_neg = z_adv; y_neg = y3                            # other-class adv embeddings (grad)
    elif neg_source == "clean":
        z_neg = model.embed(x_clean); y_neg = y              # other-class clean embeddings (grad)
    else:
        raise ValueError(neg_source)
    sim = z_adv @ z_neg.t()                                  # (3B, N)
    neg_mask = (y3[:, None] != y_neg[None, :])
    neg_logits = (sim / tau).masked_fill(~neg_mask, float("-inf"))
    row_has_neg = neg_mask.any(1)
    scaffold = torch.logsumexp(neg_logits[row_has_neg], dim=1).mean() if row_has_neg.any() \
               else z_adv.new_zeros(())
    return alpha * scaffold + beta * glue, {"glue": float(glue.detach()), "scaffold": float(scaffold.detach())}


@torch.no_grad()
def collapse_diag(model, x, x_adv):
    zc = model.embed(x); za = model.embed(x_adv)
    align = (za * zc).sum(1).mean().item()
    sq = torch.pdist(zc, p=2).pow(2)
    unif = torch.log(torch.exp(-2.0 * sq).mean() + 1e-12).item() if len(sq) else float("nan")
    norm = model.head(model.features(x)).norm(dim=1).mean().item()
    return {"alignment": align, "uniformity": unif, "embed_norm": norm}


@torch.no_grad()
def worst_union_acc(model, X, Y, idx, device, bs=250, n_iter=20):
    """val_select worst-union robust acc (== B3/B4 selection metric): AND over 3-norm APGD."""
    model.eval()
    xs = X[idx].to(device); ys = Y[idx].to(device)
    robust = torch.ones(len(idx), dtype=torch.bool, device=device)
    for nm in NORMS:
        pred_ok = torch.zeros(len(idx), dtype=torch.bool, device=device)
        for i in range(0, len(idx), bs):
            xb = xs[i:i+bs]; yb = ys[i:i+bs]
            with torch.enable_grad():
                xa = apgd_train(model, xb, yb, nm, EPS[nm], n_iter=n_iter, is_train=False)
            pred_ok[i:i+bs] = model(xa).argmax(1) == yb
        robust &= pred_ok
    return robust.float().mean().item()


def make_aug_loader(bs, seed, num_workers=4):
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    full = torchvision.datasets.CIFAR10(root=str(ROOT/"data"), train=True, download=False, transform=tf)
    core = torch.utils.data.Subset(full, list(range(*TRAIN_CORE)))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(core, batch_size=bs, shuffle=True, num_workers=num_workers,
                                       drop_last=True, generator=g)


def load_eval_tensors():
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=str(ROOT/"data"), train=True, download=False,
                                      transform=T.Compose([T.ToTensor()]))
    X = torch.stack([ds[i][0] for i in range(50000)]); Y = torch.tensor([ds[i][1] for i in range(50000)])
    return X, Y


# ------------------------- embedding-trajectory dump -------------------------
def build_tracked(X, Y, classes=(0, 1, 2), k=8, seed=0):
    rng = np.random.default_rng(seed); tracked = []
    for c in classes:
        pool = np.where((Y.numpy() == c) & (np.arange(len(Y)) >= TRAIN_CORE[0]) & (np.arange(len(Y)) < TRAIN_CORE[1]))[0]
        for i in rng.choice(pool, k, replace=False):
            tracked.append({"id": int(i), "cls": int(c)})
    return tracked


@torch.no_grad()
def dump_embeddings(model, X, Y, tracked, device, pca=None):
    model.eval()
    ids = [t["id"] for t in tracked]
    xc = X[ids].to(device)
    zc = model.embed(xc).cpu().numpy()
    adv = {}
    for nm in NORMS:
        with torch.enable_grad():
            xa = apgd_train(model, xc, Y[ids].to(device), nm, EPS[nm], n_iter=10, is_train=False)
        adv[nm] = model.embed(xa).cpu().numpy()
    # class prototypes = running mean of clean embeddings per class over train_core sample (cheap: use tracked-class means)
    protos = {}
    for c in sorted(set(t["cls"] for t in tracked)):
        idc = [t["id"] for t in tracked if t["cls"] == c]
        protos[str(c)] = model.embed(X[idc].to(device)).mean(0).cpu().numpy()
    if pca is None:  # fit PCA-2D once on epoch-0 clean embeddings
        Z = zc - zc.mean(0); U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        pca = {"mean": zc.mean(0), "components": Vt[:2]}
    def proj(v):
        return ((np.atleast_2d(v) - pca["mean"]) @ pca["components"].T).tolist()
    return pca, {
        "clean": {str(t["cls"]): proj(zc[i]) for i, t in enumerate(tracked)},
        "adv": {nm: proj(adv[nm]) for nm in NORMS},
        "prototypes": {c: proj(protos[c])[0] for c in protos},
        "tracked_clean2d": [proj(zc[i])[0] for i in range(len(tracked))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["M1a", "M1b", "M0"])  # M0 = matched-control (pure MSD-AT)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--milestones", type=int, nargs="+", default=[70])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--base", choices=["msd", "apgd"], default="msd")  # CE driver: true MSD (default) | worst-case APGD fallback
    p.add_argument("--msd-steps", type=int, default=10)      # MSD iters (recipe budget; msd_v0 default is 50)
    p.add_argument("--n-iter", type=int, default=10)          # per-norm APGD steps (pull-push positives)
    p.add_argument("--outdir", required=True)
    p.add_argument("--dump-epochs", type=int, nargs="+", default=[0, 20, 40, 80])
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("C5_NUM_WORKERS", 4)))
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    pullpush = a.variant != "M0"          # M0 = matched control: pure MSD-AT, no head/positives/scaffold/glue
    neg_source = {"M1a": "adv", "M1b": "clean", "M0": "none"}[a.variant]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.outdir); (out/"ckpt").mkdir(parents=True, exist_ok=True)
    model = BackboneHead().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=a.milestones, gamma=0.1)
    loader = make_aug_loader(a.bs, a.seed, a.num_workers)
    Xe, Ye = load_eval_tensors()
    tracked = build_tracked(Xe, Ye, seed=a.seed)
    if a.smoke:
        a.epochs = 1; a.dump_epochs = [0, 1]

    def warm(ep):
        return min(1.0, ep / max(a.warmup, 1))

    # ---- resume-from-checkpoint (restart-safe for the ~1-2 day run) ----
    pca = None; embed_traj = {"epochs": [], "frames": []}; best = -1.0; hist = []; start_ep = 0
    rp = out / "resume.pt"
    if rp.exists() and not a.smoke:
        st = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start_ep = st["epoch"] + 1; best = st["best"]; hist = st["hist"]
        embed_traj = st["embed_traj"]; pca = st["pca"]
        print(f"[{a.variant}] RESUMED from epoch {st['epoch']} (best_valWU={best:.4f})", flush=True)

    # ---- W&B (same project/tags as B3/B4; graceful if not logged in) ----
    run_name = f"C5_{a.variant}_{neg_source}neg_seed{a.seed}"
    wcfg = {"wandb": {"project": "attackdro-union", "entity": None,
                      "mode": ("disabled" if a.smoke else a.wandb_mode), "required": False,
                      "group": f"C5_fromscratch_seed{a.seed}",
                      "tags": ["cifar10", "prn18", "ramp80", "C5", "fromscratch", "pullpush",
                               f"base-{a.base}", f"neg-{neg_source}", a.variant, f"seed{a.seed}"]}}
    wb = WandbLogger(wcfg, run_name=run_name)
    wb.summary({"variant": a.variant, "neg_source": neg_source, "base": a.base,
                "msd_steps": a.msd_steps, "alpha": a.alpha, "beta": a.beta, "tau": a.tau})
    for h in hist:                                            # replay past epochs after a resume
        wb.log({f"train/{k}": v for k, v in h.items() if k != "epoch"}, step=h["epoch"])

    for ep in range(start_ep, a.epochs + 1):
        if pullpush and ep in a.dump_epochs and ep not in embed_traj["epochs"]:  # dump BEFORE training (ep0=init); M0 has no trained head
            pca, frame = dump_embeddings(model, Xe, Ye, tracked, device, pca)
            embed_traj["epochs"].append(ep); embed_traj["frames"].append(frame)
        if ep == a.epochs:
            break
        model.train(); w = warm(ep); rl = rc = rg = rs = n = 0; t0 = time.time()
        for bi, (x, y) in enumerate(loader):
            x = x.to(device); y = y.to(device)
            model.eval()
            # pull-push positives: 3 per-norm APGD adversarials — only when pull-push is on
            # (M0 matched-control skips them → pure MSD-AT, ~4x cheaper attacks)
            xs_adv = ([apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=True).detach() for nm in NORMS]
                      if (pullpush or a.base == "apgd") else None)
            if a.base == "msd":
                x_base = msd_v0(model, x, y, EPS["Linf"], EPS["L2"], EPS["L1"], steps=a.msd_steps).detach()
            model.train()
            lce = F.cross_entropy(model.logits(x_base), y) if a.base == "msd" else ce_at_loss(model, xs_adv, y)
            if pullpush:
                lpp, comp = decoupled_pullpush(model, x, xs_adv, y, w*a.alpha, w*a.beta, a.tau, neg_source)
            else:
                lpp = torch.zeros((), device=device); comp = {"glue": 0.0, "scaffold": 0.0}
            loss = lce + lpp
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss (base={a.base}) at ep{ep} bi{bi}: "
                                   f"ce={float(lce)} pp={float(lpp)} — MSD base may be unstable, fall back to --base apgd")
            opt.zero_grad(); loss.backward(); opt.step()
            bs_ = y.size(0); n += bs_; rl += loss.item()*bs_; rc += lce.item()*bs_
            rg += comp["glue"]*bs_; rs += comp["scaffold"]*bs_
            if a.smoke and bi >= 1: break
        sched.step()
        wu = worst_union_acc(model, Xe, Ye, list(range(*VAL_SELECT)), device,
                             n_iter=(3 if a.smoke else 20))
        hist.append({"epoch": ep, "loss": rl/n, "ce": rc/n, "glue": rg/n, "scaffold": rs/n,
                     "val_worst_union": wu, "lr": sched.get_last_lr()[0], "sec": round(time.time()-t0)})
        print(f"[{a.variant}] ep{ep+1}/{a.epochs} loss={rl/n:.3f} ce={rc/n:.3f} "
              f"glue={rg/n:.3f} scaf={rs/n:.3f} valWU={wu:.4f} ({time.time()-t0:.0f}s)", flush=True)
        wb.log({f"train/{k}": v for k, v in hist[-1].items() if k != "epoch"}, step=ep)
        # save periodic ckpt (robustdro format: backbone only, head discarded) + val_best
        def save(tag):
            torch.save({"cfg": MODEL_CFG, "model": model.b.state_dict()}, out/"ckpt"/f"{tag}.pt")
        if (ep+1) in (20, 40, 60, 80):
            save(f"ep{ep+1:03d}")
        if wu > best:
            best = wu; save("val_best")
        if not a.smoke:                                       # resume checkpoint (restart-safe)
            torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best, "hist": hist,
                        "embed_traj": embed_traj, "pca": pca}, out/"resume.pt")

    save = lambda tag: torch.save({"cfg": MODEL_CFG, "model": model.b.state_dict()}, out/"ckpt"/f"{tag}.pt")
    save("last")
    if pullpush:                                              # collapse/embed only meaningful with a trained head
        coll = collapse_diag(model, Xe[list(range(*VAL_SELECT))[:256]].to(device),
                             apgd_train(model, Xe[list(range(*VAL_SELECT))[:256]].to(device),
                                        Ye[list(range(*VAL_SELECT))[:256]].to(device), "Linf", EPS["Linf"], n_iter=a.n_iter, is_train=False))
        (out/"embed_dump.json").write_text(json.dumps({
            "variant": a.variant, "neg_source": neg_source, "seed": a.seed,
            "tracked": tracked, "reducer": "pca2 on epoch-0 clean", **embed_traj}, indent=2))
    else:
        coll = {"note": "M0 matched-control: no head, no embedding/collapse diagnostics"}
    (out/"train.json").write_text(json.dumps({
        "variant": a.variant, "neg_source": neg_source, "seed": a.seed, "recipe": "RAMP-80 (B3/B4-matched)",
        "arch": "preact_resnet18", "eps": EPS, "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
        "milestones": a.milestones, "alpha": a.alpha, "beta": a.beta, "warmup": a.warmup, "tau": a.tau,
        "train_apgd_niter": a.n_iter, "base": a.base, "msd_steps": a.msd_steps,
        "base_ce_at": ("TRUE MSD msd_v0 (multi-steepest-descent), CE on MSD adversarial"
                       if a.base == "msd" else "worst-case max over 3 per-norm APGD (fallback)"),
        "pullpush": pullpush,
        "loss": ("L_CE(base) + alpha*scaffold + beta*glue (decoupled; anchor A stop-grad); positives=3 per-norm APGD"
                 if pullpush else "L_CE(base) ONLY — pure MSD-AT matched control (no head/positives/scaffold/glue)"),
        "ft_train_split": f"train[{TRAIN_CORE[0]}:{TRAIN_CORE[1]}]",
        "val_select_split": f"train[{VAL_SELECT[0]}:{VAL_SELECT[1]}]",
        "best_val_worst_union": best, "final_collapse": coll, "history": hist,
        "val_best_sha256": hashlib.sha256((out/"ckpt"/"val_best.pt").read_bytes()).hexdigest()}, indent=2))
    wb.summary({"best_val_worst_union": best, **{f"final_collapse/{k}": v for k, v in coll.items()}})
    wb.finish()
    print(f"[{a.variant}] DONE best_valWU={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

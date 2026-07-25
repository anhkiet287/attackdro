"""Paper C — Stage C5 FROM-SCRATCH pull-push (CONFIRMATORY-grade, NEW training).

Authorized 2026-07-14 after C1b showed the 3-epoch fine-tune probe is under-powered
(even the RAMP positive control ties A0). This trains the pull-push method FROM SCRATCH
under the RAMP-80 recipe, so a representation loss has room to reshape geometry.

TRAIN ONLY the pull-push method (M1a/M1b). Everything with a checkpoint (MSD/RAMP/MAX/AVG/
B3/B4) is EVAL-ONLY under the frozen 12-AA harness — this script does not touch them.

Variants (Stage-1 screen, 1 seed each, SAME seed):
  M1a  neg-source=adv    — push other-class ADVERSARIAL embeddings   (C0/C1 behaviour)
  M1b  neg-source=clean  — push other-class CLEAN embeddings         (transitive-separation)

Ablation (H3, --glue-view msd): the pull-push term operates on ONE view = x_MSD (the CE-AT base
union adversarial, REUSED — no extra APGD) instead of 3 per-norm APGD views. Tests whether the
3-view multi-norm structure is necessary or whether aligning the single union adversarial
suffices (defend §3.3 novelty). Everything else is identical to M1a. Requires --base msd.

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

# Dataset branch (--dataset). CIFAR-100 keeps the CIFAR-10 headline recipe EXACTLY (same eps
# triple/epochs/lr/bs/seed/val-select metric) and only swaps the dataset + num_classes + a wider
# 2k val split (20/class over 100 classes). Everything else is shared → the M1a−M0 delta is the
# claim, absolute union is NOT (C100 is harder: expect ~15-25%). Set into module globals by main().
DATASETS = {
    "cifar10":  {"cls": "CIFAR10",  "num_classes": 10,  "train_core": (0, 49000), "val_select": (49000, 50000)},
    "cifar100": {"cls": "CIFAR100", "num_classes": 100, "train_core": (0, 48000), "val_select": (48000, 50000)},
}
_DS_CLS = "CIFAR10"           # torchvision dataset class name; set in main() from --dataset


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


def ce_at_loss(model, xs_adv, y, agg="max"):
    """CE-AT over per-norm adversarial views. agg='max' = worst-of-N (default, byte-identical);
    agg='avg' = mean-of-N (the --base avg ablation). Reuses the views already crafted (no extra attack)."""
    ce = torch.stack([F.cross_entropy(model.logits(xa), y, reduction="none") for xa in xs_adv], 0)
    return (ce.max(0).values if agg == "max" else ce.mean(0)).mean()


def decoupled_pullpush(model, x_clean, xs_adv, y, alpha, beta, tau=0.1, neg_source="adv",
                       glue_weighting="uniform", bind_T=1.0):
    """Decoupled pull-push. glue = pull own adv → own detached clean anchor;
    scaffold = push own adv away from other-class negatives (adv or clean embeddings).

    glue_weighting: 'uniform' (default; byte-identical) averages the k views equally. 'bindworst'
    reweights the k glue views PER SAMPLE by softmax over the DETACHED per-view CE loss (harder view
    → higher weight): w^i = softmax_v(CE(g(f(x_v^i)),y_i)/T), Σ_v w=1 (magnitude-matched to uniform).
    No gradient flows through the weights. Only the glue weighting changes — scaffold/anchor untouched."""
    B = x_clean.shape[0]
    k = len(xs_adv)                                          # views/sample: 3 (3-norm) or 1 (msd-glue ablation)
    bind = (glue_weighting == "bindworst" and k > 1)        # needs ≥2 views to reweight
    with torch.no_grad():
        z_anchor = model.embed(x_clean)                      # (B,d) stop-grad anchor (glue target)
    if bind:                                                 # keep per-view features (shared: head-embed + classifier-CE)
        feats = [model.features(xa) for xa in xs_adv]
        z_adv = torch.cat([F.normalize(model.head(f), dim=1) for f in feats], 0)  # (kB,d) == embed
    else:
        z_adv = torch.cat([model.embed(xa) for xa in xs_adv], 0)  # (kB,d)
    anchor_idx = torch.arange(k * B, device=x_clean.device) % B
    yk = y.repeat(k)

    pos_sim = (z_adv * z_anchor[anchor_idx]).sum(1)          # (kB,)
    extra = {}
    if bind:
        # per-sample binding-norm weights: softmax over views of the DETACHED per-view CE loss
        # (1 linear pass on the features already computed). views in NORMS order (0=Linf,1=L2,2=L1).
        with torch.no_grad():
            ce_v = torch.stack([F.cross_entropy(model.b.linear(f), y, reduction="none") for f in feats], 0)  # (k,B)
            w = torch.softmax(ce_v / bind_T, dim=0)          # (k,B), Σ over views = 1
            am = w.argmax(0)
            extra = {"w_mean": w.mean(1).tolist(),           # mean weight per view (batch)
                     "frac_argmax": [float((am == v).float().mean()) for v in range(k)]}
        glue = (w.reshape(-1) * (-pos_sim / tau)).sum() / B  # Σ_i Σ_v w·(−cos/τ) / B  (== .mean() when w=1/k)
    else:
        glue = (-pos_sim / tau).mean()

    if neg_source == "adv":
        z_neg = z_adv; y_neg = yk                            # other-class adv embeddings (grad)
    elif neg_source == "clean":
        z_neg = model.embed(x_clean); y_neg = y              # other-class clean embeddings (grad)
    else:
        raise ValueError(neg_source)
    sim = z_adv @ z_neg.t()                                  # (kB, N)
    neg_mask = (yk[:, None] != y_neg[None, :])
    neg_logits = (sim / tau).masked_fill(~neg_mask, float("-inf"))
    row_has_neg = neg_mask.any(1)
    scaffold = torch.logsumexp(neg_logits[row_has_neg], dim=1).mean() if row_has_neg.any() \
               else z_adv.new_zeros(())
    return alpha * scaffold + beta * glue, {"glue": float(glue.detach()), "scaffold": float(scaffold.detach()), **extra}


def apply_grad_surgery(lce, lpp, theta_f, theta_cls, theta_h):
    """Asymmetric PCGrad at the shared encoder θ_f. Two separate backward passes give the task
    gradient g_task=∂L_CE/∂θ and the rep gradient g_clamp=∂L_rep/∂θ. On each θ_f tensor, if the two
    CONFLICT (dot<0) the rep gradient is projected off the task gradient (Gram-Schmidt); g_task is
    NEVER touched → the task is protected. θ_cls gets task grad only, θ_h gets rep grad only. This
    matches the combined backward EXACTLY when there is no conflict (only the projection differs).
    Sets p.grad in place; returns per-step diagnostics. (retain_graph=False: L_CE and L_rep come
    from independent forwards, so their graphs are disjoint apart from the shared leaf params.)"""
    nf = len(theta_f)
    gt = torch.autograd.grad(lce, theta_f + theta_cls, retain_graph=False, allow_unused=True)
    gt_enc, gt_cls = gt[:nf], gt[nf:]
    gc = torch.autograd.grad(lpp, theta_f + theta_h, retain_graph=False, allow_unused=True)
    gc_enc, gc_h = gc[:nf], gc[nf:]
    n_conf = 0; cos_sum = 0.0; cos_cnt = 0; gc_norm2 = 0.0; gc_proj_norm2 = 0.0
    for i, p in enumerate(theta_f):
        gti, gci = gt_enc[i], gc_enc[i]
        if gci is None:                                  # rep lane doesn't touch this param
            p.grad = gti if gti is not None else None
            continue
        if gti is None:                                  # task lane doesn't touch it → rep passes through
            p.grad = gci
            gc_norm2 += float(gci.pow(2).sum()); gc_proj_norm2 += float(gci.pow(2).sum())
            continue
        dot = (gci * gti).sum()
        gc_norm2 += float(gci.pow(2).sum())
        cos_sum += float(dot / (gci.norm() * gti.norm()).clamp_min(1e-12)); cos_cnt += 1
        if dot < 0:                                      # conflict → project rep grad off task grad
            n_conf += 1
            gci = gci - (dot / gti.pow(2).sum().clamp_min(1e-12)) * gti
        gc_proj_norm2 += float(gci.pow(2).sum())
        p.grad = gti + gci
    for p, g in zip(theta_cls, gt_cls):                  # classifier: task grad only (no surgery)
        p.grad = g
    for p, g in zip(theta_h, gc_h):                      # CLAMP head: rep grad only (no surgery)
        p.grad = g
    return {"n_conf": n_conf, "cos_sum": cos_sum, "cos_cnt": cos_cnt,
            "gc_norm2": gc_norm2, "gc_proj_norm2": gc_proj_norm2}


@torch.no_grad()
def collapse_diag(model, x, x_adv):
    zc = model.embed(x); za = model.embed(x_adv)
    align = (za * zc).sum(1).mean().item()
    sq = torch.pdist(zc, p=2).pow(2)
    unif = torch.log(torch.exp(-2.0 * sq).mean() + 1e-12).item() if len(sq) else float("nan")
    norm = model.head(model.features(x)).norm(dim=1).mean().item()
    return {"alignment": align, "uniformity": unif, "embed_norm": norm}


@torch.no_grad()
def pernorm_train_metrics(model, x, y, avail_views=None, n_iter=10, is_train=True):
    """Standard per-norm TRAIN logging (spec, default ON). Robust acc/loss on the TRAINING
    adversarial — an OPTIMISTIC proxy (train-strength attack), for reading the train-val gap.
    Returns {acc_clean, loss_clean, acc_{linf,l2,l1}, loss_{...}, worst_union}.

    RNG ISOLATION (the whole point): `avail_views[norm]` (a training adversarial already crafted
    this step) is REUSED byte-identically. A MISSING norm (M0 / single-view arms) is crafted here
    inside `torch.random.fork_rng()` so the MAIN RNG stream is NOT advanced — the arm's training
    trajectory is unchanged (acceptance: M0 seed-0 re-run must still reproduce 0.3886 exact; any
    drift = RNG leak = bug). apgd_train is deterministic, but fork_rng makes isolation unconditional."""
    avail_views = avail_views or {}
    was_training = model.training
    model.eval()
    out = {}
    with torch.no_grad():
        lc = model.logits(x)
        out["acc_clean"] = (lc.argmax(1) == y).float().mean().item()
        out["loss_clean"] = F.cross_entropy(lc, y).item()
    masks = []
    for nm in NORMS:
        xa = avail_views.get(nm)
        if xa is None:                                    # generate missing view WITHOUT touching main RNG
            devs = [x.device.index] if x.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devs):
                with torch.enable_grad():
                    xa = apgd_train(model, x, y, nm, EPS[nm], n_iter=n_iter, is_train=is_train).detach()
        with torch.no_grad():
            la = model.logits(xa)
        m = la.argmax(1) == y
        masks.append(m)
        out[f"acc_{nm.lower()}"] = m.float().mean().item()
        out[f"loss_{nm.lower()}"] = F.cross_entropy(la, y).item()
    wu = masks[0].clone()
    for m in masks[1:]:
        wu &= m
    out["worst_union"] = wu.float().mean().item()
    if was_training:
        model.train()
    # NOTE: no hard asserts — logging must NEVER crash training. The invariant clean>=per-norm can
    # legitimately break early (near-random model / weak TRAIN-strength attack → per-norm robust acc
    # exceeds clean by noise). These are an optimistic PROXY; the true audit is the frozen 12-AA.
    return out


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


@torch.no_grad()
def val_metrics(model, X, Y, idx, device, bs=250, n_iter=20, has_head=True):
    """Per-norm robust acc + clean + union (AND), plus align/uniformity read off the Linf
    adversarial (REUSED — no extra attack). `union` is byte-identical to worst_union_acc (same
    attacks, same order), so using it for val-selection keeps M1a/M0/M1b parity. APGD here starts
    from the clean point (no random init) → consumes no RNG → canonical runs still reproduce."""
    model.eval()
    xs = X[idx].to(device); ys = Y[idx].to(device); N = len(idx)
    clean_ok = torch.zeros(N, dtype=torch.bool, device=device)
    per = {nm: torch.zeros(N, dtype=torch.bool, device=device) for nm in NORMS}
    for i in range(0, N, bs):
        clean_ok[i:i+bs] = model(xs[i:i+bs]).argmax(1) == ys[i:i+bs]
    z_clean0 = z_adv0 = None
    for nm in NORMS:
        for i in range(0, N, bs):
            xb = xs[i:i+bs]; yb = ys[i:i+bs]
            with torch.enable_grad():
                xa = apgd_train(model, xb, yb, nm, EPS[nm], n_iter=n_iter, is_train=False)
            per[nm][i:i+bs] = model(xa).argmax(1) == yb
            if has_head and nm == "Linf" and z_adv0 is None:   # first Linf batch → align/uniformity
                z_adv0 = model.embed(xa); z_clean0 = model.embed(xb)
    m = {"acc_clean": clean_ok.float().mean().item()}
    for nm in NORMS:
        m[f"acc_{nm.lower()}"] = per[nm].float().mean().item()
    m["union"] = torch.stack([per[nm] for nm in NORMS]).all(0).float().mean().item()
    if z_adv0 is not None:
        m["align"] = (z_adv0 * z_clean0).sum(1).mean().item()
        sq = torch.pdist(z_clean0, p=2).pow(2)
        m["uniformity"] = torch.log(torch.exp(-2.0 * sq).mean() + 1e-12).item() if len(sq) else float("nan")
    return m


def make_aug_loader(bs, seed, num_workers=4):
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    DS = getattr(torchvision.datasets, _DS_CLS)
    full = DS(root=str(ROOT/"data"), train=True, download=False, transform=tf)
    core = torch.utils.data.Subset(full, list(range(*TRAIN_CORE)))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(core, batch_size=bs, shuffle=True, num_workers=num_workers,
                                       drop_last=True, generator=g)


def load_eval_tensors():
    import torchvision, torchvision.transforms as T
    DS = getattr(torchvision.datasets, _DS_CLS)
    ds = DS(root=str(ROOT/"data"), train=True, download=False,
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
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--milestones", type=int, nargs="+", default=[70])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--base", choices=["msd", "apgd", "max", "avg", "clean"], default="msd",
                   help="CE driver: msd=true-MSD union adv (default) | apgd/max=worst-of-3 per-norm APGD "
                        "| avg=mean-of-3 per-norm APGD (#11/#12, reuse the 3 glue views) "
                        "| clean=CE on CLEAN images, NO adversarial in the CE lane (necessity ablation: "
                        "the classifier head never sees an adversarial; union is expected to collapse)")
    p.add_argument("--glue-view", choices=["3norm", "msd", "dynworst", "linf"], default="3norm",
                   help="pull-push views: 3norm=3 per-norm APGD (M1a/M1b canonical); "
                        "msd=single x_MSD view; dynworst=single APGD view of the per-epoch worst norm; "
                        "linf=single ℓ∞ APGD view (redundancy ablation #14). Any glue-view works with any base "
                        "(the view it needs is crafted; base=msd reuse when they coincide)")
    p.add_argument("--neg", choices=["adv", "clean"], default=None,
                   help="override scaffold negatives (default: from --variant; M1a=adv, M1b=clean); "
                        "the dynworst arm uses --neg clean")
    p.add_argument("--msd-steps", type=int, default=10)      # MSD iters (recipe budget; msd_v0 default is 50)
    p.add_argument("--n-iter", type=int, default=10)          # per-norm APGD steps (pull-push positives)
    p.add_argument("--ce-views-w", type=float, default=0.0,
                   help="EXPOSURE CONTROL (F1): add  w * mean_p CE(logits(x^p), y)  over the 3 per-norm "
                        "APGD-10 views to the base CE. Orthogonal to the pull-push term: with --variant M0 "
                        "this gives multi-norm EXPOSURE with NO contrastive term, NO projection head and NO "
                        "clean anchor, so it isolates 'the model saw 3 norms' from 'the geometry was shaped'. "
                        "0 (default) = byte-identical to every prior run.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--dump-epochs", type=int, nargs="+", default=[0, 20, 40, 80])
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("C5_NUM_WORKERS", 4)))
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--no-log-train-pernorm", dest="log_train_pernorm", action="store_false",
                   help="disable standard per-norm TRAIN logging (train/acc_*, loss_*, worst_union); default ON, RNG-isolated")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--grad-surgery", action="store_true",
                   help="asymmetric PCGrad at the shared encoder θ_f: project the rep gradient off "
                        "the task gradient only where they conflict (task grad untouched). GATED — "
                        "default off keeps the combined backward byte-identical. Needs a rep lane.")
    p.add_argument("--pp-schedule", choices=["simul", "push_then_pull", "pull_then_push"], default="simul",
                   help="temporal split of the two rep terms. simul (default) = both every epoch "
                        "(byte-identical to current). push_then_pull = scaffold-only for ep<switch then "
                        "glue-only; pull_then_push = the reverse. CE on x_MSD stays on all 80 ep.")
    p.add_argument("--pp-switch", type=int, default=40,
                   help="phase-boundary epoch for --pp-schedule (phase 1 = ep<switch; 1-indexed ep=switch is phase 2)")
    p.add_argument("--glue-weighting", choices=["uniform", "bindworst"], default="uniform",
                   help="glue view weighting: uniform (default, byte-identical) = equal per view; "
                        "bindworst = per-sample softmax over the detached per-view CE loss (binding-norm glue)")
    p.add_argument("--bind-T", type=float, default=1.0, help="softmax temperature for --glue-weighting bindworst")
    a = p.parse_args()
    # (glue-view no longer requires --base msd: each glue-view crafts the view it needs; x_MSD is
    #  generated for glue=msd even under base=max/avg — the #13 redundancy ablation. base=msd reuses.)
    pullpush = a.variant != "M0"          # M0 = matched control: pure MSD-AT, no head/positives/scaffold/glue
    neg_source = {"M1a": "adv", "M1b": "clean", "M0": "none"}[a.variant]
    if a.neg is not None:                 # explicit scaffold-negative override (the dynworst arm uses clean)
        if not pullpush:
            p.error("--neg has no effect with --variant M0 (no scaffold term)")
        neg_source = a.neg
    if a.grad_surgery and not pullpush:
        p.error("--grad-surgery needs a rep lane (scaffold+glue); not valid with --variant M0")
    if a.glue_weighting == "bindworst" and (not pullpush or a.glue_view != "3norm"):
        p.error("--glue-weighting bindworst needs the 3 glue views: use --variant M1a/M1b with --glue-view 3norm")
    if a.pp_schedule != "simul":
        if not pullpush:
            p.error("--pp-schedule needs a rep lane (scaffold/glue); not valid with --variant M0")
        if a.base != "msd":
            p.error("--pp-schedule requires --base msd (the push phase reuses the x_MSD adversarial as its single view)")
        if not (0 < a.pp_switch < a.epochs):
            p.error(f"--pp-switch must be in (0, epochs={a.epochs}); got {a.pp_switch}")

    # ---- dataset branch: set module globals BEFORE model/loader construction ----
    global TRAIN_CORE, VAL_SELECT, _DS_CLS
    D = DATASETS[a.dataset]
    TRAIN_CORE = D["train_core"]; VAL_SELECT = D["val_select"]; _DS_CLS = D["cls"]
    MODEL_CFG["dataset"]["num_classes"] = D["num_classes"]   # backbone final linear width; head 512→128 unchanged
    print(f"[{a.variant}] dataset={a.dataset} num_classes={D['num_classes']} "
          f"train_core={TRAIN_CORE} val_select={VAL_SELECT} ({VAL_SELECT[1]-VAL_SELECT[0]} ex)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.outdir); (out/"ckpt").mkdir(parents=True, exist_ok=True)
    model = BackboneHead().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=a.milestones, gamma=0.1)
    # gradient-surgery param groups: θ_f = shared encoder (all params EXCEPT classifier + CLAMP head),
    # θ_cls = classifier (model.b.linear), θ_h = projection head. References survive resume (in-place load).
    theta_f = theta_cls = theta_h = None
    if a.grad_surgery:
        cls_ids = {id(p) for p in model.b.linear.parameters()}
        head_ids = {id(p) for p in model.head.parameters()}
        theta_cls = list(model.b.linear.parameters())
        theta_h = list(model.head.parameters())
        theta_f = [p for p in model.parameters() if id(p) not in cls_ids and id(p) not in head_ids]
        print(f"[{a.variant}] grad-surgery ON: |θ_f|={len(theta_f)} |θ_cls|={len(theta_cls)} "
              f"|θ_h|={len(theta_h)} (project rep-grad off task-grad on θ_f where they conflict)", flush=True)
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
    gv_tag = "" if a.glue_view == "3norm" else f"_{a.glue_view}glue"
    ds_tag = "" if a.dataset == "cifar10" else f"_{a.dataset}"
    gs_tag = "_gs" if a.grad_surgery else ""
    pp_tag = "" if a.pp_schedule == "simul" else f"_{a.pp_schedule}{a.pp_switch}"
    gw_tag = "" if a.glue_weighting == "uniform" else f"_{a.glue_weighting}"
    cev_tag = "" if a.ce_views_w == 0 else f"_cev{a.ce_views_w:g}"
    base_tag = "" if a.base == "msd" else f"_{a.base}"    # base=max/avg/apgd -> DISTINCT W&B run (no collide with the msd-base seed runs)
    run_name = f"C5_{a.variant}_{neg_source}neg_seed{a.seed}{gv_tag}{base_tag}{ds_tag}{gs_tag}{pp_tag}{gw_tag}{cev_tag}"
    wcfg = {"wandb": {"project": "attackdro-union", "entity": None,
                      "mode": ("disabled" if a.smoke else a.wandb_mode), "required": False,
                      "group": f"C5_fromscratch_{a.dataset}_seed{a.seed}",
                      "tags": [a.dataset, "prn18", "ramp80", "C5", "fromscratch", "pullpush",
                               f"base-{a.base}", f"neg-{neg_source}", f"glue-{a.glue_view}",
                               f"gs-{'on' if a.grad_surgery else 'off'}", f"pp-{a.pp_schedule}",
                               f"gluew-{a.glue_weighting}", a.variant, f"seed{a.seed}"]}}
    wb = WandbLogger(wcfg, run_name=run_name)
    wb.summary({"variant": a.variant, "neg_source": neg_source, "base": a.base, "glue_view": a.glue_view,
                "grad_surgery": a.grad_surgery, "pp_schedule": a.pp_schedule, "pp_switch": a.pp_switch,
                "glue_weighting": a.glue_weighting, "bind_T": a.bind_T,
                "msd_steps": a.msd_steps, "alpha": a.alpha, "beta": a.beta, "tau": a.tau,
                "ce_views_w": a.ce_views_w})
    def log_epoch(h):                                        # split metrics into train/*, val/*, glue/*, surgery/*
        payload = {}
        for k in ("loss", "ce", "glue", "scaffold", "lr", "sec"):
            if k in h:
                payload[f"train/{k}"] = h[k]
        for k, v in h.items():
            if k.startswith("train/"):
                payload[k] = v                               # standard per-norm TRAIN logging (already prefixed)
            elif k.startswith("val_"):
                payload[f"val/{k[4:]}"] = v                  # val_acc_linf → val/acc_linf, val_union → val/union
            elif k.startswith("surg_"):
                payload[f"surgery/{k[5:]}"] = v              # surg_conflict_frac → surgery/conflict_frac
            elif k.startswith("sched_"):
                payload[f"sched/{k[6:]}"] = v                # sched_active_term → sched/active_term (0=push 1=pull)
            elif k.startswith("bind_"):
                payload[f"bind/{k[5:]}"] = v                 # bind_w_mean_linf → bind/w_mean_linf
        if "selected_norm_idx" in h:
            payload["glue/selected_norm"] = h["selected_norm_idx"]   # 0=Linf 1=L2 2=L1 (chartable; name in train.json)
        wb.log(payload, step=h["epoch"])

    for h in hist:                                            # replay past epochs after a resume
        log_epoch(h)

    for ep in range(start_ep, a.epochs + 1):
        if pullpush and ep in a.dump_epochs and ep not in embed_traj["epochs"]:  # dump BEFORE training (ep0=init); M0 has no trained head
            pca, frame = dump_embeddings(model, Xe, Ye, tracked, device, pca)
            embed_traj["epochs"].append(ep); embed_traj["frames"].append(frame)
        if ep == a.epochs:
            break
        model.train(); w = warm(ep); rl = rc = rg = rs = n = 0; t0 = time.time()
        sc_conf = sc_cos = sc_coscnt = sc_steps = 0; sc_gc2 = sc_gcp2 = 0.0   # surgery epoch accumulators
        bind_wm = {"linf": 0.0, "l2": 0.0, "l1": 0.0}; bind_fa = {"linf": 0.0, "l2": 0.0, "l1": 0.0}; n_bind = 0
        tl_pn = {}; tl_pn_n = 0                              # standard per-norm TRAIN logging accumulators
        # dynworst: this epoch's single glue view = the norm with the LOWEST val robust acc last
        # epoch (argmin). Epoch 0 has no val yet → default Linf. Derived from hist → resume-safe.
        sel_norm = None
        if pullpush and a.glue_view == "dynworst":
            sel_norm = ("Linf" if not hist else
                        min(NORMS, key=lambda nm: hist[-1].get(f"val_acc_{nm.lower()}", 1.0)))
        # pull-push schedule: effective (α,β) for this epoch. simul = both×warmup (byte-identical).
        # phased = only one term active per phase; phase-1 term ramps over warmup, phase-2 enters at
        # full strength (no re-warmup). CE on x_MSD is unaffected (active every epoch either way).
        if a.pp_schedule == "simul":
            a_eff, b_eff, active_idx = w*a.alpha, w*a.beta, None
        else:
            phase1 = ep < a.pp_switch
            push_on = phase1 if a.pp_schedule == "push_then_pull" else (not phase1)
            coef = w if phase1 else 1.0
            a_eff, b_eff = (coef*a.alpha, 0.0) if push_on else (0.0, coef*a.beta)
            active_idx = 0 if push_on else 1                  # 0=push (scaffold), 1=pull (glue)
        for bi, (x, y) in enumerate(loader):
            x = x.to(device); y = y.to(device)
            model.eval()
            # ---- CE-AT views (drive the CE loss). RNG-order preserved (MSD before APGD) so every
            #      existing config is byte-identical. base: msd=true-MSD | max/apgd=worst-of-3 | avg=mean-of-3.
            need_msd = (a.base == "msd") or (a.glue_view == "msd")   # x_MSD for CE(base=msd) and/or glue(msd)
            if need_msd:
                x_base = msd_v0(model, x, y, EPS["Linf"], EPS["L2"], EPS["L1"], steps=a.msd_steps).detach()
            if a.base == "msd":
                xs_ce = [x_base]
            elif a.base == "clean":
                xs_ce = None                                         # CE on the CLEAN batch — no adversarial crafted here
            else:                                                    # max/apgd/avg: 3 per-norm APGD, aggregated in ce_at_loss
                xs_ce = [apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=True).detach() for nm in NORMS]
            # ---- glue views (pull-push positives) ----
            #   3norm=3 per-norm APGD (reuse xs_ce when base already crafted them) · msd=x_MSD ·
            #   dynworst=worst-norm APGD · linf=ℓ∞ APGD · M0 = no positives.
            if a.pp_schedule != "simul" and pullpush:                # phased (base=msd only) — UNCHANGED
                if b_eff > 0:
                    if a.glue_view == "msd":
                        xs_adv = [x_base]
                    elif a.glue_view == "dynworst":
                        xs_adv = [apgd_train(model, x, y, sel_norm, EPS[sel_norm], n_iter=a.n_iter, is_train=True).detach()]
                    else:
                        xs_adv = [apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=True).detach() for nm in NORMS]
                else:
                    xs_adv = [x_base]
            elif not pullpush:
                xs_adv = None
            # NOTE: xs_cev (exposure views) is crafted separately below so that turning the exposure
            # term on never perturbs the pull-push view logic or its RNG order.
            elif a.glue_view == "3norm":
                # reuse the CE views ONLY when they are exactly the 3 per-norm APGD set (max/avg/apgd);
                # msd and clean bases don't produce them -> craft fresh.
                xs_adv = xs_ce if a.base in ("max", "avg", "apgd") else \
                    [apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=True).detach() for nm in NORMS]
            elif a.glue_view == "msd":
                xs_adv = [x_base]
            elif a.glue_view == "linf":
                xs_adv = [apgd_train(model, x, y, "Linf", EPS["Linf"], n_iter=a.n_iter, is_train=True).detach()]
            else:                                                    # dynworst
                xs_adv = [apgd_train(model, x, y, sel_norm, EPS[sel_norm], n_iter=a.n_iter, is_train=True).detach()]
            # ---- F1 exposure views: the SAME 3 per-norm APGD set the CLAMP term would use, but fed
            #      to the classifier as plain CE. Reused when an identical set already exists, so the
            #      arm costs exactly what CLAMP costs (4 attacks/step), not more.
            xs_cev = None
            if a.ce_views_w > 0:
                if a.base in ("max", "avg", "apgd"):
                    xs_cev = xs_ce
                elif xs_adv is not None and a.glue_view == "3norm" and len(xs_adv) == len(NORMS):
                    xs_cev = xs_adv
                else:
                    xs_cev = [apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=True).detach()
                              for nm in NORMS]
            model.train()
            if a.base == "msd":
                lce = F.cross_entropy(model.logits(x_base), y)
            elif a.base == "clean":
                lce = F.cross_entropy(model.logits(x), y)            # CLEAN CE — head never sees an adversarial
            else:
                lce = ce_at_loss(model, xs_ce, y, agg=("avg" if a.base == "avg" else "max"))
            if xs_cev is not None:                               # + w_eff * mean_p CE(view_p)
                # SAME linear warmup as the pull-push term (w = min(1, ep/warmup)), so the exposure
                # control and CLAMP differ only in WHAT the auxiliary views feed, never in WHEN.
                lce = lce + (w * a.ce_views_w) * ce_at_loss(model, xs_cev, y, agg="avg")
            if pullpush:
                lpp, comp = decoupled_pullpush(model, x, xs_adv, y, a_eff, b_eff, a.tau, neg_source,
                                               glue_weighting=a.glue_weighting, bind_T=a.bind_T)
            else:
                lpp = torch.zeros((), device=device); comp = {"glue": 0.0, "scaffold": 0.0}
            if a.grad_surgery:                               # asymmetric PCGrad at θ_f (task protected)
                if not (torch.isfinite(lce) and torch.isfinite(lpp)):
                    raise RuntimeError(f"non-finite loss (surgery, base={a.base}) at ep{ep} bi{bi}: "
                                       f"ce={float(lce)} pp={float(lpp)}")
                opt.zero_grad(set_to_none=True)
                sd = apply_grad_surgery(lce, lpp, theta_f, theta_cls, theta_h)
                opt.step()
                sc_conf += sd["n_conf"]; sc_cos += sd["cos_sum"]; sc_coscnt += sd["cos_cnt"]
                sc_gc2 += sd["gc_norm2"]; sc_gcp2 += sd["gc_proj_norm2"]; sc_steps += 1
                loss_val = lce.item() + lpp.item()
            else:                                            # default: combined backward (byte-identical)
                loss = lce + lpp
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss (base={a.base}) at ep{ep} bi{bi}: "
                                       f"ce={float(lce)} pp={float(lpp)} — MSD base may be unstable, fall back to --base apgd")
                opt.zero_grad(); loss.backward(); opt.step()
                loss_val = loss.item()
            bs_ = y.size(0); n += bs_; rl += loss_val*bs_; rc += lce.item()*bs_
            rg += comp["glue"]*bs_; rs += comp["scaffold"]*bs_
            if "w_mean" in comp:                             # bindworst per-view weight diagnostics
                n_bind += bs_
                for j, nm in enumerate(("linf", "l2", "l1")):
                    bind_wm[nm] += comp["w_mean"][j]*bs_; bind_fa[nm] += comp["frac_argmax"][j]*bs_
            if a.log_train_pernorm:                          # standard per-norm TRAIN proxy (RNG-isolated)
                if a.glue_view == "3norm" and xs_adv is not None and len(xs_adv) == 3:
                    avail = {nm: xs_adv[i] for i, nm in enumerate(NORMS)}     # reuse 3 training views (byte-identical)
                elif a.glue_view == "dynworst" and sel_norm and xs_adv:
                    avail = {sel_norm: xs_adv[0]}                             # reuse the single dynworst view
                else:
                    avail = {}                                               # msd / M0 / phased-push -> craft (isolated)
                tm = pernorm_train_metrics(model, x, y, avail, n_iter=a.n_iter)
                for kk, vv in tm.items(): tl_pn[kk] = tl_pn.get(kk, 0.0) + vv * bs_
                tl_pn_n += bs_
            if a.smoke and bi >= 1: break
        sched.step()
        vm = val_metrics(model, Xe, Ye, list(range(*VAL_SELECT)), device,
                         n_iter=(3 if a.smoke else 20), has_head=pullpush)
        wu = vm["union"]                                     # == old worst_union_acc (same attacks) → val-select parity
        rec = {"epoch": ep, "loss": rl/n, "ce": rc/n, "glue": rg/n, "scaffold": rs/n,
               "lr": sched.get_last_lr()[0], "sec": round(time.time()-t0),
               "val_worst_union": wu, **{f"val_{k}": v for k, v in vm.items()}}
        if sel_norm is not None:
            rec["selected_norm"] = sel_norm; rec["selected_norm_idx"] = NORMS.index(sel_norm)
        if active_idx is not None:
            rec["sched_active_term"] = active_idx            # 0=push (scaffold) 1=pull (glue)
        if a.grad_surgery and sc_steps > 0:
            nf = len(theta_f)
            rec["surg_conflict_frac"] = sc_conf / (sc_steps * nf)          # frac of θ_f tensors with dot<0
            rec["surg_cos_task_clamp"] = sc_cos / max(sc_coscnt, 1)        # mean cos(g_task, g_clamp) pre-proj
            rec["surg_gclamp_norm_ratio"] = (sc_gcp2 / sc_gc2) ** 0.5 if sc_gc2 > 0 else 1.0  # ‖proj‖/‖orig‖
        if a.glue_weighting == "bindworst" and n_bind > 0:
            for nm in ("linf", "l2", "l1"):
                rec[f"bind_w_mean_{nm}"] = bind_wm[nm] / n_bind            # mean glue weight per view
                rec[f"bind_frac_argmax_{nm}"] = bind_fa[nm] / n_bind       # frac of samples where view is worst
        if a.log_train_pernorm and tl_pn_n > 0:                            # train/acc_*, loss_*, worst_union
            for kk, vv in tl_pn.items(): rec[f"train/{kk}"] = vv / tl_pn_n
        hist.append(rec)
        print(f"[{a.variant}] ep{ep+1}/{a.epochs} loss={rl/n:.3f} ce={rc/n:.3f} "
              f"glue={rg/n:.3f} scaf={rs/n:.3f} valU={wu:.4f}"
              + (f" worst={sel_norm}" if sel_norm else "")
              + (f" [{['push','pull'][active_idx]}]" if active_idx is not None else "")
              + (f" conf={rec['surg_conflict_frac']:.2f}" if "surg_conflict_frac" in rec else "")
              + (f" bindfa[∞{rec['bind_frac_argmax_linf']:.2f}/2{rec['bind_frac_argmax_l2']:.2f}/1{rec['bind_frac_argmax_l1']:.2f}]"
                 if "bind_frac_argmax_linf" in rec else "")
              + f" ({time.time()-t0:.0f}s)", flush=True)
        log_epoch(hist[-1])
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
        "dataset": a.dataset, "num_classes": DATASETS[a.dataset]["num_classes"],
        "arch": "preact_resnet18", "eps": EPS, "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
        "milestones": a.milestones, "alpha": a.alpha, "beta": a.beta, "warmup": a.warmup, "tau": a.tau,
        "ce_views_w": a.ce_views_w,
        "train_apgd_niter": a.n_iter, "base": a.base, "msd_steps": a.msd_steps, "glue_view": a.glue_view,
        "grad_surgery": a.grad_surgery, "pp_schedule": a.pp_schedule, "pp_switch": a.pp_switch,
        "glue_weighting": a.glue_weighting, "bind_T": a.bind_T,
        "glue_weighting_note": ("bindworst: glue views reweighted PER SAMPLE by softmax over the DETACHED "
                                "per-view CE loss (w=softmax(CE(g(f(x_v)),y)/T)), Σ_v w=1 (magnitude-matched "
                                "to uniform); no gradient through weights; scaffold/CE/warmup/head unchanged"
                                if a.glue_weighting == "bindworst" else None),
        "pp_schedule_note": (f"phased: {a.pp_schedule} at switch ep {a.pp_switch}; phase-1 term warms up "
                             "0→coef over --warmup, phase-2 term enters at full strength (no re-warmup); "
                             "CE on x_MSD active all epochs. PRE-REGISTERED DEVIATION: the push-only "
                             "phase scaffolds on z_MSD (own + other-class MSD-adv embeddings, embed(x_MSD)), "
                             "NOT the 3-view APGD negatives of simul M1a — no APGD view crafted in the push "
                             "phase. Justification: M1b (neg-source robustness) + H3 (z_MSD is a valid view)."
                             if a.pp_schedule != "simul" else None),
        "grad_surgery_note": ("asymmetric PCGrad at shared encoder θ_f: rep-grad (∂L_rep/∂θ_f) projected "
                              "off task-grad (∂L_CE/∂θ_f) where dot<0; θ_cls=classifier task-only, "
                              "θ_h=CLAMP head rep-only; task gradient never modified"
                              if a.grad_surgery else None),
        "base_ce_at": {"msd": "TRUE MSD msd_v0 (multi-steepest-descent), CE on MSD adversarial",
                       "apgd": "worst-case MAX over 3 per-norm APGD",
                       "max": "worst-case MAX over 3 per-norm APGD (base-generality #11)",
                       "avg": "MEAN over 3 per-norm APGD (base-generality #12)",
                       "clean": "CE on CLEAN images — NO adversarial in the CE lane (necessity ablation: "
                                "the classifier head never sees an adversarial; union expected to collapse)"}[a.base],
        "pullpush": pullpush,
        "loss": (("L_CE(base) + alpha*scaffold + beta*glue (decoupled; anchor A stop-grad); positives="
                  + {"msd": "1 MSD view (x_MSD, H3 ablation)",
                     "dynworst": "1 dynamic worst-norm APGD view (argmin prev-epoch val per-norm acc)",
                     "linf": "1 ℓ∞ APGD view (redundancy ablation #14)",
                     "3norm": "3 per-norm APGD"}[a.glue_view])
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

"""Paper C — Stage C0 kill-test (fine-tune, EXPLORATORY, placebo-controlled).

Authority: docs/preregistrations/preregistration_paperC_pullpush.md §3 (LOCKED loss).
Anchor fixed to variant A (own clean embedding, STOP-GRAD). No anchor sweep here (that is C3).
Backbone = public MSD-80 (locuslab/robust_union PreActResNet18), fine-tuned 3 epochs.
No change to any locked artifact / canonical config / training code — this is new probe code.

Arms:
  A0  continued-MSD  : fine-tune with L_CE-AT only (no head, no L_pp).
  A1  placebo        : head + L_pp with RANDOMIZED negatives (no class-geometry signal).
  A2  pull-push      : head + real L_pp (other-class adversarial negatives).

Loss  L = L_CE-AT + lambda * L_pp.
  L_CE-AT  = mean_i max_{p in {inf,2,1}} CE(logits(x_i^p), y_i)   (worst-case multi-norm AT;
             held constant across arms, so it does not bias the A2-vs-A1 kill-test).
  L_pp     = NT-Xent (tau=0.1): anchor z_i^clean detached; positives {z_i^inf,z_i^2,z_i^1};
             negatives = other-class adversarial embeddings in the batch (A2) or randomized (A1).

Eval (12-AA strict union) is on a TRAIN-HOLDOUT validation split (NEVER the test set).
Backbone saw all of train, so absolute val robustness is optimistic — but the kill-test is
RELATIVE (A2 vs A1 on identical val), so the shared bias cancels.
"""
from __future__ import annotations
import argparse, json, sys, hashlib, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/robust_union/CIFAR10/models"))
from robustdro.attacks.apgd_train import apgd_train  # training-time APGD per norm
import eval_multinorm_audit as H                      # frozen harness (attacks only)

EPS = {"Linf": 8/255, "L2": 0.5, "L1": 12.0}
NORMS = ["Linf", "L2", "L1"]


# ----------------------------- model -----------------------------
class BackboneHead(nn.Module):
    """robust_union PreActResNet18 backbone + SimCLR-style projection head.
    Head is used only for L_pp during training and DISCARDED at inference."""
    def __init__(self, backbone, proj_dim=128, hid=512):
        super().__init__()
        self.b = backbone
        self.head = nn.Sequential(nn.Linear(512, hid), nn.ReLU(inplace=True),
                                  nn.Linear(hid, proj_dim))

    def features(self, x):
        b = self.b
        out = b.conv1(x)
        out = b.layer1(out); out = b.layer2(out); out = b.layer3(out); out = b.layer4(out)
        out = F.avg_pool2d(out, 4).view(out.size(0), -1)   # 512-d penultimate
        return out

    def logits(self, x):
        return self.b(x)                                    # their forward (== classifier)

    def embed(self, x):
        return F.normalize(self.head(self.features(x)), dim=1)

    def forward(self, x):
        return self.logits(x)                               # apgd_train attacks the classifier


def load_msd_backbone(device):
    from preact_resnet import PreActResNet18
    ck = ROOT / "external/robust_union/CIFAR10/Selected/MSD.pt"
    sd = torch.load(ck, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = PreActResNet18(); m.load_state_dict(sd, strict=True); m.to(device)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    return m, sha


# ----------------------------- losses -----------------------------
def ce_at_loss(model, xs_adv, y):
    """Worst-case (max over norms) CE-AT. xs_adv = list of 3 adversarial batches."""
    ce = torch.stack([F.cross_entropy(model.logits(xa), y, reduction="none") for xa in xs_adv], 0)
    return ce.max(0).values.mean()


def _select_negatives(neg_mask, sim_d, pos_sim_d, neg_mode, k, delta):
    """Restrict the valid-negative mask (C1 negative mining). Selection uses DETACHED
    similarities (non-differentiable choice); the loss still uses grad-carrying sim at kept
    positions. neg_mode: 'all' | 'hardtopk' | 'semihard'."""
    if neg_mode == "all":
        return neg_mask
    R = neg_mask.shape[0]
    ar = torch.arange(R, device=neg_mask.device)
    if neg_mode == "hardtopk":
        # per anchor keep top-k HARDEST (highest sim) among valid negatives
        masked = sim_d.masked_fill(~neg_mask, float("-inf"))
        topv, topi = masked.topk(min(k, R), dim=1)
        out = torch.zeros_like(neg_mask)
        out.scatter_(1, topi, torch.isfinite(topv))
        return out & neg_mask
    if neg_mode == "semihard":
        # FaceNet band: farther than the positive but within delta (pos_sim-δ < sim < pos_sim)
        lo = (pos_sim_d - delta)[:, None]; hi = pos_sim_d[:, None]
        band = neg_mask & (sim_d > lo) & (sim_d < hi)
        empty = band.sum(1) == 0                              # fall back to nearest valid neg
        if empty.any():
            nearest = sim_d.masked_fill(~neg_mask, float("-inf")).argmax(1)
            fb = torch.zeros_like(neg_mask); fb[ar, nearest] = True
            fb = fb & neg_mask & empty[:, None]
            band = band | fb
        return band
    raise ValueError(f"neg_mode {neg_mode!r}")


def pull_push_loss(model, x_clean, xs_adv, y, tau=0.1, mode="real", gen=None,
                   neg_mode="all", k=8, delta=0.2):
    """NT-Xent pull-push (pre-reg §3). mode: 'real' | 'placebo'.
    anchor = detached clean embedding; positives = own adv embeddings;
    negatives = other-class adv embeddings ('real') or randomized ('placebo'),
    optionally hard/semi-hard mined (neg_mode, C1)."""
    B = x_clean.shape[0]
    with torch.no_grad():
        z_clean = model.embed(x_clean)                       # (B,d) STOP-GRAD anchor
    z_adv = torch.cat([model.embed(xa) for xa in xs_adv], 0) # (3B,d) positives, WITH grad
    anchor_idx = torch.arange(3 * B, device=x_clean.device) % B
    y3 = y.repeat(3)                                          # (3B,) class of each adv row

    pos_sim = (z_adv * z_clean[anchor_idx]).sum(1)           # (3B,)
    sim = z_adv @ z_adv.t()                                  # (3B,3B) adv-adv cos sim
    if mode == "real":
        neg_mask = (y3[:, None] != y3[None, :])              # other-class => negative
    elif mode == "placebo":
        perm = torch.randperm(3 * B, generator=gen, device=x_clean.device)
        yp = y3[perm]                                        # randomized labels
        neg_mask = (yp[:, None] != yp[None, :])              # random negatives, no class signal
    else:
        raise ValueError(mode)
    neg_mask.fill_diagonal_(False)
    neg_mask = _select_negatives(neg_mask, sim.detach(), pos_sim.detach(), neg_mode, k, delta)

    logits_pos = pos_sim / tau
    logits_neg = sim / tau
    # log denom = logsumexp( [pos ; masked negs] )
    neg_term = logits_neg.masked_fill(~neg_mask, float("-inf"))
    cat = torch.cat([logits_pos[:, None], neg_term], dim=1)  # (3B, 1+3B)
    log_denom = torch.logsumexp(cat, dim=1)
    loss = (log_denom - logits_pos).mean()                   # -log(num/denom)
    return loss


def ramp_pairing_loss(model, x_inf, x_l1, y):
    """C1b positive control — RAMP logit-pairing (Jiang & Singh, NeurIPS 2024), logit-pairing
    ONLY (no gradient-projection — probe limitation, recorded). Exactly RAMP's KL:
    source = ℓ∞ adversarial, target = ℓ1 adversarial; on the source(ℓ∞)-correct subset,
    KL(softmax f(x∞) ‖ softmax f(x1)) via KLDivLoss(reduction='sum')/N_sel."""
    out_s = model.logits(x_inf)                 # source = Linf
    out_t = model.logits(x_l1)                  # target = L1
    sel = (out_s.argmax(1) == y).detach()       # correct-ℓ∞ subset
    n = int(sel.sum())
    if n == 0:
        return out_s.new_zeros(())
    s = out_s[sel]; t = out_t[sel]
    kl = F.kl_div(F.log_softmax(t + 1e-12, dim=1), F.softmax(s, dim=1), reduction="sum") / n
    return kl


# ------------------------- collapse diagnostics -------------------------
@torch.no_grad()
def collapse_diag(model, x, xs_adv):
    """alignment (mean adv-clean sim), uniformity (Wang&Isola), mean head-output norm."""
    zc = model.embed(x)
    za = model.embed(xs_adv[0])
    align = (za * zc).sum(1).mean().item()
    z = zc
    sq = torch.pdist(z, p=2).pow(2)
    unif = torch.log(torch.exp(-2.0 * sq).mean() + 1e-12).item() if len(sq) else float("nan")
    norm = model.head(model.features(x)).norm(dim=1).mean().item()
    return {"alignment": align, "uniformity": unif, "embed_norm": norm}


# ----------------------------- data -----------------------------
# Split: fine-tune train = train[0:44000]; C0 val (train-holdout) = train[44000:49000] (5k);
# val_1k = train[44000:45000]. val_select train[49000:50000] reserved untouched.
FT_TRAIN = (0, 44000)
VAL_5K   = (44000, 49000)
VAL_1K   = (44000, 45000)

def load_train_tensors(device):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=str(ROOT/"data"), train=True, download=False,
                                      transform=T.ToTensor())
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return X, Y

def subset_sha(a, b):
    return hashlib.sha256(f"cifar10_train[{a}:{b}]".encode()).hexdigest()[:16]


# ------------------------- fine-tune -------------------------
def finetune(model, X, Y, arm, lam, seed, epochs, bs, lr, n_iter, device, log=print,
             neg_mode="all", k=8, delta=0.2, aux="pullpush", lam_ramp=1.5):
    g = torch.Generator(device=device); g.manual_seed(seed)
    idx = torch.arange(FT_TRAIN[0], FT_TRAIN[1])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    model.train()
    perm_gen = torch.Generator(); perm_gen.manual_seed(seed)
    for ep in range(epochs):
        order = idx[torch.randperm(len(idx), generator=perm_gen)]
        run_loss = run_ce = run_pp = n = 0
        t0 = time.time()
        for bi in range(0, len(order), bs):
            b = order[bi:bi+bs]
            x = X[b].to(device); y = Y[b].to(device)
            model.eval()  # craft adversarials against eval-mode backbone (stable BN)
            xs_adv = [apgd_train(model, x, y, nm, EPS[nm], n_iter=n_iter, is_train=True).detach()
                      for nm in NORMS]
            model.train()
            loss_ce = ce_at_loss(model, xs_adv, y)
            if arm == "A0" or aux == "none":
                loss = loss_ce; loss_aux = torch.tensor(0.0)
            elif aux == "ramp":                                   # A3: MSD + RAMP logit-pairing
                loss_aux = ramp_pairing_loss(model, xs_adv[0], xs_adv[2], y)  # Linf source, L1 target
                loss = loss_ce + lam_ramp * loss_aux
            else:                                                 # pullpush (A1 placebo / A2 real)
                mode = "placebo" if arm == "A1" else "real"
                loss_aux = pull_push_loss(model, x, xs_adv, y, tau=0.1, mode=mode, gen=g,
                                          neg_mode=neg_mode, k=k, delta=delta)
                loss = loss_ce + lam * loss_aux
            opt.zero_grad(); loss.backward(); opt.step()
            bs_ = y.size(0); n += bs_
            run_loss += loss.item()*bs_; run_ce += loss_ce.item()*bs_; run_pp += float(loss_aux.detach())*bs_
        log(f"  [ep{ep+1}/{epochs}] loss={run_loss/n:.4f} ce={run_ce/n:.4f} pp={run_pp/n:.4f} "
            f"({time.time()-t0:.0f}s)")
    model.eval()
    return model


# ------------------------- 12-AA eval on given (x,y) -------------------------
def eval_12aa(model, x, y, device, bs, few_step_iter=10):
    cfg = H.load_audit_config(str(ROOT/"configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml"))
    H.validate_config(cfg)
    clean = H.clean_mask(model, x, y, device, bs)
    per_attack, masks = {}, {}
    for atk in H.iter_attacks(cfg):
        nm = atk["name"]
        m, _, _ = H.run_attack_mask(model, x, y, atk, float(cfg["eps"][atk["eps_key"]]), device, bs)
        masks[nm] = m; per_attack[nm] = H.mean_mask(m)
    def AND(names): 
        mm = masks[names[0]].clone()
        for nb in names[1:]: mm &= masks[nb]
        return mm
    per_norm = {}
    per_norm_masks = {}
    for norm, names in H.CANONICAL_ATTACKS.items():
        mm = AND(names); per_norm_masks[norm] = mm; per_norm[norm] = H.mean_mask(mm)
    union_mask = AND(list(masks.keys()))
    union = H.mean_mask(union_mask)
    # masking (obfuscation) diagnostics — Athalye/Carlini directionality:
    #   apgd_minus_square = apgd_acc - square_acc. NEGATIVE = white-box (APGD) STRONGER than
    #   black-box (Square) = HEALTHY. POSITIVE and large = Square breaks what APGD couldn't
    #   => gradient masking. Flag iff Square is materially stronger than APGD.
    apgd_sq = {}
    for norm in ("linf","l2","l1"):
        apgd_sq[norm] = per_attack[f"apgd_ce_{norm}"] - per_attack[f"square_{norm}"]
    # few-step vs many-step (linf): apgd_train 10-step vs harness APGD-CE 100-step.
    #   fewmany = few_acc - many_acc. Normal: few weaker => few_acc >= many_acc => fewmany >= 0.
    #   fewmany < 0 (more steps give HIGHER robust acc) => attack not converging / masking.
    x_few = apgd_train(model, x.to(device), y.to(device), "Linf", EPS["Linf"], n_iter=few_step_iter, is_train=False)
    with torch.no_grad():
        few_acc = (torch.cat([model(x_few[i:i+bs]).argmax(1).cpu() for i in range(0,len(x_few),bs)]) == y.cpu()).float().mean().item()
    fewmany_linf = few_acc - per_attack["apgd_ce_linf"]
    masking_flag = (max(apgd_sq.values()) > 0.03) or (fewmany_linf < -0.01)
    return {
        "clean_acc": H.mean_mask(clean), "union": union, "per_norm_union": per_norm,
        "per_attack": per_attack,
        "masking": {"apgd_minus_square": apgd_sq, "fewstep_minus_manystep_linf": fewmany_linf,
                    "flag": bool(masking_flag)},
        "n": int(len(y)),
        "union_mask": union_mask.int().cpu().tolist(),
        "linf_mask": per_norm_masks["linf"].int().cpu().tolist(),
    }


# ----------------------------- CLI -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["A0","A1","A2","A3"])
    p.add_argument("--aux", choices=["auto","none","pullpush","ramp"], default="auto")  # C1b
    p.add_argument("--lam-ramp", type=float, default=1.5)   # RAMP default lbd
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--n-iter", type=int, default=10)      # training APGD steps
    p.add_argument("--val", choices=["1k","5k"], default="5k")
    p.add_argument("--eval-bs", type=int, default=128)
    p.add_argument("--neg-mode", choices=["all","hardtopk","semihard"], default="all")  # C1
    p.add_argument("--k", type=int, default=8)            # hardtopk
    p.add_argument("--delta", type=float, default=0.2)    # semihard band
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")        # tiny: 1 batch train, 64 val, cpu-ok
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    backbone, msd_sha = load_msd_backbone(device)
    model = BackboneHead(backbone).to(device)

    X, Y = load_train_tensors(device)
    va, vb = (VAL_1K if a.val == "1k" else VAL_5K)
    if a.smoke:
        # tiny: fine-tune 1 batch, eval 64 val examples
        global FT_TRAIN
        FT_TRAIN = (0, a.bs); va, vb = (44000, 44064)
        a.epochs = 1
    xv = X[va:vb].to(device); yv = Y[va:vb]

    aux = a.aux if a.aux != "auto" else {"A0":"none","A1":"pullpush","A2":"pullpush","A3":"ramp"}[a.arm]
    t0 = time.time()
    model = finetune(model, X, Y, a.arm, a.lam, a.seed, a.epochs, a.bs, a.lr, a.n_iter, device,
                     neg_mode=a.neg_mode, k=a.k, delta=a.delta, aux=aux, lam_ramp=a.lam_ramp)
    # collapse diag on a val batch (apgd_train needs grad internally; do NOT wrap in no_grad)
    xb = xv[:min(256,len(xv))]
    yb = yv[:len(xb)].to(device)
    xs_adv_v = [apgd_train(model, xb, yb, nm, EPS[nm], n_iter=a.n_iter, is_train=False) for nm in NORMS]
    coll = collapse_diag(model, xb, xs_adv_v)
    ev = eval_12aa(model, xv, yv, device, a.eval_bs)

    out = {
        "stage":"C0_killtest","arm":a.arm,"aux":aux,"lam":a.lam,"lam_ramp":a.lam_ramp,"seed":a.seed,"tau":0.1,
        "neg_mode":a.neg_mode,"k":a.k,"delta":a.delta,
        "epochs":a.epochs,"lr":a.lr,"bs":a.bs,"train_apgd_niter":a.n_iter,
        "backbone":"public_MSD80_robust_union","backbone_sha256":msd_sha,
        "ft_train_split":f"cifar10_train[{FT_TRAIN[0]}:{FT_TRAIN[1]}]",
        "val_split":f"cifar10_train[{va}:{vb}]","val_split_sha":subset_sha(va,vb),
        "eval":ev,"collapse":coll,
        "ce_at_def":"worst-case max over {Linf,L2,L1} APGD adversarials (constant across arms)",
        "note":"backbone saw all train; val is train-holdout from fine-tune; kill-test is RELATIVE (A2 vs A1).",
        "elapsed_s":round(time.time()-t0,1),
    }
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"[C0] {a.arm} lam={a.lam} seed={a.seed} -> union={ev['union']:.4f} "
          f"linf={ev['per_norm_union']['linf']:.4f} clean={ev['clean_acc']:.4f} "
          f"mask_flag={ev['masking']['flag']} -> {op}")


if __name__ == "__main__":
    main()

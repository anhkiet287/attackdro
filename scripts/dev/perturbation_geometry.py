"""
perturbation_geometry.py
------------------------
Diagnostic for the hypothesis: "adversarial perturbations avoid the invariant
(discriminative) features, and different attack norms reveal which regions are
truly invariant." Answers two decision questions with numbers, not eyeballing:

  Q1  Do L-inf / L2 / L1 attacks hit the SAME pixels?  -> cross-norm correlation + top-k IoU
  Q2  Do perturbations AVOID salient (important) regions? -> perturbation-vs-saliency correlation
  (+  How concentrated is each norm's perturbation? -> top-10% energy fraction)

Run (Colab or the PC; it's light — a few batches):
  python perturbation_geometry.py --ckpt checkpoints/pgd_at_linf_best.pt
  # Use a FULLY-TRAINED / robust checkpoint for meaningful results (epoch 4 is too early).
  # Optional: also run on a standard (non-robust) model to contrast:
  python perturbation_geometry.py --ckpt none      # random/untrained -> pipeline check only

Deps: torch, torchvision, numpy, matplotlib
"""
import argparse, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- model (PreActResNet-18, CIFAR)
class PreActBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin); self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout); self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Conv2d(cin, cout, 1, stride, bias=False)
    def forward(self, x):
        out = F.relu(self.bn1(x))
        s = self.short(out) if self.short is not None else x
        out = self.conv1(out); out = self.conv2(F.relu(self.bn2(out)))
        return out + s

class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.bn = nn.BatchNorm2d(512); self.linear = nn.Linear(512, num_classes)
    def _make(self, cin, cout, n, stride):
        layers = [PreActBlock(cin, cout, stride)] + [PreActBlock(cout, cout, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.relu(self.bn(out)); out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)

# normalization done INSIDE forward so attacks live in [0,1] pixel space
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
class Normalized(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m((x - MEAN.to(x.device)) / STD.to(x.device))

def load_ckpt(model, path):
    if path in (None, "none", "None"):
        print("[warn] no checkpoint — using untrained model (pipeline check only).")
        return
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "net"):
        if isinstance(sd, dict) and k in sd: sd = sd[k]
    if isinstance(sd, nn.Module): sd = sd.state_dict()
    sd = { kk.replace("module.", "").replace("model.", ""): vv for kk, vv in sd.items() }
    miss = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {path}  (missing={len(miss.missing_keys)}, unexpected={len(miss.unexpected_keys)})")

# ----------------------------------------------------------------- attacks (in [0,1] space)
def _grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, x)[0]

def pgd_linf(model, x, y, eps=8/255, alpha=2/255, steps=50):
    d = torch.zeros_like(x)
    for _ in range(steps):
        g = _grad(model, (x + d).clamp(0, 1), y)
        d = (d + alpha * g.sign()).clamp(-eps, eps)
        d = ((x + d).clamp(0, 1) - x)
    return d.detach()

def pgd_l2(model, x, y, eps=0.5, alpha=0.1, steps=50):
    d = torch.zeros_like(x)
    for _ in range(steps):
        g = _grad(model, (x + d).clamp(0, 1), y)
        gn = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        d = d + alpha * gn
        dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        d = d * torch.clamp(eps / (dn + 1e-12), max=1.0)
        d = ((x + d).clamp(0, 1) - x)
    return d.detach()

def project_l1(v, eps):  # Duchi et al. 2008, per-sample on flattened v
    B = v.shape[0]; vf = v.view(B, -1); absv = vf.abs()
    inside = absv.sum(1) <= eps
    u, _ = torch.sort(absv, dim=1, descending=True)
    css = u.cumsum(1)
    idx = torch.arange(1, vf.shape[1] + 1, device=v.device).float()
    cond = (u - (css - eps) / idx) > 0
    rho = cond.float().sum(1).clamp(min=1)
    theta = ((css.gather(1, (rho - 1).long().unsqueeze(1)).squeeze(1) - eps) / rho).clamp(min=0)
    w = torch.sign(vf) * (absv - theta.unsqueeze(1)).clamp(min=0)
    w[inside] = vf[inside]
    return w.view_as(v)

def pgd_l1(model, x, y, eps=12.0, alpha=1.0, steps=50):
    # L1-ball projection soft-thresholds -> naturally SPARSE perturbation (the point of L1)
    d = torch.zeros_like(x)
    for _ in range(steps):
        g = _grad(model, (x + d).clamp(0, 1), y)
        d = d + alpha * g
        d = project_l1(d, eps)
        d = ((x + d).clamp(0, 1) - x)
    return d.detach()

# ----------------------------------------------------------------- maps & metrics
def energy_map(delta):                 # (B,3,H,W) -> (B,H,W) per-pixel magnitude across channels
    return delta.pow(2).sum(1).sqrt()

def saliency_map(model, x, y):         # |d logit_y / d x| summed over channels, on clean x
    x = x.clone().detach().requires_grad_(True)
    logit = model(x).gather(1, y.view(-1, 1)).sum()
    g = torch.autograd.grad(logit, x)[0]
    return g.abs().sum(1)              # (B,H,W)

def pearson(a, b):                     # per-sample corr over pixels, then mean; a,b: (B,H,W)
    A = a.flatten(1); B = b.flatten(1)
    A = A - A.mean(1, keepdim=True); B = B - B.mean(1, keepdim=True)
    num = (A * B).sum(1)
    den = A.norm(dim=1) * B.norm(dim=1) + 1e-12
    return (num / den).mean().item()

def topk_iou(a, b, frac=0.2):          # IoU of top-frac pixels
    def mask(m):
        B = m.flatten(1); k = max(1, int(frac * B.shape[1]))
        thr = B.kthvalue(B.shape[1] - k + 1, dim=1).values.view(-1, 1)
        return (B >= thr)
    ma, mb = mask(a), mask(b)
    inter = (ma & mb).float().sum(1); union = (ma | mb).float().sum(1) + 1e-12
    return (inter / union).mean().item()

def concentration(a, frac=0.1):        # fraction of energy in top-frac pixels (1.0=all in a few px)
    B = a.flatten(1); k = max(1, int(frac * B.shape[1]))
    top = B.topk(k, dim=1).values.sum(1); tot = B.sum(1) + 1e-12
    return (top / tot).mean().item()

# ----------------------------------------------------------------- viz
def visualize(x, maps, sal, path, n=6):
    n = min(n, x.shape[0])
    cols = ["clean", "L-inf", "L2", "L1", "saliency", "agree (R=Linf G=L2 B=L1)"]
    fig, ax = plt.subplots(n, 6, figsize=(13, 2.1 * n))
    if n == 1: ax = ax[None, :]
    def norm01(t):
        t = t - t.min(); return t / (t.max() + 1e-12)
    for i in range(n):
        ax[i, 0].imshow(x[i].permute(1, 2, 0).cpu().numpy())
        for j, key in enumerate(["linf", "l2", "l1"]):
            ax[i, j + 1].imshow(norm01(maps[key][i]).cpu().numpy(), cmap="magma")
        ax[i, 4].imshow(norm01(sal[i]).cpu().numpy(), cmap="viridis")
        rgb = torch.stack([norm01(maps["linf"][i]), norm01(maps["l2"][i]), norm01(maps["l1"][i])], -1)
        ax[i, 5].imshow(rgb.cpu().numpy())
        for j in range(6):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(cols[j], fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[viz] saved {path}")

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--n_metric", type=int, default=128)   # images for the numbers
    ap.add_argument("--n_viz", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eps_linf", type=float, default=8/255)
    ap.add_argument("--eps_l2", type=float, default=0.5)
    ap.add_argument("--eps_l1", type=float, default=12.0)
    ap.add_argument("--out", default="perturbation_geometry.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Normalized(PreActResNet18()).to(dev).eval()
    load_ckpt(model.m, args.ckpt)

    ds = tv.datasets.CIFAR10("./data", train=False, download=True, transform=tv.transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(args.n_metric)]).to(dev)
    y = torch.tensor([ds[i][1] for i in range(args.n_metric)]).to(dev)

    print(f"[run] {args.n_metric} imgs, {args.steps} steps, eps=(Linf {args.eps_linf:.4f}, L2 {args.eps_l2}, L1 {args.eps_l1}) on {dev}")
    d_inf = pgd_linf(model, x, y, args.eps_linf, steps=args.steps)
    d_l2 = pgd_l2(model, x, y, args.eps_l2, steps=args.steps)
    d_l1 = pgd_l1(model, x, y, args.eps_l1, steps=args.steps)
    maps = {"linf": energy_map(d_inf), "l2": energy_map(d_l2), "l1": energy_map(d_l1)}
    sal = saliency_map(model, x, y)

    # ---- metrics ----
    print("\n================ RESULTS ================")
    print("\nQ1 — do the norms attack the SAME regions?")
    print(f"  corr(Linf,L2) = {pearson(maps['linf'], maps['l2']):+.3f}    IoU@top20% = {topk_iou(maps['linf'], maps['l2']):.3f}")
    print(f"  corr(Linf,L1) = {pearson(maps['linf'], maps['l1']):+.3f}    IoU@top20% = {topk_iou(maps['linf'], maps['l1']):.3f}")
    print(f"  corr(L2,  L1) = {pearson(maps['l2'],   maps['l1']):+.3f}    IoU@top20% = {topk_iou(maps['l2'], maps['l1']):.3f}")
    print("  -> LOW corr / LOW IoU  => norms hit different regions => 'cross-norm invariant feature' angle has room")
    print("  -> HIGH corr / HIGH IoU => same regions => that angle is weak, reconsider")

    print("\nQ2 — do perturbations AVOID salient (important) regions?")
    for k in ("linf", "l2", "l1"):
        print(f"  corr(perturb_{k}, saliency) = {pearson(maps[k], sal):+.3f}")
    print("  -> NEGATIVE => attacks avoid salient features => supports your masking premise")
    print("  -> POSITIVE => attacks HIT salient features => the naive 'perturb=background' premise is wrong")

    print("\nConcentration (energy fraction in top-10% pixels):")
    for k in ("linf", "l2", "l1"):
        print(f"  {k:5s} = {concentration(maps[k]):.3f}")
    print("  -> expect L1 >> Linf (L1 is sparse). Confirms the 'concentrated perturbation' story is an L1 phenomenon.")
    print("=========================================\n")

    visualize(x, maps, sal, args.out, n=args.n_viz)

if __name__ == "__main__":
    main()
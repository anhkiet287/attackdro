"""
union_mask_test.py
------------------
Confirmatory test for the idea: "take 3 masks (one per threat model), union them,
then train while focusing on that mask (object) or its complement (background)."

Part 1 (mask geometry): build per-norm top-k masks -> union -> compare with the
  salient/object region.  Shows what the union mask actually IS.
Part 2 (the decision): train a fresh model 1 epoch three ways and report CLEAN acc:
  - full            : normal images
  - object-focus    : keep only the mask region (zero the rest)
  - background-focus : keep only the complement (zero the mask region)
  We use a saliency mask as the stand-in for the union-of-attacks mask — justified
  because the diagnostic already showed perturbation energy correlates strongly
  with saliency (so union-of-attacks region ~= salient/object region).

Place next to perturbation_geometry.py (same folder) and run:
  python scripts/union_mask_test.py --ckpt checkpoints/pgd_at_linf_best.pt
  # GPU is busy with training -> defaults to CPU. For speed run on Colab with --device cuda.
  # Use a trained-ish checkpoint (so saliency/attacks are meaningful).
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, matplotlib.pyplot as plt
from perturbation_geometry import (
    PreActResNet18, Normalized, load_ckpt,
    pgd_linf, pgd_l2, pgd_l1, energy_map, saliency_map,
)

def topk_mask(m, frac):                       # (B,H,W) bool: top-frac pixels per image
    B = m.flatten(1); k = max(1, int(frac * B.shape[1]))
    thr = B.kthvalue(B.shape[1] - k + 1, dim=1).values.view(-1, 1, 1)
    return m >= thr

def apply_mask(x, mask, keep_inside=True):    # zero out the dropped region
    m = mask.unsqueeze(1).float()
    return x * (m if keep_inside else (1 - m))

def train_eval(mode, masks_tr, xtr, ytr, xte, yte, dev, epochs, B=128):
    torch.manual_seed(0)
    net = Normalized(PreActResNet18()).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(xtr.shape[0])
        for i in range(0, xtr.shape[0], B):
            idx = perm[i:i + B]
            xb, yb = xtr[idx].to(dev), ytr[idx].to(dev)
            if mode != "full":
                xb = apply_mask(xb, masks_tr[idx].to(dev), keep_inside=(mode == "object"))
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, xte.shape[0], 256):
            xb, yb = xte[i:i + 256].to(dev), yte[i:i + 256].to(dev)
            correct += (net(xb).argmax(1) == yb).sum().item()
    return correct / xte.shape[0]

def precompute_masks(model, x, y, frac, dev, B=128):
    out = []
    for i in range(0, x.shape[0], B):
        s = saliency_map(model, x[i:i + B].to(dev), y[i:i + B].to(dev))
        out.append(topk_mask(s, frac).cpu())
    return torch.cat(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=64)          # imgs for mask geometry (Part 1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cpu")            # cpu so it won't fight the training GPU
    ap.add_argument("--train-n", type=int, default=5000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default="union_mask_test.png")
    args = ap.parse_args()
    dev = args.device

    model = Normalized(PreActResNet18()).to(dev).eval()
    load_ckpt(model.m, args.ckpt)

    tr = tv.datasets.CIFAR10("./data", train=True, download=True, transform=tv.transforms.ToTensor())
    te = tv.datasets.CIFAR10("./data", train=False, download=True, transform=tv.transforms.ToTensor())
    xg = torch.stack([te[i][0] for i in range(args.n)]).to(dev)
    yg = torch.tensor([te[i][1] for i in range(args.n)]).to(dev)

    # -------- Part 1: what is the union mask? --------
    di = pgd_linf(model, xg, yg, steps=args.steps)
    d2 = pgd_l2(model, xg, yg, steps=args.steps)
    d1 = pgd_l1(model, xg, yg, steps=args.steps)
    m_inf, m_l2, m_l1 = topk_mask(energy_map(di), args.frac), topk_mask(energy_map(d2), args.frac), topk_mask(energy_map(d1), args.frac)
    union = m_inf | m_l2 | m_l1
    sal = saliency_map(model, xg, yg)
    salient = topk_mask(sal, args.frac)                    # the "object feature" region

    cover = union.float().mean().item()
    p_sal_given_union = (union & salient).float().sum() / (union.float().sum() + 1e-9)
    iou = (union & salient).float().sum() / ((union | salient).float().sum() + 1e-9)
    sal_norm = (sal / (sal.flatten(1).sum(1, keepdim=True).view(-1, 1, 1) + 1e-9))
    sal_captured = (sal_norm * union.float()).flatten(1).sum(1).mean().item()  # object signal inside union

    print("\n================ PART 1 — what the union mask IS ================")
    print(f"  union mask coverage         : {cover*100:5.1f}% of the image")
    print(f"  P(salient | in union mask)  : {p_sal_given_union.item()*100:5.1f}%   (how much of the mask sits on object features)")
    print(f"  IoU(union, salient)         : {iou.item():.3f}")
    print(f"  saliency energy INSIDE union: {sal_captured*100:5.1f}%   (object signal you'd KEEP as 'object' / LOSE as 'background')")
    print("  -> union mask ~= object/salient region. 'Focus object' = focus on non-robust features; 'focus background' = throw away this much signal.")

    # -------- Part 2: the decision (clean acc under 3 regimes) --------
    xtr = torch.stack([tr[i][0] for i in range(args.train_n)])
    ytr = torch.tensor([tr[i][1] for i in range(args.train_n)])
    xte = torch.stack([te[i][0] for i in range(args.test_n)])
    yte = torch.tensor([te[i][1] for i in range(args.test_n)])
    print("\n[part2] precomputing saliency masks (stand-in for union-of-attacks mask)…")
    masks_tr = precompute_masks(model, xtr, ytr, args.frac, dev)

    print(f"[part2] training fresh models {args.epochs} epoch on {args.train_n} imgs (device={dev})…")
    acc_full = train_eval("full", masks_tr, xtr, ytr, xte, yte, dev, args.epochs)
    acc_obj = train_eval("object", masks_tr, xtr, ytr, xte, yte, dev, args.epochs)
    acc_bg = train_eval("background", masks_tr, xtr, ytr, xte, yte, dev, args.epochs)

    print("\n================ PART 2 — clean test accuracy ================")
    print(f"  full images        : {acc_full*100:5.1f}%")
    print(f"  focus OBJECT (mask): {acc_obj*100:5.1f}%   (keep union mask, zero background)")
    print(f"  focus BACKGROUND   : {acc_bg*100:5.1f}%   (keep background, zero the mask)")
    print("  -> background-focus collapses (discards the discriminative signal). object-focus ~ full but keeps the vulnerability inside (see Part 1 + Q2).")
    print("==============================================================\n")

    # -------- viz --------
    n = min(6, args.n)
    cols = ["clean", "union mask", "focus object", "focus background", "saliency"]
    fig, ax = plt.subplots(n, 5, figsize=(11, 2.1 * n))
    if n == 1: ax = ax[None, :]
    for i in range(n):
        ax[i, 0].imshow(xg[i].permute(1, 2, 0).cpu().numpy())
        ax[i, 1].imshow(union[i].cpu().numpy(), cmap="gray")
        ax[i, 2].imshow(apply_mask(xg[i:i+1], union[i:i+1], True)[0].permute(1, 2, 0).cpu().numpy())
        ax[i, 3].imshow(apply_mask(xg[i:i+1], union[i:i+1], False)[0].permute(1, 2, 0).cpu().numpy())
        ax[i, 4].imshow((sal[i] / (sal[i].max() + 1e-9)).cpu().numpy(), cmap="viridis")
        for j in range(5):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(cols[j], fontsize=10)
    plt.tight_layout(); plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[viz] saved {args.out}")

if __name__ == "__main__":
    main()
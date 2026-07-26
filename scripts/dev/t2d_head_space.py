"""T2b — head-space (h) vs encoder-space (f) alignment, measured at the SAME checkpoint.

Pre-registered in docs/preregistrations/preregistration_T2b_head_space.md. Read that first.

CLAMP's glue acts on  z = normalize(head(features(x)))  — the head h, which is DISCARDED at eval.
T2 showed the encoder f moves adv AWAY from clean. This asks whether the alignment the loss actually
optimises exists in h.

Checkpoint constraint (pre-registered §2): val_best.pt stores the backbone only, so h is available
only in resume.pt / ckpt_latest.pt = LAST epoch. f is therefore RE-measured here at the last epoch
too, and the adversarial views are regenerated against those same weights. T2's val_best f-numbers
are printed for reference only.

  python scripts/dev/t2d_head_space.py [--n-iter 50] [--bs 128]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
from robustdro.models import build_model                     # noqa: E402
from robustdro.attacks.apgd_train import apgd_train          # noqa: E402
import importlib.util                                        # noqa: E402
_s = importlib.util.spec_from_file_location("t2", os.path.join(ROOT, "scripts/dev/t2_geometry.py"))
t2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t2)

OUT = t2.OUT
NORMS, VIEWS, EPS, APGD_NORM = t2.NORMS, t2.VIEWS, t2.EPS, t2.APGD_NORM
MODEL_CFG = {"model": {"arch": "preact_resnet18", "normalize": False},
             "dataset": {"num_classes": 10}}

# (arm, role, resume-checkpoint, head trained?)
MODELS = [
    ("M1a",      "PRIMARY  weak   CLAMP  (MSD-10, 80ep)",  "results/fromscratch/C5/M1a_advneg/seed0/resume.pt",     True),
    ("M1a_full", "PRIMARY  strong CLAMP-50 (MSD-50, 50ep)", "results/fromscratch/C5_full/M1a_full/ckpt_latest.pt",  True),
    ("M0",       "control  weak   term OFF -> head UNTRAINED (seed2)", "results/fromscratch/C5/M0_matched/seed2/resume.pt", False),
    ("M0_full",  "control  strong term OFF -> head UNTRAINED",         "results/fromscratch/C5_full/M0_full/ckpt_latest.pt", False),
]


class BackboneHead(nn.Module):
    """Exact replica of scripts/dev/c5_fromscratch.py:BackboneHead (same module names, so the
    resume state_dict loads strictly)."""

    def __init__(self):
        super().__init__()
        self.b = build_model(MODEL_CFG)
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))

    def features(self, x):
        b = self.b
        out = b.conv1(x)
        for L in (b.layer1, b.layer2, b.layer3, b.layer4):
            out = L(out)
        out = F.relu(b.bn(out))
        return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)      # f, 512-d

    def embed(self, x):
        return F.normalize(self.head(self.features(x)), dim=1)          # z = h-space, 128-d

    def forward(self, x):
        return self.b(x)                                                 # logits — what APGD attacks


def load(path, device):
    ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
    m = BackboneHead()
    m.load_state_dict(ck["model"], strict=True)          # strict: head must be present
    return m.to(device).eval(), int(ck.get("epoch", -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260709)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    xs, ys, _ = t2.subset("1k", device)
    print(f"[T2b] n={len(xs)} device={device} apgd_iter={a.n_iter}  (LAST-EPOCH weights, pre-registered)\n")

    R, PEf, PEh = {}, {}, {}
    for arm, role, ck, trained in MODELS:
        m, ep = load(ck, device)
        sha = hashlib.sha256(open(os.path.join(ROOT, ck), "rb").read()).hexdigest()
        hstd = float(m.head[2].weight.std())
        print(f"[T2b] {arm:9s} {role}\n      ep={ep} head.2.std={hstd:.5f} "
              f"({'TRAINED' if trained else 'UNTRAINED = random projection'})  sha={sha[:16]}")
        Zf = {v: [] for v in VIEWS}
        Zh = {v: [] for v in VIEWS}
        acc = {v: [] for v in VIEWS}
        for i in range(0, len(xs), a.bs):
            x, y = xs[i:i + a.bs].to(device), ys[i:i + a.bs].to(device)
            batch = {"clean": x}
            for nm in NORMS:                       # regenerated against THESE weights
                batch[nm] = apgd_train(m, x, y, APGD_NORM[nm], EPS[nm],
                                       n_iter=a.n_iter, is_train=False).detach()
            with torch.no_grad():
                for v in VIEWS:
                    Zf[v].append(m.features(batch[v]).float().cpu())
                    Zh[v].append(m.embed(batch[v]).float().cpu())
                    acc[v].append((m(batch[v]).argmax(1) == y).cpu())
            print(f"      [{i + len(x):>5}/{len(xs)}]", end="\r", flush=True)
        print(" " * 32, end="\r")
        Zf = {v: torch.cat(Zf[v]).numpy() for v in VIEWS}
        Zh = {v: torch.cat(Zh[v]).numpy() for v in VIEWS}
        PEf[arm] = {nm: t2.dist(Zf["clean"], Zf[nm], "cos") for nm in NORMS}
        PEh[arm] = {nm: t2.dist(Zh["clean"], Zh[nm], "cos") for nm in NORMS}
        R[arm] = {"role": role, "ckpt": ck, "ckpt_sha256": sha, "epoch": ep,
                  "head_trained": trained, "head2_std": hstd,
                  "acc": {v: float(torch.cat(acc[v]).float().mean()) for v in VIEWS},
                  "f": t2.geometry(Zf), "h": t2.geometry(Zh)}
        del m
        torch.cuda.empty_cache()
        f, h = R[arm]["f"]["cos"], R[arm]["h"]["cos"]
        print(f"      d_f(cl,·) ℓ∞ {f['clean-linf']:.4f} ℓ2 {f['clean-l2']:.4f} ℓ1 {f['clean-l1']:.4f}"
              f"   |   d_h(cl,·) ℓ∞ {h['clean-linf']:.4f} ℓ2 {h['clean-l2']:.4f} ℓ1 {h['clean-l1']:.4f}\n")

    # ---- primary: paired h-vs-f, per example ----
    rng = np.random.default_rng(0)
    HF = {}
    for arm, _, _, _ in MODELS:
        e = {}
        for nm in NORMS:
            d = PEh[arm][nm] - PEf[arm][nm]
            n = len(d)
            b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
            e[nm] = {"d_h": float(PEh[arm][nm].mean()), "d_f": float(PEf[arm][nm].mean()),
                     "h_minus_f": float(d.mean()),
                     "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
        HF[arm] = e

    T2VB = json.load(open(os.path.join(OUT, "geometry.json")))["models"]

    print("=" * 104)
    print("T2b.1  PRIMARY — head-space vs encoder-space, SAME checkpoint (last epoch), cosine distance")
    print("=" * 104)
    print(f"{'model':<10}{'norm':<6}{'d_h (head)':>12}{'d_f (enc)':>11}{'h − f':>10}{'95% CI':>24}   verdict")
    for arm, _, _, trained in MODELS:
        for nm in NORMS:
            e = HF[arm][nm]
            v = ("h TIGHTER" if e["ci95"][1] < 0 else "h LOOSER" if e["ci95"][0] > 0 else "no diff")
            print(f"{arm:<10}{nm:<6}{e['d_h']:>12.4f}{e['d_f']:>11.4f}{e['h_minus_f']:>+10.4f}"
                  f"  [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]   {v}"
                  + ("" if trained else "   (untrained head)"))
    print("\n  P1 predicted d_h < d_f (h TIGHTER) for the two PRIMARY arms.")

    print("\n" + "=" * 104)
    print("T2b.2  CONTROL (P3) — trained head vs UNTRAINED head (random projection of the same f)")
    print("=" * 104)
    print(f"  {'pair':<34}{'norm':<6}{'d_h trained':>13}{'d_h untrained':>15}{'diff':>10}")
    for A, B, lbl in [("M1a", "M0", "weak:   M1a (trained) vs M0"),
                      ("M1a_full", "M0_full", "strong: M1a_full vs M0_full")]:
        for nm in NORMS:
            ha, hb = HF[A][nm]["d_h"], HF[B][nm]["d_h"]
            print(f"  {lbl:<34}{nm:<6}{ha:>13.4f}{hb:>15.4f}{ha - hb:>+10.4f}")

    print("\n" + "=" * 104)
    print("T2b.3  adv–adv pairwise in HEAD space (are the three views pulled onto one anchor?)")
    print("=" * 104)
    print(f"{'model':<10}{'ℓ∞-ℓ2':>9}{'ℓ∞-ℓ1':>9}{'ℓ2-ℓ1':>9}   |  f-space: {'ℓ∞-ℓ2':>7}{'ℓ∞-ℓ1':>7}{'ℓ2-ℓ1':>7}")
    for arm, _, _, _ in MODELS:
        h, f = R[arm]["h"]["cos"], R[arm]["f"]["cos"]
        print(f"{arm:<10}{h['linf-l2']:>9.4f}{h['linf-l1']:>9.4f}{h['l2-l1']:>9.4f}   |"
              f"           {f['linf-l2']:>7.4f}{f['linf-l1']:>7.4f}{f['l2-l1']:>7.4f}")

    print("\n" + "=" * 104)
    print("T2b.4  reference only — T2's f-space at VAL_BEST vs this run's f-space at LAST epoch")
    print("=" * 104)
    print(f"{'model':<10}{'norm':<6}{'f @val_best (T2)':>18}{'f @last (T2b)':>16}")
    for arm in ("M1a", "M1a_full", "M0", "M0_full"):
        if arm not in T2VB:
            continue
        for nm in NORMS:
            print(f"{arm:<10}{nm:<6}{T2VB[arm]['all']['cos'][f'clean-{nm}']:>18.4f}"
                  f"{R[arm]['f']['cos'][f'clean-{nm}']:>16.4f}")

    json.dump({"config": {"n": len(xs), "apgd_n_iter": a.n_iter, "weights": "last epoch (resume/ckpt_latest)",
                          "h": "normalize(head(features(x))) 128-d", "f": "pooled 512-d",
                          "bootstrap": "B=10000 rng(0) 2-sided 95%, paired per example"},
               "models": R, "h_vs_f": HF},
              open(os.path.join(OUT, "head_space.json"), "w"), indent=2)
    print(f"\n  saved -> results/analysis/T2_geometry/head_space.json")


if __name__ == "__main__":
    main()

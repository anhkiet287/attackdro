"""T2e — POST-HOC (exploratory, not pre-registered). Disambiguates the T2b head-space result.

T2b (pre-registered) measured d_h on TEST data with APGD-50 views: d_h(clean,adv) = 0.22 (M1a),
far LOOSER than f. But the training logs show the glue term reached cos = 0.918 (d_h = 0.082) on its
TRAINING views, which are APGD-10 on TRAIN data. Two things differ at once. This separates them:

  (i) attack strength : same test data, APGD-10 (as in training) vs APGD-50 (as in T2b)
  (ii) train vs test  : APGD-10 on TRAIN images vs APGD-10 on TEST images

Everything else is held at the T2b setting (last-epoch weights, same subset, same metric).
The pre-registered T2b numbers are unchanged and remain the primary result.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import importlib.util
_s = importlib.util.spec_from_file_location("td", os.path.join(ROOT, "scripts/dev/t2d_head_space.py"))
td = importlib.util.module_from_spec(_s); _s.loader.exec_module(td)
t2 = td.t2
NORMS, EPS, APGD_NORM = t2.NORMS, t2.EPS, t2.APGD_NORM
from robustdro.attacks.apgd_train import apgd_train   # noqa: E402

PRIMARY = [m for m in td.MODELS if m[3]]              # trained heads only


def train_images(n, device):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=True,
                                      download=False, transform=T.Compose([T.ToTensor()]))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return torch.stack([ds[i][0] for i in idx]), torch.tensor([ds[i][1] for i in idx])


def dh(m, xs, ys, n_iter, bs, device):
    out = {}
    Zc, Za = [], {n: [] for n in NORMS}
    for i in range(0, len(xs), bs):
        x, y = xs[i:i + bs].to(device), ys[i:i + bs].to(device)
        with torch.no_grad():
            Zc.append(m.embed(x).float().cpu())
        for n in NORMS:
            xa = apgd_train(m, x, y, APGD_NORM[n], EPS[n], n_iter=n_iter, is_train=False).detach()
            with torch.no_grad():
                Za[n].append(m.embed(xa).float().cpu())
    Zc = torch.cat(Zc).numpy()
    for n in NORMS:
        out[n] = float(t2.dist(Zc, torch.cat(Za[n]).numpy(), "cos").mean())
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xs_te, ys_te, _ = t2.subset("1k", device)
    xs_tr, ys_tr = train_images(1000, device)
    T2B = json.load(open(os.path.join(t2.OUT, "head_space.json")))["h_vs_f"]
    R = {}
    for arm, role, ck, _ in PRIMARY:
        m, _ = td.load(ck, device)
        R[arm] = {"test_apgd10": dh(m, xs_te, ys_te, 10, 128, device),
                  "train_apgd10": dh(m, xs_tr, ys_tr, 10, 128, device),
                  "test_apgd50": {n: T2B[arm][n]["d_h"] for n in NORMS},
                  "f_test_apgd50": {n: T2B[arm][n]["d_f"] for n in NORMS}}
        del m; torch.cuda.empty_cache()
        print(f"  {arm} done")

    print("\n" + "=" * 100)
    print("T2e  HEAD-SPACE d(clean,adv) — where does the train↔eval gap come from? (post-hoc)")
    print("=" * 100)
    print(f"{'model':<10}{'norm':<6}{'TRAIN apgd10':>14}{'TEST apgd10':>13}{'TEST apgd50':>13}"
          f"{'  Δ(gen)':>10}{'  Δ(attack)':>12}")
    for arm, _, _, _ in PRIMARY:
        r = R[arm]
        for n in NORMS:
            gen = r["test_apgd10"][n] - r["train_apgd10"][n]
            atk = r["test_apgd50"][n] - r["test_apgd10"][n]
            print(f"{arm:<10}{n:<6}{r['train_apgd10'][n]:>14.4f}{r['test_apgd10'][n]:>13.4f}"
                  f"{r['test_apgd50'][n]:>13.4f}{gen:>+10.4f}{atk:>+12.4f}")
    print("\n  Δ(gen)    = test − train, both at APGD-10  -> generalisation gap of the alignment")
    print("  Δ(attack) = APGD-50 − APGD-10, both on test -> how much a stronger attack breaks it")
    print("  train-log reference: glue at last epoch -> cos 0.918, i.e. d_h_train ≈ 0.082 (M1a)")
    json.dump(R, open(os.path.join(t2.OUT, "head_attack_strength.json"), "w"), indent=2)
    print(f"\n  saved -> results/analysis/T2_geometry/head_attack_strength.json")


if __name__ == "__main__":
    main()

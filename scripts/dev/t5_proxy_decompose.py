"""T5 — decompose the val-proxy over-read into (a) val->test distribution, (b) attack strength.

The val-selection proxy is c5_fromscratch.worst_union_acc: APGD-CE, 3 norms (Linf/L2/L1),
n_iter=20, is_train=False, per-example AND. That exact function is applied here to BOTH:
  * the val split  train[49000:50000]  -> val_proxy   (recomputed, not read from a log)
  * the frozen test subset (n=1000)    -> test_proxy  (new number)
and compared with the strong audit already on disk (9 attacks x 100 steps, same test subset).

  val_proxy  - test_proxy   = distribution effect  (same weak attack, val vs test)
  test_proxy - test_strong  = attack-strength effect (same test data, weak vs strong)

val_proxy is recomputed rather than taken from train.json so all three arms come from one code
path (M1a_max was trained on Colab and has no local train.json); where a logged value exists it is
printed alongside as a self-check.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import importlib.util
_s = importlib.util.spec_from_file_location("t2", os.path.join(ROOT, "scripts/dev/t2_geometry.py"))
t2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t2)
from robustdro.attacks.apgd_train import apgd_train        # noqa: E402

EPS = {"Linf": 8/255, "L2": 0.5, "L1": 12.0}
NORMS = ["Linf", "L2", "L1"]
VAL_SELECT = (49000, 50000)

ARMS = [
    ("M1a",     "MSD + CLAMP (reference)", "results/fromscratch/C5/M1a_advneg/seed0/ckpt/val_best.pt",
     "results/fromscratch/C5/M1a_advneg/seed0/train.json"),
    ("M1a_max", "MAX + CLAMP",             "results/fromscratch/C5_ablations/M1a_max/ckpt/val_best.pt", None),
    ("M1a_avg", "AVG + CLAMP",             "results/fromscratch/C5_ablations/M1a_avg/ckpt/val_best.pt",
     "results/fromscratch/C5_ablations/M1a_avg/train.json"),
]


def worst_union(model, xs, ys, device, bs=250, n_iter=20):
    """Byte-for-byte the val-selection proxy (c5_fromscratch.worst_union_acc)."""
    model.eval()
    N = len(xs)
    robust = torch.ones(N, dtype=torch.bool, device=device)
    per = {}
    for nm in NORMS:
        ok = torch.zeros(N, dtype=torch.bool, device=device)
        for i in range(0, N, bs):
            xb, yb = xs[i:i+bs].to(device), ys[i:i+bs].to(device)
            with torch.enable_grad():
                xa = apgd_train(model, xb, yb, nm, EPS[nm], n_iter=n_iter, is_train=False)
            with torch.no_grad():
                ok[i:i+bs] = model(xa).argmax(1) == yb
        per[nm] = float(ok.float().mean())
        robust &= ok
    return float(robust.float().mean()), per


def val_split(device):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=True,
                                      download=False, transform=T.Compose([T.ToTensor()]))
    idx = list(range(*VAL_SELECT))
    return torch.stack([ds[i][0] for i in idx]), torch.tensor([ds[i][1] for i in idx])


def strong(arm):
    """9-attack (no-square) union on the frozen test subset, from masks already on disk."""
    for sub in ("1k_nosq", "", "1k"):
        p = os.path.join(ROOT, "results/eval/union_bench", arm, sub, "masks_multinorm_v1.npz")
        if os.path.exists(p):
            d = np.load(p)
            m = {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}
            ks = [k for k in m if "square" not in k]
            u = np.ones(len(m[ks[0]]), bool)
            for k in ks:
                u &= m[k]
            return float(u.mean()), len(ks), (sub or "root")
    return None, None, None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xs_te, ys_te, _ = t2.subset("1k", device)
    xs_va, ys_va = val_split(device)
    print(f"[T5] val split train{VAL_SELECT} n={len(xs_va)} | test subset n={len(xs_te)} | proxy=APGD-CE 20it AND-3norm\n")

    R = {}
    for arm, desc, ck, tj in ARMS:
        m = t2.load(os.path.join(ROOT, ck), device)
        vp, vper = worst_union(m, xs_va, ys_va, device)
        tp, tper = worst_union(m, xs_te, ys_te, device)
        ts, natk, src = strong(arm)
        logged = None
        if tj and os.path.exists(os.path.join(ROOT, tj)):
            logged = json.load(open(os.path.join(ROOT, tj))).get("best_val_worst_union")
        R[arm] = dict(desc=desc, val_proxy=vp, val_per=vper, test_proxy=tp, test_per=tper,
                      test_strong=ts, n_attacks=natk, strong_src=src, logged_val=logged)
        chk = ("  [logged %.4f  %s]" % (logged, "MATCH" if abs(logged - vp) < 5e-4 else "DIFFERS")) if logged else ""
        print(f"  {arm:9s} val_proxy={vp:.4f}{chk}  test_proxy={tp:.4f}  test_strong={ts:.4f} ({natk}-atk, {src})")
        del m; torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("T5  VAL-PROXY OVER-READ, DECOMPOSED")
    print("=" * 100)
    print(f"{'arm':<10}{'val_proxy':>11}{'test_proxy':>12}{'test_strong':>13}  |{'distribution':>14}{'attack':>10}{'total':>10}")
    print(f"{'':10}{'weak/VAL':>11}{'weak/TEST':>12}{'strong/TEST':>13}  |{'val-test':>14}{'weak-strong':>12}{'':>8}")
    print("-" * 100)
    for arm, _, _, _ in ARMS:
        r = R[arm]
        d, a_, t = r["val_proxy"] - r["test_proxy"], r["test_proxy"] - r["test_strong"], r["val_proxy"] - r["test_strong"]
        print(f"{arm:<10}{r['val_proxy']:>11.4f}{r['test_proxy']:>12.4f}{r['test_strong']:>13.4f}  |"
              f"{d:>+14.4f}{a_:>10.4f}{t:>+10.4f}")
    print("-" * 100)
    print("  distribution = val_proxy - test_proxy   (same weak attack; val is train-held-out, test is unseen)")
    print("  attack       = test_proxy - test_strong (same test examples; 20-step x3 vs 100-step x9)")
    print("\nper-norm under the weak proxy (TEST):")
    for arm, _, _, _ in ARMS:
        p = R[arm]["test_per"]
        q = R[arm]["val_per"]
        print(f"  {arm:9s} TEST  " + "  ".join(f"{n} {p[n]:.4f}" for n in NORMS)
              + "   |  VAL  " + "  ".join(f"{n} {q[n]:.4f}" for n in NORMS))
    out = os.path.join(ROOT, "results/analysis/T5_proxy_decompose")
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, "proxy_decompose.json"), "w"), indent=2)
    print(f"\n  saved -> results/analysis/T5_proxy_decompose/proxy_decompose.json")


if __name__ == "__main__":
    main()

"""T2 — representation geometry of CLAMP, measured in ENCODER f-space.

Pre-registered in docs/preregistrations/preregistration_T2_representation_geometry.md.
Read that FIRST — hypotheses and decision rules are fixed there and are not revised here.

f-space = pooled 512-d encoder output, BEFORE the classifier g (`self.linear`) and BEFORE the
SimCLR projection head h. CLAMP's glue/scaffold act on normalize(h(f(x))), so measuring in h would
be circular; f asks whether the effect propagates into the shared encoder.

For each model, three white-box APGD-CE views are crafted against THAT model at the standard triple,
then embedded. Distances are paired per example across models (identical frozen audit subset).

  python scripts/dev/t2_geometry.py [--n-iter 50] [--bs 128] [--scale 1k|10k]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from robustdro.models import build_model                     # noqa: E402
from robustdro.attacks.apgd_train import apgd_train          # noqa: E402

OUT = os.path.join(ROOT, "results/analysis/T2_geometry")
SUBSETS = {"1k": "results/audit/subsets/cifar10_testfinal_1000_seed20260709_v3A.json",
           "10k": "results/audit/subsets/cifar10_test_10000_full_v3A.json"}
EPS = {"linf": 8 / 255, "l2": 0.5, "l1": 12.0}
APGD_NORM = {"linf": "Linf", "l2": "L2", "l1": "L1"}
NORMS = ("linf", "l2", "l1")
VIEWS = ("clean",) + NORMS

MODELS = [
    ("M0",       "weak-control  MSD-10 / 80ep / term off", "results/external_ckpt/M0_seed0.pt"),
    ("M1a",      "weak-term     MSD-10 / 80ep / CLAMP",    "results/fromscratch/C5/M1a_advneg/seed0/ckpt/val_best.pt"),
    ("M0_full",  "strong-control MSD-50 / 50ep / term off", "results/fromscratch/C5_full/M0_full/val_best.pt"),
    ("M1a_full", "strong-term   MSD-50 / 50ep / CLAMP",    "results/fromscratch/C5_full/M1a_full/val_best.pt"),
]
PAIRS = [("M1a", "M0", "WEAK  base (MSD-10)"), ("M1a_full", "M0_full", "STRONG base (MSD-50)")]

# per-norm accuracy deltas @10k full tier, from results/main (used for the correlation panel only)
ACC_DELTA = {"weak": {"linf": +0.0229, "l2": +0.0015, "l1": +0.0481},
             "strong": {"linf": -0.0077, "l2": +0.0075, "l1": +0.0008}}


# ----------------------------------------------------------------------------- model
class Enc(torch.nn.Module):
    """Frozen backbone exposing f (pooled 512-d, pre-g, pre-h) and logits. No projection head."""

    def __init__(self, backbone):
        super().__init__()
        self.b = backbone

    def features(self, x):
        b = self.b
        out = b.conv1(x)
        for L in (b.layer1, b.layer2, b.layer3, b.layer4):
            out = L(out)
        out = F.relu(b.bn(out))                       # robustdro has a final BN+ReLU
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)              # (B, 512)

    def forward(self, x):
        return self.b.linear(self.features(x))        # logits — what APGD attacks


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    m = build_model({"model": {"arch": "preact_resnet18", "normalize": False}})
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "num_batches" not in k], f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return Enc(m).to(device).eval()


def subset(scale, device):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=False,
                                      download=False, transform=T.Compose([T.ToTensor()]))
    idx = json.load(open(os.path.join(ROOT, SUBSETS[scale])))["indices"]
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx])
    return xs, ys, [int(i) for i in idx]


# ----------------------------------------------------------------------------- geometry
def embed_all(model, xs, ys, n_iter, bs, device):
    """Return {view: (N,512) float32 embeddings} and the per-view correctness mask."""
    Z = {v: [] for v in VIEWS}
    correct = {v: [] for v in VIEWS}
    for i in range(0, len(xs), bs):
        x, y = xs[i:i + bs].to(device), ys[i:i + bs].to(device)
        batch = {"clean": x}
        for nm in NORMS:                              # views crafted against THIS model
            batch[nm] = apgd_train(model, x, y, APGD_NORM[nm], EPS[nm],
                                   n_iter=n_iter, is_train=False).detach()
        with torch.no_grad():
            for v in VIEWS:
                Z[v].append(model.features(batch[v]).float().cpu())
                correct[v].append((model(batch[v]).argmax(1) == y).cpu())
        print(f"    [{i + len(x):>5}/{len(xs)}]", end="\r", flush=True)
    print(" " * 30, end="\r")
    return ({v: torch.cat(Z[v]).numpy() for v in VIEWS},
            {v: torch.cat(correct[v]).numpy() for v in VIEWS})


def dist(a, b, metric):
    if metric == "cos":
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return 1.0 - (an * bn).sum(1)
    return np.linalg.norm(a - b, axis=1)


def geometry(Z, keep=None):
    """Per-example distances -> means. keep = optional boolean subgroup mask."""
    sel = slice(None) if keep is None else keep
    g = {"n": int(len(Z["clean"]) if keep is None else keep.sum()), "cos": {}, "l2": {}, "fnorm": {}}
    for v in VIEWS:
        g["fnorm"][v] = float(np.linalg.norm(Z[v][sel], axis=1).mean())
    for metric in ("cos", "l2"):
        for i, a in enumerate(VIEWS):
            for b in VIEWS[i + 1:]:
                g[metric][f"{a}-{b}"] = float(dist(Z[a][sel], Z[b][sel], metric).mean())
    return g


def per_example(Z, metric="cos"):
    return {nm: dist(Z["clean"], Z[nm], metric) for nm in NORMS}


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--scale", default="1k", choices=["1k", "10k"])
    p.add_argument("--seed", type=int, default=20260709)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    xs, ys, idx = subset(a.scale, device)
    print(f"[T2] subset={a.scale} n={len(xs)} device={device} apgd_iter={a.n_iter}")
    print(f"[T2] f-space = pooled 512-d, pre-g (linear) and pre-h (projection head)\n")

    R, PE, ACC = {}, {}, {}
    for arm, desc, ck in MODELS:
        ckp = os.path.join(ROOT, ck)
        assert os.path.exists(ckp), f"ckpt missing: {ckp}"
        sha = hashlib.sha256(open(ckp, "rb").read()).hexdigest()
        print(f"[T2] {arm:9s} {desc}\n     sha={sha[:16]}  crafting {len(NORMS)} views ...")
        m = load(ckp, device)
        Z, corr = embed_all(m, xs, ys, a.n_iter, a.bs, device)
        uni = corr["clean"] & corr["linf"] & corr["l2"] & corr["l1"]   # survives all 3 white-box views
        R[arm] = {"desc": desc, "ckpt": ck, "ckpt_sha256": sha,
                  "all": geometry(Z), "survivors": geometry(Z, uni), "failures": geometry(Z, ~uni),
                  "acc": {v: float(corr[v].mean()) for v in VIEWS},
                  "apgd3_union": float(uni.mean())}
        PE[arm] = per_example(Z)
        ACC[arm] = corr
        np.savez_compressed(os.path.join(OUT, f"emb_{arm}.npz"),
                            **{v: Z[v].astype(np.float16) for v in VIEWS},
                            **{f"correct_{v}": corr[v] for v in VIEWS})
        del m, Z
        torch.cuda.empty_cache()
        c = R[arm]["all"]["cos"]
        print(f"     d_cos(clean,·): ℓ∞ {c['clean-linf']:.4f}  ℓ2 {c['clean-l2']:.4f}  ℓ1 {c['clean-l1']:.4f}"
              f"   |  APGD-3 union {R[arm]['apgd3_union']:.4f}\n")

    # ---- relative movement + paired CI over examples ----
    rng = np.random.default_rng(0)
    rel = {}
    for A, B, lbl in PAIRS:
        e = {}
        for nm in NORMS:
            d = PE[A][nm] - PE[B][nm]                       # paired per example
            n = len(d)
            boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
            e[nm] = {"delta_cos": float(d.mean()),
                     "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                     "delta_l2": float((dist(np.load(f"{OUT}/emb_{A}.npz")["clean"].astype(np.float32),
                                             np.load(f"{OUT}/emb_{A}.npz")[nm].astype(np.float32), "l2")
                                        - dist(np.load(f"{OUT}/emb_{B}.npz")["clean"].astype(np.float32),
                                               np.load(f"{OUT}/emb_{B}.npz")[nm].astype(np.float32), "l2")).mean())}
        rel[lbl] = {"term_on": A, "term_off": B, "per_norm": e}

    payload = {"config": {"scale": a.scale, "n": len(xs), "apgd_n_iter": a.n_iter,
                          "eps": EPS, "space": "pooled 512-d encoder f, pre-g, pre-h",
                          "subset": SUBSETS[a.scale], "bootstrap": "B=10000 rng(0), 2-sided 95%"},
               "models": R, "relative_movement": rel, "acc_delta_10k_full": ACC_DELTA}
    json.dump(payload, open(os.path.join(OUT, "geometry.json"), "w"), indent=2)

    # ---- report ----
    def row(arm, key="all", metric="cos"):
        g = R[arm][key][metric]
        return (f"{g['clean-linf']:.4f} {g['clean-l2']:.4f} {g['clean-l1']:.4f} | "
                f"{g['linf-l2']:.4f} {g['linf-l1']:.4f} {g['l2-l1']:.4f}")

    print("=" * 100)
    print("T2.1  DISTANCE-TO-CLEAN and ADV-ADV, cosine distance (1-cos), f-space, all examples")
    print("=" * 100)
    print(f"{'model':<10} {'d(cl,ℓ∞)':>9}{'d(cl,ℓ2)':>9}{'d(cl,ℓ1)':>9} | {'ℓ∞-ℓ2':>7}{'ℓ∞-ℓ1':>7}{'ℓ2-ℓ1':>7} | {'‖z_cl‖':>7}")
    for arm, _, _ in MODELS:
        g = R[arm]["all"]
        print(f"{arm:<10} {g['cos']['clean-linf']:>9.4f}{g['cos']['clean-l2']:>9.4f}{g['cos']['clean-l1']:>9.4f} | "
              f"{g['cos']['linf-l2']:>7.4f}{g['cos']['linf-l1']:>7.4f}{g['cos']['l2-l1']:>7.4f} | "
              f"{g['fnorm']['clean']:>7.2f}")

    print("\n" + "=" * 100)
    print("T2.2  RELATIVE MOVEMENT  Δd = d(CLAMP) − d(control)      NEGATIVE = term pulls adv CLOSER to clean")
    print("=" * 100)
    for lbl, r in rel.items():
        print(f"  {lbl}   [{r['term_on']} − {r['term_off']}]")
        for nm in NORMS:
            e = r["per_norm"][nm]
            sig = "" if e["ci95"][0] <= 0 <= e["ci95"][1] else "  *"
            print(f"    ℓ{nm[1:]:<4} Δd_cos={e['delta_cos']:+.5f}  95%[{e['ci95'][0]:+.5f},{e['ci95'][1]:+.5f}]"
                  f"   Δd_l2={e['delta_l2']:+.4f}{sig}")

    print("\n" + "=" * 100)
    print("T2.3  REDUNDANCY TEST — does the STRONG base already have the geometry?")
    print("=" * 100)
    print(f"  {'norm':<6}{'d10 (M0)':>10}{'d50 (M0_full)':>15}{'d50−d10':>10}  |  "
          f"{'Δd weak':>9}{'Δd strong':>11}{'  saturation':>13}")
    for nm in NORMS:
        d10 = R["M0"]["all"]["cos"][f"clean-{nm}"]
        d50 = R["M0_full"]["all"]["cos"][f"clean-{nm}"]
        dw = rel["WEAK  base (MSD-10)"]["per_norm"][nm]["delta_cos"]
        dsn = rel["STRONG base (MSD-50)"]["per_norm"][nm]["delta_cos"]
        sat = "n/a" if abs(dw) < 1e-9 else f"{100*(1 - abs(dsn)/abs(dw)):+.0f}%"
        print(f"  {nm:<6}{d10:>10.4f}{d50:>15.4f}{d50-d10:>+10.4f}  |  {dw:>+9.5f}{dsn:>+11.5f}{sat:>13}")
    print("  (a) d50 < d10  -> strong base already closer.   (b) |Δd strong| << |Δd weak| -> term saturated.")
    print("  H-redundancy needs BOTH.")

    print("\n" + "=" * 100)
    print("T2.4  CORRELATION PANEL — per-norm distance reduction vs per-norm accuracy gain @10k full")
    print("=" * 100)
    for key, lbl in (("weak", "WEAK  base (MSD-10)"), ("strong", "STRONG base (MSD-50)")):
        xs_ = [-rel[lbl]["per_norm"][nm]["delta_cos"] for nm in NORMS]      # +ve = pulled closer
        ys_ = [ACC_DELTA[key][nm] for nm in NORMS]
        r = float(np.corrcoef(xs_, ys_)[0, 1]) if np.std(xs_) > 0 and np.std(ys_) > 0 else float("nan")
        print(f"  {lbl}:")
        for nm, xv, yv in zip(NORMS, xs_, ys_):
            print(f"    ℓ{nm[1:]:<4} pull-closer={xv:+.5f}   Δacc@10k={yv:+.4f}")
        print(f"    Pearson r = {r:+.3f}   (3 points — DESCRIPTIVE ONLY, no p-value is claimed)")

    print(f"\n  saved -> {os.path.relpath(os.path.join(OUT, 'geometry.json'), ROOT)}")
    print(f"  embeddings (fp16) -> {os.path.relpath(OUT, ROOT)}/emb_<arm>.npz")


if __name__ == "__main__":
    main()

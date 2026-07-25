"""Clean-margin vs blocker-count analysis (masks + 1 clean forward pass).

margin_i = logit_y(x_i) − max_{k≠y} logit_k(x_i)   on the CLEAN audit inputs (no attack).
Joined per-example with |B_i| = number of norms that block example i (from the audit masks:
example fails norm n ⇔ any of that norm's attacks succeeds; B_i counts failing norms, 0..3).

Question: does the clean margin predict multi-norm blocking? If AUC(margin → |B_i|≥2) is high
(≳0.85), the union frontier is largely explained by intrinsic example DIFFICULTY (low clean margin),
which motivates a margin-aware union objective rather than per-norm geometry.

Read-only w.r.t. masks/checkpoints; writes only to --out. One forward pass per model.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

NORMS = ["linf", "l2", "l1"]
ATTACKS = {n: [f"apgd_ce_{n}", f"apgd_dlr_{n}", f"fab_t_{n}", f"square_{n}"] for n in NORMS}


def build_model(arch, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if arch == "robustdro":
        from robustdro.models import build_model as bm
        m = bm(ck["cfg"]); m.load_state_dict(ck["model"])
    elif arch == "robust_union_preact":
        sys.path.insert(0, str(ROOT / "external/robust_union/CIFAR10/models"))
        from preact_resnet import PreActResNet18
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        m = PreActResNet18(); m.load_state_dict(sd, strict=True)
    elif arch == "ramp":
        sys.path.insert(0, str(ROOT / "external/RAMP"))
        from model_zoo.fast_models import PreActResNet18
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        m = PreActResNet18(10); m.load_state_dict(sd, strict=True)
    else:
        raise ValueError(arch)
    return m.to(device).eval()


def load_subset_images(subset_path):
    import torchvision, torchvision.transforms as T
    d = json.load(open(subset_path))
    idx = d["indices"]
    is_train = d.get("split", "test") == "train"
    ds = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=is_train, download=False,
                                      transform=T.ToTensor())
    X = torch.stack([ds[i][0] for i in idx])
    Y = torch.tensor([ds[i][1] for i in idx])
    return X, Y, d


@torch.no_grad()
def clean_margin(model, X, Y, device, bs=500):
    out = []
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs].to(device))
        yb = Y[i:i + bs].to(device)
        ly = logits.gather(1, yb[:, None]).squeeze(1)
        other = logits.clone().scatter_(1, yb[:, None], float("-inf"))
        out.append((ly - other.max(1).values).cpu())
    return torch.cat(out).numpy()


def block_count(masks_path):
    d = np.load(masks_path, allow_pickle=True)
    masks = {k: np.asarray(d[k]).astype(bool) for k in d.files if k != "metadata_json"}
    per = {n: np.logical_and.reduce([masks[a] for a in ATTACKS[n] if a in masks]) for n in NORMS}
    fail = {n: ~per[n] for n in NORMS}
    B = sum(fail[n].astype(int) for n in NORMS)
    return B, fail, len(B)


def auc(score, label):
    """AUC that higher `score` predicts positive `label` (bool). Tie-safe via average ranks."""
    from scipy.stats import rankdata
    label = np.asarray(label, dtype=bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def quantiles(v):
    return {str(q): float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(v) else {}


def analyse(arm, arch, ckpt, masks_path, subset_path, device):
    X, Y, subset = load_subset_images(subset_path)
    model = build_model(arch, ckpt, device)
    margin = clean_margin(model, X, Y, device)
    B, fail, n = block_count(masks_path)
    assert len(margin) == n == len(X), f"length mismatch: margin {len(margin)} masks {n} imgs {len(X)}"
    clean_correct = margin > 0
    res = {"arm": arm, "arch": arch, "n": n, "checkpoint": str(ckpt),
           "clean_acc": float(clean_correct.mean()),
           "block_dist": {str(k): int((B == k).sum()) for k in (0, 1, 2, 3)},
           "margin_by_block": {}, "auc": {}}
    for k in (0, 1, 2, 3):
        m = margin[B == k]
        res["margin_by_block"][str(k)] = {"n": int(len(m)), "mean": float(m.mean()) if len(m) else None,
                                          "median": float(np.median(m)) if len(m) else None,
                                          "std": float(m.std()) if len(m) else None, "quantiles": quantiles(m)}
    # AUC: LOW margin predicts blocking → score = −margin. Report several targets.
    res["auc"]["any_blocked(|B|>=1)"] = auc(-margin, B >= 1)
    res["auc"]["multi_blocked(|B|>=2)"] = auc(-margin, B >= 2)
    res["auc"]["all3_blocked(|B|==3)"] = auc(-margin, B == 3)
    # multi vs single WITHIN the failing set (is multi-blocked harder than single-blocked, among fails?)
    failing = B >= 1
    res["auc"]["multi_vs_single|failing"] = auc(-margin[failing], (B[failing] >= 2))
    return res, margin, B


def print_report(all_res, threshold):
    P = print
    P("\n" + "=" * 74)
    P("CLEAN MARGIN vs BLOCKER COUNT  (margin = logit_y − max_{k≠y} logit_k, clean)")
    P("=" * 74)
    for r in all_res:
        P(f"\n### {r['arm']} [{r['arch']}]  n={r['n']}  clean_acc={r['clean_acc']:.4f}")
        bd = r["block_dist"]
        tot = r["n"]
        P(f"  |B| distribution:  0(robust) {bd['0']} ({bd['0']/tot*100:.1f}%)  1 {bd['1']} ({bd['1']/tot*100:.1f}%)"
          f"  2 {bd['2']} ({bd['2']/tot*100:.1f}%)  3 {bd['3']} ({bd['3']/tot*100:.1f}%)")
        P(f"  {'|B|':<5}{'n':>7}{'mean':>9}{'median':>9}{'p10':>9}{'p90':>9}")
        for k in ("0", "1", "2", "3"):
            g = r["margin_by_block"][k]
            if g["n"]:
                qs = g["quantiles"]
                P(f"  {k:<5}{g['n']:>7}{g['mean']:>9.3f}{g['median']:>9.3f}{qs['0.1']:>9.3f}{qs['0.9']:>9.3f}")
        a = r["auc"]
        P("  AUC (low clean margin → blocked):")
        P(f"    any-blocked |B|≥1     : {a['any_blocked(|B|>=1)']:.4f}")
        P(f"    MULTI-blocked |B|≥2   : {a['multi_blocked(|B|>=2)']:.4f}   <-- headline")
        P(f"    all-3 |B|=3           : {a['all3_blocked(|B|==3)']:.4f}")
        P(f"    multi-vs-single|fail  : {a['multi_vs_single|failing']:.4f}")
        verdict = ("FRONTIER≈DIFFICULTY (margin-aware union motivated)"
                   if a['multi_blocked(|B|>=2)'] >= threshold else "margin only partly explains blocking")
        P(f"  >>> multi-blocked AUC {a['multi_blocked(|B|>=2)']:.3f} vs {threshold} → {verdict}")
    P("=" * 74 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="results/eval/union_bench")
    ap.add_argument("--subset", default="results/audit/subsets/cifar10_test_10000_full_v3A.json")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--date", default=None)
    ap.add_argument("--auc-threshold", type=float, default=0.85)
    # one or more arms as arm:arch:ckpt (masks taken from <base>/<arm>/10k or <base>/<arm>)
    ap.add_argument("--arm", action="append", default=[], metavar="ARM:ARCH:CKPT",
                    help="e.g. M0_full:robustdro:results/fromscratch/C5_full/M0_full/val_best.pt")
    a = ap.parse_args()
    import datetime
    date = a.date or datetime.date.today().isoformat()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path(a.base_dir)

    all_res, missing = [], []
    for spec in a.arm:
        arm, arch, ckpt = spec.split(":", 2)
        mpath = None
        for rel in (f"{arm}/10k/masks_multinorm_v1.npz", f"{arm}/masks_multinorm_v1.npz"):
            if (base / rel).exists():
                mpath = base / rel; break
        if mpath is None:
            missing.append(f"{arm} (masks not found under {base})"); continue
        if not Path(ckpt).exists():
            missing.append(f"{arm} (checkpoint not found: {ckpt})"); continue
        print(f"[margin] {arm}: model={arch} ckpt={ckpt}", flush=True)
        res, _, _ = analyse(arm, arch, ckpt, mpath, a.subset, device)
        all_res.append(res)

    print_report(all_res, a.auc_threshold)
    for m in missing:
        print(f"  SKIPPED: {m}")
    out = {"date": date, "subset": a.subset, "auc_threshold": a.auc_threshold,
           "margin_def": "logit_y - max_{k!=y} logit_k on clean inputs",
           "results": all_res, "missing": missing}
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    jp = od / f"margin_analysis_{date}.json"
    jp.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved: {jp}")


if __name__ == "__main__":
    main()

"""Per-class x norm breakdown between two audited arms, from existing 12-AA mask sidecars.

CPU-only: reads `masks_multinorm_v1.npz` (12 per-attack bool masks, convention
true_means_robust_false_means_failed) for arm A and arm B, maps mask position -> CIFAR-10 test
label (subset indices verified identity for the 10k full-test subset), and reports per class:
  * union accuracy  = per-example AND over all 12 components
  * per-norm robust = per-example AND over that norm's 4 attacks (linf/l2/l1)
  * Delta_c = A_c - B_c, with a per-class paired bootstrap CI (resample within class)
Plus: sign test across the 10 classes (union + each norm), and "bottom-3 lift" = mean Delta over
the 3 classes where arm B is weakest on union.

No clean column: the mask sidecar stores only the 12 attack masks (no clean mask), and we do NOT
run new inference. Emits a long-form CSV + a markdown delta table.

Usage:
  python scripts/dev/perclass_breakdown.py \
      --a results/eval/union_bench/M1a/10k --a-name M1a \
      --b results/eval/union_bench/M0/10k  --b-name M0 \
      --outdir results/eval/union_bench/perclass
"""
import argparse, json, os
from math import comb
from pathlib import Path

import numpy as np

NORMS = ["linf", "l2", "l1"]
ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))


def load_masks(d: Path):
    p = d / "masks_multinorm_v1.npz"
    if not p.exists():
        raise FileNotFoundError(f"masks not found: {p}")
    z = np.load(p)
    m = {k: z[k].astype(bool) for k in z.files if k != "metadata_json"}
    meta = {}
    if "metadata_json" in z.files:
        raw = z["metadata_json"]
        meta = json.loads(raw.item() if raw.shape == () else str(raw))
    if meta.get("mask_convention") not in (None, "true_means_robust_false_means_failed"):
        raise ValueError(f"unexpected mask_convention: {meta.get('mask_convention')}")
    return m, meta


def union_all(m):
    u = np.ones(len(next(iter(m.values()))), bool)
    for v in m.values():
        u &= v
    return u


def per_norm(m, norm):
    ks = [k for k in m if k.endswith(norm)]          # 'linf' never matches '...l1' (suffix-safe)
    if len(ks) != 4:
        raise ValueError(f"expected 4 attacks for {norm}, got {ks}")
    u = np.ones(len(m[ks[0]]), bool)
    for k in ks:
        u &= m[k]
    return u


def labels_for(meta, n):
    """mask position -> CIFAR-10 test label, via the audit subset indices (verified)."""
    import torchvision
    sub = ROOT / meta.get("audit_subset_path", "results/audit/subsets/cifar10_test_10000_full_v3A.json")
    idx = json.load(open(sub))["indices"]
    if len(idx) != n:
        raise ValueError(f"subset n={len(idx)} != masks n={n}")
    ds = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=False)
    y = np.array(ds.targets)[np.array(idx)]
    return y, ds.classes


def boot_ci(a, b, B=2000, seed=0):
    """paired bootstrap on (a-b) within one class; returns (delta, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(B)
    for i in range(B):
        j = rng.integers(0, n, n)
        d[i] = a[j].mean() - b[j].mean()
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def sign_test(deltas):
    """exact two-sided binomial sign test on the signs of per-class deltas (ties dropped)."""
    pos = int(sum(1 for d in deltas if d > 0))
    neg = int(sum(1 for d in deltas if d < 0))
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    def cdf_le(k):
        return sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p = 2 * min(cdf_le(pos), 1 - cdf_le(pos - 1) if pos > 0 else 1.0)
    return pos, neg, float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a-name", required=True)
    ap.add_argument("--b", required=True); ap.add_argument("--b-name", required=True)
    ap.add_argument("--outdir", default="results/eval/union_bench/perclass")
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A, metaA = load_masks(ROOT / a.a)
    B, metaB = load_masks(ROOT / a.b)
    n = len(next(iter(A.values())))
    if len(next(iter(B.values()))) != n:
        raise ValueError("arm A/B mask lengths differ")
    if metaA.get("audit_subset_sha256") != metaB.get("audit_subset_sha256"):
        raise ValueError("arms audited on DIFFERENT subsets — refusing to pair")
    y, classes = labels_for(metaA, n)

    uA, uB = union_all(A), union_all(B)
    nA = {nm: per_norm(A, nm) for nm in NORMS}
    nB = {nm: per_norm(B, nm) for nm in NORMS}

    rows, md = [], []
    per_class = {}
    for c in range(10):
        s = (y == c)
        dU, lo, hi = boot_ci(uA[s], uB[s], a.boot, a.seed)
        rec = {"class": classes[c], "n": int(s.sum()),
               f"{a.a_name}_union": float(uA[s].mean()), f"{a.b_name}_union": float(uB[s].mean()),
               "delta_union": dU, "ci_lo": lo, "ci_hi": hi}
        for nm in NORMS:
            rec[f"delta_{nm}"] = float(nA[nm][s].mean() - nB[nm][s].mean())
            rec[f"{a.a_name}_{nm}"] = float(nA[nm][s].mean())
            rec[f"{a.b_name}_{nm}"] = float(nB[nm][s].mean())
        per_class[classes[c]] = rec
        for arm, U, N in [(a.a_name, uA, nA), (a.b_name, uB, nB)]:
            rows.append({"class": classes[c], "arm": arm, "union": float(U[s].mean()),
                         **{nm: float(N[nm][s].mean()) for nm in NORMS}})
        md.append(f"| {classes[c]:10s} | {uB[s].mean():.3f} | {uA[s].mean():.3f} | "
                  f"{dU:+.3f} [{lo:+.3f},{hi:+.3f}] | {rec['delta_linf']:+.3f} | "
                  f"{rec['delta_l2']:+.3f} | {rec['delta_l1']:+.3f} |")

    out = ROOT / a.outdir; out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / f"perclass_{a.a_name.lower()}_{a.b_name.lower()}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "arm", "union"] + NORMS); w.writeheader(); w.writerows(rows)

    dU_all = [per_class[c]["delta_union"] for c in per_class]
    pos, neg, p_union = sign_test(dU_all)
    sign_norms = {nm: sign_test([per_class[c][f"delta_{nm}"] for c in per_class]) for nm in NORMS}
    weakest = sorted(per_class, key=lambda c: per_class[c][f"{a.b_name}_union"])[:3]
    bottom3 = float(np.mean([per_class[c]["delta_union"] for c in weakest]))
    cmax = max(per_class, key=lambda c: per_class[c]["delta_union"])
    cmin = min(per_class, key=lambda c: per_class[c]["delta_union"])

    hdr = (f"| class | {a.b_name} union | {a.a_name} union | Δ (95% CI) | Δℓ∞ | Δℓ₂ | Δℓ₁ |\n"
           f"|---|---|---|---|---|---|---|")
    summary = [
        f"- **Sign test (union, n=10 classes):** {pos} positive / {neg} negative → exact two-sided **p = {p_union:.4f}**",
        "- **Sign test per norm:** " + " · ".join(
            f"ℓ{'∞' if nm=='linf' else nm[-1]} {sign_norms[nm][0]}+/{sign_norms[nm][1]}− (p={sign_norms[nm][2]:.4f})" for nm in NORMS),
        f"- **Bottom-3 lift** (3 classes where {a.b_name} is weakest on union: {', '.join(weakest)}): **Δ = {bottom3:+.4f}**",
        f"- **Max Δ:** {cmax} ({per_class[cmax]['delta_union']:+.4f})  ·  **Min Δ:** {cmin} ({per_class[cmin]['delta_union']:+.4f})",
    ]
    body = (f"# Per-class × norm — {a.a_name} vs {a.b_name} (10k, frozen 12-AA)\n\n"
            f"Union = per-example AND over 12 components; per-norm = AND over that norm's 4 attacks. "
            f"Per-class paired bootstrap B={a.boot}, seed={a.seed}, resampled within class (n=1000/class). "
            f"No clean column (mask sidecar stores no clean mask; no new inference run).\n\n"
            + hdr + "\n" + "\n".join(md) + "\n\n" + "\n".join(summary) + "\n")
    (out / f"perclass_delta.md").write_text(body)

    print(hdr); print("\n".join(md)); print(); print("\n".join(summary))
    print(f"\nsaved -> {out}/perclass_{a.a_name.lower()}_{a.b_name.lower()}.csv")
    print(f"saved -> {out}/perclass_delta.md")


if __name__ == "__main__":
    main()

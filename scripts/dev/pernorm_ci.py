"""Per-norm paired-bootstrap CIs (L13b). Same machinery as the union CIs: per-example masks,
B=10000, np.random.default_rng(0), percentile CI, LCB = q05.

A per-norm block = AND over that norm's attacks in the tier (4 at tier=full). The bootstrap is
paired: one resampled index vector indexes BOTH arms, so it is the same estimator the union deltas
use, applied to a different mask.

Multi-seed: there is NO pooled multi-seed bootstrap in this repo (dossier §7). This script reports
per-seed CIs, the mean of the per-seed point estimates, and the seed-MINIMUM LCB — the same
convention the union claim already uses. It does not invent a pooled interval.
"""
from __future__ import annotations
import numpy as np, os, sys, json
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
MAIN = os.path.join(ROOT, "results/main")
NORMS = ("linf", "l2", "l1")
B, SEED = 10000, 0


def masks(arm, scale="10k"):
    p = os.path.join(MAIN, arm, scale, "masks_multinorm_v1.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}


def block(m, nm):
    ks = [k for k in m if k.endswith(nm)]
    u = np.ones(len(m[ks[0]]), bool)
    for k in ks:
        u &= m[k]
    return u


def boot(a, b):
    assert len(a) == len(b)
    rng = np.random.default_rng(SEED)
    n = len(a)
    D = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        D[i] = a[idx].mean() - b[idx].mean()
    return float(D.mean()), float(np.quantile(D, .05)), float(np.quantile(D, .95))


def row(A, Bm, lbl, out):
    ma, mb = masks(A), masks(Bm)
    if ma is None or mb is None:
        print(f"  {lbl:<30} MISSING masks ({A if ma is None else Bm})")
        return
    cells = []
    for nm in NORMS:
        d, lo, hi = boot(block(ma, nm), block(mb, nm))
        sig = "+" if lo > 0 else ("−" if hi < 0 else "ns")
        cells.append((nm, d, lo, hi, sig))
    out[lbl] = {nm: dict(delta=d, lo=lo, hi=hi, sig=s) for nm, d, lo, hi, s in cells}
    txt = "  ".join(f"{nm}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] {s:>2}" for nm, d, lo, hi, s in cells)
    print(f"  {lbl:<30} {txt}")
    return cells


ROWS = [
    ("M1a", "M0", "RQ1/MSD-10 seed0"),
    ("M1a_seed2", "M0_seed2", "MSD-10 seed2"),
    ("M1a_seed3", "M0_seed3", "MSD-10 seed3"),
    ("M1a_avg", "M0_avg", "AVG"),
    ("M1a_max", "M0_max", "MAX"),
    ("M1a_full", "M0_full", "MSD-50 (views-10) s0"),
    ("M1a_full_seed1", "M0_full_seed1", "MSD-50 (views-10) s1"),
    ("B1", "Rprime", "RAMP (B1-Rprime)"),
    ("ft_clamp", "ft_none", "FT-linf"),
    ("M1a_full_v50", "M0_full", "F2 v50 - MSD-50"),
    ("M1a_full_v50", "M1a_full", "F2 v50 - views-10"),
]

out = {}
print(f"per-norm paired bootstrap  B={B}  rng({SEED})  percentile CI  LCB=q05\n")
for A, Bm, lbl in ROWS:
    row(A, Bm, lbl, out)

print("\n--- MSD-10 multi-seed (no pooled bootstrap exists; per-seed + seed-min LCB) ---")
for nm in NORMS:
    ds = [out[k][nm]["delta"] for k in ("RQ1/MSD-10 seed0", "MSD-10 seed2", "MSD-10 seed3")]
    ls = [out[k][nm]["lo"] for k in ("RQ1/MSD-10 seed0", "MSD-10 seed2", "MSD-10 seed3")]
    print(f"  {nm:<5} per-seed D = {', '.join(f'{d:+.4f}' for d in ds)}   mean {np.mean(ds):+.4f}"
          f"   seed-min LCB {min(ls):+.4f}   {'all-seed +' if min(ls) > 0 else 'NOT all-seed +'}")

os.makedirs(os.path.join(ROOT, "results/analysis/pernorm_ci"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/analysis/pernorm_ci/pernorm_ci.json"), "w"), indent=2)
print(f"\n  saved -> results/analysis/pernorm_ci/pernorm_ci.json")

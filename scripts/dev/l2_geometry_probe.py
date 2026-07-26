"""L2-geometry probe (Option B) — why ℓ₂ is never the sole blocker. CPU only, no GPU, no attacks.

Two independent pieces, deliberately kept separate:

  TASK 1 (pure geometry, no data): three inscribed-ball radii in R^d for centred, UNCLIPPED balls.
    r_box1  = largest ℓ₂ ball inside B_1(ε₁)                  = ε₁/√d
    r_union = largest ℓ₂ ball inside B_1(ε₁) ∪ B_∞(ε_∞)       = min_u max(ε₁/‖u‖₁, ε_∞/‖u‖_∞)
    r_hull  = largest ℓ₂ ball inside conv(B_1 ∪ B_∞)          = min_u max(ε₁‖u‖_∞, ε_∞‖u‖₁)
  Both radial functions are minimised over unit-ℓ₂ directions u. For the union the radial function
  of a ray is the max of the two exit distances (both balls are convex and contain the origin, so
  each ray meets each ball in an interval [0, ·]). For the hull it is the support function, whose
  dual-norm form gives the ε‖u‖ expressions above.

  TASK 2 (data): per-example blocking anatomy over union-FAILING examples, from the frozen @10k
    12-component masks. A norm "blocks" an example when the AND over that norm's 4 components is
    False. No attack is re-run; this only re-reads masks.

  TASK 3: joins them — ε₂ = 0.5 sits ABOVE r_union (so the raw union does not contain it) yet ℓ₂ is
    never the sole blocker empirically. Guaranteed containment at that radius exists only for the
    convex HULL, i.e. only for affine classifiers; for deep nets it is an empirical question.

  python scripts/dev/l2_geometry_probe.py [--mc 1000000] [--d 3072]
"""
from __future__ import annotations
import argparse, os
import numpy as np

D_DEFAULT = 3 * 32 * 32          # 3072
EPS_1, EPS_INF, EPS_2 = 12.0, 8.0 / 255.0, 0.5
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
MAIN = os.path.join(ROOT, "results/main")
NORMS = ("linf", "l2", "l1")


# ------------------------------------------------------------------ TASK 1
def radial_union(l1, linf):
    """Exit distance of a ray from the ORIGIN through the union: max of the two exit distances."""
    return np.maximum(EPS_1 / l1, EPS_INF / linf)


def radial_hull(l1, linf):
    """Support function of conv(B_1 ∪ B_inf) — max of the two support functions (dual norms)."""
    return np.maximum(EPS_1 * linf, EPS_INF * l1)


def spike_family(d, s, betas):
    """u: `s` coords each carrying beta/sqrt(s) of the ℓ₂ mass, the rest spread equally.
    Returns (‖u‖₁, ‖u‖_∞) for each beta. beta = ℓ₂ mass on the spikes."""
    a = betas / np.sqrt(s)                                    # magnitude on each spike
    rest = np.sqrt(np.maximum(1.0 - betas ** 2, 0.0))
    b = rest / np.sqrt(d - s)                                 # magnitude on each remaining coord
    l1 = s * a + (d - s) * b
    linf = np.maximum(a, b)
    return l1, linf


def minimise(radial, d, s_max=10, verbose=True):
    """Structured minimisation over the s-spike family, coarse grid then refinement."""
    best = (np.inf, None, None)
    for s in range(1, s_max + 1):
        betas = np.linspace(1e-9, 1 - 1e-12, 200_001)
        v = radial(*spike_family(d, s, betas))
        i = int(np.argmin(v))
        lo = betas[max(i - 1, 0)]
        hi = betas[min(i + 1, len(betas) - 1)]
        for _ in range(6):                                     # refine around the argmin
            betas = np.linspace(lo, hi, 20_001)
            v = radial(*spike_family(d, s, betas))
            i = int(np.argmin(v))
            lo = betas[max(i - 1, 0)]
            hi = betas[min(i + 1, len(betas) - 1)]
        if v[i] < best[0]:
            best = (float(v[i]), s, float(betas[i]))
        if verbose:
            print(f"      s={s:<3} min={v[i]:.6f}  at beta={betas[i]:.6f}")
    return best


def ksparse_scan(d):
    """Exact scan over k-sparse equal-magnitude u for the HULL radius (integer k)."""
    k = np.arange(1, d + 1, dtype=np.float64)
    val = np.maximum(EPS_1 / np.sqrt(k), EPS_INF * np.sqrt(k))
    i = int(np.argmin(val))
    return float(val[i]), int(k[i])


def monte_carlo(radial, d, n, rng, batch=4000):
    """Random unit directions; must never undercut the structured minimum."""
    lo = np.inf
    done = 0
    while done < n:
        m = min(batch, n - done)
        g = rng.standard_normal((m, d)).astype(np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        v = radial(np.abs(g).sum(1).astype(np.float64), np.abs(g).max(1).astype(np.float64))
        lo = min(lo, float(v.min()))
        done += m
    return lo


def task1(d, n_mc):
    print("=" * 96)
    print(f"TASK 1 — inscribed ℓ₂ radii, d={d}, ε₁={EPS_1}, ε_∞=8/255={EPS_INF:.10f}, ε₂={EPS_2}")
    print("=" * 96)
    r_box1 = EPS_1 / np.sqrt(d)
    print(f"\n  (a) r_box1  = ε₁/√d = {r_box1:.4f}")

    print("\n  (b) r_union — structured minimisation over s-spike family:")
    u_val, u_s, u_beta = minimise(radial_union, d)
    print(f"      -> structured min = {u_val:.6f}  (s={u_s}, beta={u_beta:.6f})")

    print("\n  (c) r_hull — structured minimisation:")
    h_val, h_s, h_beta = minimise(radial_hull, d)
    cf = float(np.sqrt(EPS_1 * EPS_INF))
    ks_val, ks_k = ksparse_scan(d)
    print(f"      -> structured min = {h_val:.6f}  (s={h_s}, beta={h_beta:.6f})")
    print(f"      -> exact k-sparse scan: min={ks_val:.6f} at k={ks_k}   (k* = ε₁/ε_∞ = {EPS_1/EPS_INF:.1f})")
    print(f"      -> closed form √(ε₁ε_∞) = {cf:.6f}   |scan − closed| = {abs(ks_val - cf):.2e}")

    rng = np.random.default_rng(0)
    print(f"\n  Monte-Carlo sanity, {n_mc:,} random unit directions (must NOT undercut):")
    mc_u = monte_carlo(radial_union, d, n_mc, rng)
    mc_h = monte_carlo(radial_hull, d, n_mc, rng)
    print(f"      union: MC min {mc_u:.6f}  vs structured {u_val:.6f}   "
          f"{'OK' if mc_u >= u_val - 1e-9 else '*** UNDERCUT ***'}")
    print(f"      hull : MC min {mc_h:.6f}  vs structured {min(h_val, ks_val):.6f}   "
          f"{'OK' if mc_h >= min(h_val, ks_val) - 1e-9 else '*** UNDERCUT ***'}")

    r_union, r_hull = u_val, min(h_val, ks_val)
    print("\n" + "-" * 96)
    print(f"  r_box1  = {r_box1:.4f}")
    print(f"  r_union = {r_union:.4f}")
    print(f"  r_hull  = {r_hull:.4f}")
    print("-" * 96)
    print("\n  ZONES (ε₂ = 0.5):")
    print(f"    zone 1  ε₂ ≤ r_union            : ε₂ ≤ {r_union:.4f}  -> {'YES' if EPS_2 <= r_union else 'no'}")
    print(f"    zone 2  r_union < ε₂ ≤ r_hull   : {r_union:.4f} < ε₂ ≤ {r_hull:.4f}  -> "
          f"{'YES' if r_union < EPS_2 <= r_hull else 'no'}")
    print(f"    zone 3  ε₂ > r_hull             : ε₂ > {r_hull:.4f}  -> {'YES' if EPS_2 > r_hull else 'no'}")
    print(f"\n    ε₂/r_union = {EPS_2/r_union:.4f}      ε₂/r_hull = {EPS_2/r_hull:.4f}")

    flags = []
    if abs(r_hull - cf) > 1e-3:
        flags.append(f"r_hull {r_hull:.6f} deviates from √(ε₁ε_∞) {cf:.6f} by {abs(r_hull-cf):.2e} > 1e-3")
    if r_union >= EPS_2:
        flags.append(f"r_union {r_union:.6f} >= ε₂ {EPS_2} — ε₂ would be inside the RAW union")
    if mc_u < u_val - 1e-9 or mc_h < min(h_val, ks_val) - 1e-9:
        flags.append("Monte-Carlo undercut a structured minimum")
    return r_box1, r_union, r_hull, flags


# ------------------------------------------------------------------ TASK 2
def load(arm):
    p = os.path.join(MAIN, arm, "10k", "masks_multinorm_v1.npz")
    d = np.load(p, allow_pickle=True)
    return {k: d[k].astype(bool) for k in d.files if k != "metadata_json"}


def task2():
    arms = sorted(a for a in os.listdir(MAIN)
                  if os.path.exists(os.path.join(MAIN, a, "10k", "masks_multinorm_v1.npz")))
    print("\n" + "=" * 96)
    print("TASK 2 — per-example ℓ₂ anatomy over UNION-FAILING examples (@10k, 12 components)")
    print("=" * 96)
    print(f"{'model':<16}{'n_atk':>6}{'n_fail':>8}{'ℓ₂ blocks%':>12}{'ℓ₂ SOLE%':>10}"
          f"{'{ℓ₂,ℓ∞}%':>10}{'{ℓ₂,ℓ₁}%':>10}{'{all3}%':>9}")
    print("-" * 96)
    rows, flags = [], []
    for a in arms:
        m = load(a)
        if len(m) != 12:
            flags.append(f"{a}: {len(m)} mask keys, expected 12 — skipped")
            continue
        blk = {}
        for nm in NORMS:
            ks = [k for k in m if k.endswith(nm)]
            u = np.ones(len(m[ks[0]]), bool)
            for k in ks:
                u &= m[k]
            blk[nm] = ~u                                   # True = this norm blocks the example
        uni = np.ones(len(blk["linf"]), bool)
        for k in m:
            uni &= m[k]
        fail = ~uni
        n = int(fail.sum())
        li, l2, l1 = blk["linf"][fail], blk["l2"][fail], blk["l1"][fail]
        pct = lambda x: 100.0 * float(x.mean())
        r = dict(arm=a, n=n, any=pct(l2), sole=pct(l2 & ~li & ~l1),
                 l2linf=pct(l2 & li & ~l1), l2l1=pct(l2 & l1 & ~li), all3=pct(l2 & li & l1))
        rows.append(r)
        print(f"{a:<16}{len(m):>6}{n:>8}{r['any']:>12.2f}{r['sole']:>10.2f}"
              f"{r['l2linf']:>10.2f}{r['l2l1']:>10.2f}{r['all3']:>9.2f}")
        if r["sole"] > 0.0:
            flags.append(f"{a}: ℓ₂-SOLE = {r['sole']:.4f}% > 0")
    print("-" * 96)
    print(f"  models covered: {len(rows)}   (every arm under results/main with a 12-key @10k mask set)")
    return rows, flags


# ------------------------------------------------------------------ TASK 3
def task3(r_union, r_hull, rows):
    print("\n" + "=" * 96)
    print("TASK 3 — consistency check")
    print("=" * 96)
    c1 = r_union < EPS_2 <= r_hull
    worst = max((r["sole"] for r in rows), default=float("nan"))
    c2 = worst == 0.0
    print(f"  (i)  ε₂ = {EPS_2} lies in zone 2 (outside the raw union, inside the hull): "
          f"{r_union:.4f} < {EPS_2} ≤ {r_hull:.4f}  -> {'PASS' if c1 else 'FAIL'}")
    print(f"  (ii) ℓ₂-sole = 0.00% across all {len(rows)} models   (max observed {worst:.4f}%)  "
          f"-> {'PASS' if c2 else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if (c1 and c2) else 'FAIL'}")
    return c1 and c2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=D_DEFAULT)
    p.add_argument("--mc", type=int, default=1_000_000)
    a = p.parse_args()
    r_box1, r_union, r_hull, f1 = task1(a.d, a.mc)
    rows, f2 = task2()
    ok = task3(r_union, r_hull, rows)
    flags = f1 + f2
    print("\n" + "=" * 96)
    print("FLAGS: " + ("no flags" if not flags else ""))
    for x in flags:
        print("  🔴 " + x)
    print("=" * 96)


if __name__ == "__main__":
    main()

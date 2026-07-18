"""Claim A attribution — paired gain of a pull-push arm over the matched MSD control M0.

Reads union masks from results/eval/union_bench/{arm,M0}/{eval.json,10k/eval.json},
computes paired sample-level bootstrap (B=10000, seed 0, 1-sided 95% LCB) of
union(arm) - union(M0) at 1k and 10k. Writes results/fromscratch/C5/attribution.json + .md.

Usage: c5_attribution.py <arm1> [<arm2> ...]   (arm dirs under union_bench, e.g. M1a M1b)
"""
import sys, json
from pathlib import Path
import numpy as np

UB = Path("/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO/results/eval/union_bench")
OUT = Path("/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO/results/fromscratch/C5")
ARMS = sys.argv[1:] or ["M1a"]

def union_mask(method, grade):
    npz = UB / method / ("masks_multinorm_v1.npz" if grade == "1k" else "10k/masks_multinorm_v1.npz")
    if not npz.exists():
        return None
    d = np.load(npz)
    names = [k for k in d.files if k != "metadata_json"]
    u = np.ones(len(d[names[0]]), bool)
    for n in names:
        u &= d[n].astype(bool)
    return u

def paired(a, b, q=0.05):
    rng = np.random.default_rng(0); n = len(a); D = []
    for _ in range(10000):
        i = rng.integers(0, n, n); D.append(a[i].mean() - b[i].mean())
    return float(np.mean(D)), float(np.quantile(D, q)), float(a.mean()), float(b.mean())

rows = []
for arm in ARMS:
    for grade in ("1k", "10k"):
        a = union_mask(arm, grade); m0 = union_mask("M0", grade)
        if a is None or m0 is None:
            rows.append({"arm": arm, "grade": grade, "status": "pending",
                         "have_arm": a is not None, "have_M0": m0 is not None})
            continue
        if len(a) != len(m0):
            rows.append({"arm": arm, "grade": grade, "status": "mask-length-mismatch"}); continue
        mean, lcb, ua, um0 = paired(a, m0)
        rows.append({"arm": arm, "grade": grade, "union_arm": round(ua, 4), "union_M0": round(um0, 4),
                     "gain_mean": round(mean, 4), "lcb95": round(lcb, 4),
                     "significant": bool(lcb > 0), "n": len(a)})

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "attribution.json").write_text(json.dumps({"comparator": "M0 (matched MSD-AT control)", "rows": rows}, indent=2))
L = ["# Claim A — pull-push attribution vs matched MSD control M0", "",
     "Paired sample-level bootstrap (B=10000, seed 0, 1-sided 95% LCB) of union(arm) − union(M0), same examples.",
     "M0 = pure MSD-AT (base MSD steps=10, NO pull-push), same recipe/seed as M1a → isolates the pull-push contribution.", "",
     "| arm | grade | union(arm) | union(M0) | gain | LCB95 | signif |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get("status"):
        L.append(f"| {r['arm']} | {r['grade']} | — | — | — | — | {r['status']} |")
    else:
        L.append(f"| {r['arm']} | {r['grade']} | {r['union_arm']} | {r['union_M0']} | {r['gain_mean']:+.4f} | "
                 f"{r['lcb95']:+.4f} | {'YES' if r['significant'] else 'no (CI incl 0)'} |")
(OUT / "attribution.md").write_text("\n".join(L))
print("\n".join(L)); print("\nwrote", OUT / "attribution.json", "+ .md")

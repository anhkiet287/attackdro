"""T2g — POST-HOC (exploratory, NOT pre-registered). Written 2026-07-23.

Designed after the T2 / T2b / t2e / T2f results were known. It is a measurement-space robustness
check on T2, not a new hypothesis test, and no decision rule is committed here.

T2 measured d(clean, adv) in f = adaptive_avg_pool2d(relu(bn(...))) — pooled features AFTER the
final ReLU. f is therefore non-negative elementwise, so cosine between any two f vectors is
confined to [0, 1] and cosine distance to [0, 1]. T2g re-measures the SAME quantity in the
PRE-ReLU space, pool(bn(...)) without the relu, where entries are signed and cosine spans
[-1, 1], to test whether T2's sign is an artifact of the measurement space.

Exactly one thing changes relative to t2_geometry.py: a second feature extractor that omits
F.relu. Same 4 val_best checkpoints, same frozen 1000-example audit subset, same APGD-CE
n_iter=50, same eps triple, same cosine metric, same paired bootstrap. Both spaces are read off
the SAME forward pass, so the adversarial views are shared and the two spaces are exactly paired
per example.

Range diagnostic (same pass): fraction of negative entries in the pre-ReLU pooled features, and
cosine between randomly paired DIFFERENT images in pre-ReLU, post-ReLU f, and head h — i.e. how
much room a repulsion term actually has in each space.

WEIGHTS CAVEAT: f and pre-ReLU are measured at val_best (T2's checkpoints). h exists only in the
resume checkpoints (val_best stores the backbone only), so h is measured at LAST-EPOCH weights,
exactly as T2b did. The h column is therefore a different set of weights and is reported as a
space-range diagnostic only, never as a within-checkpoint contrast against f.

  python scripts/dev/t2g_prerelu_geometry.py [--n-iter 50] [--bs 128]
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
_s = importlib.util.spec_from_file_location("td", os.path.join(ROOT, "scripts/dev/t2d_head_space.py"))
td = importlib.util.module_from_spec(_s); _s.loader.exec_module(td)
t2 = td.t2
from robustdro.attacks.apgd_train import apgd_train            # noqa: E402

OUT = t2.OUT
NORMS, VIEWS, EPS, APGD_NORM = t2.NORMS, t2.VIEWS, t2.EPS, t2.APGD_NORM
SPACES = ("pre", "post")
B_BOOT = 10000


def features_pre(b, x):
    """Identical to t2.Enc.features but WITHOUT F.relu: pool(bn(...)), signed, 512-d."""
    out = b.conv1(x)
    for L in (b.layer1, b.layer2, b.layer3, b.layer4):
        out = L(out)
    out = b.bn(out)                                   # <- the one change: no F.relu
    return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)


def boot_ci(d, nboot=B_BOOT, seed=0):
    """Paired bootstrap over examples, fresh default_rng(seed) per call, 2-sided 95% percentile."""
    rng = np.random.default_rng(seed)
    n = len(d)
    D = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    return {"delta": float(d.mean()),
            "ci95": [float(np.quantile(D, .025)), float(np.quantile(D, .975))]}


def rand_partner(n, seed=0):
    """j[i] uniform over {0..n-1} \\ {i}: guarantees a DIFFERENT image, deterministic."""
    rng = np.random.default_rng(seed)
    j = rng.integers(0, n - 1, n)
    j[j >= np.arange(n)] += 1
    return j


def randpair_stats(Z, j):
    zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    c = (zn * zn[j]).sum(1)
    return c, {"mean": float(c.mean()), "min": float(c.min()), "max": float(c.max()),
               "q01": float(np.quantile(c, .01)), "q99": float(np.quantile(c, .99)),
               "frac_negative": float((c < 0).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260709)      # == t2_geometry default
    p.add_argument("--boot", type=int, default=B_BOOT)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)                                 # same stream as t2_geometry.main

    xs, ys, idx = t2.subset("1k", device)
    j = rand_partner(len(xs), 0)
    print(f"[T2g] POST-HOC / exploratory — not pre-registered.  n={len(xs)} device={device} "
          f"apgd_iter={a.n_iter}")
    print(f"[T2g] post = pool(relu(bn(.))) (T2's f, non-negative)   pre = pool(bn(.)) (signed)\n")

    R, PE = {}, {}
    for arm, desc, ck in t2.MODELS:                           # same order as T2 -> same RNG stream
        ckp = os.path.join(ROOT, ck)
        assert os.path.exists(ckp), f"ckpt missing: {ck}"
        sha = hashlib.sha256(open(ckp, "rb").read()).hexdigest()
        m = t2.load(ckp, device)                              # read-only load, eval mode
        print(f"[T2g] {arm:9s} {desc}\n      sha={sha[:16]}  crafting {len(NORMS)} views ...")
        Z = {s: {v: [] for v in VIEWS} for s in SPACES}
        corr = {v: [] for v in VIEWS}
        for i in range(0, len(xs), a.bs):
            x, y = xs[i:i + a.bs].to(device), ys[i:i + a.bs].to(device)
            batch = {"clean": x}
            for nm in NORMS:                                  # crafted against THIS model, as in T2
                batch[nm] = apgd_train(m, x, y, APGD_NORM[nm], EPS[nm],
                                       n_iter=a.n_iter, is_train=False).detach()
            with torch.no_grad():
                for v in VIEWS:                               # BOTH spaces from the same forward batch
                    Z["post"][v].append(m.features(batch[v]).float().cpu())
                    Z["pre"][v].append(features_pre(m.b, batch[v]).float().cpu())
                    corr[v].append((m(batch[v]).argmax(1) == y).cpu())
            print(f"      [{i + len(x):>5}/{len(xs)}]", end="\r", flush=True)
        print(" " * 32, end="\r")
        Z = {s: {v: torch.cat(Z[s][v]).numpy() for v in VIEWS} for s in SPACES}
        corr = {v: torch.cat(corr[v]).numpy() for v in VIEWS}
        PE[arm] = {s: {nm: t2.dist(Z[s]["clean"], Z[s][nm], "cos") for nm in NORMS} for s in SPACES}

        neg_frac = {v: float((Z["pre"][v] < 0).mean()) for v in VIEWS}
        rp = {}
        for s in SPACES:
            c, st = randpair_stats(Z[s]["clean"], j)
            rp[s] = st
            rp[s]["_c"] = c
        R[arm] = {"desc": desc, "ckpt": ck, "ckpt_sha256": sha,
                  "weights": "val_best",
                  "d_post": {nm: float(PE[arm]["post"][nm].mean()) for nm in NORMS},
                  "d_pre": {nm: float(PE[arm]["pre"][nm].mean()) for nm in NORMS},
                  "acc": {v: float(corr[v].mean()) for v in VIEWS},
                  "prerelu_negative_entry_fraction": neg_frac,
                  "randpair_cos": {s: {k: v for k, v in rp[s].items() if k != "_c"} for s in SPACES}}
        R[arm]["_rp"] = rp
        del m, Z
        torch.cuda.empty_cache()
        print(f"      d_post ℓ∞ {R[arm]['d_post']['linf']:.4f} ℓ2 {R[arm]['d_post']['l2']:.4f} "
              f"ℓ1 {R[arm]['d_post']['l1']:.4f}   |   d_pre ℓ∞ {R[arm]['d_pre']['linf']:.4f} "
              f"ℓ2 {R[arm]['d_pre']['l2']:.4f} ℓ1 {R[arm]['d_pre']['l1']:.4f}")
        print(f"      pre-ReLU negative entries (clean): {neg_frac['clean']:.4f}\n")

    # ---- head-space range diagnostic: LAST-EPOCH resume weights (T2b's checkpoints) ----
    print("[T2g] head-space range diagnostic — LAST-EPOCH resume weights (h is absent from val_best)")
    H = {}
    for arm, role, ck, trained in td.MODELS:
        mh, ep = td.load(ck, device)                          # strict=True; head must be present
        sha = hashlib.sha256(open(os.path.join(ROOT, ck), "rb").read()).hexdigest()
        zs = []
        with torch.no_grad():
            for i in range(0, len(xs), a.bs):
                zs.append(mh.embed(xs[i:i + a.bs].to(device)).float().cpu())
        c, st = randpair_stats(torch.cat(zs).numpy(), j)
        H[arm] = {"ckpt": ck, "ckpt_sha256": sha, "epoch": ep, "head_trained": bool(trained),
                  "weights": "last epoch (resume/ckpt_latest)", "randpair_cos_h": st}
        H[arm]["_c"] = c
        del mh
        torch.cuda.empty_cache()
        print(f"      {arm:9s} ep={ep} head={'TRAINED' if trained else 'UNTRAINED'}  "
              f"randpair cos_h mean {st['mean']:+.4f} min {st['min']:+.4f} frac<0 {st['frac_negative']:.4f}")

    # ---- relative movement, both spaces, paired per example ----
    rel = {}
    for A, Bm, lbl in t2.PAIRS:
        rel[lbl] = {"term_on": A, "term_off": Bm, "per_space": {}}
        for s in SPACES:
            rel[lbl]["per_space"][s] = {nm: boot_ci(PE[A][s][nm] - PE[Bm][s][nm], a.boot, 0)
                                        for nm in NORMS}

    # ---- cross-check: d_post must reproduce T2's stored f-space means exactly ----
    xchk = {"reference": "results/analysis/T2_geometry/geometry.json", "checked": [], "max_abs_diff": None}
    gp = os.path.join(OUT, "geometry.json")
    if os.path.exists(gp):
        ref = json.load(open(gp))["models"]
        diffs = []
        for arm in R:
            if arm in ref:
                for nm in NORMS:
                    d = abs(R[arm]["d_post"][nm] - ref[arm]["all"]["cos"][f"clean-{nm}"])
                    diffs.append(d)
                    xchk["checked"].append({"arm": arm, "norm": nm, "abs_diff": d})
        xchk["max_abs_diff"] = float(max(diffs)) if diffs else None
        xchk["n_cells"] = len(diffs)
        xchk["n_bit_exact"] = int(sum(d == 0.0 for d in diffs))
        # 1e-4 = far below any reported digit; float32 GPU reductions are not bit-deterministic
        # across runs, so exact equality is not expected on every cell.
        xchk["reproduces_within_1e_4"] = bool(diffs and max(diffs) < 1e-4)
        xchk["note"] = ("same 4 checkpoints, same subset, same n_iter, same torch seed and model "
                        "order as t2_geometry.py, so the adversarial views are re-crafted from the "
                        "same RNG stream; residual differences are float32 GPU reduction "
                        "nondeterminism, not a different attack")
        print(f"\n[T2g] cross-check vs T2 geometry.json: max |d_post - d_f(T2)| = {max(diffs):.3e} "
              f"over {len(diffs)} cells ({xchk['n_bit_exact']} bit-exact) — "
              f"{'reproduces T2 within 1e-4' if xchk['reproduces_within_1e_4'] else 'DIFFERS'}")

    # ---- outputs ----
    np.savez_compressed(
        os.path.join(OUT, "t2g_pe.npz"),
        **{f"d_{arm}_{s}_{nm}": PE[arm][s][nm].astype(np.float16)
           for arm in PE for s in SPACES for nm in NORMS},
        **{f"randpair_{arm}_{s}": R[arm]["_rp"][s]["_c"].astype(np.float16)
           for arm in R for s in SPACES},
        **{f"randpair_h_{arm}": H[arm]["_c"].astype(np.float16) for arm in H},
        rand_partner_index=j.astype(np.int32))

    payload = {
        "status": "POST-HOC / exploratory — NOT pre-registered (written 2026-07-23, after the "
                  "T2 / T2b / t2e / T2f results were known)",
        "config": {
            "n": len(xs), "apgd_n_iter": a.n_iter, "bs": a.bs, "eps": EPS,
            "apgd": "APGD-CE, is_train=False, crafted per batch against each checkpoint itself, "
                    "shared by both spaces (identical views -> exactly paired comparison)",
            "spaces": {"post": "adaptive_avg_pool2d(relu(bn(.)),1) — T2's f, non-negative, cos in [0,1]",
                       "pre": "adaptive_avg_pool2d(bn(.),1) — signed, cos in [-1,1]",
                       "h": "normalize(head(features(x))) 128-d — range diagnostic ONLY"},
            "metric": "cosine distance 1 - cos, per example, then mean",
            "bootstrap": f"paired per example, B={a.boot}, fresh np.random.default_rng(0) per call, "
                         "2-sided 95% percentile",
            "subset": t2.SUBSETS["1k"], "subset_indices": idx,
            "weights": "f and pre-ReLU at val_best (T2's checkpoints); h at LAST-EPOCH resume "
                       "weights (val_best stores the backbone only, so h does not exist there). "
                       "The h column is a space-range diagnostic and is NOT a within-checkpoint "
                       "contrast against f.",
            "rand_partner": "j[i] drawn uniformly from {0..n-1} minus {i} via np.random.default_rng(0); "
                            "the SAME j is used in every space and every arm",
            "bootstrap_note": "t2_geometry.py draws its CIs from ONE continuous rng(0) stream across "
                              "pairs and norms; this script uses a fresh rng(0) per call. Point "
                              "estimates are unaffected; CI endpoints differ only by Monte-Carlo noise.",
        },
        "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in R.items()},
        "head_diagnostic": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                            for k, v in H.items()},
        "relative_movement": rel,
        "crosscheck_vs_t2": xchk,
    }
    json.dump(payload, open(os.path.join(OUT, "t2g_prerelu.json"), "w"), indent=2)

    # ---- console ----
    print("\n" + "=" * 112)
    print("T2g.1  DISTANCE-TO-CLEAN, cosine distance (1-cos), PRE-ReLU vs POST-ReLU, same views")
    print("=" * 112)
    print(f"{'model':<10}{'':2}{'d_post ℓ∞':>10}{'d_post ℓ2':>10}{'d_post ℓ1':>10}{'':4}"
          f"{'d_pre ℓ∞':>10}{'d_pre ℓ2':>10}{'d_pre ℓ1':>10}")
    for arm, _, _ in t2.MODELS:
        r = R[arm]
        print(f"{arm:<10}{'':2}{r['d_post']['linf']:>10.4f}{r['d_post']['l2']:>10.4f}"
              f"{r['d_post']['l1']:>10.4f}{'':4}{r['d_pre']['linf']:>10.4f}"
              f"{r['d_pre']['l2']:>10.4f}{r['d_pre']['l1']:>10.4f}")

    print("\n" + "=" * 112)
    print("T2g.2  RELATIVE MOVEMENT  Δd = d(CLAMP) − d(control), paired per example, 2-sided 95% CI")
    print("=" * 112)
    for lbl, r in rel.items():
        print(f"  {lbl}   [{r['term_on']} − {r['term_off']}]")
        print(f"    {'norm':<6}{'POST-ReLU Δd':>14}{'95% CI':>26}{'':3}{'PRE-ReLU Δd':>13}{'95% CI':>26}")
        for nm in NORMS:
            e_po, e_pr = r["per_space"]["post"][nm], r["per_space"]["pre"][nm]
            s_po = "*" if not (e_po["ci95"][0] <= 0 <= e_po["ci95"][1]) else " "
            s_pr = "*" if not (e_pr["ci95"][0] <= 0 <= e_pr["ci95"][1]) else " "
            print(f"    {nm:<6}{e_po['delta']:>+14.5f}  [{e_po['ci95'][0]:+.5f},{e_po['ci95'][1]:+.5f}]{s_po}"
                  f"{'':2}{e_pr['delta']:>+13.5f}  [{e_pr['ci95'][0]:+.5f},{e_pr['ci95'][1]:+.5f}]{s_pr}")
    print("  * = 95% CI excludes 0")

    print("\n" + "=" * 112)
    print("T2g.3  RANGE DIAGNOSTIC — how much room does a repulsion term have?")
    print("=" * 112)
    print(f"  pre-ReLU negative-entry fraction of the pooled 512-d feature (clean view):")
    for arm, _, _ in t2.MODELS:
        nf = R[arm]["prerelu_negative_entry_fraction"]
        print(f"    {arm:<10} clean {nf['clean']:.4f}   ℓ∞ {nf['linf']:.4f}  ℓ2 {nf['l2']:.4f}  ℓ1 {nf['l1']:.4f}")
    print(f"\n  cosine between randomly paired DIFFERENT images (clean), same partner index j:")
    print(f"    {'model':<10}{'space':<15}{'mean':>9}{'min':>9}{'q01':>9}{'q99':>9}{'max':>9}"
          f"{'frac<0':>9}   weights")
    for arm, _, _ in t2.MODELS:
        for s in SPACES:
            st = R[arm]["randpair_cos"][s]
            print(f"    {arm:<10}{s:<15}{st['mean']:>+9.4f}{st['min']:>+9.4f}{st['q01']:>+9.4f}"
                  f"{st['q99']:>+9.4f}{st['max']:>+9.4f}{st['frac_negative']:>9.4f}   val_best")
    for arm in H:
        st = H[arm]["randpair_cos_h"]
        tag = "h (trained)" if H[arm]["head_trained"] else "h (UNtrained)"
        print(f"    {arm:<10}{tag:<15}{st['mean']:>+9.4f}{st['min']:>+9.4f}{st['q01']:>+9.4f}"
              f"{st['q99']:>+9.4f}{st['max']:>+9.4f}{st['frac_negative']:>9.4f}   LAST EPOCH (ep={H[arm]['epoch']})")
    print("\n  NOTE  h rows are LAST-EPOCH resume weights; the pre/post rows are val_best. Different")
    print("        weights — the h row is a range diagnostic for the space, not a paired contrast.")

    print(f"\n  saved -> results/analysis/T2_geometry/t2g_prerelu.json")
    print(f"           results/analysis/T2_geometry/t2g_pe.npz  (per-example distances, fp16)")


if __name__ == "__main__":
    main()

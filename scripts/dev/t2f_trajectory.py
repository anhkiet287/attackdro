"""T2f — POST-HOC (exploratory, NOT pre-registered). Written 2026-07-22.

Designed after the T2, T2b and t2e results were known; it is a follow-up probe, not a
pre-registered test, and no hypothesis or decision rule is committed here.

Question: does the CLAMP term reduce d_f(clean, adv) on TRAINING data at any point during
training, relative to its matched control, and does that reduction transfer to TEST data?
t2e answered a two-point version of this in h-space, CLAMP arms only, with no control. T2f
turns it into a per-epoch trajectory in f-space WITH the matched control.

Checkpoint constraint (Phase-0 discovery, 2026-07-22): the periodic ep0NN.pt snapshots are
saved as `model.b.state_dict()` (c5_fromscratch.py:683) — BACKBONE ONLY, no head.* keys and
no stored `epoch` field (the epoch is the filename number = completed epochs). h-space is
therefore not measurable at intermediate epochs and T2f is f-space only; the h branch below
is kept and would activate if head.* keys were ever present.

Seed constraint: M1a has epoch snapshots at seed0; M0_matched exists only at seed2/seed3
(there is no from-scratch M0 seed0 — results/external_ckpt/M0_seed0.pt is a final-weights
download with no epoch series). seed2 is used, the same control T2b used. The pairing is
therefore CROSS-SEED and is flagged as such in the JSON and on the console.

The strong pair (M1a_full / M0_full, C5_full, 50 ep) has NO intermediate snapshots at all,
so no strong-pair trajectory is produced.

  python scripts/dev/t2f_trajectory.py [--n-iter 10] [--bs 128] [--n 1000]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np
import torch

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import importlib.util                                          # noqa: E402
_s = importlib.util.spec_from_file_location("td", os.path.join(ROOT, "scripts/dev/t2d_head_space.py"))
td = importlib.util.module_from_spec(_s); _s.loader.exec_module(td)
t2 = td.t2
from robustdro.attacks.apgd_train import apgd_train             # noqa: E402

OUT = t2.OUT
NORMS, VIEWS, EPS, APGD_NORM = t2.NORMS, t2.VIEWS, t2.EPS, t2.APGD_NORM
TEST_SUBSET = t2.SUBSETS["1k"]
TRAIN_CORE = (0, 49000)          # c5_fromscratch.py:51 — train[49000:50000] is the val split
EPOCHS = (20, 40, 60, 80)
DATASETS = ("train", "test")

# (arm, role, seed, ckpt-dir)
ARMS = [
    ("M1a", "CLAMP    weak (MSD-10, 80ep)", 0, "results/fromscratch/C5/M1a_advneg/seed0/ckpt"),
    ("M0",  "control  weak (MSD-10, 80ep, term off)", 2, "results/fromscratch/C5/M0_matched/seed2/ckpt"),
]
CLAMP_ARM, CTRL_ARM = "M1a", "M0"


# ----------------------------------------------------------------------------- data
def train_images(n, seed=0):
    """n images sampled from the ACTUAL training split, train[0:49000].

    Deliberate correction relative to t2e.train_images, which drew from all 50k and so could
    include val-split indices (train[49000:50000]) that the model selected on but did not fit.
    """
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=True,
                                      download=False, transform=T.Compose([T.ToTensor()]))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(TRAIN_CORE[1] - TRAIN_CORE[0], generator=g)[:n].add(TRAIN_CORE[0]).tolist()
    return (torch.stack([ds[i][0] for i in idx]), torch.tensor([ds[i][1] for i in idx]),
            [int(i) for i in idx])


# ----------------------------------------------------------------------------- model
def load_ckpt(rel, device):
    """Backbone-only -> t2.Enc (f only). backbone+head -> td.BackboneHead strict=True (f and h).

    No silent strict=False fallback for head models: a strict failure propagates.
    """
    ck = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    stored_epoch = ck.get("epoch") if isinstance(ck, dict) else None
    if any(k.startswith("head.") for k in sd):
        m = td.BackboneHead()
        m.load_state_dict(sd, strict=True)
        return m.to(device).eval(), True, stored_epoch
    return t2.load(os.path.join(ROOT, rel), device), False, stored_epoch


def embed(m, xs, ys, has_head, n_iter, bs, device):
    """Per-example cosine distances d(clean, adv_p) in f (and h if present), + per-view acc.

    Views are crafted against THIS checkpoint, per batch, exactly as t2_geometry.embed_all.
    """
    Zf = {v: [] for v in VIEWS}
    Zh = {v: [] for v in VIEWS}
    acc = {v: [] for v in VIEWS}
    for i in range(0, len(xs), bs):
        x, y = xs[i:i + bs].to(device), ys[i:i + bs].to(device)
        batch = {"clean": x}
        for nm in NORMS:
            batch[nm] = apgd_train(m, x, y, APGD_NORM[nm], EPS[nm],
                                   n_iter=n_iter, is_train=False).detach()
        with torch.no_grad():
            for v in VIEWS:
                Zf[v].append(m.features(batch[v]).float().cpu())
                if has_head:
                    Zh[v].append(m.embed(batch[v]).float().cpu())
                acc[v].append((m(batch[v]).argmax(1) == y).cpu())
        print(f"      [{i + len(x):>5}/{len(xs)}]", end="\r", flush=True)
    print(" " * 32, end="\r")
    Zf = {v: torch.cat(Zf[v]).numpy() for v in VIEWS}
    pe = {"f": {nm: t2.dist(Zf["clean"], Zf[nm], "cos") for nm in NORMS}}
    if has_head:
        Zh = {v: torch.cat(Zh[v]).numpy() for v in VIEWS}
        pe["h"] = {nm: t2.dist(Zh["clean"], Zh[nm], "cos") for nm in NORMS}
    return pe, {v: float(torch.cat(acc[v]).float().mean()) for v in VIEWS}


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=10, help="APGD iters (10 = the training-time condition)")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--boot", type=int, default=10000)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    xs_te, ys_te, idx_te = t2.subset("1k", device)
    xs_tr, ys_tr, idx_tr = train_images(a.n)
    if a.n < len(xs_te):                                  # --n < 1000 is a smoke knob only
        xs_te, ys_te, idx_te = xs_te[:a.n], ys_te[:a.n], idx_te[:a.n]
    assert len(xs_te) == a.n and len(xs_tr) == a.n
    assert max(idx_tr) < TRAIN_CORE[1], "train sample leaked into the val split"
    DATA = {"train": (xs_tr, ys_tr), "test": (xs_te, ys_te)}

    seeds = {arm: sd for arm, _, sd, _ in ARMS}
    cross_seed = seeds[CLAMP_ARM] != seeds[CTRL_ARM]

    print(f"[T2f] POST-HOC / exploratory — not pre-registered.  {a.n} train + {a.n} test images")
    print(f"[T2f] device={device} apgd_iter={a.n_iter} (training-time condition) bs={a.bs}")
    print(f"[T2f] f-space = pooled 512-d encoder, pre-g, pre-h")
    if cross_seed:
        print(f"[T2f] WARNING cross-seed pair: {CLAMP_ARM} seed{seeds[CLAMP_ARM]} vs "
              f"{CTRL_ARM} seed{seeds[CTRL_ARM]} — M0 has no seed0 epoch snapshots on disk.")
    print()

    CK, PE = {}, {}                       # PE[arm][ep][dataset][space][norm] = (N,) distances
    for arm, role, sd, ckdir in ARMS:
        PE[arm] = {}
        for ep in EPOCHS:
            rel = f"{ckdir}/ep{ep:03d}.pt"
            path = os.path.join(ROOT, rel)
            assert os.path.exists(path), f"ckpt missing: {rel}"
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            m, has_head, stored_ep = load_ckpt(rel, device)
            CK[f"{arm}@{ep}"] = {"arm": arm, "role": role, "seed": sd, "path": rel,
                                 "sha256": sha, "epoch": ep, "epoch_source": "filename",
                                 "epoch_stored": stored_ep, "head_present": has_head,
                                 "spaces": ["f", "h"] if has_head else ["f"], "acc": {}}
            print(f"[T2f] {arm:4s} ep{ep:03d} seed{sd}  head={'yes' if has_head else 'NO (backbone-only)'}"
                  f"  sha={sha[:16]}")
            PE[arm][ep] = {}
            for ds in DATASETS:
                x, y = DATA[ds]
                pe, acc = embed(m, x, y, has_head, a.n_iter, a.bs, device)
                PE[arm][ep][ds] = pe
                CK[f"{arm}@{ep}"]["acc"][ds] = acc
                print(f"        {ds:<5} d_f(cl,·) ℓ∞ {pe['f']['linf'].mean():.4f}"
                      f"  ℓ2 {pe['f']['l2'].mean():.4f}  ℓ1 {pe['f']['l1'].mean():.4f}"
                      f"   | acc clean {acc['clean']:.3f}")
            del m
            torch.cuda.empty_cache()

    # ---- paired deltas: delta = d(CLAMP) - d(control), per example ----
    rng = np.random.default_rng(0)        # instantiated once, consumed in the loop order below
    SPACES = [s for s in ("f", "h") if all(s in PE[arm][ep][ds]
                                           for arm in PE for ep in EPOCHS for ds in DATASETS)]
    DEL = {}
    for ep in EPOCHS:
        for ds in DATASETS:
            for sp in SPACES:
                for nm in NORMS:
                    dA, dB = PE[CLAMP_ARM][ep][ds][sp][nm], PE[CTRL_ARM][ep][ds][sp][nm]
                    d = dA - dB
                    n = len(d)
                    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(a.boot)])
                    lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
                    DEL[f"ep{ep}|{ds}|{sp}|{nm}"] = {
                        "epoch": ep, "dataset": ds, "space": sp, "norm": nm,
                        "d_clamp": float(dA.mean()), "d_control": float(dB.mean()),
                        "delta": float(d.mean()), "ci95": [lo, hi],
                        "excludes_zero": not (lo <= 0.0 <= hi)}

    # ---- outputs ----
    np.savez_compressed(
        os.path.join(OUT, "t2f_pe.npz"),
        **{f"{arm}_ep{ep}_{ds}_{sp}_{nm}": PE[arm][ep][ds][sp][nm].astype(np.float16)
           for arm in PE for ep in EPOCHS for ds in DATASETS
           for sp in PE[arm][ep][ds] for nm in NORMS})

    payload = {
        "status": "POST-HOC / exploratory — NOT pre-registered (written 2026-07-22, after "
                  "T2 / T2b / t2e results were known)",
        "config": {
            "n": a.n, "apgd_n_iter": a.n_iter, "apgd": "APGD-CE, is_train=False, crafted per "
            "batch against each checkpoint itself", "eps": EPS, "bs": a.bs,
            "epochs": list(EPOCHS), "spaces": SPACES,
            "f": "pooled 512-d encoder output, pre-g (linear), pre-h (projection head)",
            "metric": "cosine distance 1 - cos, per example, then mean",
            "bootstrap": f"paired per example, B={a.boot}, np.random.default_rng(0), 2-sided 95% percentile",
            "test_subset": TEST_SUBSET, "test_indices": idx_te,
            "train_split": {"range": list(TRAIN_CORE), "sampler": "torch.Generator().manual_seed(0), "
                            "randperm over range(49000)[:n]", "indices": idx_tr,
                            "note": "differs from t2e.train_images, which sampled from all 50k; this "
                                    "is a deliberate correction — train[49000:50000] is the val split "
                                    "(c5_fromscratch.py VAL_SELECT) and is excluded here"},
            "h_space": "not measurable: the periodic ep0NN.pt snapshots store model.b.state_dict() "
                       "(backbone only, no head.* keys), so no checkpoint in this run carries a head",
            "strong_pair": "not run — C5_full (M1a_full / M0_full) has no intermediate epoch "
                           "checkpoints on disk, only ckpt_latest / ckpt_prev / val_best",
        },
        "cross_seed_pair": cross_seed,
        "cross_seed_note": (f"{CLAMP_ARM} epoch snapshots are seed{seeds[CLAMP_ARM]}; "
                            f"{CTRL_ARM} epoch snapshots exist only at seed2/seed3 and seed"
                            f"{seeds[CTRL_ARM]} is used (the control T2b used). Deltas are "
                            "therefore not same-seed paired.") if cross_seed else None,
        "checkpoints": CK,
        "means": {f"{arm}|ep{ep}|{ds}|{sp}|{nm}": float(PE[arm][ep][ds][sp][nm].mean())
                  for arm in PE for ep in EPOCHS for ds in DATASETS
                  for sp in PE[arm][ep][ds] for nm in NORMS},
        "deltas": DEL,
    }
    json.dump(payload, open(os.path.join(OUT, "t2f_trajectory.json"), "w"), indent=2)

    # ---- console table ----
    for sp in SPACES:
        for nm in NORMS:
            print("\n" + "=" * 118)
            print(f"T2f  d_{sp}(clean, adv)  norm={nm}   Δ = d({CLAMP_ARM} CLAMP) − d({CTRL_ARM} control), "
                  f"paired per example, 95% CI B={a.boot}")
            print("=" * 118)
            print(f"{'':<6}{'':<2}{'TRAIN (train[0:49000] sample)':^54}|{'TEST (frozen audit 1k)':^54}")
            print(f"{'epoch':<6}{'':<2}{'CLAMP':>9}{'control':>10}{'Δ':>10}{'95% CI':>24}|"
                  f"{'CLAMP':>9}{'control':>10}{'Δ':>10}{'95% CI':>24}")
            for ep in EPOCHS:
                cells = []
                for ds in DATASETS:
                    e = DEL[f"ep{ep}|{ds}|{sp}|{nm}"]
                    cells.append(f"{e['d_clamp']:>9.4f}{e['d_control']:>10.4f}{e['delta']:>+10.5f}"
                                 f"  [{e['ci95'][0]:>+8.5f},{e['ci95'][1]:>+8.5f}]"
                                 + ("*" if e["excludes_zero"] else " "))
                print(f"{ep:<6}{'':<2}{cells[0]}|{cells[1]}")
            print("  * = bootstrap 95% CI excludes 0")

    if cross_seed:
        print(f"\n  WARNING cross_seed_pair=true — {CLAMP_ARM} is seed{seeds[CLAMP_ARM]}, "
              f"{CTRL_ARM} is seed{seeds[CTRL_ARM]}; deltas are not same-seed paired.")
    print(f"\n  saved -> results/analysis/T2_geometry/t2f_trajectory.json")
    print(f"  per-example distances (fp16) -> results/analysis/T2_geometry/t2f_pe.npz")


if __name__ == "__main__":
    main()

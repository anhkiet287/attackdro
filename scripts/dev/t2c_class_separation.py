"""T2c — POST-HOC (exploratory, not pre-registered). Zero GPU: reuses T2's saved embeddings.

T2 falsified the PULL hypothesis (adv moves further from clean in f, not closer). CLAMP has a second
term: the scaffold PUSHES each adv view away from other-class negatives. If the encoder-level effect
is push rather than pull, it should show up as CLASS SEPARATION in f, not as clean-adv alignment.

Measured per view, in the same 512-d f-space, on the same 1000 examples:
  within  = mean cosine similarity between two embeddings of the SAME class
  between = mean cosine similarity between embeddings of DIFFERENT classes
  margin  = within - between      (higher = classes better separated)
  fisher  = tr(S_b) / tr(S_w)     (classical between/within scatter ratio)
"""
from __future__ import annotations
import json, os, sys
import numpy as np
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import importlib.util
_s = importlib.util.spec_from_file_location("t2", os.path.join(ROOT, "scripts/dev/t2_geometry.py"))
t2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t2)
OUT = t2.OUT


def stats(Z, y):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    S = Zn @ Zn.T
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(y), dtype=bool)
    within, between = float(S[same & off].mean()), float(S[~same].mean())
    mu = Z.mean(0)
    sw = sum(((Z[y == c] - Z[y == c].mean(0)) ** 2).sum() for c in np.unique(y))
    sb = sum((y == c).sum() * ((Z[y == c].mean(0) - mu) ** 2).sum() for c in np.unique(y))
    return dict(within=within, between=between, margin=within - between, fisher=float(sb / sw))


def main():
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=False,
                                      download=False, transform=T.Compose([T.ToTensor()]))
    idx = json.load(open(os.path.join(ROOT, t2.SUBSETS["1k"])))["indices"]
    y = np.array([ds[int(i)][1] for i in idx])

    R = {}
    for arm, _, _ in t2.MODELS:
        d = np.load(os.path.join(OUT, f"emb_{arm}.npz"))
        R[arm] = {v: stats(d[v].astype(np.float32), y) for v in t2.VIEWS}

    print("=" * 96)
    print("T2c  CLASS SEPARATION in f-space (post-hoc, exploratory)   margin = within-cos − between-cos")
    print("=" * 96)
    print(f"{'model':<10}" + "".join(f"{'margin ' + v:>14}" for v in t2.VIEWS) + f"{'fisher(clean)':>15}")
    for arm, _, _ in t2.MODELS:
        print(f"{arm:<10}" + "".join(f"{R[arm][v]['margin']:>14.4f}" for v in t2.VIEWS)
              + f"{R[arm]['clean']['fisher']:>15.4f}")
    print("-" * 96)
    print("\nΔ (term ON − term OFF):")
    print(f"  {'pair':<26}" + "".join(f"{'Δmargin ' + v:>15}" for v in t2.VIEWS) + f"{'Δfisher':>11}")
    for A, B, lbl in t2.PAIRS:
        print(f"  {lbl:<26}" + "".join(f"{R[A][v]['margin'] - R[B][v]['margin']:>+15.4f}" for v in t2.VIEWS)
              + f"{R[A]['clean']['fisher'] - R[B]['clean']['fisher']:>+11.4f}")
    print("\n  positive Δmargin => the term separates classes MORE in the encoder (push mechanism)")
    json.dump(R, open(os.path.join(OUT, "class_separation.json"), "w"), indent=2)
    print(f"  saved -> results/analysis/T2_geometry/class_separation.json")


if __name__ == "__main__":
    main()

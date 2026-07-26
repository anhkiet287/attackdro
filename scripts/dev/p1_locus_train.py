"""P1 LOCUS pilot — train arm E: both CLAMP terms computed on f instead of on normalize(head(f)).

Pre-registered in docs/preregistrations/preregistration_P1_locus.md (committed 2026-07-23, before
this file existed). Read it first: the reading rule for E-M1a is asymmetric and is fixed there.

scripts/dev/c5_fromscratch.py is NOT edited. This is a thin wrapper that imports it and swaps ONE
function at runtime; with the default --clamp-space h it swaps nothing at all — the code path is
literally `return c5.main()` on unmodified argv — so every prior arm stays reproducible.

MEASURED CAVEAT, do not overclaim reproducibility: c5_fromscratch.py is NOT bit-deterministic
run-to-run on this hardware. Two identical direct invocations (--smoke, 2 batches, seed 0) gave
loss 2.290/2.290, scaffold 14.592/14.591, val proxy 0.0050/0.0110. The wrapper's h path sits
inside that spread. "Same code path", not "byte-identical output".

  --clamp-space h   (default)  z = normalize(head(features(x)))   == every existing arm
  --clamp-space f              z = normalize(features(x))         == arm E, the pilot

f = pooled 512-d POST-ReLU encoder output, identical to t2_geometry.Enc.features — the features
the linear classifier actually consumes. normalize() is kept because the loss is a cosine and
pos_sim is a plain dot product; cosine on f IS the dot product on normalize(f), the same metric
T2 measured.

KNOWN PROPERTY, recorded and NOT engineered around (pre-registration §7.1): f is non-negative
after the final ReLU, so cosine on f lies in [0,1] and the scaffold (push) branch cannot drive a
pair past orthogonality. No signed projection, no tau retune, no Euclidean push, no ReLU removal.

Under --clamp-space f the head must receive NO gradient. Asserted three ways (§7.2):
  1. every head parameter has .grad is None, checked once per epoch;
  2. the head state_dict at the end is BITWISE identical to its snapshot at initialisation;
  3. head[2].weight.std() at the end equals the initialisation value (T2b's 0.02549 check).

Usage (all other flags pass straight through to c5_fromscratch.py):
  python scripts/dev/p1_locus_train.py --clamp-space f --variant M1a --seed 0 --epochs 80 \
      --base msd --glue-view 3norm --neg adv --outdir results/fromscratch/P1_locus/E
"""
from __future__ import annotations
import argparse, copy, importlib.util, json, os, sys
import torch
import torch.nn.functional as F

ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
_s = importlib.util.spec_from_file_location("c5", os.path.join(ROOT, "scripts/dev/c5_fromscratch.py"))
c5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c5)

# per-epoch f-space diagnostics, drained into the training record by the val_metrics patch
STATS = {"cos_anchor_sum": 0.0, "cos_negpair_sum": 0.0, "n_anchor": 0.0, "n_negpair": 0.0}
STATE = {"model": None, "head_init": None, "space": "h"}


def _reset():
    for k in STATS:
        STATS[k] = 0.0


def pullpush_f(model, x_clean, xs_adv, y, alpha, beta, tau=0.1, neg_source="adv",
               glue_weighting="uniform", bind_T=1.0):
    """decoupled_pullpush with the locus on f. Same algebra as c5_fromscratch.py:99-148 with
    model.embed -> normalize(model.features); anchor still stop-grad, scaffold still decoupled
    (negatives only in the logsumexp), same masking by label."""
    if glue_weighting != "uniform":
        raise ValueError("--clamp-space f supports only --glue-weighting uniform "
                         "(bindworst calls model.head directly; P1 uses uniform)")
    B = x_clean.shape[0]
    k = len(xs_adv)
    with torch.no_grad():
        z_anchor = F.normalize(model.features(x_clean), dim=1)      # stop-grad anchor, on f
    z_adv = torch.cat([F.normalize(model.features(xa), dim=1) for xa in xs_adv], 0)
    anchor_idx = torch.arange(k * B, device=x_clean.device) % B
    yk = y.repeat(k)

    pos_sim = (z_adv * z_anchor[anchor_idx]).sum(1)
    glue = (-pos_sim / tau).mean()

    if neg_source == "adv":
        z_neg, y_neg = z_adv, yk
    elif neg_source == "clean":
        z_neg, y_neg = F.normalize(model.features(x_clean), dim=1), y
    else:
        raise ValueError(neg_source)
    sim = z_adv @ z_neg.t()
    neg_mask = (yk[:, None] != y_neg[None, :])
    neg_logits = (sim / tau).masked_fill(~neg_mask, float("-inf"))
    row_has_neg = neg_mask.any(1)
    scaffold = torch.logsumexp(neg_logits[row_has_neg], dim=1).mean() if row_has_neg.any() \
        else z_adv.new_zeros(())

    # ---- pre-registered §8 diagnostics, both on f, both RNG-free ----
    with torch.no_grad():
        STATS["cos_anchor_sum"] += float(pos_sim.sum())             # (i) view vs its clean anchor
        STATS["n_anchor"] += float(pos_sim.numel())
        # (ii) different-class pairs. Deterministic fixed shift, NOT a fresh permutation: drawing
        # one here would advance the main RNG stream and change the training trajectory.
        sh = torch.roll(torch.arange(B, device=x_clean.device), 1)
        dif = y != y[sh]
        if dif.any():
            c = (z_anchor * z_anchor[sh]).sum(1)[dif]
            STATS["cos_negpair_sum"] += float(c.sum())
            STATS["n_negpair"] += float(c.numel())

    return alpha * scaffold + beta * glue, {"glue": float(glue.detach()),
                                            "scaffold": float(scaffold.detach())}


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--clamp-space", choices=["h", "f"], default="h")
    a, rest = ap.parse_known_args()
    STATE["space"] = a.clamp_space

    if a.clamp_space == "h":
        print("[P1] --clamp-space h -> no patch applied; identical to running c5_fromscratch.py")
        sys.argv = [sys.argv[0]] + rest
        return c5.main()

    print("[P1] --clamp-space f -> arm E: glue AND scaffold computed on normalize(f), "
          "f = pooled 512-d POST-ReLU encoder output (t2_geometry.Enc.features)")
    print("[P1] KNOWN PROPERTY (pre-registration §7.1): f >= 0 after the final ReLU, so cosine on f")
    print("[P1]   is confined to [0,1] and the push branch cannot pass orthogonality. NOT tuned around.")

    # ---- patch 1: the loss locus ----
    c5.decoupled_pullpush = pullpush_f

    # ---- patch 2: snapshot the head at init, so the frozen-head assertion has a reference ----
    _init = c5.BackboneHead.__init__

    def init_snap(self):
        _init(self)
        STATE["model"] = self
        STATE["head_init"] = copy.deepcopy(self.head.state_dict())
    c5.BackboneHead.__init__ = init_snap

    # ---- patch 3: val_metrics — force has_head=False (never touch the head), drain diagnostics ----
    _val = c5.val_metrics

    def val_patched(model, X, Y, idx, device, bs=250, n_iter=20, has_head=True):
        m = _val(model, X, Y, idx, device, bs=bs, n_iter=n_iter, has_head=False)
        head_params = list(model.head.parameters())
        bad = [i for i, p in enumerate(head_params) if p.grad is not None]
        assert not bad, f"head parameters {bad} received a gradient under --clamp-space f"
        m["p1_cos_f_anchor"] = (STATS["cos_anchor_sum"] / STATS["n_anchor"]) if STATS["n_anchor"] else float("nan")
        m["p1_cos_f_negpair"] = (STATS["cos_negpair_sum"] / STATS["n_negpair"]) if STATS["n_negpair"] else float("nan")
        m["p1_head_grad_none"] = True
        _reset()
        return m
    c5.val_metrics = val_patched

    # ---- patch 4: collapse_diag calls model.embed (head). Disable it for arm E. ----
    c5.collapse_diag = lambda model, x, x_adv: {"alignment": float("nan"), "uniformity": float("nan"),
                                                "embed_norm": float("nan"),
                                                "note": "disabled under --clamp-space f (uses the head)"}

    sys.argv = [sys.argv[0]] + rest
    if "--dump-epochs" not in rest:      # embedding dumps also go through model.embed
        sys.argv += ["--dump-epochs", "999"]
    if "--glue-weighting" in rest and rest[rest.index("--glue-weighting") + 1] != "uniform":
        sys.exit("[P1] --clamp-space f requires --glue-weighting uniform")

    rc = c5.main()

    # ---- final assertions (pre-registration §7.2) ----
    m = STATE["model"]
    assert m is not None, "no BackboneHead was constructed"
    now = m.head.state_dict()
    diffs = {k: float((now[k].float().cpu() - STATE["head_init"][k].float()).abs().max())
             for k in STATE["head_init"]}
    worst = max(diffs.values())
    std = float(m.head[2].weight.detach().std())
    print("\n" + "=" * 78)
    print("[P1] FROZEN-HEAD ASSERTION (pre-registration §7.2)")
    print("=" * 78)
    for k, v in diffs.items():
        print(f"  head.{k:<16} max|final - init| = {v:.3e}")
    print(f"  head[2].weight.std() = {std:.5f}   (T2b's untrained-head reference: 0.02549)")
    assert worst == 0.0, f"head CHANGED during training (max|diff|={worst:.3e}) — locus switch leaked"
    print("  [ok] head state_dict is BITWISE identical to its initialisation snapshot")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    main()

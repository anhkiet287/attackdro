#!/usr/bin/env python3
"""Binding-norm trait-vs-state diagnostic from the per-epoch [G,B] loss dumps.

Gates the "predictive per-sample weighting" novelty (revival of thesis clustering
with an observable target): IF a sample's binding norm is a stable *trait* it can
be predicted and pre-weighted; if it is a *state* that migrates during training it
must be measured online (which is what F6 already argues at the population level).

CPU-only. Reads results/loss_mats/<run>/epNNN.pt (each = {'epoch','norms','L'} with
L = [G, B] per-sample losses of that epoch's FIRST TRAIN BATCH).

⚠ HONEST LIMITATION (verified in code): the trainer's first batch is drawn from a
`shuffle=True` loader and carries NO sample indices, so the B samples DIFFER every
epoch. Per-sample identity is therefore NOT trackable across epochs from these
dumps -- true per-sample trait-vs-state stability CANNOT be measured here. What we
CAN measure is population-level:
  (1) binding-norm frequency trajectory across epochs  -> F6 migration, aggregate
  (2) binding decisiveness  (L_max - L_2nd)/L_max      -> how forced the max is
  (3) binding entropy over norms per epoch             -> concentration vs spread
For a real per-sample trait-vs-state study, add a FIXED held-out probe batch dumped
with indices every epoch (1-line trainer change, needs one re-run) -- see NOTE at
the end of the printout.

Usage:  python scripts/dev/binding_trait_vs_state.py [run ...]   # default: all runs
"""
from __future__ import annotations

import glob
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "results", "loss_mats")


def analyse(run_dir):
    eps = sorted(glob.glob(os.path.join(run_dir, "ep*.pt")))
    if not eps:
        return None
    rows = []
    norms = None
    for p in eps:
        d = torch.load(p, map_location="cpu")
        norms = d["norms"]
        L = d["L"]                      # [G, B]
        G, B = L.shape
        bind = L.argmax(dim=0)          # weakest (highest-loss) norm per sample
        freq = torch.bincount(bind, minlength=G).float() / B
        top2 = L.topk(2, dim=0).values  # [2, B]
        decis = ((top2[0] - top2[1]) / top2[0].clamp(min=1e-6)).mean().item()
        p_ = freq.clamp(min=1e-9)
        ent = float(-(p_ * p_.log()).sum() / math.log(G))   # normalised [0,1]
        rows.append((d["epoch"], freq.tolist(), decis, ent))
    return norms, rows


def main():
    runs = sys.argv[1:] or [os.path.basename(d.rstrip("/"))
                            for d in sorted(glob.glob(os.path.join(ROOT, "*/")))]
    print("BINDING-NORM TRAIT-VS-STATE DIAGNOSTIC (population-level; see limitation)")
    print("=" * 78)
    for run in runs:
        rd = os.path.join(ROOT, run)
        res = analyse(rd)
        if not res:
            print(f"\n{run}: no dumps"); continue
        norms, rows = res
        first, last = rows[0], rows[-1]
        # aggregate migration = how much the binding-frequency vector moved end-to-end
        drift = sum(abs(a - b) for a, b in zip(first[1], last[1])) / 2  # total-variation
        print(f"\n### {run}   ({len(rows)} epochs, norms={norms})")
        print(f"  ep{first[0]:>3}  freq={[f'{x:.2f}' for x in first[1]]}  "
              f"decisiveness={first[2]:.3f}  entropy={first[3]:.2f}")
        mid = rows[len(rows)//2]
        print(f"  ep{mid[0]:>3}  freq={[f'{x:.2f}' for x in mid[1]]}  "
              f"decisiveness={mid[2]:.3f}  entropy={mid[3]:.2f}")
        print(f"  ep{last[0]:>3}  freq={[f'{x:.2f}' for x in last[1]]}  "
              f"decisiveness={last[2]:.3f}  entropy={last[3]:.2f}")
        dom_first = norms[max(range(len(norms)), key=lambda i: first[1][i])]
        dom_last = norms[max(range(len(norms)), key=lambda i: last[1][i])]
        migr = "MIGRATED" if dom_first != dom_last else "stable-dominant"
        print(f"  -> binding-freq total-variation drift={drift:.2f} | "
              f"dominant {dom_first}->{dom_last} [{migr}]")
    print("\n" + "=" * 78)
    print("READING: high drift / dominant-norm migration = STATE (F6: measure online).")
    print("  Low drift + one dominant norm throughout = population looks trait-like,")
    print("  but this is AGGREGATE only -- it does NOT establish per-sample stability.")
    print("NOTE: to settle per-sample trait-vs-state, dump a FIXED probe batch (same")
    print("  samples + indices every epoch). 1-line trainer change + 1 re-run (GPU).")


if __name__ == "__main__":
    main()

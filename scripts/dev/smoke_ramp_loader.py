#!/usr/bin/env python
"""Smoke-test the upstream RAMP checkpoint loader.

This only imports the RAMP model and loads one checkpoint. It does not load
data or run attacks, so it is safe to use as a quick integration check.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.models.ramp_loader import load_ramp_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="external/RAMP/models/pretr_Linf.pth")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.checkpoint)
    if not Path("external/RAMP").exists():
        print("[skip] external/RAMP is missing; clone RAMP before running this smoke test.")
        return
    if not path.exists():
        print(f"[skip] checkpoint missing: {path}")
        return

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = load_ramp_checkpoint(str(path), device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] loaded RAMP checkpoint: {path}")
    print(f"[ok] device={device}  params={n_params:,}  training={model.training}")


if __name__ == "__main__":
    main()

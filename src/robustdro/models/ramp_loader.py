"""Loader for upstream RAMP CIFAR-10 checkpoints.

RAMP uses its own PreActResNet18 implementation in external/RAMP with
activation="softplus1" and normal="none". The model keeps its normalization
tensors as plain attributes instead of registered buffers, so we instantiate it
directly on the requested device instead of building on CPU and moving later.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
_RAMP_ROOT = _REPO_ROOT / "external" / "RAMP"


@contextmanager
def _ramp_on_path():
    if not _RAMP_ROOT.exists():
        raise FileNotFoundError(
            f"RAMP checkout not found at {_RAMP_ROOT}. "
            "Clone it with: git clone https://github.com/uiuc-focal-lab/RAMP external/RAMP"
        )
    root = str(_RAMP_ROOT)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass


def _resolve_device(device: str) -> torch.device:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("RAMP checkpoint must be a state_dict or contain a 'state_dict' key")
    return {k.removeprefix("module."): v for k, v in checkpoint.items()}


def load_ramp_checkpoint(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load a RAMP PreActResNet18 checkpoint for our eval harness.

    Supports both RAMP's usual plain state_dict files and wrapped
    {"state_dict": ...} checkpoints. Inputs are expected in raw [0,1] space;
    the instantiated RAMP model uses normal="none".
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RAMP checkpoint not found: {ckpt_path}")

    dev = _resolve_device(device)
    with _ramp_on_path():
        from model_zoo.fast_models import PreActResNet18  # noqa: WPS433

        model = PreActResNet18(
            10,
            cuda=(dev.type == "cuda"),
            activation="softplus1",
            normal="none",
        )

    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model

"""Deterministic seeding (see CLAUDE.md: deterministic seeding everywhere)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (CPU + CUDA).

    With `deterministic=True` we also request deterministic cuDNN kernels. This
    can slow training slightly but makes runs reproducible — worth it for a
    paper. Set to False for a small speed bump during throwaway exploration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

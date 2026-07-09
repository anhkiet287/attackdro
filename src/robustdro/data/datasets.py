"""CIFAR-10 data loaders.

Images are kept in raw [0,1] pixel space (ToTensor only) — NO normalization
transform. Standardization, if any, happens inside the model so that adversarial
epsilons keep their literal pixel meaning (see models/preact_resnet.py and
the active paper config). Train-time augmentation is the standard random-crop + flip.
"""

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

_DATASETS = {"cifar10": torchvision.datasets.CIFAR10}


def build_loaders(cfg: dict, download: bool = True):
    dcfg = cfg["dataset"]
    name = dcfg["name"]
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {list(_DATASETS)}")

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),                 # -> [0,1], no normalization
    ])
    test_tf = T.Compose([T.ToTensor()])

    ds_cls = _DATASETS[name]
    train_ds = ds_cls(root=dcfg["root"], train=True, download=download, transform=train_tf)
    test_ds = ds_cls(root=dcfg["root"], train=False, download=download, transform=test_tf)

    # Optional held-out val split (CARD-3a-v2): the LAST `val_holdout` train
    # images are excluded from training and served by get_train_holdout().
    holdout = dcfg.get("val_holdout", 0)
    if holdout:
        train_ds = torch.utils.data.Subset(train_ds, range(len(train_ds) - holdout))

    bs = cfg["train"]["batch_size"]
    nw = dcfg.get("num_workers", 4)
    # Seeded generator so shuffling is reproducible under set_seed.
    g = torch.Generator().manual_seed(cfg.get("seed", 0))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=True, generator=g,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
    )
    return train_loader, test_loader


def get_train_holdout(cfg: dict, download: bool = False):
    """Tensors (x, y) for the held-out val split: the LAST dataset.val_holdout
    train images, loaded with ToTensor only (no augmentation). Deterministic,
    matches the exclusion in build_loaders."""
    dcfg = cfg["dataset"]
    holdout = dcfg.get("val_holdout", 0)
    if not holdout:
        raise ValueError("dataset.val_holdout not set")
    ds = _DATASETS[dcfg["name"]](root=dcfg["root"], train=True, download=download,
                                 transform=T.Compose([T.ToTensor()]))
    idx = range(len(ds) - holdout, len(ds))
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx])
    return xs, ys


def get_train_probe_batch(cfg: dict, size: int = 512, seed: int | None = None,
                          download: bool = False):
    """Fixed, deterministic train-set probe tensors with sample indices.

    This is for binding-norm trait-vs-state diagnostics: use the same raw train
    images every epoch, with no random crop/flip, so per-sample identities are
    trackable over time. If a held-out validation tail is configured, sample only
    from the actual training pool.
    """
    dcfg = cfg["dataset"]
    ds = _DATASETS[dcfg["name"]](root=dcfg["root"], train=True, download=download,
                                 transform=T.Compose([T.ToTensor()]))
    holdout = dcfg.get("val_holdout", 0)
    train_n = len(ds) - holdout
    if train_n <= 0:
        raise ValueError(f"Invalid train probe pool: len={len(ds)}, holdout={holdout}")
    n = min(int(size), train_n)
    g = torch.Generator().manual_seed(cfg.get("seed", 0) if seed is None else int(seed))
    indices = torch.randperm(train_n, generator=g)[:n].sort().values
    xs = torch.stack([ds[int(i)][0] for i in indices])
    ys = torch.tensor([ds[int(i)][1] for i in indices], dtype=torch.long)
    return xs, ys, indices.to(torch.long)

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


def get_test_subset(cfg: dict, n_examples: int | None = None, download: bool = False):
    """Raw CIFAR-10 test tensors. ``n_examples=None`` returns the full test set."""
    dcfg = cfg["dataset"]
    ds = _DATASETS[dcfg["name"]](root=dcfg["root"], train=False, download=download,
                                 transform=T.Compose([T.ToTensor()]))
    n = len(ds) if n_examples is None else min(int(n_examples), len(ds))
    xs = torch.stack([ds[i][0] for i in range(n)])
    ys = torch.tensor([ds[i][1] for i in range(n)], dtype=torch.long)
    return xs, ys


# ---------------------------------------------------------------------------
# Canonical CIFAR-10 split index ranges (pre-reg v3-A / Program A lock).
# Half-open [lo, hi) in untouched dataset order; mutually disjoint per source.
#   train:  train_core   [0, 49000)      val_select [49000, 50000)
#   test:   test_monitor [0, 1000)       test_final [1000, 9000)  cal [9000, 10000)
# `val_select/worst_union` is the ONLY selection-eligible signal; cal /
# test_monitor / test_final must NEVER drive checkpoint selection.
# ---------------------------------------------------------------------------
CANONICAL_SPLIT_RANGES: dict[str, tuple[str, int, int]] = {
    "train_core":   ("train", 0, 49000),
    "val_select":   ("train", 49000, 50000),
    "test_monitor": ("test", 0, 1000),
    "test_final":   ("test", 1000, 9000),
    "cal":          ("test", 9000, 10000),
}
SELECTION_ELIGIBLE_SPLITS = frozenset({"val_select"})
NON_SELECTION_SPLITS = frozenset({"cal", "test_monitor", "test_final"})
EVAL_SPLITS = frozenset({"val_select", "cal", "test_monitor", "test_final"})


def assert_canonical_splits_disjoint() -> bool:
    """Fail-closed: canonical split index ranges must be mutually disjoint
    within each source domain (train vs test). Raises on any collision."""
    by_source: dict[str, list] = {}
    for name, (src, lo, hi) in CANONICAL_SPLIT_RANGES.items():
        if lo >= hi:
            raise ValueError(f"Split {name!r} has empty/inverted range [{lo}, {hi})")
        by_source.setdefault(src, []).append((lo, hi, name))
    for src, ranges in by_source.items():
        ranges.sort()
        for (lo1, hi1, n1), (lo2, hi2, n2) in zip(ranges, ranges[1:]):
            if hi1 > lo2:
                raise ValueError(
                    f"Split index collision in {src}: {n1}[{lo1}, {hi1}) "
                    f"overlaps {n2}[{lo2}, {hi2})"
                )
    return True


def is_selection_split(split: str) -> bool:
    """True only for splits allowed to drive checkpoint selection (val_select)."""
    return split.lower() in SELECTION_ELIGIBLE_SPLITS


def assert_not_selection_split(split: str) -> None:
    """Gatekeeper: cal / test_monitor / test_final may NEVER drive checkpoint
    selection (selection stays val_select/worst_union). Fail-closed."""
    s = split.lower()
    if s in NON_SELECTION_SPLITS:
        raise ValueError(
            f"Split {s!r} is not selection-eligible; checkpoint selection must "
            f"use val_select/worst_union only."
        )


def _test_index_slice(cfg: dict, lo: int, hi: int, n_examples: int | None = None,
                      download: bool = False):
    """Raw CIFAR-10 test tensors for the half-open index range [lo, hi), in
    dataset order. ``n_examples`` (if given) caps the range from ``lo``; it can
    never extend past ``hi`` (keeps splits strictly disjoint)."""
    dcfg = cfg["dataset"]
    ds = _DATASETS[dcfg["name"]](root=dcfg["root"], train=False, download=download,
                                 transform=T.Compose([T.ToTensor()]))
    hi = min(int(hi), len(ds))
    if n_examples is not None:
        hi = min(hi, int(lo) + int(n_examples))
    idx = range(int(lo), hi)
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    return xs, ys


def get_eval_split(cfg: dict, split: str, n_examples: int | None = None,
                   download: bool = False):
    """Canonical eval-role tensors, sliced from frozen, disjoint index ranges.

    ``val_select`` = train 49000-49999 (held-out selection tail configured by
    ``dataset.val_holdout``). ``cal`` = test 9000-9999. ``test_monitor`` = test
    0-999. ``test_final`` = test 1000-8999 (8k). ``n_examples`` caps a split
    within its own range and can never spill into an adjacent split.

    cal / test_monitor / test_final NEVER drive checkpoint selection (see
    ``assert_not_selection_split``); selection stays ``val_select/worst_union``.
    """
    assert_canonical_splits_disjoint()
    split = split.lower()
    if split not in EVAL_SPLITS:
        raise ValueError(
            f"Unknown eval split {split!r}; expected one of "
            f"val_select, cal, test_monitor, test_final"
        )
    if split == "val_select":
        # Frozen semantics: the last dataset.val_holdout train rows == train
        # [49000, 50000) when val_holdout == 1000.
        xs, ys = get_train_holdout(cfg, download=download)
        if n_examples is not None:
            n = min(int(n_examples), xs.shape[0])
            xs, ys = xs[:n], ys[:n]
        return xs, ys
    _src, lo, hi = CANONICAL_SPLIT_RANGES[split]
    return _test_index_slice(cfg, lo, hi, n_examples=n_examples, download=download)


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

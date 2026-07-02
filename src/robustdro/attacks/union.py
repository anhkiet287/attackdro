"""The union rule (golden rule #2).

A sample is robust under the UNION of threat models only if it survives an
attack in *every* norm — the attacker picks, per sample, the strongest norm.
Given per-norm boolean "still correct after attack" masks, union robustness is
their logical AND.

    robust_under_union[i] = correct_under_linf[i] AND correct_under_l2[i] AND correct_under_l1[i]

Worst-case union accuracy is the mean of that mask. It is <= every per-norm
accuracy and <= the average-case accuracy, by construction.
"""

from __future__ import annotations

import torch


def union_mask(per_norm_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    """AND together per-norm robustness masks -> per-sample union robustness."""
    if not per_norm_masks:
        raise ValueError("per_norm_masks is empty")
    masks = list(per_norm_masks.values())
    out = masks[0].clone()
    for m in masks[1:]:
        out &= m
    return out


def summarize(clean_mask: torch.Tensor,
              per_norm_masks: dict[str, torch.Tensor]) -> dict:
    """Build the metrics dict: clean, per-norm robust, average, worst-case union."""
    umask = union_mask(per_norm_masks)
    per_norm_acc = {k: v.float().mean().item() for k, v in per_norm_masks.items()}
    return {
        "n": int(clean_mask.numel()),
        "clean_acc": clean_mask.float().mean().item(),
        "per_norm_robust_acc": per_norm_acc,
        # Average-case = mean over norms of the per-norm robust accuracy (secondary).
        "avg_robust_acc": sum(per_norm_acc.values()) / len(per_norm_acc),
        # PRIMARY metric: survives every norm.
        "worst_union_acc": umask.float().mean().item(),
    }

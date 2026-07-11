"""eval_union — the P0 linchpin.

Evaluate a model's worst-case robust accuracy over the UNION of (linf, l2, l1)
threat models under the shared paper protocol. This harness is
kept independent of training (golden rule #4) and is used for BOTH self-trained
checkpoints and downloaded baselines re-evaluated with our attack config.

Primary metric = worst_union_acc: a sample counts as robust only if it survives
a strong attack in every norm (attacker picks the strongest per sample).
"""

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T

from ..attacks.union import summarize
from .attacks_aa import robust_mask


@torch.no_grad()
def _clean_mask(model, x, y, device, bs=500):
    preds = []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        preds.append(model(xb).argmax(1).cpu())
    return torch.cat(preds) == y.cpu()


def load_test_subset(cfg, n_examples=None, device="cpu"):
    """Load the CIFAR-10 test set (raw [0,1]) as tensors, optionally a subset.

    The subset is the first `n_examples` images — deterministic, so re-evals of
    different checkpoints see identical samples (fair comparison).
    """
    ds = torchvision.datasets.CIFAR10(
        root=cfg["dataset"]["root"], train=False, download=False, transform=T.ToTensor()
    )
    n = len(ds) if n_examples is None else min(n_examples, len(ds))
    xs = torch.stack([ds[i][0] for i in range(n)])
    ys = torch.tensor([ds[i][1] for i in range(n)])
    return xs, ys


def load_eval_checkpoint(checkpoint_path: str, cfg: dict, model_family: str = "robustdro",
                         device: str = "cuda"):
    """Load a checkpoint for evaluation.

    model_family="robustdro" preserves the existing in-repo checkpoint format.
    model_family="ramp" loads upstream RAMP PreActResNet18 checkpoints from
    external/RAMP with activation="softplus1" and no input normalization.
    """
    if model_family == "ramp":
        from ..models import load_ramp_checkpoint

        return load_ramp_checkpoint(checkpoint_path, device=device), None
    if model_family != "robustdro":
        raise ValueError("Unknown model_family {!r}. Use 'robustdro' or 'ramp'.".format(model_family))

    from ..models import build_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("cfg", cfg)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def evaluate_union(model, x, y, cfg, norms=("linf", "l2", "l1"),
                   version="apgd", device="cuda", bs=250, seed=0, log_fn=print):
    """Run per-norm strong attacks and apply the union rule.

    Returns a metrics dict (see attacks.union.summarize) plus per-norm detail.
    """
    model.eval()
    tm = cfg["threat_model"]
    eval_cfg = cfg.get("eval_attack", {})

    clean = _clean_mask(model, x, y, device)
    if log_fn:
        log_fn(f"[eval] n={x.shape[0]}  clean_acc={clean.float().mean().item():.4f}")

    per_norm_masks = {}
    for norm in norms:
        eps = tm[norm]["eps"]
        steps = eval_cfg.get(norm, {}).get("steps", 100)
        restarts = eval_cfg.get(norm, {}).get("restarts", 1)
        mask, _ = robust_mask(model, x, y, norm, eps, steps=steps, restarts=restarts,
                              device=device, version=version, seed=seed, bs=bs)
        per_norm_masks[norm] = mask
        if log_fn:
            log_fn(f"[eval] {norm:>4s} (eps={eps:g}, {steps} it, "
                   f"{version}) robust_acc={mask.float().mean().item():.4f}")

    metrics = summarize(clean, per_norm_masks)
    metrics.update({"norms": list(norms), "version": version})
    if log_fn:
        log_fn(f"[eval] avg_robust_acc={metrics['avg_robust_acc']:.4f}  "
               f"WORST_UNION_acc={metrics['worst_union_acc']:.4f}")
    return metrics

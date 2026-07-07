"""Build a training-time source attack (a callable) from a config spec.

A spec is a dict like {norm: linf|l2|l1, steps: 10, step_size: 0.0078, ...}. The
epsilon is NOT in the spec — it comes from the locked protocol
(cfg.threat_model[norm].eps) so a method can never train under a different eps
than it is evaluated at (golden rule #1).
"""

from __future__ import annotations

from .apgd_train import apgd_train as _apgd_train
from .norms import pgd_l1_topk, pgd_l2, pgd_linf

_RAMP_NORM = {"linf": "Linf", "l2": "L2", "l1": "L1"}


def build_source_attack(spec: dict, eps: float, attack: str = "pgd"):
    """Return a training source-attack callable atk(model, x, y) -> x_adv.
    attack='pgd'  -> plain fixed-step PGD (per-norm; l1 = top-k).
    attack='apgd' -> APGD (RAMP-matched: momentum + adaptive step + best-iterate; l1 =
                     adaptive top-k + L1 projection). `steps` = APGD n_iter (10 = RAMP)."""
    norm = spec["norm"].lower()
    steps = spec.get("steps", 10)

    if attack == "apgd":
        ramp_norm = _RAMP_NORM[norm]

        def atk(model, x, y):
            was_training = model.training
            model.eval()                      # BN frozen while crafting (apgd_train needs eval)
            x_adv = _apgd_train(model, x, y, norm=ramp_norm, eps=eps, n_iter=steps, is_train=True)
            if was_training:
                model.train()
            return x_adv
        return atk

    step_size = spec["step_size"]
    random_start = spec.get("random_start", True)

    if norm == "linf":
        def atk(model, x, y):
            return pgd_linf(model, x, y, eps=eps, step_size=step_size,
                            steps=steps, random_start=random_start)
    elif norm == "l2":
        def atk(model, x, y):
            return pgd_l2(model, x, y, eps=eps, step_size=step_size,
                          steps=steps, random_start=random_start)
    elif norm == "l1":
        k = spec.get("k", 20)
        gap = spec.get("gap", 0.05)
        random_k = spec.get("random_k", True)
        def atk(model, x, y):
            return pgd_l1_topk(model, x, y, eps=eps, step_size=step_size,
                               steps=steps, k=k, gap=gap, random_k=random_k,
                               random_start=random_start)
    else:
        raise ValueError(f"Unknown source-attack norm: {norm!r}")
    return atk

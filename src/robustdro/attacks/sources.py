"""Build a training-time source attack (a callable) from a config spec.

A spec is a dict like {norm: linf|l2|l1, steps: 10, step_size: 0.0078, ...}. The
epsilon is NOT in the spec — it comes from the locked protocol
(cfg.threat_model[norm].eps) so a method can never train under a different eps
than it is evaluated at (golden rule #1).
"""

from __future__ import annotations

from .norms import pgd_l1_topk, pgd_l2, pgd_linf


def build_source_attack(spec: dict, eps: float):
    norm = spec["norm"].lower()
    steps = spec.get("steps", 10)
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

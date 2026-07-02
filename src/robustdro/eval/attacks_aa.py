"""Strong per-norm evaluation attacks via the official AutoAttack library.

Golden rule #4: evaluation is where robustness results break, so the eval
adversary must be strong and trustworthy. We use the maintained AutoAttack
(fra31/auto-attack) rather than a hand-rolled APGD — a subtly weak attack would
inflate robustness, the classic adversarial-eval failure. AutoAttack supports
Linf / L2 / L1, which is exactly the union we care about.

Two `version`s are exposed:
  * "apgd"     : APGD-CE + APGD-T only. Fast, strong; the default for iteration.
  * "standard" : full AutoAttack (APGD-CE, APGD-T, FAB-T, Square). Slower; use
                 for final/reported numbers.

All attacks operate in raw [0,1] pixel space.
"""

from __future__ import annotations

import contextlib
import io

import torch

from autoattack import AutoAttack

# Our lowercase norm keys -> AutoAttack's names.
NORM_MAP = {"linf": "Linf", "l2": "L2", "l1": "L1"}

_VERSION_ATTACKS = {
    "apgd-ce": ["apgd-ce"],   # CE only — cheap val-calibration signal (CARD-3a-v2)
    "apgd": ["apgd-ce", "apgd-t"],
    "standard": ["apgd-ce", "apgd-t", "fab-t", "square"],
}


def _build_adversary(model, norm, eps, steps, restarts, device, version, seed, verbose):
    if norm not in NORM_MAP:
        raise ValueError(f"Unknown norm '{norm}'. Use one of {list(NORM_MAP)}")
    attacks = _VERSION_ATTACKS.get(version)
    if attacks is None:
        raise ValueError(f"Unknown version '{version}'. Use {list(_VERSION_ATTACKS)}")

    aa = AutoAttack(
        model, norm=NORM_MAP[norm], eps=float(eps), version="custom",
        attacks_to_run=attacks, device=device, seed=seed, verbose=verbose,
    )
    # Match the protocol's iteration/restart budget for the APGD components.
    aa.apgd.n_iter = steps
    aa.apgd.n_restarts = restarts
    if hasattr(aa, "apgd_targeted"):
        aa.apgd_targeted.n_iter = steps
        aa.apgd_targeted.n_restarts = restarts
    return aa


@torch.no_grad()
def _predict(model, x, device):
    return model(x.to(device)).argmax(1).cpu()


def craft_adv(model, x, y, norm, eps, steps=100, restarts=1, device="cuda",
              version="apgd", seed=0, bs=None, verbose=False):
    """Craft adversarial examples for one norm. Returns x_adv (on CPU)."""
    aa = _build_adversary(model, norm, eps, steps, restarts, device, version, seed, verbose)
    bs = bs or x.shape[0]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if not verbose else io.StringIO()):
        x_adv = aa.run_standard_evaluation(x.to(device), y.to(device), bs=bs)
    return x_adv.detach().cpu()


def robust_mask(model, x, y, norm, eps, steps=100, restarts=1, device="cuda",
                version="apgd", seed=0, bs=250, verbose=False):
    """Return (robust_mask, x_adv) for one norm.

    robust_mask[i] is True iff sample i is STILL correctly classified after the
    attack (i.e. the attack failed to break it within the eps budget).
    """
    x_adv = craft_adv(model, x, y, norm, eps, steps, restarts, device, version,
                      seed, bs, verbose)
    pred = _predict(model, x_adv, device)
    mask = pred == y.cpu()
    return mask, x_adv

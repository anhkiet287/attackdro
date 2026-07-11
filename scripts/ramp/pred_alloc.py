"""Predictive per-norm budget allocation grafted onto RAMP (EXPLORATION lane).

Kiet forced the RAMP generality test despite the structural mismatch (RAMP crafts source+target
via a min-max curriculum + weight-space GP, NOT craft-3-then-allocate — see
results/exploration/generality_predictive_on_ramp.md). The graft keeps RAMP intact:

  phi = per-sample per-norm loss EMA (CARD-PB's predictor idea), keyed by dataset index, over
  the TWO crafted norms {source, target}. Each batch predicts ONCE (phi.predict); the crafts
  then give FULL apgd iters to the predicted-binding norm and a FLOOR to the other. So every
  sample spends exactly (full + floor) instead of (2*full) -> attack_flops_ratio = (full+floor)/
  (2*full) by construction (e.g. 0.60 with full=10, floor=2). RAMP's max + KL logit-pairing +
  weight-GP are untouched; only the per-sample attack budget shrinks.

IMPORTANT (fixed for the 5070ti run): `pred` is computed ONCE per batch and passed to both
alloc_craft calls. The earlier version recomputed pred inside each call, so phi.update on the
source column mutated Lbar before the target craft's predict — some samples then landed full-on-
both (inflating FLOPs to ~0.68) and, symmetrically, floor-on-both (under-attacked on every norm,
hurting union). Computing pred once eliminates both. Device-agnostic (CPU-smokable)."""
import torch

try:
    from autopgd_train import apgd_train           # RAMP's own APGD (run from external/RAMP)
except Exception:                                   # pragma: no cover
    from external.RAMP.autopgd_train import apgd_train


class Phi:
    """Per-sample binding predictor over the crafted norms (cols 0=source, 1=target)."""

    def __init__(self, n, norm_names, device, beta=0.5, cold_epochs=1, floor_iters=2):
        self.Lbar = torch.zeros(n, len(norm_names), device=device)
        self.seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.norm_names = list(norm_names)
        self.beta = float(beta)
        self.cold_epochs = int(cold_epochs)
        self.floor_iters = int(floor_iters)

    def predict(self, idx, epoch):
        """None during cold epochs (=> craft all full); else per-sample argmax col in {0,1}.
        Call ONCE per batch and pass the result to both alloc_craft calls."""
        if epoch < self.cold_epochs:
            return None
        return self.Lbar[idx].argmax(dim=1)

    def update(self, idx, col, loss_vec):
        prev = self.Lbar[idx, col]
        new = torch.where(self.seen[idx], self.beta * loss_vec + (1 - self.beta) * prev, loss_vec)
        self.Lbar[idx, col] = new
        self.seen[idx] = True


def alloc_craft(model, x, y, idx, col, norm_name, eps, full_iters, phi, pred):
    """Craft `norm_name` (column `col`): FULL iters where `pred`==col, FLOOR elsewhere.
    `pred` is None (cold: all full) or the batch's per-sample argmax col, computed ONCE by the
    caller. Returns (x_adv, loss_best[B], passes) matching RAMP's apgd_train unpacking; updates phi."""
    B = x.shape[0]
    if pred is None:                                 # cold: everyone full
        x_adv, _, _, loss_best, _ = apgd_train(model, x, y, norm=norm_name, eps=eps,
                                               n_iter=full_iters, is_train=True)
        phi.update(idx, col, loss_best.detach())
        return x_adv, loss_best, full_iters * B

    full_mask = (pred == col)
    x_adv = x.clone()
    loss_best = torch.zeros(B, device=x.device)
    passes = 0
    for mask, iters in ((full_mask, full_iters), (~full_mask, phi.floor_iters)):
        if mask.any():
            sub = mask.nonzero(as_tuple=True)[0]
            xa, _, _, lb, _ = apgd_train(model, x[sub], y[sub], norm=norm_name, eps=eps,
                                         n_iter=int(iters), is_train=True)
            x_adv[sub] = xa
            loss_best[sub] = lb
            passes += int(iters) * int(mask.sum())
    phi.update(idx, col, loss_best.detach())
    return x_adv, loss_best, passes

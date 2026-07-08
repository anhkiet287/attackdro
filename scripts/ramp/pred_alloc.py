"""Predictive per-norm budget allocation grafted onto RAMP (EXPLORATION — Colab idea-lane).

Kiet forced this despite the structural mismatch (RAMP crafts source+target via a min-max
curriculum + weight-space GP, NOT craft-3-then-allocate — see
results/exploration/generality_predictive_on_ramp.md). The graft that keeps RAMP intact:

  phi = per-sample per-norm loss EMA (CARD-PB's predictor idea), keyed by dataset index, over
  the TWO crafted norms {source, target}. For each craft, samples phi predicts BIND that norm
  get FULL apgd iters; the rest get a FLOOR. So every sample spends (full + floor) instead of
  (2 * full) steps -> pb/attack_flops_ratio ~ (full+floor)/(2*full) by construction. RAMP's
  max + KL logit-pairing + weight-GP are untouched; only the per-sample attack budget shrinks.

STEP-1 smoke question: does it RUN, is the ratio < 1.0, and does l-inf (RAMP's GP-strong norm,
crafted full only for samples phi routes to it) survive? Device-agnostic (CPU-smokable)."""
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
        """None during cold epochs (=> craft all full); else per-sample argmax col in {0,1}."""
        if epoch < self.cold_epochs:
            return None
        return self.Lbar[idx].argmax(dim=1)

    def update(self, idx, col, loss_vec):
        prev = self.Lbar[idx, col]
        new = torch.where(self.seen[idx], self.beta * loss_vec + (1 - self.beta) * prev, loss_vec)
        self.Lbar[idx, col] = new
        self.seen[idx] = True


def alloc_craft(model, x, y, idx, col, norm_name, eps, full_iters, phi, epoch):
    """Craft `norm_name` (column `col`): FULL iters where phi predicts it binds, FLOOR elsewhere.
    Returns (x_adv, loss_best[B], passes) matching RAMP's apgd_train unpacking. Updates phi."""
    B = x.shape[0]
    pred = phi.predict(idx, epoch)
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

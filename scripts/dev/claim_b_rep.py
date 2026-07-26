"""Claim-B representation terms added on top of the RAMP loss (reuse RAMP's own adversarials).

Two arms (see RAMP_claimB.py for the training-loop injection):
  B1 = RAMP + pull-push : decoupled alpha*scaffold + beta*glue.
        anchor  = clean embedding, STOP-GRAD (anchor A)
        positives = RAMP's own adversarials {x_linf, x_l1 [, x_l2 optional]} embeddings
        negatives = other-class adversarial embeddings in the batch (tau=0.1, alpha=beta=0.5)
  B2 = RAMP + worst-case SupCon : gamma*SupCon on the l-inf adversarial embeddings only
        (pull same-class, push other-class, batch labels; no clean/multi-lp), gamma small.

Head is a SimCLR projection (pooled 512-d penultimate -> 128), L2-normalized, DISCARDED at eval.
`features` here = fast_models `forward(x, return_features=True)` (8192-d pre-pool); we pool -> 512.
Pure torch → unit-testable on the PC before the Colab RAMP run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_head(proj_dim=128, hid=512):
    return nn.Sequential(nn.Linear(512, hid), nn.ReLU(inplace=True), nn.Linear(hid, proj_dim))


def _pool(feat):
    """fast_models return_features -> 512-d post-pool (mean over the 4x4 spatial grid)."""
    if feat.dim() == 2 and feat.shape[1] == 8192:
        return feat.view(feat.shape[0], 512, 4, 4).mean(dim=(2, 3))
    if feat.dim() == 2 and feat.shape[1] == 512:
        return feat
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    raise ValueError(f"unexpected feature shape {tuple(feat.shape)}")


def embed(head, feat):
    return F.normalize(head(_pool(feat)), dim=1)


def rep_b1_pullpush(head, feat_clean, feats_adv, y, alpha=0.5, beta=0.5, tau=0.1):
    """B1. feats_adv = list of per-norm adversarial features (l-inf, l-1[, l-2]).
    glue = pull each adv -> own detached clean anchor; scaffold = logsumexp-push over
    other-class adversarial embeddings. Returns (loss, {glue, scaffold})."""
    B = feat_clean.shape[0]; K = len(feats_adv)
    with torch.no_grad():
        z_clean = embed(head, feat_clean)                    # (B,d) STOP-GRAD anchor
    z_adv = torch.cat([embed(head, f) for f in feats_adv], 0)  # (K*B, d), grad
    anchor_idx = torch.arange(K * B, device=z_adv.device) % B
    yk = y.repeat(K)
    pos_sim = (z_adv * z_clean[anchor_idx]).sum(1)           # (K*B,)
    glue = (-pos_sim / tau).mean()
    sim = z_adv @ z_adv.t()                                  # (K*B,K*B) adv-adv
    neg_mask = (yk[:, None] != yk[None, :])                  # other-class => negative
    row_has = neg_mask.any(1)
    neg_logits = (sim / tau).masked_fill(~neg_mask, float("-inf"))
    scaffold = torch.logsumexp(neg_logits[row_has], dim=1).mean() if row_has.any() else z_adv.new_zeros(())
    return alpha * scaffold + beta * glue, {"glue": float(glue.detach()), "scaffold": float(scaffold.detach())}


def rep_b2_supcon(head, feat_adv_linf, y, gamma=0.2, tau=0.1):
    """B2. Supervised contrastive (Khosla et al.) on the l-inf adversarial embeddings.
    Returns (gamma*SupCon, {supcon}). pull same-class, push other-class, batch labels."""
    z = embed(head, feat_adv_linf)                           # (B,d)
    B = z.shape[0]
    sim = z @ z.t() / tau
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # over all a!=i
    pos_mask = (y[:, None] == y[None, :]) & ~self_mask
    denom = pos_mask.sum(1).clamp(min=1)
    # mask-fill non-positives to 0 BEFORE summing (avoid -inf*0 = NaN from the diagonal)
    loss_i = -logprob.masked_fill(~pos_mask, 0.0).sum(1) / denom
    valid = pos_mask.any(1)                                   # anchors with >=1 same-class positive
    supcon = loss_i[valid].mean() if valid.any() else z.new_zeros(())
    return gamma * supcon, {"supcon": float(supcon.detach())}

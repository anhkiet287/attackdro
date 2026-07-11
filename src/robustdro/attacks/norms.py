"""Per-norm attacks. For now: PGD-linf, used both for adversarial training and
as a cheap in-training robustness probe.

The strong evaluation attacks (APGD-linf/l2, DDN, APGD-l1) and the union rule
live in src/robustdro/eval and are built in P0 alongside eval_union. Keeping
train-time and eval-time attacks separate is deliberate (golden rule #4): the
final numbers must come from the independent, strong eval harness.

All attacks operate in raw [0,1] pixel space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pgd_linf(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Standard L-infinity PGD (Madry et al.). Returns adversarial images.

    Untargeted, maximizing cross-entropy. Perturbation is projected to the
    linf ball of radius `eps` and the image is clamped to [0,1] each step.
    """
    was_training = model.training
    model.eval()  # freeze BN statistics while crafting the attack

    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)  # project to linf ball
            x_adv = torch.clamp(x_adv, 0.0, 1.0)                   # valid image

    if was_training:
        model.train()
    return x_adv.detach()


# --------------------------------------------------------------------------- #
# L2 and L1 training-time attacks.
#
# Ported (fp32, [0,1] convention, always-attack) from locuslab/robust_union's
# cifar_funcs.py (pgd_l2, pgd_l1_topk + the l1 top-k steepest-descent helpers
# and L1-ball / simplex projection). These are the SOURCE attacks for union
# adversarial training. Adding L1 here is the P1 fix — the AttackDRO++ thesis
# used only Linf and L2. The independent EVAL attacks live in eval/ (AutoAttack).
# --------------------------------------------------------------------------- #

def _l2n(z):
    return z.view(z.shape[0], -1).norm(dim=1)[:, None, None, None]


def _l1n(z):
    return z.view(z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]


def pgd_l2(model, x, y, eps, step_size, steps, random_start=True):
    """L2 PGD (normalized steepest descent, project to L2 ball, clamp to [0,1])."""
    was_training = model.training
    model.eval()
    delta = torch.zeros_like(x)
    if random_start:
        delta = (2.0 * torch.rand_like(x) - 1.0) * eps
        delta = delta / _l2n(delta).clamp(min=eps)
        delta = torch.min(torch.max(delta, -x), 1 - x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + step_size * grad / _l2n(grad).clamp(min=1e-12)
            delta = delta * (eps / _l2n(delta).clamp(min=eps))       # project to L2 ball
            delta = torch.min(torch.max(delta, -x), 1 - x)           # keep x+delta in [0,1]
    if was_training:
        model.train()
    return (x + delta).detach()


def _kth_largest(t, k, dim=-1):
    return t.topk(k, dim=dim)[0][:, :, -1]


def _l1_dir_topk(grad, delta, x, gap, k):
    """Top-k signed steepest-descent direction for the L1 attack.

    Zero out coordinates that are already clipped (at 0/1) or that would push
    past the [gap, 1-gap] box in the wrong direction, then keep the k largest
    |grad| coordinates (per image) and step in their sign.
    """
    x_curr = x + delta
    b, c, p = x.shape[0], x.shape[1], x.shape[2]
    neg1 = (grad < 0) * (x_curr <= gap)
    neg2 = (grad > 0) * (x_curr >= 1 - gap)
    neg3 = x_curr <= 0
    neg4 = x_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(b, 1, -1)
    g = grad.view(b, 1, -1).clone()
    g[u] = 0
    kval = _kth_largest(g.abs().float(), k, dim=2).unsqueeze(1)
    k_hot = (g.abs() >= kval).float() * g.sign()
    return k_hot.view(b, c, p, p)


def _proj_simplex(v, s):
    b = v.shape[0]
    u = v.view(b, 1, -1)
    n = u.shape[2]
    u, _ = torch.sort(u, descending=True)
    cssv = u.cumsum(dim=2)
    vec = u * torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    comp = (vec > (cssv - s)).to(v.dtype)
    uu = comp.cumsum(dim=2)
    w = (comp - 1).cumsum(dim=2)
    uu = uu + w
    rho = torch.argmax(uu, dim=2).view(b)
    c = torch.stack([cssv[i, 0, rho[i]] for i in range(b)]).to(v.device) - s
    theta = (c / (rho.to(v.dtype) + 1)).view(b, 1, 1, 1)
    return (v - theta).clamp(min=0)


def _proj_l1ball(x, eps):
    u = x.abs()
    if (u.view(u.shape[0], -1).sum(dim=1) <= eps).all():
        return x
    y = _proj_simplex(u, s=eps).view_as(x)
    return y * x.sign()


def msd_v0(model, x, y, eps_linf, eps_l2, eps_l1,
           alpha_linf=0.003, alpha_l2=0.05, alpha_l1=0.05, steps=50):
    """MSD (multi steepest descent) training attack — faithful fp32 port of
    locuslab/robust_union CIFAR10/cifar_funcs.py::msd_v0 at commit ef34194
    (2024-07-16). Per step: ONE gradient, three candidate steps (l2/linf/l1),
    keep per-sample the candidate with max loss (3 extra forwards). Their
    defaults kept: alphas (0.003, 0.05, 0.05), steps=50, k~U{5..20} for l1.
    """
    was_training = model.training
    model.eval()
    delta = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            d = delta.detach()
            # l2 candidate
            d_l2 = d + alpha_l2 * grad / _l2n(grad).clamp(min=1e-12)
            d_l2 = d_l2 * (eps_l2 / _l2n(d_l2).clamp(min=eps_l2))
            d_l2 = torch.min(torch.max(d_l2, -x), 1 - x)
            # linf candidate
            d_linf = (d + alpha_linf * grad.sign()).clamp(-eps_linf, eps_linf)
            d_linf = torch.min(torch.max(d_linf, -x), 1 - x)
            # l1 candidate (their in-loop k/alpha schedule)
            kk = int(torch.randint(5, 21, (1,)).item())
            a1 = (alpha_l1 / kk) * 20.0
            d_l1 = d + a1 * _l1_dir_topk(grad, d, x, a1, kk)
            d_l1 = _proj_l1ball(d_l1, eps_l1)
            d_l1 = torch.min(torch.max(d_l1, -x), 1 - x)
            # pick per-sample max-loss candidate
            best = d.clone()
            best_loss = torch.full((x.shape[0],), -1.0, device=x.device)
            for cand in (d_l1, d_l2, d_linf):
                lc = nn.functional.cross_entropy(model(x + cand), y, reduction="none")
                m = lc >= best_loss
                best[m] = cand[m]
                best_loss = torch.max(best_loss, lc)
            delta = best
    if was_training:
        model.train()
    return (x + delta).detach()


def pgd_l1_topk(model, x, y, eps, step_size, steps, k=20, gap=0.05,
                random_k=True, random_start=False):
    """L1 top-k PGD (sparse steepest descent + L1-ball projection).

    Follows robust_union's pgd_l1_topk: each step keeps the top-k |grad|
    coordinates, steps in their sign, and projects the perturbation back onto
    the L1 ball of radius eps. `random_k` picks k in [5,20] per step (their
    schedule) which diversifies sparsity; set False for a fixed k.
    """
    was_training = model.training
    model.eval()
    delta = torch.zeros_like(x)
    if random_start:
        delta = (2.0 * torch.rand_like(x) - 1.0) * eps
        delta = delta / _l1n(delta).clamp(min=eps)
        delta = torch.min(torch.max(delta, -x), 1 - x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            kk = int(torch.randint(5, 21, (1,)).item()) if random_k else k
            alpha = step_size / kk * 20.0
            delta = delta + alpha * _l1_dir_topk(grad, delta, x, gap, kk)
            if (_l1n(delta) > eps).any():
                delta = _proj_l1ball(delta, eps)
            delta = torch.min(torch.max(delta, -x), 1 - x)
    if was_training:
        model.train()
    return (x + delta).detach()

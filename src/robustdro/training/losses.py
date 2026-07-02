"""Training objectives for union adversarial training.

GroupDRO (attacks-as-groups) — the AttackDRO++ mechanism ported from the thesis
(`ardg` GroupDRO), generalized to the (linf, l2, l1) union. Each source attack is
a group; a difficulty-aware weight q_g is raised on whichever attack is currently
hardest, and the batch loss is the q-weighted sum of per-group CE losses:

    L = sum_g q_g * CE(f(A_g(x)), y),
    log_q[g] <- log_q[g] + eta_q * detach(CE_g),   q = softmax(log_q).

q is detached in the loss (gradients flow only through the CE terms). log_q starts
at 0 (uniform q). Softmax is shift-invariant, so unbounded log_q accumulation is
numerically safe.
"""

from __future__ import annotations

import torch


class GroupDRO:
    def __init__(self, num_groups: int, eta_q: float, device="cpu"):
        self.num_groups = num_groups
        self.eta_q = float(eta_q)
        # log_q = 0 -> softmax gives the uniform distribution.
        self.log_q = torch.zeros(num_groups, dtype=torch.float32, device=device)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_q, dim=0)

    def update(self, group_losses: torch.Tensor) -> torch.Tensor:
        """Multiplicative log-space update from detached per-group losses [G]."""
        gl = group_losses.detach().to(self.log_q.dtype).to(self.log_q.device)
        self.log_q = self.log_q + self.eta_q * gl
        return self.weights()

    def set_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Set log_q directly so weights() == softmax(scores).

        Used by the binding-aware signal (CARD-3a): scores = -robust_acc_g / tau,
        so the weakest norm (lowest robust acc) gets the highest weight. This
        REPLACES the per-step loss update (a different weighting signal, not a
        running accumulation)."""
        self.log_q = scores.detach().to(self.log_q.dtype).to(self.log_q.device)
        return self.weights()

    def state_dict(self) -> dict:
        return {"log_q": self.log_q.detach().cpu(), "eta_q": self.eta_q}

    def load_state_dict(self, sd: dict) -> None:
        self.log_q = sd["log_q"].to(self.log_q.device)
        self.eta_q = sd["eta_q"]

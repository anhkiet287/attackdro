# AttackDRO

AttackDRO studies compute-efficient predictive versus static allocation for multi-norm adversarial training on CIFAR-10 with PreActResNet-18 and ℓ∞/ℓ2/ℓ1 threats.

Start here:

- [`docs/STATE.md`](docs/STATE.md) — current-state authority and active gates.
- [`docs/REPO_MAP.md`](docs/REPO_MAP.md) — repository index.
- [`results/MANIFEST.md`](results/MANIFEST.md) — run retention and canonical-status index.
- [`docs/governance/`](docs/governance/) — research charter, roles, and workflow.
- [`docs/STATE_LOG.md`](docs/STATE_LOG.md) — append-only history, not routine state.

Current decision-grade evaluation uses APGD 20/20/100 on `val_select`; final full AutoAttack is reserved for an explicitly frozen winner.

# AttackDRO++ — worst-case robustness over the union of (ℓ∞, ℓ2, ℓ1)

Difficulty-aware weighting for **worst-case union adversarial robustness** on
CIFAR-10 / PreActResNet-18. A sample counts robust only if it survives APGD in
**every** norm (per-sample AND). Locked protocol: ε = (ℓ∞ 0.03, ℓ2 0.5, ℓ1 12),
train == eval; see [`configs/base.yaml`](configs/base.yaml).

**Project state:** read [`PLAYBOOK.md`](PLAYBOOK.md) →
[`docs/MEMORY.md`](docs/MEMORY.md) → tail of [`docs/LOG.md`](docs/LOG.md), in
that order. The local dashboard is [`docs/dashboard.html`](docs/dashboard.html),
regenerated from `docs/MEMORY.md`, `docs/LOG.md`, and `results/*.json` by:

```bash
.venv/bin/python scripts/make_dashboard.py
```

## Layout
- `src/robustdro/` — attacks (`norms.py`), eval (`eval_union.py` + AutoAttack), training (`groupdro.py`), models.
- `scripts/` — `train.py`, `evaluate.py`, `eval_standard_pack.py`, `validate_baseline.py`, `make_experiment_table.py`, `figures/make_all.py`, `run_final_pipeline.sh`. Dev/diagnostic tools in `scripts/dev/`.
- `configs/` — `base.yaml` + one file per paper method. `docs/` — compact memory/log/dashboard plus `archive/`. `external/` — RAMP + robust_union checkouts (baselines).

## Reproduce one number end-to-end (MSD baseline, worst-∪ = 42.5)
```bash
python -m venv .venv && source .venv/bin/activate      # PyTorch cu128, see SETUP.md
pip install -r requirements.txt
python scripts/evaluate.py \
  --config configs/base.yaml \
  --checkpoint external/robust_union/CIFAR10/Selected/MSD.pt \
  -n 1000 --version apgd --out results/eval_msd_check.json
# -> metrics.worst_union_acc ≈ 0.425  (tier-2 row in EXPERIMENT_TABLE.md)
```
Setup (WSL2 + GPU): [`SETUP.md`](SETUP.md). Operating rules start at [`PLAYBOOK.md`](PLAYBOOK.md).

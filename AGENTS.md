# Agent Contract

1. Source of truth: read `docs/PROJECT_STATE.md` before any work. `docs/dashboard.html` is a generated view; if it disagrees with `PROJECT_STATE.md`, regenerate the dashboard. `configs/legacy/`, `docs/legacy/`, and `results/archive/` are provenance only.

2. Do not touch active runs: do not edit `configs/paper/`, main-run `results/`, or active writer folders; check `pgrep -af scripts/train.py` before cleanup or run orchestration. Do not launch train/eval from cleanup turns unless explicitly asked. `results/` is gitignored.

3. Coding rules: do not hardcode values parseable from config. FLOPs denominators come from `sum(train.attacks[*].steps)`. Use canonical keys `efficiency/attack_flops_ratio`, `eval/worst_union`, and `eval/robust_{linf,l2,l1}`.

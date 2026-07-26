# Codex repository rules

This file is the compact Codex entry contract. Detailed governance and history live at the pointers below.

## Read first

1. The director's current explicit instruction.
2. `docs/STATE.md` — current research state and active gates.
3. The approved task or experiment contract.
4. `docs/governance/RESEARCH_CHARTER.md` and `docs/governance/AI_LAB_ROLES.md`.
5. `docs/REPO_MAP.md` and `results/MANIFEST.md` when locating code or artifacts.

Historical state is append-only in `docs/STATE_LOG.md`; do not use it as current authority.

## Operating boundary

- Default to read-only. Mutate, launch, sync, clean, commit, or publish only when explicitly authorized.
- Use the narrowest authorized mode. A request to inspect, prepare, review, or preflight does not authorize training or evaluation.
- Stop on conflicts with `docs/STATE.md`, missing contract fields, existing output targets, active writers, or unexpected repository changes.
- The LAB-DIRECTOR owns protocol, thresholds, launches, retries, checkpoint choice, final evaluation, and claims.

## Artifact and Git safety

- Treat `results/`, `checkpoints/`, `wandb/`, `dumps/`, `logs/`, `data/`, and `external/` as protected provenance.
- Never overwrite historical artifacts or reuse an existing result directory without explicit authorization.
- Before edits or execution, inspect `git status --short`; preserve unrelated dirty work.
- Never reset, restore, clean, stash, switch destructively, commit, push, or merge unless the active contract explicitly authorizes it.

## Scientific invariants

- Canonical split roles: `train_core`, `val_select`, `test_monitor`, `test_final`.
- Checkpoint selection uses only `val_select/worst_union`; monitor, audit, and final-test results never select or tune.
- Canonical develop comparison is APGD 20/20/100, one restart, n=1000, eps 8/255, 0.5, 12, with grade-equivalent checkpoints and splits.
- Parse attack and compute settings from resolved configs. Report all attack calls, steps, refreshes, probes, and extra work.
- Main training requires its approved W&B policy; never silently fall back or sync/login without authorization.

## Execution gate

Before any authorized writer operation, confirm exact command/config/output behavior, active processes, environment, split/checkpoint protocol, applicable checker/smoke/preflight passes, and W&B mode. On failure, stop; do not auto-retry or change parameters.

At completion report commands and exits, files changed, artifacts, metrics, blockers, protected-path safety, and whether training/eval/audit/sync/dashboard operations occurred.

# Repository Agent Contract

This file governs agent behavior in this repository. It is standing governance,
not experiment status. Current status belongs in `docs/STATE.md`.

## 1. Authority and Source of Truth
Always read `docs/STATE.md` before work.

Standing reference order:

1. explicit current instruction from Kiet;
2. `docs/STATE.md`;
3. approved active task or experiment contract;
4. root `AGENTS.md`;
5. current non-legacy code, configs, and docs;
6. legacy material.

Kiet's explicit current instruction defines what actions are permitted.
`docs/STATE.md` and the approved active contract define the locked
gates, protocol, and conditions under which the action is valid.

If a current instruction requests an action blocked by `STATE.md` or
the approved contract, do not silently treat it as an override. Stop and report
the exact conflict. Proceed only if Kiet explicitly overrides that exact gate or
updates the governing contract/state.

An instruction to execute an action does not silently redefine scientific
history, locked protocol, evaluation grade, or previous gate outcome.

`docs/dashboard.html` is a generated view. If it disagrees with
`STATE.md`, report the mismatch; do not regenerate it unless explicitly
authorized. Treat `configs/legacy/`, `docs/legacy/`, and `results/archive/` as
provenance unless Kiet explicitly says otherwise.

## 2. Default Read-Only Mode
The default mode is read-only.

Without explicit authorization, agents may inspect files, summarize evidence,
compare configs, run non-mutating discovery commands, and report findings.

Without explicit authorization, agents must not edit files, create files,
delete files, train, eval, audit, sync W&B, install dependencies, clean outputs,
commit, push, stash, reset, restore, switch branches, or regenerate generated
artifacts. Silence is not authorization.

## 3. Human Decision Rights
Kiet owns scientific and project decisions.

Do not make or silently change decisions about research claims, paper framing,
locked protocols, attack suites, thresholds, gates, pass/fail criteria,
checkpoint selection, baseline inclusion, final evaluation, new training runs,
changed-parameter reruns, protocol-mismatch recovery, W&B publication, or
destructive cleanup.

If the active task contract already resolves a decision, follow it without
asking again. If a decision is unresolved, would expand scope, would change
protocol, or would alter scientific interpretation, stop and report the
decision required. Do not repeatedly request confirmation for details already
explicitly authorized.

## 4. Task Modes
Use the narrowest mode authorized by the current request.

- Read-only governance/audit mode: inspect and report only; do not write,
  launch, sync, or clean; if repository state changes unexpectedly, stop.
- Scoped implementation mode: edit only explicitly allowed files; create
  support files only when their path/category is explicitly allowed; do not
  train, eval, audit, sync, launch, or opportunistically refactor.
- Experiment-operator mode: requires an approved run/eval contract with exact
  config path, output path, command, and gates; permits only that run or batch.
- Cleanup mode: requires explicit cleanup authorization; never launches train,
  eval, or audit unless separately authorized; never deletes active writers or
  historical artifacts.

## 5. Authorization and Scope Boundaries
A task authorizes only the actions it names.

If a request says "prepare", "inspect", "review", "draft", "preflight", or
"read-only", do not launch training, evaluation, audit, W&B sync, or final
claim evaluation. If a request authorizes one run, do not launch a second run.
If it authorizes one config or result path, do not substitute another without
explicit approval.

If a target artifact already exists, stop and report unless the task explicitly
authorizes overwrite or resume behavior.

New checkers, smoke tests, configs, docs, or derived outputs are allowed only
when explicitly named or when the allowed path/category is stated in the
contract. "Directly required" is not permission for opportunistic file
creation. If a required support file is outside the allowed set, stop and
report the needed scope expansion.

## 6. Git and User-Work Safety
The worktree may contain user or prior-agent changes. Treat all existing dirty
or untracked files as user work unless you created them in the current task.

A dirty worktree is not automatically a blocker. It becomes a blocker when
authorized edits overlap existing changes that cannot be safely preserved,
launch provenance cannot be determined, the active contract requires a clean
tree, or unexpected changes appear during the task.

Never clean the tree merely to satisfy a gate. Before file edits or run
orchestration, inspect `git status --short`.

Never run destructive Git commands unless Kiet explicitly requests them:
`git reset`, `git reset --hard`, `git checkout --`, `git restore`, `git clean`,
`git stash`, or branch switches that would overwrite local work. Do not commit,
tag, push, or open a PR unless explicitly requested.

If required changes overlap dirty files, preserve unrelated user changes. If
that is not possible, stop and ask. After authorized edits, report changed files
and whether unrelated dirty files were present before the task.

## 7. High-Risk Files
Treat these paths as high-risk: `docs/STATE.md`, `docs/STATE_LOG.md`,
`docs/dashboard.html`, `configs/paper/**`, `configs/eval/**`,
`scripts/train.py`, `scripts/evaluate.py`, `scripts/post_train_develop_eval.py`,
`scripts/preflight_check.py`, `scripts/eval_multinorm_audit.py`,
`src/robustdro/training/**`, `src/robustdro/data/**`,
`src/robustdro/eval/**`, `src/robustdro/attacks/**`,
`src/robustdro/utils/io.py`, `src/robustdro/utils/wandb_log.py`, dependency and
environment files, `external/RAMP/**`, active and historical artifacts under
`results/**`, and W&B offline or synced directories.

High-risk modification requires a scoped objective, allowed files, forbidden
changes, acceptance checks, and explicit launch prohibition unless launch is
separately authorized. Do not edit active writer folders.

## 8. Result and Artifact Immutability
Historical results are protected even when gitignored.

Do not overwrite or mutate existing checkpoints, `train.json`, eval JSONs,
audit JSONs, mask sidecars, subset files, W&B run directories, logs, generated
result summaries, or external RAMP artifacts.

Do not reuse an existing target result directory for a new run. Derived outputs
must use a new explicitly named path. If a derived path already exists, stop and
report unless overwrite is explicitly approved. Gitignored does not mean
disposable.

## 9. Data-Split and Selection Invariants
Preserve canonical CIFAR-10 split roles unless an approved protocol says
otherwise:

- `train_core`: 49k training examples;
- `val_select`: 1k validation-selection examples;
- `test_monitor`: fixed monitor-only test subset;
- `test_final`: final-only test set.

Checkpoint selection must use `val_select/worst_union` unless an approved
protocol explicitly defines a different selection metric. Do not use
`test_monitor`, audit metrics, sidecar-derived metrics, or final test metrics
for checkpoint selection.

Do not tune thresholds, attack choices, schedules, or method choices after
looking at monitor, audit, or final-test results. JSON fields named
`used_for_selection` or `selected_by_test` must match the actual protocol.

## 10. Canonical Comparison Contract
Direct method comparisons require grade-equivalent evaluation.

Canonical develop-eval comparison grade: split `val_select`, checkpoint role
`val_best`, APGD, n = 1000, steps Linf/L2/L1 = 20/20/100, eps Linf/L2/L1 =
8/255, 0.5, 12, and `train_eval_eps_mismatch=false`.

Report clean accuracy, robust Linf, robust L2, robust L1, worst-union accuracy,
checkpoint role, split, n, attack steps, epsilons, and compute ratio. Do not
compare non-equivalent grades as if they were equivalent.

Full standard AutoAttack final evaluation requires explicit frozen-winner or
final-claim authorization.

## 11. Attack and Compute Accounting
Do not hardcode values that can be parsed from config.

For training compute, use training-time attack settings, not eval or audit
settings. Distinguish training, develop-eval, audit, and final-eval attack
steps.

The attack FLOPs denominator comes from `sum(train.attacks[*].steps)` unless an
approved config defines an equivalent structured source of truth.

Report `efficiency/attack_flops_ratio` when available. If compute accounting is
missing or ambiguous, stop the compute analysis and report the missing fields.
Do not hide extra attacks, probes, refreshes, restarts, or calibration passes.
Compute must be parsed from resolved config and actual attack work.

## 12. Optimizer and Scheduler Semantics
Optimizer, backward, gradient accumulation, scheduler, EMA, and allocation
semantics must match the approved contract for the run.

Do not assume one optimizer step per batch unless the approved protocol and
code confirm it.

When changing loss or allocation behavior, verify optimizer stepping,
scheduler stepping, attack-budget accounting, logging keys, and checkpoint
resume state. If semantics are ambiguous, stop and report before launch.

## 13. Pre-Launch Gates
Before any authorized training, eval, audit, W&B sync, or cleanup operation,
perform the applicable gates: read `docs/STATE.md`; check
`git status --short`; confirm exact command, config path, output path, and
target output behavior; check active processes; parse config rather than memory;
confirm dataset, model, split, attack protocol, checkpoint role, and selection
rule; run applicable preflight/config/smoke/split checks; confirm environment,
`.venv/bin/python` usage, W&B mode, and sync intent.

The standard relevant-writer check is:

`pgrep -af '[s]cripts/(train|evaluate|post_train_develop_eval|eval_multinorm_audit).py' || true`

Adapt the process check for other exact approved entrypoints. External RAMP may
use a different command and must be checked separately. If another relevant
writer is active, do not launch unless the active contract explicitly permits
concurrency.

Run applicable config guards, smoke tests, split checks, and preflight tools
required by the active contract. If an expected utility exists and is
applicable, it must pass before launch. If no applicable utility exists, report
that limitation rather than inventing a substitute. A successful checker proves
only the properties it actually checks.

If any gate fails, stop and report. Do not patch around the gate unless Kiet
explicitly authorizes a new task.

## 14. No Automatic Recovery
If training, eval, audit, sync, or validation fails, stop and report.

Do not automatically retry with changed config, seed, checkpoint, batch size,
attack steps, restarts, epsilons, schedule, threshold, subset, model
normalization, or output path.

Do not silently resume, repair, or reinterpret failed results. A rerun with
changed parameters requires explicit approval and documentation.

## 15. W&B Rules
Classify every execution operation as exactly one of: `MAIN_RUN`,
`SMOKE_OR_VALIDATION`, `DIAGNOSTIC_OR_DEBUG`, `EVAL_OR_AUDIT`, `W&B_SYNC`, or
`OTHER`.

`MAIN_RUN` means a main training run and must use W&B mode `online`. Before
launch, verify W&B authentication/login availability without automatic login,
and check expected project, entity, run name, and tags against the resolved
config or approved contract. If online logging is unavailable, the operation is
blocked.

Do not silently fall back from online to offline, disabled, stdout-only, or
local-only logging. A main run without the required online W&B record is not
fully contract-compliant, even when local checkpoints and JSON artifacts exist.

`SMOKE_OR_VALIDATION` and `DIAGNOSTIC_OR_DEBUG` may use offline or disabled
mode only when the active contract specifies it, and must not create a
misleading main-run W&B record. `EVAL_OR_AUDIT` must use the exact W&B mode in
the approved contract; do not infer it from the associated training run.

`W&B_SYNC` always requires explicit authorization and is not part of a main run
unless explicitly included. Do not run `wandb login`, W&B sync, or online
publication unless explicitly authorized. If W&B login is missing, report the
blocker; do not attempt interactive login. Do not delete local W&B offline
directories.

W&B mode must not be changed automatically after approval. Report W&B contract
failure separately from scientific execution failure and local artifact success.
When syncing is authorized, record offline path, command, exit code, status,
and URL if available.

## 16. Environment and Execution Location
Use `.venv/bin/python` for repository Python commands unless the task
explicitly authorizes another interpreter.

Do not install or upgrade dependencies unless explicitly authorized. Report
environment blockers rather than working around them silently.

Be careful with execution under `/mnt/c`; large training and data workloads may
have filesystem-performance implications. Follow `SETUP.md` and the approved
task contract for execution location.

Distinguish local sandbox limitations from actual host limitations when
reporting GPU, CUDA, filesystem, or network availability.

## 17. External RAMP Safety
Treat `external/RAMP/` as external provenance and high-risk code.

Do not modify external RAMP code, checkpoints, logs, or outputs unless
explicitly authorized. Before running any RAMP script, inspect its output and
deletion behavior; some scripts may delete or rewrite files in their target
save directory.

RAMP reproduce gates must be grade-matched to the stated reference metric. Do
not substitute internal audit metrics for a reference-grade reproduce gate. If
a RAMP reproduce gate fails, stop and report; do not report downstream audit
results as scientific results until the gate is resolved.

## 18. Documentation Updates
Documentation updates are not implicitly authorized merely because a run, eval,
audit, or sync was authorized.

The active task or experiment contract must state whether a
`docs/STATE.md` update is included. If included, update it after
verified artifacts are available and before any authorized dashboard
regeneration. If not included, do not modify `STATE.md`; report that
synchronization remains pending.

Dashboard regeneration always requires explicit authorization. Do not manually
edit dashboard truth. `docs/dashboard.html` must not carry independent protocol
truth. Do not use legacy docs as active state unless `STATE.md` or Kiet
says to.

When state updates are authorized, record command/action, artifact paths,
metrics or gate result, safety constraints, caveats, and whether any expected
action was intentionally not run.

## 19. Scientific Reporting Discipline
Separate direct observations, supported inferences, and unsupported claims.

Do not overclaim from small samples, monitor-only data, audit-only data, or
non-equivalent eval grades.

Always report enough protocol detail for review: config path, checkpoint path
and role, split, n examples, attack version, attack steps, epsilons, clean and
robust metrics, union metric, compute ratio, skipped or missing optional
attacks, selection source, and known caveats.

If a result is exploratory, repaired, monitor-only, audit-only, or not
grade-equivalent, label it that way.

## 20. Required Completion Report
At the end of a task, report operating mode, commands run and exit codes, files
read/changed/created/left untouched, hashes when integrity matters, artifacts
generated or verified, key metrics or validation results, blockers and
unresolved risks, safety confirmations, and whether training, eval, audit, W&B
sync, dashboard regeneration, or result mutation occurred.

If no files were changed, say so explicitly. If repository state changed
unexpectedly during the task, stop and report the change before continuing.

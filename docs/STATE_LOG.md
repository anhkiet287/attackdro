> Append-only history. Current state -> `STATE.md`.

# STATE_LOG.md — historical project record

## 0. Source-of-truth / gate rules

- Last sync: 2026-07-10 14:23 UTC.
- This file is the single active agent state file for Codex, ChatGPT, Claude, and Kiet.
- After every run/eval/frontier update, update this file first. The dashboard must be regenerated from this file plus result JSONs. If dashboard and this file disagree, this file is the source of truth and dashboard must be regenerated.
- `docs/dashboard.html` is the human-facing dashboard; it must not carry independent protocol truth.
- Historical notes may live in `docs/legacy/`, but legacy Markdown must not be used as active planning/state.
- Do not train, evaluate, clean results, or change configs from sync/cleanup mode unless Kiet explicitly asks.
- Research context in section 9 is advisory only; it cannot block runs or override the locked protocol, run queue, or experiment table.

## 1. Quick Context Snapshot

- Project: Predictive Allocation for Reliable Multi-Source Adversarial Training on CIFAR-10 / PreActResNet-18, reusing the existing multi-norm linf/l2/l1 AT asset base.
- Claim target: deployment-cost multi-source robust training with mandatory reliability audit. The previous accuracy-per-attack-FLOP framing under RAMP80 is retained as an asset/baseline lineage, but is no longer the main story by itself.
- Main comparator/control: B1 full/reactive multi-source internal control. RAMP is a strong multi-norm reference/motivation point, not the enemy comparator.
- Locked train recipe: RAMP80, APGD train 10/10/10, lr 0.05 -> 0.005 at epoch 70.
- Split: CIFAR train 49k train_core + 1k val_select; test_monitor is fixed 1k diagnostic; test_final is full 10k final-only.
- Checkpoint rule: val_best selected only by val_select/worst_union; always eval val_best and last on val_select and test_monitor.
- Active batch: seed-0 full80 method screen.
- Active run status: Reactive Soft-T Full-10 seed 0 completed locally in `results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0/`; `train.json` has 80 records through stored epoch 79, with `ckpt/ep080.pt`, `ckpt/last.pt`, and `ckpt/val_best.pt` present. Provenance check: `ckpt/val_best.pt` was selected at stored epoch 71 by `val_select/worst_union = 0.4490000009536743`; `ckpt/best.pt` is a byte-distinct legacy alias with identical loaded checkpoint content and must not be used for upcoming eval/audit. Post-train develop eval and multinorm audit v1 are complete for reactive `val_best` and `last`.
- B4 Route A-Diagnostic one-run contract completed on 2026-07-10 in `results/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k/s0/`; no further B4 run is authorized without a renewed protocol gate. Canonical APGD 20/20/100 `val_select/val_best` union is 0.4230000079 at realized compute ratio 0.4000, so this is B1/B3-parity diagnostic evidence rather than a canonical strong-positive predictive result.
- Active methods: Reactive Soft-T Full-10, Reactive Soft-T Flat-6, CARD-PB v2 kspan-4, Reactive Soft-T Curriculum 3-6-10, Reactive Soft-T Fail-Rate 50%.
- Dashboard: generated from PROJECT_STATE.md + result JSONs.
- Do not add ablations, early LR drop, extra protocols, or test-based selection unless Kiet explicitly unlocks them.

## 2. Current phase and allowed next actions

- Current phase: seed-0 full80 method screen under locked RAMP80 protocol.
- Strategic scope lock: Pre-registration v1 for **Predictive Allocation for Reliable Multi-Source AT** has critic PASS. Primary success is **Deployment Success only**; audit is mandatory reliability layer.
- RQ0/RQ1 are cleared as infrastructure/evaluation-only tasks, but must not write into or disturb result directories. Reactive Soft-T Full-10 completion/provenance is now confirmed; RQ0/RQ1 must still use isolated audit/eval output paths and must not mutate the completed run directory.
- RQ2+ training is blocked until analytic compute model predicts B4 saving >=50% before runtime overhead; if <50%, stop and patch refresh R, then gate again.
- Paper lane stays clean: RAMP recipe, APGD train 10/10/10, decision eval APGD 20/20/100, final full AutoAttack only for the selected winner/final claim.
- Exploration ideas may inform decisions, but must not contaminate paper configs/results.
- W&B preflight passed on 2026-07-08 for the three idea-compare names: `idea1_failrate_ramp80_apgd_8255_s0`, `idea2_kmin_recovery_ramp80_apgd_8255_s0`, `idea3_curriculum_ramp80_apgd_8255_s0`.
- Idea1 fail-rate completed to `results/idea1_failrate_ramp80_apgd_8255/s0/` with `ckpt/ep080.pt`, `ckpt/last.pt`, `train.json`, and `eval.json`; do not overwrite these files.
- Current Reactive Soft-T Full-10 post-train develop eval is complete; canonical eval references are `ckpt/val_best.pt` for `checkpoint_role=val_best` and `ckpt/last.pt` for `checkpoint_role=last`. Do not use `ckpt/best.pt`.
- Pre-registration placeholders for threshold decisions must remain `__TBD__` until Kiet explicitly fills them.
- Previous missing-config blocker for the reactive-first run is resolved by `configs/paper/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k.yaml`; the single allowed baseline has launched.
- Reactive Soft-T Full-10 baseline completed in `results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0/`. Provenance sync at 2026-07-09 16:52 UTC found 80 train records through stored epoch 79, `ckpt/ep080.pt`, `ckpt/last.pt`, `ckpt/val_best.pt`, and `ckpt/best.pt`.
- Checkpoint provenance: `run_meta.json` confirms train_core 49k, val_select 1k, selection metric `val_select/worst_union`, APGD train steps 10/10/10, and `train.json` confirms canonical `val_select/*`, `test_monitor/*`, `efficiency/attack_flops_ratio = 1.0`, `test_monitor/used_for_selection = false`, and lr 0.005 after the epoch-70 milestone. `ckpt/val_best.pt` selected stored epoch 71 by `val_select/worst_union = 0.4490000009536743` with clean/linf/l2/l1 = 0.834/0.453000009059906/0.6890000104904175/0.6019999980926514.
- `ckpt/best.pt` provenance: code path in `src/robustdro/training/groupdro.py` saves `val_best` and then `best` from the same `val_select/worst_union` event; loaded checkpoint content matches `val_best.pt` for epoch, selection metadata, model, optimizer, scheduler, DRO state, cfg, and RNG. Its SHA256 differs because it was serialized as a separate legacy alias archive. Classification: legacy alias of `val_best.pt`, not clean-best, not test-selected, not stale earlier val-best, and not unknown.
- Reactive Soft-T Full-10 post-train develop eval complete at APGD 20/20/100 r1 n1000:
  - `eval_val_select_val_best.json`: clean/linf/l2/l1/union = 0.8339999914169312/0.43299999833106995/0.6729999780654907/0.49300000071525574/0.4230000078678131, `used_for_selection=true`.
  - `eval_val_select_last.json`: clean/linf/l2/l1/union = 0.8270000219345093/0.39500001072883606/0.6740000247955322/0.4830000102519989/0.3880000114440918, `used_for_selection=false`.
  - `eval_test_monitor_val_best.json`: clean/linf/l2/l1/union = 0.8240000009536743/0.4309999942779541/0.6629999876022339/0.5040000081062317/0.421999990940094, monitor-only.
  - `eval_test_monitor_last.json`: clean/linf/l2/l1/union = 0.8320000171661377/0.4000000059604645/0.6650000214576721/0.4869999885559082/0.39500001072883606, monitor-only.
- Idea1 Fail-Rate develop-eval repair complete at APGD 20/20/100 r1 n1000. `ckpt/val_best.pt` is absent, so legacy `ckpt/best.pt` was mapped to `checkpoint_role=val_best`; historical `eval.json` and `train.json` were not overwritten.
  - `eval_val_select_val_best.json`: clean/linf/l2/l1/union = 0.8069999814033508/0.421999990940094/0.6660000085830688/0.49900001287460327/0.4099999964237213, `used_for_selection=true`.
  - `eval_val_select_last.json`: clean/linf/l2/l1/union = 0.9810000061988831/0.34200000762939453/0.8149999976158142/0.5649999976158142/0.3370000123977661, `used_for_selection=false`.
  - `eval_test_monitor_val_best.json`: clean/linf/l2/l1/union = 0.7599999904632568/0.40400001406669617/0.5899999737739563/0.4569999873638153/0.38999998569488525, monitor-only.
  - `eval_test_monitor_last.json`: clean/linf/l2/l1/union = 0.871999979019165/0.26100000739097595/0.6100000143051147/0.4269999861717224/0.2540000081062317, monitor-only.
- RAMP Arm A reproduce gate PASS before RAMP audit reporting. Gate used existing grade-matched decision APGD 20/20/100 r1 n1000 result `results/ramp/eval_armA_rampfull_ep80_apgd_2020100_n1000.json`; target union 0.448, measured union 0.4480000138282776, absolute difference 1.3828277578564752e-08.
- Multinorm audit v1 complete with config `configs/eval/audit_cifar10_preactrn18_multinorm_v1.yaml`; fixed subset `results/audit/subsets/cifar10_test_1000_seed20260709.json`; no attacks skipped and no masking caveats for all three completed audits.
  - `results/audit/reactive_softT_full10_s0/val_best/audit_multinorm_v1.json`: clean 0.8339999914169312, primary APGD linf/l2/l1/union = 0.43799999356269836/0.6880000233650208/0.5360000133514404/0.43299999833106995, full_audit_union = 0.4129999876022339, audit gaps linf/l2/l1 = 0.023000001907348633/0.01100003719329834/0.016000032424926758.
  - `results/audit/reactive_softT_full10_s0/last/audit_multinorm_v1.json`: clean 0.8399999737739563, primary APGD linf/l2/l1/union = 0.4090000092983246/0.6830000281333923/0.5109999775886536/0.4020000100135803, full_audit_union = 0.38600000739097595, audit gaps linf/l2/l1 = 0.018000006675720215/0.013000011444091797/0.014999985694885254.
  - `results/audit/ramp_armA_ep80/final/audit_multinorm_v1.json`: clean 0.8130000233650208, primary APGD linf/l2/l1/union = 0.4909999966621399/0.6660000085830688/0.5239999890327454/0.48100000619888306, full_audit_union = 0.4560000002384186, audit gaps linf/l2/l1 = 0.02399998903274536/0.016000032424926758/0.02399998903274536.
- No checkpoint selection used `test_monitor` or audit results. No thresholds, method choices, pass/fail criteria, or audit-suite settings were tuned from these results. Dashboard regeneration remains deferred.
- Standalone deterministic gatekeeper `scripts/preflight_check.py` exists as a read-only PASS/FAIL utility. It is not wired into `scripts/train.py`; wiring is deferred until after the current Reactive Soft-T Full-10 baseline completes.

### Post-B1 analysis batch (2026-07-09, no training / no B2/B4 launch)

This batch ran no GPU eval — all reactive/idea1 develop evals, the RAMP reproduce gate, and the three Tier 1 audits already existed and were re-verified read-only (metrics unchanged; consistent). New derived artifacts only.

- RAMP reproduce gate (re-confirmed): canonical grade = decision APGD 20/20/100 r1 n1000 (Option A), record `results/ramp/eval_armA_rampfull_ep80_apgd_2020100_n1000.json`, target union 0.448 ± 1pp, measured 0.4480000138, |diff| ≈ 1.38e-8 → **PASS**. This is NOT the audit `primary_apgd_union` (which is APGD-CE-only 100/100/100 = 0.481). Secondary cross-checks consistent: RAMP's own full-AA log (`external/RAMP/.../log_eval_final.txt`) union 44.7 / per-norm 46.1/65.9/48.8; our multinorm `full_audit_union` on RAMP ep80 = 0.456 (|diff| 0.9pp vs 44.7). `reproduce_gate_metric = decision_apgd_20_20_100_worst_union`.
- Analytic compute model (Step 5) → `results/analysis/analytic_compute_model_pre_b4.json`. Read from config (base_ramp_apgd_8255 train.attacks = APGD 10/10/10 r1, inherited by reactive/idea1). B1 = 30 step-units/batch (ratio 1.0, all 3 sources/batch; efficiency ratio 1.0 confirms). B2 static = 10 units/batch (one full source/batch), ratio 0.333, saving 66.7% (resolved). B4 predictive v1 (R=10, S=30, no cheap probe): avg = ((R−1)·10 + 30)/R = 12 units/batch, ratio 0.400, **predicted saving 60.0%**; theoretical max (R→∞) 66.7%; starvation floor redirects selection only (no compute change). **≥50% gate: PASS (10 pp margin)**; no R/refresh patch required. Values derived from config, not the prompt; DO NOT launch B4/B2 from this PASS.
- Tier 1 CI review (Step 6) → `results/analysis/tier1_ci_review_v1.json` (read existing bootstrap CIs; not recomputed). All three checkpoints (reactive val_best, reactive last, RAMP final): audit_gap_linf, audit_gap_l2, audit_gap_l1 95% CIs all **exclude zero** (extra AA components find genuinely more adversarials at every norm). full_audit_union 95% CIs: reactive val_best [0.382, 0.443], reactive last [0.354, 0.417], RAMP final [0.422, 0.488].
- Tier 1 coverage baseline (Step 7): **BLOCKED — per-sample robust masks are NOT persisted** by `scripts/eval_multinorm_audit.py` (masks are computed in-memory in `build_result` but only aggregate `robust_acc`/unions/gaps/CIs are written to the audit JSON; `per_attack` entries lack any mask field). No coverage JSON written; no fake coverage metrics computed. Minimal future patch: export per-attack boolean masks (packed bitset or `.npz` sidecar keyed by attack name) so unique_fail_count/rate, marginal_union_drop, and Jaccard are recoverable.
- idea1 last.pt provenance (Step 8) → `results/analysis/idea1_last_provenance_v1.json`. sha256 last.pt `ec613f80…`, best.pt `2158f383…`, ep080.pt `1d6c5cda…` (last.pt is the lighter weights-only final-epoch save; not byte-identical to ep080.pt but same epoch-79 state). Verdict: **expected final-epoch checkpoint, natural robust-overfit / allocation drift — not corruption, not wrong file.** In-training probe worst_union peaks 0.420 @ep26 (val-best point) then decays monotonically to 0.183 @ep79 while probe clean rises 0.763→0.873, accelerating after the milestone-70 LR step; `exp/steps_*` show the fail-rate controller starved linf steps (~1–4) while pinning l2/l1 near 10, consistent with linf-driven union collapse. Checkpoints unmodified; last.pt **not** used for calibration.
- Tier 2 Option A setup / dry-run only (Step 9): added `configs/eval/audit_cifar10_preactrn18_multiattack_v1.yaml` (L2-only: cw_l2 + ddn_l2; success = misclassification AND ‖δ‖₂ ≤ 0.5, **min-norm-then-threshold** required) and standalone checker `scripts/dev/check_multiattack_audit_config.py`. Scope caveat recorded: "Tier 2 v1 adds independent multi-attack checks only for L2; Linf and L1 remain AA-component-only in this version." Implementation availability: **cw_l2 and ddn_l2 UNAVAILABLE** — torchattacks / foolbox / ART all missing (autoattack + robustbench present but provide neither a min-norm CW-L2 nor DDN-L2). Tier 2 eval **BLOCKED** pending an implementation asset AND an explicitly supplied/approved calibration checkpoint. No robust-checkpoint Tier 2 eval; no calibration; idea1 last.pt explicitly excluded as calibration asset. The locked Tier 1 audit harness was NOT modified.
- Batch safety confirmations: no training; no B4 launch; no B2 launch; no `train.json` mutation; no edits to train/core code or the Tier 1 audit script; no Tier 1 audit JSON mutation; no `docs/dashboard.html` regeneration; no full test_final/full-AA claim eval run (reused existing files); no `test_monitor`/audit used for checkpoint selection; no post-hoc threshold/method/audit-suite tuning.

### Tier 1 audit mask sidecar patch (2026-07-09, no audit rerun)

- Implemented opt-in mask sidecar export in `scripts/eval_multinorm_audit.py` and added read-only checker `scripts/dev/check_audit_mask_sidecar.py`.
- Sidecar schema: `masks_multinorm_v1`, default path next to the audit JSON as `masks_multinorm_v1.npz`, one boolean vector per actually run canonical attack, plus `metadata_json`.
- Mask convention: `true_means_robust_false_means_failed`; coverage analyzers must invert masks when counting failures.
- Validation rule: recompute per-attack `robust_acc`, primary APGD linf/l2/l1 and union, per-norm audit accuracies, and `full_audit_union` from masks; require abs diff <= 1e-6 against the audit JSON before marking the sidecar valid.
- Existing Tier 1 JSON status: reactive val_best, reactive last, and RAMP final JSONs contain no per-sample masks and have no sibling `masks_multinorm_v1.npz` sidecars.
- Sidecar generation status: not generated for existing Tier 1 audits. Generating these sidecars requires an explicit future audit rerun with `--export-masks`; no attack rerun was launched in this patch.
- Validation status: implementation syntax/import checks passed, checker help passed, config dry-run passed, and a synthetic `/tmp` sidecar validation passed. Existing Tier 1 sidecar validation was not run because sidecars are missing.
- Safety confirmations: no training; no B2/B4 launch; no Tier 2 eval; no existing `audit_multinorm_v1.json` mutation; no `train.json` mutation; no `docs/dashboard.html` regeneration.

### Tier 1 sidecar-only CLI patch (2026-07-09, code only / no real audit rerun)

- Added explicit `--sidecar-only` mode to `scripts/eval_multinorm_audit.py`.
- Sidecar-only CLI semantics: requires `--existing-json` and `--export-masks`; rejects `--out`; reruns attacks only when later approved; writes only `masks_multinorm_v1.npz`; validates masks against `--existing-json`; never calls `save_json` for the audit JSON.
- JSON mutation guard: sidecar-only computes SHA256 of `--existing-json` before and after sidecar generation and exits nonzero if the hash changes.
- Seed provenance behavior: sidecar-only prints original JSON top-level seed fields, original JSON per-attack seed fields including nested `params.seed`, subset seed/path/SHA256, config subset seed, config attack seeds, and rerun seed. If old JSON lacks attack seeds, provenance status is `config_inferred_seed_not_json_recorded` rather than a hard blocker when config/subset seeds are fixed.
- Validation diagnostics now include per-attack and aggregate metric diffs plus mismatch classification: `square_only_possible_stochasticity`, `apgd_or_fab_reproducibility_issue`, `union_logic_issue`, or `unknown`. The checker reports the same classification on failure.
- Synthetic `/tmp` sidecar-only writer test passed: temporary existing JSON SHA256 stayed unchanged, sidecar validation passed, and `scripts/dev/check_audit_mask_sidecar.py` passed on the synthetic sidecar.
- No real audit attacks were run. Existing Tier 1 audit JSONs and result directories were not mutated. No training, B2/B4 launch, Tier 2 eval, `train.json` mutation, or dashboard regeneration occurred.

### Tier 1 sidecar generation attempt (2026-07-10, stopped on validation failure)

- Approved real sidecar generation began for `reactive_softT_full10_s0 / val_best` using `scripts/eval_multinorm_audit.py --sidecar-only --existing-json ... --export-masks --validate-mask-sidecar` and no `--out`.
- Sandbox CUDA check reported `cuda_available=false`; the first sandbox attempt was interrupted before sidecar write. Unsandboxed CUDA probe reported `cuda_available=true`, `cuda_device_count=1`, so the target was rerun with the approved `scripts/eval_multinorm_audit.py` command prefix.
- `reactive_softT_full10_s0 / val_best` failed pre-write sidecar validation; no `masks_multinorm_v1.npz` was written. Mismatch classification: `apgd_or_fab_reproducibility_issue`.
- Significant diffs from the failed rerun:
  - `per_attack.fab_t_linf.robust_acc`: JSON 0.41999998688697815, recomputed 0.426, diff 0.00600001311302184.
  - `per_attack.fab_t_l1.robust_acc`: JSON 0.5659999847412109, recomputed 0.576, diff 0.010000015258789019.
  - `per_norm_audit.audit_acc_l1`: JSON 0.5199999809265137, recomputed 0.521, diff 0.0010000190734863468.
- The batch stopped immediately after the reactive val_best validation failure. Reactive last and RAMP final sidecar generation were not run.
- Seed provenance for the failed target: config attack seeds and subset seed were 20260709; existing JSON records nested `params.seed=20260709` for all attacks; subset path `results/audit/subsets/cifar10_test_1000_seed20260709.json`; subset SHA256 `0ad0c27b9c91aec8f87398de6224d15431a70c96e47fe9c5e1d8e8e36f3137d3`; provenance status `json_attack_seed_recorded`.
- JSON mutation check passed for all three existing audit JSONs:
  - reactive val_best: `1a2fe398e392a43cc4a3ab57fe0bbbef468ef2d5e256b9cdac7d2caf96e1bfad` before/after.
  - reactive last: `0337a4124904cb7000554678bdb999fc150a511e74481726e53168ea2b1b036c` before/after.
  - RAMP final: `a8b1a1818c55bde242a3fd3c897aad268b6c185fc9f12d3bbed44e91af6bd5f5` before/after.
- Sidecar status remains missing for all three Tier 1 targets. No coverage metrics were computed.
- Safety confirmations: no training; no B2/B4 launch; no Tier 2 eval; no `train.json` mutation; no existing audit JSON mutation; no `docs/dashboard.html` regeneration.

### FAB-T determinism diagnostic (2026-07-10, isolated / no official mutation)

- Ran isolated FAB-T-only diagnostic for `reactive_softT_full10_s0 / val_best`; output: `results/diagnostics/fab_determinism/reactive_val_best/fab_determinism_v1.json`.
- Command scope: 3 repeats each for `fab_t_linf` and `fab_t_l1` on the fixed audit subset, same checkpoint, same config, seed 20260709, batch size 250. Optional `fab_t_l2` was skipped to keep runtime bounded after the full sidecar rerun proved slow.
- Environment: `.venv`, torch 2.11.0+cu128, CUDA 12.8, device `cuda`, GPU `NVIDIA GeForce RTX 5070 Ti`, AutoAttack 0.1 from `.venv/lib/python3.12/site-packages/autoattack/`; FAB implementation `autoattack.fab_pt.FABAttack_PT` in `fab_pt.py`. Exact FAB restart/init RNG state is not exposed without invasive instrumentation.
- Classification: `fab_repeat_nondeterminism`. Repeated FAB-T runs with the same seed/config produced different robust accuracies, so the previous sidecar failure is not evidence that the entire audit harness is broken.
- Repeat robust accuracies:
  - `fab_t_linf`: 0.426, 0.424, 0.424; canonical JSON 0.41999998688697815; pairwise max diff 0.002; std 0.0009428090415820643.
  - `fab_t_l1`: 0.574, 0.578, 0.568; canonical JSON 0.5659999847412109; pairwise max diff 0.010; std 0.0041096093353126546.
- Interpretation: FAB-T varies by about 0.2pp for linf and up to 1.0pp for l1 under repeated same-seed reruns; this is attack-restart/environment uncertainty not captured by bootstrap subset CI.
- Recommended next action: A - keep old Tier 1 aggregate-only audit results as canonical, and require future audits to export masks at first run. Do not attach rerun FAB masks to old canonical JSON unless validation exactly matches.
- Safety confirmations: no training; no B2/B4 launch; no Tier 2 eval; no official `masks_multinorm_v1.npz` sidecars generated; no coverage metrics computed; no existing `audit_multinorm_v1.json` mutation; no `train.json` mutation; no `docs/dashboard.html` regeneration.

### B2 static-cycle launch precheck (2026-07-10, blocked / no training)

- Requested single run: `b2_static_cycle_ramp80_apgd_8255_t49k_v1k_s0`, intended result path `results/b2_static_cycle_ramp80_apgd_8255_t49k_v1k/s0/`, intended config path `configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml`.
- Precheck passed: no active `scripts/train.py` writer found (`pgrep -af '[s]cripts/train.py'` empty); target result directory does not exist; B1 reactive control has `ckpt/ep080.pt`, `ckpt/last.pt`, `ckpt/val_best.pt`, complete develop-eval JSONs, and Tier 1 audit JSONs; analytic compute model remains PASS with B2 static 10 attack-step units/batch, ratio 0.3333, saving 66.7%, and B4 predicted saving 60.0% >= 50%.
- Locked protocol confirmed from resolved B1 config: CIFAR-10, PreActResNet-18, `normalize=false`, eps linf/l2/l1 = 8/255, 0.5, 12, epochs 80, lr 0.05 with milestone [70], APGD train 10/10/10, train split 49k train_core + 1k val_select, test_monitor fixed 1k monitor-only.
- **BLOCKER:** `configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml` is missing, and the current `GroupDROTrainer` has no static-cycle one-source-per-batch objective. The non-predictive path iterates over every source attack each batch before aggregation, so a config-only B2 would silently run the B1/reactive all-source craft path and log `efficiency/attack_flops_ratio=1.0`, violating the B2 contract.
- Required before B2 launch: add/review an explicit static-cycle training implementation and config that selects exactly one full APGD source per batch (`linf -> l2 -> l1 -> repeat`), logs selected-source counts/step units/fractions per epoch, records `efficiency/attack_flops_ratio ~= 0.333`, and fails closed on unknown allocation objectives.
- No B2 config was created, no training command was launched, no post-train develop eval was run, and no previous result/audit JSONs were mutated. No B4 launch, no Tier 2 eval, no full test_final/full-AA final eval, no `train.json` mutation, no dashboard regeneration, and no post-hoc schedule/method tuning occurred.

### B2 static-cycle implementation patch (2026-07-10, code/config only / no training)

- Files changed for B2 support: `src/robustdro/training/groupdro.py`, `configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml`, `scripts/dev/check_b2_static_cycle_config.py`, `scripts/dev/smoke_b2_static_cycle.py`, and this state file.
- Added config-controlled allocation mode in training: `train.allocation_mode` defaults to full/all-source behavior for existing configs; `train.allocation_mode: static_cycle` enables the B2 path. Added `train.static_cycle_order: [linf, l2, l1]`.
- Static-cycle implementation location: `GroupDROTrainer.train_epoch`. The `static_cycle` branch selects one source from `linf -> l2 -> l1`, invokes exactly `self.attacks[g]` for that selected source, updates on that selected adversarial loss, and `continue`s before the all-source `for g, atk in enumerate(self.attacks)` loop.
- B2 config created at `configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml`. It inherits the B1 locked RAMP80/APGD 10/10/10 train49k/val1k config and changes only run identity, W&B tags, `train.allocation_mode`, and `train.static_cycle_order`.
- Config diff guard PASS: `.venv/bin/python scripts/dev/check_b2_static_cycle_config.py --b1 configs/paper/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k.yaml --b2 configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml` exited 0. Allowed differences: `run_name`, `train.allocation_mode`, `train.static_cycle_order`, and `wandb.tags`. Unexpected differences: none. Locked protocol differences: none.
- B1 regression check PASS: resolved B1 allocation mode is `full`; B1 remains full/all-source reactive and would execute linf + l2 + l1 every batch; B1 expected compute remains 30 attack-step units/batch and `efficiency/attack_flops_ratio = 1.0`. No default behavior changed for configs without `train.allocation_mode`.
- B2 dry smoke PASS: `.venv/bin/python scripts/dev/smoke_b2_static_cycle.py --config configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml --num-batches 9` exited 0. First 9 selected sources: `linf, l2, l1, linf, l2, l1, linf, l2, l1`. Attack-call counts: linf=3, l2=3, l1=3, total=9. Static units = 90; B1 all-source reference = 270; ratio = 0.3333333333333333.
- Real-compute proof status: smoke uses the same static-cycle helper as training, counts one fake source-attack invocation per selected batch, and statically asserts the `static_cycle` training branch occurs before the all-source loop and has `continue` before that loop. This proves the B2 code path bypasses hidden all-source APGD computation.
- Expected B2 compute: B1 = 30 step-units/batch; B2 static cycle = 10 step-units/batch; ratio = 0.3333333333333333; saving = 66.66666666666667%.
- B2 launch status: code/config blocker resolved. B2 is ready for a new launch precheck, but B2 training was not launched in this task.
- Commands/checks run: py_compile for `src/robustdro/training/groupdro.py`, `scripts/dev/check_b2_static_cycle_config.py`, and `scripts/dev/smoke_b2_static_cycle.py` all exited 0; config parse command exited 0; B2 config diff guard exited 0; B2 smoke dry-run exited 0; B1/B2 allocation parse sanity check exited 0; `pgrep -af '[s]cripts/train.py'` exited 1 with no active trainer found.
- Safety confirmations: no training; no B2 launch; no B4 launch; no second run; no Tier 2 eval; no full test_final/final full AutoAttack; no existing result JSON mutation; no `train.json` mutation; no B1/Reactive, Idea1, RAMP, or audit output mutation; no `docs/dashboard.html` regeneration.

### B2 static-cycle develop eval + canonical B1 comparison (2026-07-10, eval-only)

- Command: `WANDB_MODE=offline .venv/bin/python scripts/post_train_develop_eval.py --config configs/paper/b2_static_cycle_ramp80_apgd_8255_t49k_v1k.yaml --version apgd --n-examples 1000 --bs 250` → **exit 0**. B2 = `b2_static_cycle_ramp80_apgd_8255_t49k_v1k_s0`, `allocation_mode=static_cycle`, `static_cycle_order=[linf,l2,l1]`, `efficiency/attack_flops_ratio=0.3333`, train.json 80 epochs (best epoch 78 by `val_select/worst_union`, final 79), `best_val_select_worst_union=0.423`.
- Output JSONs (B2 `s0/`): `eval_val_select_val_best.json`, `eval_val_select_last.json`, `eval_test_monitor_val_best.json`, `eval_test_monitor_last.json`.
- B2 develop eval (APGD 20/20/100 r1 n1000) — clean / linf / l2 / l1 / **union** / sel:
  - val_select/val_best (ep78): 0.851 / 0.409 / 0.688 / 0.530 / **0.399** / **true**
  - val_select/last (ep79): 0.860 / 0.394 / 0.687 / 0.526 / **0.385** / false
  - test_monitor/val_best (ep78): 0.858 / 0.396 / 0.665 / 0.517 / **0.391** / monitor
  - test_monitor/last (ep79): 0.849 / 0.382 / 0.666 / 0.506 / **0.378** / monitor
- Grade equivalence: **IDENTICAL** — both apgd, n=1000, eval steps linf/l2/l1 = 20/20/100 r1, split `val_select`, role `val_best`; bs=250 both (bs not stored in JSON, harmless).
- Canonical B2-vs-B1 (val_select/val_best, diff = B2−B1): clean +1.70 pp, linf **−2.40 pp**, l2 +1.50 pp, l1 +3.70 pp, **union −2.40 pp**, compute ratio 0.333 vs 1.0 (−66.7 pp compute). B2 develop union 0.399 is **2.40 pp below** its own train.json 0.423 and **2.40 pp below** B1 (0.423) → **NOT within the 2 pp match threshold.**
- Bottleneck analysis (B2 train.json val_select per-epoch, 80 epochs): **linf is the bottleneck in 80/80 epochs** (l2=0, l1=0); switch_count=0; longest same-bottleneck streak=80; margin (2nd-lowest − lowest) mean 0.1542 / median 0.1580 / min 0.0650 / max 0.2190; best-epoch(78) margin 0.184 (linf 0.425 vs l1 0.609); final-epoch(79) margin 0.204 (linf 0.402 vs l1 0.606). The bottleneck is **static and strong** (linf persistently ~15 pp below the second-worst norm; never switches).
- Interpretation (decision rule → option 2): **B2 train.json 0.423 was not canonical; predictive-allocation central claim remains open.** Under the identical develop-eval grade B2 does not match B1 (−2.4 pp union at 1/3 compute), and the entire gap is concentrated on linf — the persistent static bottleneck. Because the bottleneck never switches and its margin is large, this setting does **not** by itself motivate *dynamic* prediction; a stronger *static* linf-weighted allocation is the more direct lever. Do NOT conclude B4 is useful merely because it could send more compute to linf.
- Safety confirmations: no training; no B4 launch; no new baseline; no Tier 2 eval; no full test_final/final-AA; no `test_monitor`/audit used for selection; no B2 tuning or config change; no mutation of B1/Idea1/RAMP/audit JSONs.

### B2 Static-Cycle Develop Eval Packet + W&B Sync -- 2026-07-10

- Resume/safety precheck: `pwd` confirmed repo root; `pgrep -af '[s]cripts/train.py'` returned no active trainer; no B4 result directory was found under `results/`; `git status --short` was dirty from existing repo work, including preexisting `docs/dashboard.html` modifications, but this resume workflow did not regenerate or edit the dashboard.
- B2 packet completeness: PASS. Existing JSONs were collected read-only; no eval rerun was needed. Paths: `results/b2_static_cycle_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_val_best.json`, `eval_val_select_last.json`, `eval_test_monitor_val_best.json`, and `eval_test_monitor_last.json`.
- B2 four-event metadata and metrics (all `version=apgd`, n=1000, eval steps linf/l2/l1 = 20/20/100, eps linf/l2/l1 = 0.03137254901960784/0.5/12.0, `train_eval_eps_mismatch=false`, `selected_by_test=false`):
  - `val_select/val_best`: checkpoint `val_best.pt`, epoch 78, `used_for_selection=true`, clean/linf/l2/l1/union = 0.8510000109672546/0.4090000092983246/0.6880000233650208/0.5299999713897705/0.39899998903274536.
  - `val_select/last`: checkpoint `last.pt`, epoch 79, `used_for_selection=false`, clean/linf/l2/l1/union = 0.8600000143051147/0.39399999380111694/0.6869999766349792/0.5260000228881836/0.38499999046325684.
  - `test_monitor/val_best`: checkpoint `val_best.pt`, epoch 78, `used_for_selection=false`, clean/linf/l2/l1/union = 0.8579999804496765/0.3959999978542328/0.6650000214576721/0.5170000195503235/0.39100000262260437.
  - `test_monitor/last`: checkpoint `last.pt`, epoch 79, `used_for_selection=false`, clean/linf/l2/l1/union = 0.8489999771118164/0.38199999928474426/0.6660000085830688/0.5059999823570251/0.3779999911785126.
- B1 grade verification source: `results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_val_best.json`, checkpoint `val_best.pt`, epoch 71, split `val_select`, `version=apgd`, n=1000, eval steps 20/20/100, eps 0.03137254901960784/0.5/12.0, `used_for_selection=true`, `selected_by_test=false`, clean/linf/l2/l1/union = 0.8339999914169312/0.43299999833106995/0.6729999780654907/0.49300000071525574/0.4230000078678131.
- GRADE_EQUIVALENCE: PASS. B1 and B2 have identical grade fields for version, n_examples, linf/l2/l1 steps, linf/l2/l1 eps, eval split, and comparable `checkpoint_role=val_best`.
- Canonical B2 minus B1 deltas (`val_select/val_best`, B2 - B1): clean +0.017000019550323486, linf -0.02399998903274536, l2 +0.01500004529953003, l1 +0.03699997067451477, union -0.02400001883506775; compute ratio 0.3333333333333333 vs 1.0.
- B2 bottleneck/margin analysis from 80 `train.json` epochs: linf bottleneck 80/80, l2 0/80, l1 0/80; bottleneck switch count 0; longest same-bottleneck streak 80; margin mean 0.15417500045150517, median 0.15800002217292786, min 0.06499999761581421, max 0.21900001168251038. Best train epoch 78 margin 0.18400001525878906 (linf 0.42500001192092896 vs l1 0.609000027179718); final epoch 79 margin 0.20399999618530273 (linf 0.4020000100135803 vs l1 0.6060000061988831). Develop-eval `val_best` bottleneck is linf 0.4090000092983246 vs second-lowest l1 0.5299999713897705, margin 0.12099996209144592.
- Interpretation guard: B2 train.json 0.423 was not canonical; canonical B2 `val_select/val_best` is 0.39899998903274536 and `test_monitor/val_best` is 0.39100000262260437. B2 is a strong static baseline at one-third compute. B4 must beat B2 meaningfully enough to justify its higher predicted compute ratio; B4 merely passing a B1-minus-2pp threshold may not be sufficient if it barely beats B2. Static-weighted-linf is a serious confound because the B2 linf bottleneck is static and large. B4 remains blocked pending Critic review and explicit B4-vs-B2 criteria.
- W&B sync status: LOGIN-BLOCKED. W&B CLI exists (`.venv/bin/python -m wandb --version` -> 0.28.0), but `.venv/bin/python -m wandb status` returned `api_key: null`; per contract, no automatic login was attempted and no `wandb sync` command was run. Expected training offline run path exists at `wandb/offline-run-20260710_015753-b2_static_cycle_ramp80_apgd_8255_t49k_v1k_s0`; additional repo-local B2 offline dirs also exist at timestamps 043134, 043414, 043651, and 043927. User must run `wandb login` before sync/resync. Local offline W&B files were not deleted.
- Safety confirmations: no training; no B4 launch; no static-weighted-linf launch; no Tier 2; no full test_final/final-AA; no result JSON mutation; no B2 tuning; no dashboard regeneration.

### B2 W&B Sync and B4 Gate Status -- 2026-07-10

- Safety precheck: repo root confirmed at `/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO`; `pgrep -af '[s]cripts/train.py'` found no active trainer; no B4 result directory and no static-weighted-linf result directory were found under `results/`. Existing worktree is dirty from prior repo work; this task did not touch result JSONs or regenerate `docs/dashboard.html`.
- B2 W&B sync status: **blocked by login**. Offline path exists: `wandb/offline-run-20260710_015753-b2_static_cycle_ramp80_apgd_8255_t49k_v1k_s0`. W&B CLI check `.venv/bin/python -m wandb --version` exited 0 with version 0.28.0. Login check `.venv/bin/python -m wandb status` exited 0 but reported `api_key: null`, so no `wandb sync` command was attempted, no URL is available, and no automatic login was attempted. Local offline W&B files were not deleted.
- B2 packet status: PASS, all four develop-eval JSON events remain present. Canonical B2 `val_select/val_best` union = 0.39899998903274536; canonical B2 `test_monitor/val_best` union = 0.39100000262260437; compute ratio = 0.3333333333333333. All four B2 eval files remain APGD n=1000 with eval steps linf/l2/l1 = 20/20/100, eps 8/255, 0.5, 12, and `train_eval_eps_mismatch=false`.
- B4 preflight status: no launch-ready B4 config was found. Exact expected config `configs/paper/b4_predictive_alloc_v1_ramp80_apgd_8255_t49k_v1k.yaml` is absent; config/result search found only `results/analysis/analytic_compute_model_pre_b4.json` for B4 naming. Existing `configs/paper/cardpb_v2.yaml` is a prior predictive-binding paper config, not a vetted B4 predictive-allocation-v1 launch contract.
- B4 gate: B4 remains blocked because Critic review is unavailable / not yet passed for explicit B4-vs-B2 criteria. Do not launch B4 until Critic pass or an explicit documented protocol decision. B4 must beat B2 meaningfully enough to justify its higher predicted compute ratio; static-weighted-linf remains a serious confound because B2's bottleneck is linf in 80/80 epochs with a large margin.
- Safety confirmations: no training; no B4 launch; no static-weighted-linf launch; no Tier 2; no full final eval; no tuning; no result JSON mutation; no dashboard regeneration.

## Critic Gate — B2 Packet and B3/B4 Sequencing

### Decision

- Critic verdict: 🔴 Blocker
- B4-next decision: REJECTED
- B3-before-B4 decision: ACCEPTED and REQUIRED
- B4 remains blocked until:
  1. B3/static-weighted-linf is pre-registered,
  2. B3 receives the required execution gate,
  3. B3 completes with the canonical four-event develop-eval packet,
  4. B4-vs-B2-vs-B3 criteria are patched and receive Critic 🟢 Pass.

### Scientific reason

B2 has a persistent linf bottleneck:

- training bottleneck: linf in 80/80 epochs
- switch count: 0
- mean bottleneck margin: 0.1542
- canonical val_select/val_best linf: 0.409
- nearest attacked norm, l1: 0.530
- canonical bottleneck margin: 0.121

Under this regime, a predictive allocator may appear successful merely by
assigning additional fixed capacity to linf. Because planned B4 compute is
approximately 0.400 versus B2 compute 0.3333333333, attribution requires an
equal-budget non-predictive static-weighted-linf baseline before B4.

### Updated central framing

Static allocation is unexpectedly strong. Predictive allocation must
demonstrate value beyond both:

1. B2 balanced static cycling at compute ratio 0.3333333333, and
2. B3 bottleneck-informed static weighting at compute ratio approximately 0.400.

The project must not claim that predictive allocation recovers robustness
unless B4 clears the pre-registered B2 and B3 comparisons.

## B3 — Static-Weighted-Linf Baseline

### Purpose

Test whether a deterministic increase in linf allocation explains the gain
that B4 is intended to obtain. B3 is an attribution baseline, not a candidate
predictive method.

### Locked allocation

- policy: deterministic static cycle
- cycle: [linf, linf, l2, l1]
- expected source fractions:
  - linf: 0.50
  - l2: 0.25
  - l1: 0.25
- target compute ratio: approximately 0.400
- acceptable realized compute ratio: 0.390 to 0.410

### Prohibited B3 behavior

B3 must not use:

- online bottleneck scores to change allocation
- validation or test feedback
- predictive models
- adaptive switching
- learned allocation policies
- dynamic thresholds selected from B2 outcomes

All non-allocation training settings must match the planned B4 comparison.

### Canonical B3 evaluation packet

B3 must produce:

1. val_select / val_best
2. val_select / last
3. test_monitor / val_best
4. test_monitor / last

All four events must use:

- attack: APGD
- n_examples: 1000
- linf steps: 20
- l2 steps: 20
- l1 steps: 100
- linf epsilon: 8/255
- l2 epsilon: 0.5
- l1 epsilon: 12
- train_eval_eps_mismatch: false

Only val_select / val_best is canonical for primary comparison.
test_monitor remains monitor-only and must not affect selection or tuning.

### B3 interpretation thresholds

- B3 union improvement over B2 >= 0.010:
  static bottleneck reweighting is empirically important.
- B3 union >= 0.403:
  B3 reaches the B1-minus-2pp threshold at approximately B4 compute.
- B3 below these thresholds:
  fixed linf reweighting is insufficient, strengthening the justification for B4.

These thresholds are interpretive. B3 remains a valid required baseline
regardless of whether it improves over B2.

### B3 config and validation status

- Config drafted: `configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml`.
- Implementation support patch: `resolve_static_cycle_order` now allows repeated configured sources while still requiring every configured source to appear at least once and rejecting unknown sources. B2 regression checks still pass.
- Validation script: `scripts/check_b3_static_weighted_linf_config.py`.
- Smoke script: `scripts/smoke_b3_static_weighted_linf.py`.
- Config validation command: `.venv/bin/python scripts/check_b3_static_weighted_linf_config.py` exited 0 after the optimizer-semantics patch. Allowed B2->B3 diffs are only `run_name`, `train.static_cycle_order`, `train.static_extra_source`, `train.static_extra_every_n_batches`, `train.static_extra_mode`, `train.static_extra_primary_weight`, `train.static_extra_source_weight`, and `wandb.tags`; locked protocol diffs are empty; predictive fields are disabled; eval attack steps are 20/20/100; train attacks remain 10/10/10; eps match 8/255, 0.5, 12. `train.static_extra_update` is rejected as invalid.
- Smoke command: `.venv/bin/python scripts/smoke_b3_static_weighted_linf.py` exited 0. First 12 scheduled sources were `linf, linf, l2, l1, linf, linf, l2, l1, linf, linf, l2, l1`; deterministic extra linf aggregate-loss attacks fired at dry-run batch indices 0, 5, 10, and 15 over a 20-batch window; primary calls were linf=10, l2=5, l1=5; extra calls were linf=4, l2=0, l1=0; total calls were linf=14, l2=5, l1=5; optimizer steps = 20, backward calls = 20, and train-epoch scheduler steps = 0 over 20 batches; total units 240 vs B1 reference 600; projected ratio 0.4. The static code-path proof confirms the branch bypasses the all-source loop and contains one `zero_grad`, one `backward`, and one `optimizer.step` in that order.
- Compute-ratio assessment: B3 equal-budget now uses deterministic periodic extra linf attack work aggregated into the same optimizer step. Scheduled static cycle contributes 10 units/batch; extra linf attack work contributes 10/5 = 2 units/batch; total = 12 units/batch; ratio = 12/30 = 0.4. This resolves the previous 0.333 vs 0.400 mismatch without prediction, adaptation, validation/test feedback, hidden compute, or an extra optimizer update.

## B3 Launch Gate — Optimizer-Semantics Blocker

- Critic verdict: 🔴 Blocker accepted.
- B3 launch status: prohibited until the one-run launch prechecks are re-run after this patch and Kiet gives an explicit later launch instruction.
- B4 status: blocked until B3 completes and B4-vs-B2-vs-B3 criteria receive the required Critic pass or an explicit documented protocol decision.
- Blocker substance: the attack-compute schedule passed, but optimizer semantics failed if `train.static_extra_update: true` meant a second optimizer update every fifth batch. That flag is invalid/ambiguous and must not be used.
- Required replacement: same-step `aggregate_loss` mode.
- Ordinary batch loss: `primary_source_loss`.
- Extra batch loss: `0.5 * primary_source_loss + 0.5 * extra_linf_loss`.
- Optimizer invariant: exactly one `backward()` and one `optimizer.step()` per training batch; no second step for the extra linf attack.
- Scheduler invariant: scheduler progression unchanged relative to B2/static-cycle; no train-batch scheduler step is added.
- Trigger: `global_batch_index % 5 == 0`.
- Trigger inputs: only deterministic global batch index; no model outputs, losses, validation/test metrics, prediction, adaptation, or threshold logic participate.
- Renewed required checks before any B3 launch: py_compile for the trainer/check/smoke scripts, B3 config checker PASS, B3 smoke PASS showing 20 optimizer steps and 20 backward calls over 20 batches plus one `zero_grad` before the single backward, B2 config guard PASS, and B2 smoke PASS.

## Critic Gate — B3 Equal-Budget Launch

- B3 launch gate existed before B3 training.
- Critic verdict: 🟢 Pass.
- Authorized action: launch exactly one B3 training run.
- Authorized config: `configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml`.
- B4 status: BLOCKED.
- Tier 2 status: BLOCKED.
- Seed sweep status: NOT AUTHORIZED.
- final/full-AutoAttack evaluation: NOT AUTHORIZED.
- Threshold tuning: NOT AUTHORIZED.
- B3 was launched only after this pass, using the authorized config above. This resolves the post-B3 protocol-integrity flag.

Accepted B3 mechanism:

- primary cycle: `[linf, linf, l2, l1]`.
- extra linf trigger: `global_batch_index % 5 == 0`.
- ordinary batch: `loss = primary_source_loss`.
- extra-linf batch: `loss = 0.5 * primary_source_loss + 0.5 * extra_linf_loss`.
- exactly one `zero_grad`, one `backward`, and one `optimizer.step` per batch.
- no extra scheduler step.
- no prediction.
- no adaptive switching.
- no validation/test feedback.

Accepted config fields:

- `train.static_extra_source: linf`.
- `train.static_extra_every_n_batches: 5`.
- `train.static_extra_mode: aggregate_loss`.
- `train.static_extra_primary_weight: 0.5`.
- `train.static_extra_source_weight: 0.5`.

Invalid field removed:

- `train.static_extra_update`.

Required post-train:

- four-event develop eval packet.
- B3-vs-B2 deltas.
- B3-vs-B1 deltas.
- realized compute ratio.
- optimizer/scheduler-step audit.
- source-count audit.

## B1/B2/B3 Grade Verification — 2026-07-10

- Canonical comparison source: `val_select / val_best` develop-eval JSONs only.
- GRADE_EQUIVALENCE_B1_B2_B3: PASS.
- PASS criteria checked: APGD, n_examples 1000, eval steps linf/l2/l1 = 20/20/100, eps linf/l2/l1 = 8/255 / 0.5 / 12, `eval_split=val_select`, `checkpoint_role=val_best`, and `train_eval_eps_mismatch=false`.

| run | JSON path | role | split | version | n | steps linf/l2/l1 | eps linf/l2/l1 | train_eval_eps_mismatch | used_for_selection | selected_by_test | clean | linf | l2 | l1 | union | ratio |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | `results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_val_best.json` | val_best | val_select | apgd | 1000 | 20/20/100 | 0.03137254901960784 / 0.5 / 12.0 | false | true | false | 0.8339999914 | 0.4329999983 | 0.6729999781 | 0.4930000007 | 0.4230000079 | 1.0 |
| B2 | `results/b2_static_cycle_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_val_best.json` | val_best | val_select | apgd | 1000 | 20/20/100 | 0.03137254901960784 / 0.5 / 12.0 | false | true | false | 0.8510000110 | 0.4090000093 | 0.6880000234 | 0.5299999714 | 0.3989999890 | 0.3333333333 |
| B3 | `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_val_best.json` | val_best | val_select | apgd | 1000 | 20/20/100 | 0.03137254901960784 / 0.5 / 12.0 | false | true | false | 0.8479999900 | 0.4510000050 | 0.6790000200 | 0.4930000007 | 0.4219999909 | 0.3996509599 |

- Since grade equivalence is PASS, B4 design discussion may proceed only as an efficiency-design draft under the post-B3 Critic gate below. It must not become a launch-ready config or task contract without renewed Critic pass.

## B3 Equal-Budget Design Resolution

- Previous mismatch: `[linf, linf, l2, l1]` changed source frequency only, so the previous implementation still ran exactly one 10-step source attack per batch and projected to ratio 0.3333333333, below the Critic-required 0.390-0.410 range.
- Accepted design: deterministic periodic extra linf attack aggregated into the same optimizer step, configured only for B3 via `train.static_extra_source: linf`, `train.static_extra_every_n_batches: 5`, `train.static_extra_mode: aggregate_loss`, `train.static_extra_primary_weight: 0.5`, and `train.static_extra_source_weight: 0.5`.
- Compute target: baseline scheduled-source work = 10 units/batch; extra linf work = 10 units every 5 global batches = 2 units/batch; target total = 12 units/batch; ratio = 12/30 = 0.400.
- Static-only guarantees: no online scores, no prediction, no adaptive switching, no learned policy, no validation/test feedback, no `test_monitor` use, and no threshold tuning. The extra linf attack fires deterministically from the global batch index only.
- Accounting: extra linf calls increment source counts, attack call counts, attack-step units, source update fractions, and `efficiency/attack_flops_ratio` through the existing measured static-cycle accounting.
- B2/B1 isolation: B1 and B2 configs do not contain the `static_extra_*` fields. B2 config guard and B2 smoke both still pass; B2 remains one selected 10-step source per batch at ratio 0.3333333333333333.
- B3 completed on 2026-07-10; see `## B3 Static-Weighted-Linf Run Complete`. B4 remains blocked until the completed B3 packet and B4-vs-B2-vs-B3 criteria receive the required Critic pass.

## Critic Gate — Post-B3 B4 Direction

- Critic verdict: ⚠️ Concern.
- Same-compute B4 route: CLOSED.
- B4 efficiency route: CONDITIONALLY OPEN for design discussion only.
- B3 result: union 0.4219999909, realized aggregate compute ratio 0.4000000000.
- B1 reference: union 0.4230000079, compute ratio 1.0.
- B2 reference: union 0.3989999890, compute ratio 0.3333333333.
- Interpretation: B3 is B1-parity at 40% compute. Its gain over B2 is concentrated in linf, consistent with static bottleneck reweighting rather than evidence that predictive allocation is needed.
- Central claim status: downgraded to open question. B4 can no longer be justified as a same-compute method chasing B3; it must be redesigned, if pursued at all, as a stricter efficiency test.
- Conditional B4 efficiency criteria before any launch contract: realized compute ratio <= 0.370 and `val_select/val_best` union >= 0.417.
- B1-minus-2pp alone is necessary but insufficient.
- If B4 fails the efficiency route, reframe the paper around B3/static bottleneck reweighting and report the B4 null honestly.
- No B4 config, launch-ready task contract, training run, Tier 2, final/full-AA eval, threshold tuning, or method tuning before renewed Critic pass.

## Predictive Leverage Analysis — B2/B3 Train Dynamics

- Task type: read-only feasibility diagnostic from existing train logs; not a final locked success criterion.
- Inputs:
  - B2 train: `results/b2_static_cycle_ramp80_apgd_8255_t49k_v1k/s0/train.json`.
  - B3 train: `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/train.json`.
- Outputs:
  - `results/analysis/predictive_leverage_b2_b3_v1.json`.
  - `results/analysis/predictive_leverage_b2_b3_v1.md`.
- Safety: no training, no eval, no B4 launch/config, no Tier 2, no final/full-AA, no threshold tuning, no result JSON mutation, no config mutation, and no dashboard regeneration.

B2 train dynamics:

- Epochs analyzed: 80.
- Bottleneck identity: linf 80/80, l2 0/80, l1 0/80; switch count 0; longest same-bottleneck streak 80.
- Bottleneck margin: mean 0.1541750005, std 0.0285800518, CV 0.1853740988, median 0.1580000222, min 0.0649999976, max 0.2190000117, range 0.1540000141, p10 0.1153999984, p25 0.1377500147, p75 0.1722499728, p90 0.1841000170.
- Margin first/selected/last: epoch 0 = 0.0649999976; selected epoch 78 = 0.1840000153; epoch 79 = 0.2039999962.
- Linf dynamics: mean robust_linf 0.3510250008, std 0.0372545222, min 0.2370000035, max 0.4250000119, range 0.1880000085; linf was never not-bottleneck; linf gap to nearest was never below 0.03 or 0.05.
- Phase means, margin / linf robust: 0-19 = 0.1181999989 / 0.3089000009; 20-39 = 0.1530000031 / 0.3577999994; 40-59 = 0.1661999986 / 0.3542500004; 60-79 = 0.1793000013 / 0.3831500024.
- LR-drop check: mean margin epochs 60-69 = 0.1714000046, epochs 70-79 = 0.1871999979, post-minus-pre = +0.0157999933; margin does not narrow after LR drop.
- Correlations: corr(epoch, margin) = 0.8268476736; corr(epoch, linf robust) = 0.7044289109.
- Run-level leverage classification: HIGH by the transparent heuristic, because margin range is 0.154 with phase range 0.061.

B3 train dynamics:

- Epochs analyzed: 80.
- Bottleneck identity: linf 80/80, l2 0/80, l1 0/80; switch count 0; longest same-bottleneck streak 80.
- Bottleneck margin: mean 0.0883124992, std 0.0192130147, CV 0.2175571388, median 0.0890000015, min 0.0279999971, max 0.1250000298, range 0.0970000327, p10 0.0628999949, p25 0.0787499920, p75 0.1012499854, p90 0.1161000252.
- Margin first/selected/last: epoch 0 = 0.0279999971; selected epoch 72 = 0.0949999988; epoch 79 = 0.1180000007.
- Linf dynamics: mean robust_linf 0.3989875000, std 0.0405880184, min 0.2599999905, max 0.4650000036, range 0.2050000131; linf was never not-bottleneck; linf gap to nearest was below 0.03 in 1 epoch and below 0.05 in 3 epochs.
- Phase means, margin / linf robust: 0-19 = 0.0693499953 / 0.3475500017; 20-39 = 0.0856500015 / 0.4066499978; 40-59 = 0.0945499986 / 0.4071000025; 60-79 = 0.1037000015 / 0.4346499979.
- LR-drop check: mean margin epochs 60-69 = 0.0979000002, epochs 70-79 = 0.1095000029, post-minus-pre = +0.0116000026; margin does not narrow after LR drop.
- Correlations: corr(epoch, margin) = 0.7045623298; corr(epoch, linf robust) = 0.7709583291.
- Run-level leverage classification: MODERATE by the transparent heuristic, because margin CV is 0.2176, margin range is 0.0970, and corr(epoch, margin) is 0.7046.

Overall interpretation:

- Overall predictive-leverage classification: MODERATE.
- B4 efficiency route mechanistically plausible: WEAK.
- Reason: linf is static as the bottleneck in both B2 and B3, so there is no evidence for norm switching; however, bottleneck margin and linf robust dynamics vary by phase enough that a cheaper B4 design might try to reduce fixed linf-heavy compute outside high-demand phases.
- Recommendation: a B4 efficiency-design draft is justified only as a cautious feasibility draft, not as a launch. B4 remains blocked pending owner decision and renewed Critic pass.
- If owner declines a null-confirming B4 risk, reframe around B3/static bottleneck reweighting.
- Same-compute B4 criteria remain obsolete/closed as predictive-win criteria. The owner decision below reopens same-compute B4 only as Route A-Diagnostic, not as a pure efficiency route or launch approval.

## Owner Decision — B4 Route A-Diagnostic

- Owner-selected route: same-compute diagnostic B4, not the previous pure efficiency route.
- This owner decision supersedes the post-B3 "same-compute route closed" rule only for diagnostic design. It does **not** reopen a same-compute predictive-win claim.
- Risk posture: high-risk / high-insight.
- B4 remains blocked pending renewed Critic pass. No B4 launch-ready config or task contract exists.
- Target realized compute ratio: `[0.390, 0.410]`.
- Expected Route A setting: `R=10`, giving expected compute ratio approximately `0.400`.
- Diagnostic goal: test whether predictive allocation breaks the stable linf bottleneck seen in B2/B3, or merely reproduces the same linf-heavy behavior as B3.
- If B4 fails to beat B3, make no predictive win claim.
- If B4 allocation is linf-heavy and union is approximately B3, interpret it as bottleneck-ceiling evidence: predictive allocation did not add robustness beyond static bottleneck allocation in this setting.

## B4 Route A-Diagnostic Design Draft — Not Launch-Ready

- Status: design/pre-registration draft only. B4 remains blocked pending Critic pass.
- Do not create a launch-ready B4 config, launch B4, train, evaluate, tune thresholds, use `test_monitor` for design choices, run Tier 2, run final/full-AA, mutate result JSONs, or regenerate the dashboard from this draft.
- Canonical references for the diagnostic:
  - B1 `val_select/val_best` union = `0.4230000079`, compute ratio = `1.000`.
  - B2 `val_select/val_best` union = `0.3989999890`, compute ratio = `0.3333333333`.
  - B3 `val_select/val_best` union = `0.4219999909`, realized compute ratio approximately `0.400`.
- Route A target: same compute as B3, target realized ratio `[0.390, 0.410]`.

Predictive mechanism draft:

- B1 denominator: `10 + 10 + 10 = 30` attack-step units/batch.
- Route A parameters: `R=10`, `S=30`, no cheap probe.
- Non-refresh batch: run one predicted full 10-step source attack.
- Refresh batch: run all three full source attacks, 10/10/10, to update binding/EMA state.
- Starvation floor `S` constrains source selection recency only; it must not add hidden attacks.
- Compute formula: `units_per_batch = ((R - 1) * 10 + 30) / R`.
- For `R=10`: `units_per_batch = 12`, ratio `12/30 = 0.400`, saving vs B1 `60.0%`.
- This is a same-compute diagnostic against B3, not an efficiency route.

Existing predictive implementation inspection:

- `configs/paper/cardpb_v2.yaml` points to `train.groupdro.objective: predictive_binding`, a prior CARD-PB v2 mechanism, not a B4 Route A launch contract.
- `src/robustdro/training/groupdro.py` currently has per-sample EMA loss state (`pb_Lbar`, `pb_seen`) and a predicted binding source `b_hat = argmax(Lbar)`.
- Existing CARD-PB has cold-start full attacks, random recalibration subset (`pb_recal_rho`), optional full-reactive epoch recalibration (`pb_recal_every`), confidence/fixed floor attacks for non-predicted norms, `phi/misprediction_rate`, `phi/miss_vol_*`, `phi/miss_rate_*`, `floor/*`, `pb/attack_passes`, `pb/attack_flops_ratio`, and canonical `efficiency/attack_flops_ratio`.
- Existing CARD-PB does **not** implement the Route A deterministic `R=10` refresh schedule.
- Existing CARD-PB does **not** implement starvation floor `S=30` as a source-age constraint with no extra attack cost.
- Existing CARD-PB floor semantics run floor attacks on non-predicted norms; that is different from Route A's one-source non-refresh batch.
- Existing CARD-PB does not log selected source per batch, source counts/fractions per epoch, refresh batch indices, refresh binding trace, prediction accuracy by refresh batch, EMA loss/bind/score traces, source age, starvation override events, or per-source compute units.
- Existing CARD-PB predictive state is not saved in checkpoints (`pb_Lbar`, `pb_seen`, floor/miss EMA state, source ages would need explicit checkpoint state for valid resume/provenance).

Required implementation/logging patches before any launch:

- Add explicit Route A allocation mode separate from historical `predictive_binding`.
- Add deterministic refresh scheduler with `R=10`.
- Add source-age/starvation floor with `S=30`, no hidden extra attacks.
- Log selected source per batch, selected source counts/fractions per epoch, refresh batch indices, binding/worst source on refresh batches, prediction accuracy on refresh batches, EMA loss per source, EMA bind per source, final score per source, starvation-floor override events, source age/time since last selected, per-epoch attack-step units, per-source attack-step units, refresh-call units, and per-epoch compute ratio.
- Persist predictor/allocation state needed for resume and provenance.
- Add a config checker/smoke test before launch; the checker must reject any config that uses test/monitor/audit feedback, cheap probes, hidden extra attacks, or a realized-ratio target outside `[0.390, 0.410]`.

Required post-run analysis if later launched:

- Four-event develop eval packet only: `val_select/val_best`, `val_select/last`, `test_monitor/val_best`, `test_monitor/last`.
- All four events must use APGD, `n_examples=1000`, eval steps `20/20/100`, eps `8/255 / 0.5 / 12`, and `train_eval_eps_mismatch=false`.
- Primary robustness comparison uses `val_select/val_best` only.
- Report B4-vs-B3, B4-vs-B1, and B4-vs-B2 union deltas.
- Report B4 source fractions vs B3 source fractions, B4 linf fraction vs B3 linf call fraction, refresh prediction accuracy, bottleneck identity/margin by epoch, whether dynamic prediction differs meaningfully from static weighted-linf, and whether any dynamic difference translates into robustness gain.

Interpretation rules:

- If `val_select/val_best` union >= `0.432`: strong positive predictive result.
- If union is close to B3 and allocation is linf-heavy: diagnostic ceiling result; supports bottleneck-limited interpretation.
- If union is below B3 and allocation is noisy: negative predictive result.
- If union is below B2: clear failure.
- If B4 union is within `±0.014` of B3 or within `±0.014` of `0.432`, do not make a strong one-seed claim. Patch `PROJECT_STATE.md` and request a replication gate.

## B4 Route A-Diagnostic Implementation Support

- Status: code/config support added only. B4 remains blocked pending Critic review/pass.
- No B4 training was launched, no eval was run, no result JSONs were mutated, no Tier 2/final/full-AA was run, and `docs/dashboard.html` was not regenerated.
- No launch-ready B4 config exists.
- Draft-only config path: `configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.DRAFT.yaml`.
- Check script path: `scripts/check_b4_predictive_refresh_config.py`.
- Smoke script path: `scripts/smoke_b4_predictive_refresh.py`.

Implementation behavior:

- Added explicit `train.allocation_mode: predictive_refresh`, separate from legacy `train.groupdro.objective: predictive_binding`.
- Required Route A fields: `train.predictive_refresh_every_n_batches: 10`, `train.predictive_starvation_floor_batches: 30`, `train.predictive_ema_alpha: 0.9`, `train.predictive_use_cheap_probe: false`, `train.predictive_score_rule: loss_plus_bind`, and `train.predictive_log_batch_trace: true`.
- Cold start and every `R=10` global batches run a refresh batch with all three full 10-step source attacks.
- Refresh batches update scalar EMA loss and EMA bind state from all-source losses, record binding/worst source, and train with the inherited B1 reactive aggregation rule (`per_sample_soft`, T=0.25 for the draft config).
- Non-refresh batches compute `score_g = normalized_EMA_loss_g + EMA_bind_g`, apply starvation-floor redirection if any source age is at least `S=30`, run exactly one selected 10-step source attack, and train on that selected source.
- Starvation floor redirects selection only; it does not add attacks or optimizer steps.
- Checkpoints now include predictive-refresh state for this mode: EMA loss, EMA bind, initialized flag, last selected batch/source ages, last global batch, Route A config, and group norms. Resume in predictive-refresh mode fails closed if the checkpoint lacks this state.
- Per-epoch diagnostics include selected-source counts/fractions, refresh indices/counts, prediction accuracy against refresh worst/binding source, EMA loss/bind/final score by source, source ages, floor override count, source attack call counts, attack-step units by source, total attack-step units, and canonical `efficiency/attack_flops_ratio`. Optional batch trace records selected source, refresh flag, scores, EMA state, source age, and floor override flag per batch.

Validation commands:

- `.venv/bin/python -m py_compile src/robustdro/training/groupdro.py` → exit 0.
- `.venv/bin/python -m py_compile scripts/check_b4_predictive_refresh_config.py` → exit 0.
- `.venv/bin/python -m py_compile scripts/smoke_b4_predictive_refresh.py` → exit 0.
- `.venv/bin/python scripts/check_b4_predictive_refresh_config.py` → exit 0.
- `.venv/bin/python scripts/smoke_b4_predictive_refresh.py` → exit 0.

Checker result:

- B4 draft config is `.DRAFT.yaml`, run name contains `DRAFT`, and W&B tags include `draft-only`.
- Locked B1 protocol differences: none.
- Allowed differences only: run name, W&B tags, `train.allocation_mode`, and Route A predictive-refresh fields.
- APGD train steps: 10/10/10.
- Eval steps: 20/20/100.
- Eps: 8/255, 0.5, 12.
- `R=10`, `S=30`, no cheap probe, score rule `loss_plus_bind`.
- Projected units/batch: 12.0; projected ratio vs B1: 0.4.
- No `test_monitor` selection fields detected.
- Launch-ready status: false.

Smoke result over 20 fake batches:

- Refresh indices: `[0, 10]`.
- Refresh count: 2.
- Non-refresh selected-source batches: 18.
- Total attack-step units: 240.
- B1 reference units: 600.
- Ratio: 0.400.
- Optimizer steps/backward calls: 20/20.
- Starvation floor extra optimizer steps: false.
- Batch trace length: 20.
- Required trace fields present: selected source, refresh flag, scores, EMA loss, EMA bind, source age, and floor override flag.
- Attack-step units by source sum to total.
- Checkpoint predictor-state test passed: torch save/load roundtrip, `save_checkpoint` includes predictor state, resume loads predictor state, and resume fails closed without predictor state.
- Missing pieces before any launch: Critic review/pass, launch contract, and a non-DRAFT config created only after approval.

## Critic Gate — B4 Route A-Diagnostic One-Run Launch Contract

- Critic verdict: 🟢 Pass.
- Authorized action: launch exactly one B4 Route A-Diagnostic training run after final preflight/check/smoke pass.
- Route A-Diagnostic purpose: test whether predictive allocation can break the stable linf bottleneck seen in B2/B3, or whether it converges to B3-like linf-heavy behavior at the same compute.
- Config transition: promote draft config `configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.DRAFT.yaml` to launch config `configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml`.
- Authorized run name: `b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k_s0`.
- Expected result dir: `results/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k/s0/`.
- Compute target: realized `efficiency/attack_flops_ratio` in `[0.390, 0.410]`; expected ratio `0.400`.
- Required diagnostics: selected-source trace, source counts/fractions, refresh indices/counts, binding/worst source on refresh, prediction accuracy on refresh, EMA loss/bind/score by source, starvation-floor events, source ages, attack-step units by source, total attack-step units, realized ratio, and checkpoint predictor-state provenance.
- Required preflight:
  - `pwd`.
  - `git status --short`.
  - `pgrep -af '[s]cripts/train.py' || true`.
  - `pgrep -af '[s]cripts/post_train_develop_eval.py' || true`.
  - `pgrep -af '[s]cripts/evaluate.py' || true`.
  - `test ! -d results/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k/s0`.
  - `.venv/bin/python -m py_compile src/robustdro/training/groupdro.py`.
  - `.venv/bin/python -m py_compile scripts/check_b4_predictive_refresh_config.py`.
  - `.venv/bin/python -m py_compile scripts/smoke_b4_predictive_refresh.py`.
  - `.venv/bin/python scripts/check_b4_predictive_refresh_config.py --config configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml`.
  - `.venv/bin/python scripts/smoke_b4_predictive_refresh.py --config configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml`.
- Training command:

```bash
WANDB_MODE=offline .venv/bin/python scripts/train.py \
  --config configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml
```

- Post-train four-event develop eval command:

```bash
WANDB_MODE=offline .venv/bin/python scripts/post_train_develop_eval.py \
  --config configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml \
  --version apgd \
  --n-examples 1000 \
  --bs 250
```

- Required eval outputs: `eval_val_select_val_best.json`, `eval_val_select_last.json`, `eval_test_monitor_val_best.json`, and `eval_test_monitor_last.json`, all APGD n=1000, steps 20/20/100, eps 8/255 / 0.5 / 12, `train_eval_eps_mismatch=false`, and `selected_by_test=false`.
- Interpretation rules:
  - Strong positive predictive result: ratio in `[0.390,0.410]` and `val_select/val_best` union >= `0.432`.
  - Diagnostic ceiling result: ratio in `[0.390,0.410]`, union close to B3, and allocation linf-heavy/B3-like.
  - Negative result: below B3 with noisy allocation, below B2, clean drop >2pp vs B1, or ratio outside interval.
- Prohibited actions: no second B4 run, no seed sweep, no Route B, no Tier 2, no final/full-AA, no `test_final`, no threshold tuning, no mutation of historical result JSONs, no `test_monitor` selection/tuning, and no dashboard regeneration.

## B4 Route A-Diagnostic One-Run Result -- 2026-07-10

Status: **complete**. The Critic-approved one-run Route A-Diagnostic contract was executed exactly once.

Safety and launch:

- Final launch config promoted from draft: `configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml`.
- Result path: `results/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k/s0/`.
- Pre-launch checks passed: no active `scripts/train.py`, `scripts/post_train_develop_eval.py`, or `scripts/evaluate.py`; target result directory absent before launch.
- Preflight command exits: py_compile `groupdro.py` = 0, py_compile config checker = 0, py_compile smoke = 0, config checker = 0, smoke = 0.
- Training command exit: 0.
- Post-train four-event develop eval command exit: 0.
- W&B caveat: W&B initialization/summary logging rejected the long `run:` tag (>64 chars) and fell back to stdout/JSON-only logging. Local `train.json` and eval JSONs were written and are the source of truth.

Training completion and checkpoints:

- `train.json` has 80 history records, stored epochs 0 through 79.
- Checkpoints present: `ckpt/ep080.pt`, `ckpt/last.pt`, `ckpt/val_best.pt`, and legacy `ckpt/best.pt`.
- `val_best.pt` selected stored epoch 74 by train-time `val_select/worst_union = 0.4740000069141388`.
- Important grade caveat: train-time selector uses the training loop's `train.eval_pgd_steps = 20`, while canonical post-train develop eval uses APGD linf/l2/l1 steps 20/20/100. Do not compare the train-time selector metric as if it were the canonical develop-eval grade.
- Final stored epoch 79 train-time `val_select/worst_union = 0.453000009059906`.

Compute and diagnostic trace:

- Total attack-step units: 366720.
- B1 all-source reference units: 916800.
- Realized total compute ratio: 0.4000000000.
- Per-epoch `efficiency/attack_flops_ratio`: mean 0.4000000000, min 0.3996509599, max 0.4013961606.
- Refresh count total: 3056; non-refresh selected-source batches: 27504.
- Starvation-floor overrides: 0.
- Selected-source counts: linf 25038, l2 0, l1 5522; selected-source fractions: linf 0.8193, l2 0.0000, l1 0.1807.
- Attack-step units by source, including refresh attacks: linf 267350, l2 30560, l1 68810.
- Source-count fractions by attack calls, including refresh: linf 0.7290, l2 0.0833, l1 0.1876.
- Weighted refresh prediction accuracy: worst-source 0.3132569558, binding-source 0.3839607201 over denominator 3055.
- Per-epoch trace sidecars present: `predictive_refresh_traces/ep000.json` through `ep079.json`; each trace has 382 batch records and fields for selected source, refresh flag, scores, EMA loss, EMA bind, source age, binding/worst source, prediction correctness, and floor override.
- Checkpoint predictor-state provenance: both `val_best.pt` and `last.pt` include initialized `predictive_refresh` state with config, EMA loss/bind, group norms, and last-selected/source-age state. `val_best.pt` has `last_global_batch=28649`; `last.pt` has `last_global_batch=30559`.

Canonical post-train develop eval, APGD 20/20/100 r1 n1000:

- `eval_val_select_val_best.json` (`used_for_selection=true`, checkpoint epoch 74): clean/linf/l2/l1/union = 0.8209999799728394/0.4659999907016754/0.671999990940094/0.46299999952316284/0.4230000078678131.
- `eval_val_select_last.json` (`used_for_selection=false`, checkpoint epoch 79): clean/linf/l2/l1/union = 0.8349999785423279/0.4490000009536743/0.6690000295639038/0.4339999854564667/0.40299999713897705.
- `eval_test_monitor_val_best.json` (monitor-only): clean/linf/l2/l1/union = 0.8330000042915344/0.45100000500679016/0.6610000133514404/0.45500001311302185/0.4129999876022339.
- `eval_test_monitor_last.json` (monitor-only): clean/linf/l2/l1/union = 0.8339999914169312/0.4490000009536743/0.6570000052452087/0.42399999499320984/0.39800000190734863.

Canonical B4 vs B1/B2/B3, `val_select/val_best` APGD 20/20/100:

- B1 reference: clean/linf/l2/l1/union = 0.8339999914169312/0.43299999833106995/0.6729999780654907/0.49300000071525574/0.4230000078678131.
- B2 reference: clean/linf/l2/l1/union = 0.8510000109672546/0.4090000092983246/0.6880000233650208/0.5299999713897705/0.39899998903274536.
- B3 reference: clean/linf/l2/l1/union = 0.8479999899864197/0.45100000500679016/0.6790000200271606/0.49300000071525574/0.421999990940094.
- B4 minus B1: clean -0.0130000114, linf +0.0329999924, l2 -0.0009999871, l1 -0.0300000012, union +0.0000000000.
- B4 minus B2: clean -0.0300000310, linf +0.0569999814, l2 -0.0160000324, l1 -0.0669999719, union +0.0240000188.
- B4 minus B3: clean -0.0270000100, linf +0.0149999857, l2 -0.0070000291, l1 -0.0300000012, union +0.0010000169.

Interpretation:

- Realized compute ratio PASS: 0.4000 is inside `[0.390, 0.410]`.
- Canonical strong-positive threshold NOT met: canonical `val_select/val_best` union is 0.4230000079, below the registered 0.432 threshold.
- Diagnostic classification: B4 Route A-Diagnostic is B1/B3-parity at 40% compute, improves linf relative to B1/B2/B3, but gives back clean and l1 robustness. Allocation is strongly linf-heavy and B3-like, with l2 never selected outside refresh.
- This result should be reported as diagnostic ceiling / parity evidence, not as a clean predictive-allocation win under the canonical APGD 20/20/100 grade.
- No checkpoint selection used `test_monitor`; no audit or final/full-AA was run.

Safety confirmations:

- No second B4 run.
- No B2/B3 rerun or seed sweep.
- No Route B.
- No Tier 2.
- No final/full-AA or `test_final`.
- No threshold/method/suite tuning.
- No mutation of historical result JSONs.
- No `docs/dashboard.html` regeneration.

## B4 Pre-Registration Constraints

The previous pre-launch B4 block was superseded by the Critic-approved Route A-Diagnostic one-run launch contract above. That one run is now complete. B4 is again blocked for any additional run, Route B, seed sweep, or tuning unless Kiet records a renewed protocol decision and Critic/owner gate.

### B4 hard validity requirements

- canonical grade equivalent to B1/B2/B3
- four-event develop-eval packet
- realized compute ratio within the locked tolerance
- no test_monitor use for selection, controller updates, early stopping,
  threshold tuning, or allocation decisions
- report realized source counts and fractions
- report allocation changes and switch statistics
- report controller predictions and audit fields
- report all outcomes, including failed criteria

### Historical Same-Compute B4 Thresholds — Closed

The following pre-B3 same-compute thresholds are retained only as provenance.
They are superseded by `Critic Gate — Post-B3 B4 Direction` and must not be
used to justify a B4 launch.

- old same-compute target ratio: approximately 0.400.
- old val_select/val_best union threshold: >= 0.409.
- old test_monitor/val_best union threshold: >= 0.391.
- old val_select/val_best linf threshold: >= 0.399.

Current rule: B1-minus-2pp alone is necessary but insufficient; the active
owner-selected B4 path is Route A-Diagnostic with target compute ratio
`[0.390, 0.410]`. It still requires renewed Critic pass before config or
task-contract work.

### Historical Same-Compute B4-vs-B3 Comparison — Closed

The equal-compute B4-vs-B3 comparison route is closed after B3 reached
B1-parity at 40% compute. The old constraints below are provenance only:

- val_select/val_best union >= B3 union + 0.010.
- no attacked per-norm accuracy below B3 by more than 0.010.
- test_monitor/val_best union >= B3 test_monitor/val_best union.

Current rule: Route A-Diagnostic may be discussed only under the owner decision
above; it may not become launch-ready before renewed Critic pass.

### Interpretation guard

If B4 primarily assigns more capacity to linf and does not outperform B3
under the locked criteria, the result must be interpreted as evidence for
static bottleneck reweighting rather than predictive intelligence.

### Next permitted action

The next permitted protocol action is:

- submit/review the completed B3 packet for B4-vs-B2-vs-B3 pre-registration. Do not launch B4 until Critic pass or an explicit documented protocol decision.

## B3 Static-Weighted-Linf One-Run Task Contract

### Input

- Config path: `configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml`
- Run name: `b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`
- Expected result dir: `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/`
- Expected allocation order: `[linf, linf, l2, l1]`
- Expected source fractions: linf 0.50, l2 0.25, l1 0.25
- Equal-budget extra attack: deterministic `linf` attack every 5 global batches, aggregated into the same optimizer step (`train.static_extra_source=linf`, `train.static_extra_every_n_batches=5`, `train.static_extra_mode=aggregate_loss`, `train.static_extra_primary_weight=0.5`, `train.static_extra_source_weight=0.5`)
- Expected projected compute: scheduled 10 units/batch + extra 2 units/batch = 12 units/batch; ratio 0.400 vs B1.
- Expected canonical eval packet: `val_select/val_best`, `val_select/last`, `test_monitor/val_best`, `test_monitor/last`, all APGD n=1000 with eval steps 20/20/100 and eps 8/255, 0.5, 12.

### Prechecks

- Confirm repo root with `pwd`.
- Record `git status --short`.
- Confirm no active trainer with `pgrep -af '[s]cripts/train.py'`.
- Confirm no B4 result directory exists.
- Confirm no previous B3 result directory exists.
- Run `.venv/bin/python scripts/check_b3_static_weighted_linf_config.py`.
- Run `.venv/bin/python scripts/smoke_b3_static_weighted_linf.py`.
- Choose W&B mode explicitly.
- Current execution gate: completed. Do not relaunch B3 unless Kiet explicitly opens a new run contract.

### Execution command

Completed B3 training command:

```bash
WANDB_MODE=offline .venv/bin/python scripts/train.py \
  --config configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml
```

- Completed training run artifacts are in `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/`.
- Codex command exit code was not captured because the client turn was interrupted while the unsandboxed process continued; completion is verified by 80 train records, `ckpt/ep080.pt`, `ckpt/last.pt`, `ckpt/val_best.pt`, and the completed four-event develop-eval packet.

### Post-train required

- `train.json` with 80 epochs.
- Honest recorded compute ratio and attack-step units.
- Source counts, source fractions, and attack-step units by source.
- Four develop-eval JSONs from:

```bash
WANDB_MODE=offline .venv/bin/python scripts/post_train_develop_eval.py \
  --config configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml \
  --version apgd \
  --n-examples 1000 \
  --bs 250
```

- Patch `docs/PROJECT_STATE.md` after completion before reporting.

### Safety

- No `test_monitor` selection.
- No B4 launch.
- No Tier 2.
- No full test_final/final AutoAttack.
- No threshold, method, or schedule tuning from B3 results.

## B3 Static-Weighted-Linf Run Complete

- Training command: `WANDB_MODE=offline .venv/bin/python scripts/train.py --config configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml`.
- Training command exit code: not captured by Codex because the turn was interrupted while the unsandboxed process continued; completion verified from artifacts. A sandboxed first attempt exited 130 after CUDA/local-socket restrictions and was interrupted before `s0/train.json` or checkpoints were written; local W&B files were not deleted.
- Result dir: `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/`.
- W&B offline dirs: completed training run `wandb/offline-run-20260710_072652-b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`; interrupted sandbox attempt `wandb/offline-run-20260710_070938-b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`; develop-eval runs `wandb/offline-run-20260710_095051-b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`, `wandb/offline-run-20260710_095328-b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`, `wandb/offline-run-20260710_095605-b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`, and `wandb/offline-run-20260710_095840-b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k_s0`.
- Train output: `train.json` contains 80 records through epoch 79; checkpoints present: `ckpt/ep080.pt`, `ckpt/last.pt`, `ckpt/val_best.pt`, and legacy alias `ckpt/best.pt`.
- Selected checkpoint: `ckpt/val_best.pt`, selected at epoch 72 by `val_select/worst_union = 0.45100000500679016` in train-time probes. Train-time selected metrics: clean 0.848, linf 0.4650000035762787, l2 0.6930000185966492, l1 0.5600000023841858, union 0.45100000500679016.
- Final train-time epoch 79 metrics: val_select clean/linf/l2/l1/union = 0.851/0.4429999887943268/0.6850000023841858/0.5609999895095825/0.4339999854564667; test_monitor clean/linf/l2/l1/union = 0.835/0.44699999690055847/0.7020000219345093/0.5559999942779541/0.43700000643730164, monitor-only.
- Train audit: total train batches = 30560; computed primary counts = linf 15280, l2 7640, l1 7640; computed extra linf trigger count = 6112, frequency = 0.2, min/max per epoch = 76/77; logged total source/attack calls = linf 21392, l2 7640, l1 7640.
- Attack-step accounting: linf 213920, l2 76400, l1 76400, total 366720; reactive reference 916800; realized aggregate compute ratio = 0.4000000000. Per-epoch ratio range = 0.3996509599 to 0.4005235602, mean = 0.4000000000. Accepted interval 0.390 to 0.410: PASS.
- Optimizer/scheduler audit: one-update-per-batch semantics are enforced by the static branch and B3 smoke proof; expected optimizer steps = 30560. Scheduler is epoch-level as in B2; expected scheduler steps = 80. No extra scheduler step is introduced for extra linf attacks.
- Post-train develop eval command: `WANDB_MODE=offline .venv/bin/python scripts/post_train_develop_eval.py --config configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml --version apgd --n-examples 1000 --bs 250`, exit code 0.
- Four-event develop eval packet, all APGD n=1000 with steps linf/l2/l1 = 20/20/100, eps 8/255, 0.5, 12, `train_eval_eps_mismatch=false`, `selected_by_test=false`:

| event | epoch | used_for_selection | clean | linf | l2 | l1 | union | efficiency ratio | path |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| val_select / val_best | 72 | true | 0.8479999900 | 0.4510000050 | 0.6790000200 | 0.4930000007 | 0.4219999909 | 0.3996509599 | `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_val_best.json` |
| val_select / last | 79 | false | 0.8510000110 | 0.4350000024 | 0.6779999733 | 0.4709999859 | 0.4099999964 | 0.3996509599 | `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/eval_val_select_last.json` |
| test_monitor / val_best | 72 | false | 0.8429999948 | 0.4480000138 | 0.6710000038 | 0.4830000103 | 0.4280000031 | 0.3996509599 | `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/eval_test_monitor_val_best.json` |
| test_monitor / last | 79 | false | 0.8349999785 | 0.4280000031 | 0.6830000281 | 0.4779999852 | 0.4059999883 | 0.3996509599 | `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/eval_test_monitor_last.json` |

- Canonical B3-vs-B2 deltas, using val_select/val_best and realized aggregate compute ratio 0.4000000000: clean -0.0030000210, linf +0.0419999957, l2 -0.0090000033, l1 -0.0369999707, union +0.0230000019, compute ratio +0.0666666667.
- Canonical B3-vs-B1 deltas, using val_select/val_best and realized aggregate compute ratio 0.4000000000: clean +0.0139999986, linf +0.0180000067, l2 +0.0060000420, l1 +0.0000000000, union -0.0010000169, compute ratio -0.6000000000.
- Threshold interpretation: B3 improves over B2 union by +0.023, exceeding the +0.010 static-bottleneck-reweighting threshold. B3 union 0.4219999909 also exceeds the B1-minus-2pp threshold 0.403. B3 is essentially B1-union parity on canonical val_select/val_best while using realized compute ratio 0.400.
- Safety confirmations: no B4 launch; no Tier 2; no final/full-AA; no threshold tuning; no config changes based on B3 results; no `test_monitor` selection, stopping, tuning, allocation, or B4 design; no dashboard regeneration.
- Next gate: submit this B3 packet to Critic for B4-vs-B2-vs-B3 pre-registration. B4 remains blocked.

Allowed immediate action:
- Review/report the completed B3 packet and prepare Critic review material for B4-vs-B2-vs-B3 pre-registration. Do not regenerate dashboard unless Kiet explicitly asks or a safe dashboard refresh is requested.

Future training runs blocked until gatekeeper exists and passes:
- Reactive Soft-T Flat-6.
- CARD-PB v2 kspan-4.
- Reactive Soft-T Curriculum 3-6-10.
- Reactive Soft-T Fail-Rate 50% rerun/new batch.
- Any batch with more than one training run.

Critic resolution:
- Concern: do not let gatekeeper implementation delay the single reactive baseline run, but define minimal preflight and avoid editing `train.py` while reactive is running.
- Resolution: ACCEPTED.
- Reactive-first launch state: the baseline completed locally; continue avoiding checkpoint mutation or repair. Upcoming eval/audit should use only `ckpt/val_best.pt` and `ckpt/last.pt`.
- Required before threshold lock: confirm `val_best.pt` was selected by `val_select/worst_union`, and add gatekeeper checks preventing test-driven selection.

## 3. Locked protocol

- Dataset: CIFAR-10.
- Model: PreActResNet-18 family.
- Pixels: raw `[0,1]`, `normalize: false`.
- Threat model: eps `(linf 8/255, l2 0.5, l1 12)`.
- Optimizer/schedule: SGD, epochs 80, lr 0.05, milestone `[70]`, gamma 0.1, lr becomes 0.005 after epoch 70, momentum 0.9, weight_decay 5e-4, save_freq 10.
- Training attack: APGD train, steps `10/10/10` for linf/l2/l1.
- Reactive FLOPs denominator: 30 attack steps.
- Develop eval: APGD `20/20/100`, restarts1, decision grade.
- Final claim eval: full AutoAttack via `--version standard`, final grade only for selected winner/final claim.
- Canonical frontier grid: `{2,4,8,16,24}`.
- Do not propose or use early LR drop unless Kiet explicitly unlocks a new protocol.

## 4. Split and checkpoint-selection protocol

- Internal method comparison uses CIFAR train split into 49k `train_core` plus 1k `val_select`.
- `val_select`: 1,000 held-out images from CIFAR-10 train. Logged inside the training W&B run every epoch. Used for checkpoint selection only through `val_select/worst_union`. Selected checkpoint is `ckpt/val_best.pt`.
- `val_select` train-run keys: `val_select/clean_acc`, `val_select/robust_linf`, `val_select/robust_l2`, `val_select/robust_l1`, `val_select/worst_union`, `val_select/n_examples = 1000`, `val_select/frequency = 1`.
- `test_monitor`: fixed 1,000-image CIFAR-10 test subset. Logged inside the training W&B run every 10 epochs at epochs 10, 20, 30, 40, 50, 60, 70, and 80. It is monitor-only and must never select checkpoints, methods, configs, thresholds, or queue order.
- `test_monitor` train-run keys: `test_monitor/clean_acc`, `test_monitor/robust_linf`, `test_monitor/robust_l2`, `test_monitor/robust_l1`, `test_monitor/worst_union`, `test_monitor/n_examples = 1000`, `test_monitor/frequency = 10`, `test_monitor/used_for_selection = false`.
- `test_final`: full 10,000-image CIFAR-10 test set. Run only after Kiet freezes the winner/final comparator. It is report/final-claim only, logged to a separate W&B eval run, and must never select checkpoints, methods, configs, thresholds, or queue order.
- Final W&B run name format: `<run_name>_s<seed>_final_fullAA`.
- Final W&B group: `ramp80_t49k_v1k_final`.
- Final W&B tags: `final`, `test-final`, `full-autoattack`, `selected-winner`.
- Final W&B keys: `final/test_final/clean_acc`, `final/test_final/robust_linf`, `final/test_final/robust_l2`, `final/test_final/robust_l1`, `final/test_final/worst_union`, `final/test_final/n_examples = 10000`, `final/test_final/eval_grade = full_autoattack_standard`, `final/test_final/used_for_selection = false`.
- After each training run completes, evaluate both `ckpt/val_best.pt` and `ckpt/last.pt` on both develop splits `val_select` and `test_monitor`. Log these four develop eval events to the same training W&B run under `eval/val_select/val_best/*`, `eval/val_select/last/*`, `eval/test_monitor/val_best/*`, and `eval/test_monitor/last/*`.
- Required post-train develop eval keys: `eval/<split>/<checkpoint>/clean_acc`, `eval/<split>/<checkpoint>/robust_linf`, `eval/<split>/<checkpoint>/robust_l2`, `eval/<split>/<checkpoint>/robust_l1`, `eval/<split>/<checkpoint>/worst_union`, `eval/<split>/<checkpoint>/n_examples`, `eval/<split>/<checkpoint>/used_for_selection`.
- `eval/val_select/val_best/used_for_selection = true`; `eval/val_select/last/used_for_selection = false`; `eval/test_monitor/val_best/used_for_selection = false`; `eval/test_monitor/last/used_for_selection = false`.
- Do not evaluate full 10k test during every training epoch. Dashboard may show both `test_monitor` and `test_final`, but must label them separately. Report final claims use `test_final`, not `test_monitor`.

## 5. Naming convention

- Paper configs live under `configs/paper/`.
- Exploration configs live under `configs/exploration/`.
- Dev/smoke configs live under `configs/dev/` or `configs/dev/smoke/`.
- Legacy configs/scripts/docs may live under explicit `legacy/` paths only and must not be active defaults.
- Per-run layout: `results/<run_name>/s<seed>/`.
- RAMP external-anchor evals use `results/ramp/eval_*.json`.
- W&B run names must match local result folders for active idea-compare runs.
- Canonical run names in the active screen include `reactive_ramprecipe`, `flat6_ramp80_apgd_8255_s0`, `idea3_curriculum_ramp80_apgd_8255_s0`, and `idea1_failrate_ramp80_apgd_8255_s0`.

## 6. Single experiment table

| run | epoch | grade/source | clean | union | linf/l2/l1 | FLOPs | role/verdict |
|---|---:|---|---:|---:|---|---:|---|
| RAMP paper ep50 | 50 | APGD-only 100/100/100 n1000 | 77.5 | 42.9 | 43.9/61.9/47.1 | 1.00 | reference |
| RAMP paper ep80 | 80 | APGD-only 100/100/100 n1000, original ckpt `RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`, `results/ramp/eval_ramp_ep80_eps8255_apgd_n1000.json` | 81.2 | 46.0 | 47.3/65.8/49.6 | 1.00 | APGD-grade over-report; not decision/final |
| RAMP paper ep80 decision | 80 | APGD 20/20/100 r1 n1000, same original ckpt, `results/ramp/eval_ramp_ep80_apgd_2020100_n1000.json` | 81.2 | 46.2 | 47.3/65.8/49.7 | 1.00 | decision-grade anchor |
| RAMP paper ep80 standard | 80 | `--version standard` n1000, same original ckpt, `results/ramp/eval_ramp_ep80_fullAA_n1000.json` | 81.2 | 46.2 | 47.3/65.8/49.7 | 1.00 | standard eval on this n1000 subset; did not confirm expected 44.6 |
| Arm A ep80 full-AA log | 80 | full AutoAttack log, `results/exploration/ramp_armA_full_5070ti.log` | 80.9 | 44.7 | 46.1/65.9/48.8 | 1.00 | RAMP reproduce final-grade log; matches paper lambda=5 44.6 +/-0.6 |
| Arm A ep80 decision | 80 | APGD 20/20/100 r1 n1000, `external/RAMP/trained_models/armA_rampfull_5070ti/ep_80_0.pth`, `results/ramp/eval_armA_rampfull_ep80_apgd_2020100_n1000.json` | 81.2 | 44.8 | 45.9/66.4/50.1 | 1.00 | reproduce decision-grade cross-check |
| Reactive Soft-T Full-10 s0 | val_best@71; completed 80/80 | develop APGD 20/20/100 r1 n1000 on `val_select/val_best`; audit v1 complete | 83.4 | 42.3 | 43.3/67.3/49.3 | 1.00 | paper control completed; audit full union 41.3 for val_best, 38.6 for last; `best.pt` is legacy alias only, use `val_best.pt`/`last.pt` |
| Idea1 Fail-Rate 50% s0 | val_best role uses best@26; last@79 | develop APGD 20/20/100 r1 n1000 repaired on all four develop events; no audit | 80.7 | 41.0 | 42.2/66.6/49.9 | 0.759 | exploration screen completed; `ckpt/val_best.pt` absent so `ckpt/best.pt` mapped to val_best; test_monitor monitor-only |
| reactive_apgd_8255 | 50 | APGD 20/20/100 n1000 | 76.7 | 38.8 | 39.2/60.1/45.6 | 1.00 | reactive ref |
| predictive ks16 | 50 | APGD 20/20/100 n1000 | 76.1 | 38.7 | 40.3/59.9/44.8 | 0.665 | l1 holds, FLOPs high |
| predictive ks2 | 50 | APGD 20/20/100 n1000 | 77.2 | 39.4 | 41.4/59.9/44.2 | 0.584 | current frontier winner |
| predictive ks4 | 50 | APGD 20/20/100 n1000 | 78.4 | 39.8 | 41.1/61.2/46.9 | 0.603 | l1 strong, just over target |
| predictive ks8 | 50 | APGD 20/20/100 n1000 | 77.7 | 39.3 | 41.0/59.9/45.7 | 0.623 | holds, over target |
| predictive ks24 | 37/50 at sync | partial run | pending | pending | pending | ~0.706 so far | active/incomplete at original sync |

RAMP grade reconciliation:

- `results/ramp/eval_ramp_ep80_eps8255_apgd_n1000.json` is the original paper checkpoint evaluated with APGD-only 100/100/100 restarts1 n1000. It is APGD-grade, not decision-grade 20/20/100, and not final-grade full AutoAttack.
- T1a decision re-eval on the same paper checkpoint produced clean 81.2, linf/l2/l1 47.3/65.8/49.7, union 46.2.
- T1b `--version standard` on the same paper checkpoint and n1000 subset produced the same clean 81.2, linf/l2/l1 47.3/65.8/49.7, union 46.2; it did not confirm the expected 44.6-44.7 for that checkpoint/subset.
- Arm A full-AA log resolved the reproduce question: clean 80.9, linf/l2/l1 46.1/65.9/48.8, union 44.7. The `46.1` number is linf-only, not union.
- Arm A decision APGD 20/20/100 produced clean 81.2, linf/l2/l1 45.9/66.4/50.1, union 44.8, consistent with the full-AA log union within 0.1pp.
- RAMP Reproduce ep80 is treated as a full50k external anchor. The active method screen uses train49k/val1k for leakage-safe checkpoint selection. Therefore RAMP is not an exact split-matched internal baseline. Internal thresholding for the screen must use Reactive Soft-T Full-10 train49k/val1k as the control anchor.

## 7. Metric schema

- Primary robustness metric: `worst_union`, robust only if linf and l2 and l1 all hold.
- Canonical train keys: `epoch`, `lr`, `train/loss`, `train/loss_linf`, `train/loss_l2`, `train/loss_l1`, `train/adv_acc_linf`, `train/adv_acc_l2`, `train/adv_acc_l1`, `val_select/clean_acc`, `val_select/robust_linf`, `val_select/robust_l2`, `val_select/robust_l1`, `val_select/worst_union`, `val_select/n_examples`, `val_select/frequency`, `test_monitor/clean_acc`, `test_monitor/robust_linf`, `test_monitor/robust_l2`, `test_monitor/robust_l1`, `test_monitor/worst_union`, `test_monitor/n_examples`, `test_monitor/frequency`, `test_monitor/used_for_selection`, `efficiency/attack_flops_ratio`, `train/epoch_time_s`.
- Canonical post-train develop eval keys: `eval/split`, `eval/checkpoint`, `eval/checkpoint_role`, `eval/clean_acc`, `eval/robust_linf`, `eval/robust_l2`, `eval/robust_l1`, `eval/worst_union`, `eval/selected_by_test`, `eval/used_for_selection`, `eval/attack_linf_steps`, `eval/attack_l2_steps`, `eval/attack_l1_steps`, `eval/seed`.
- Canonical nested develop eval keys: `eval/<split>/<checkpoint>/clean_acc`, `eval/<split>/<checkpoint>/robust_linf`, `eval/<split>/<checkpoint>/robust_l2`, `eval/<split>/<checkpoint>/robust_l1`, `eval/<split>/<checkpoint>/worst_union`, `eval/<split>/<checkpoint>/n_examples`, `eval/<split>/<checkpoint>/used_for_selection`, where split is `val_select` or `test_monitor` and checkpoint is `val_best` or `last`.
- Canonical final eval keys: `final/test_final/clean_acc`, `final/test_final/robust_linf`, `final/test_final/robust_l2`, `final/test_final/robust_l1`, `final/test_final/worst_union`, `final/test_final/n_examples`, `final/test_final/eval_grade`, `final/test_final/used_for_selection`.
- `efficiency/attack_flops_ratio` is the canonical comparable FLOPs key. Method aliases such as `pb/attack_flops_ratio` or `exp/attack_flops_ratio` may also be logged but must not replace it.
- `probe/*` is legacy diagnostic only. Do not alias `probe/*` to `val_select/*` for active selection-grade runs.
- Backward-compatible aliases may be emitted in JSON/W&B, but dashboards and comparison tables should use canonical keys.

## 8. Method registry

| method | config/result anchor | objective and key knobs | lane/status |
|---|---|---|---|
| Reactive Soft-T Full-10 | `configs/paper/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k.yaml` inheriting `configs/paper/reactive_ramprecipe.yaml` | reactive per-sample-soft, T=0.25, APGD 10/10/10 | paper control; fixed-10 anchor |
| Reactive Soft-T Flat-6 | `configs/exploration/flat6_ramp80.yaml` | reactive APGD 6/6/6, target FLOPs about 0.60 | method screen |
| CARD-PB v2 kspan-4 | `configs/paper/cardpb_v2.yaml` with kspan-4 pin before launch | predictive binding, confidence floor, miss_vol floor driver | main method screen |
| Reactive Soft-T Curriculum 3-6-10 | `configs/exploration/idea3_curriculum_ramp80.yaml` | fixed curriculum 3 -> 6 -> 10, mean FLOPs about 0.63375 | control/ablation |
| Reactive Soft-T Fail-Rate 50% | `configs/exploration/idea1_failrate_ramp80.yaml`, `results/idea1_failrate_ramp80_apgd_8255/s0/` | per-norm fail-rate controller, threshold 50% | exploration screen; completed seed0 |
| CARD-PB v2 frontier ks2/4/8/16/24 | historical predictive result folders | frontier sweep over confidence-floor kspan grid | development evidence |

CARD-PB v2 mechanics:

- Predictor gives full APGD budget to predicted binding norm and floor budget to non-predicted norms.
- Confidence floor formula: `floor_g = clamp(k_min + round(kspan * miss_vol_g), k_min, full_g)`.
- Driver: `miss_vol_g = P(true_bind=g and b_hat != g)`, measured on full-attack recalibration and EMA-smoothed.
- `miss_rate_g = 1 - recall_g` is diagnostic only and must not drive the floor.
- Adaptive probe guard is OFF or bypassed in confidence mode.
- `Lbar` and predictor state must be updated only from full-budget norms, not under-attacked floored losses.
- FLOPs ratio denominator is the reactive APGD 10/10/10 budget, i.e. 30.

## 9. Research Context & Literature Insights

This section is advisory context, not a run blocker.
It informs decisions, framing, and report writing.
It must not override Locked Protocol, Run Queue, or Single Experiment Table.

Add new literature/prior-work insight here instead of creating new active Markdown memos. Long notes or paper summaries may go to `docs/legacy/` only if necessary, but active decision-relevant summaries must be condensed here. Status must be one of `reference`, `locked-support`, `protocol-support`, `hypothesis`, `parked`, or `superseded`. Items with status `reference`, `hypothesis`, or `parked` must not block runs. Items with status `locked-support` or `protocol-support` explain existing locked decisions but still do not override the locked protocol.

| id | insight | evidence/source | decision influence | status |
|---|---|---|---|---|
| L1 | RAMP scratch CIFAR-10 likely trains on full 50k train images, with no held-out validation split. | RAMP script/code check. | Treat RAMP as external anchor; internal method comparison uses our locked train49k/val1k split. | reference |
| L2 | RAMP/RN18 union robustness should be framed as a cluster/curve, not only a single headline point. | RAMP ep50/ep80 and Arm A reconciliation. | Claim accuracy-per-FLOP, not pure leaderboard SOTA. | reference |
| L3 | l1 evaluation needs stronger APGD budget than linf/l2. | Eval convergence curve: linf/l2 plateau earlier, l1 needs around 100 steps. | Decision eval remains APGD 20/20/100. | locked-support |
| L4 | Earlier l1 collapse was caused by train-weak/eval-strong mismatch plus CARD-PB bugs, not necessarily by the predictive idea itself. | APGD rerun and CARD-PB v2 fixes. | APGD train 10/10/10 and physical floor rebuild remain locked. | locked-support |
| L5 | Test monitoring may reveal train-test shift, but using test for checkpoint or method selection creates leakage. | Kiet protocol decision. | Dashboard may show test_monitor; selection uses val_select only. | protocol-support |

## 10. Run queue

Active seed-0 full80 method screen, locked to RAMP80/APGD 10/10/10:

| order | run | config | status/next action |
|---:|---|---|---|
| 1 | Reactive Soft-T Full-10 | `configs/paper/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k.yaml` | DEVELOP_EVAL_AND_AUDIT_DONE; `train.json` through stored epoch 79, `ckpt/ep080.pt`, `ckpt/last.pt`, and `ckpt/val_best.pt` present; `val_best.pt` selected at stored epoch 71 by `val_select/worst_union`; `best.pt` is legacy alias and must not be used for eval/audit |
| 2 | Reactive Soft-T Flat-6 | `configs/exploration/flat6_ramp80.yaml` | BLOCKED_PENDING_GATEKEEPER |
| 3 | CARD-PB v2 kspan-4 | `configs/paper/cardpb_v2.yaml` plus explicit kspan/floor pin before launch | BLOCKED_PENDING_GATEKEEPER; placeholder remains `__TBD__` until Kiet fills |
| 4 | Reactive Soft-T Curriculum 3-6-10 | `configs/exploration/idea3_curriculum_ramp80.yaml` | BLOCKED_PENDING_GATEKEEPER |
| 5 | Reactive Soft-T Fail-Rate 50% | `configs/exploration/idea1_failrate_ramp80.yaml` | DEVELOP_EVAL_REPAIR_DONE; completed seed0 result remains preserved; `ckpt/val_best.pt` absent, so `ckpt/best.pt` was mapped to `checkpoint_role=val_best`; no audit by default |

Completed control post-train develop eval command, using `ckpt/val_best.pt` and `ckpt/last.pt`:

```bash
python scripts/post_train_develop_eval.py --config configs/paper/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k.yaml --version apgd --n-examples 1000 --bs 250
```

This ran the four develop eval events: `val_select/val_best`, `val_select/last`, `test_monitor/val_best`, and `test_monitor/last`. For a single manual event, use `scripts/evaluate.py --eval-split <val_select|test_monitor> --checkpoint-role <val_best|last>` and write a split-specific output such as `eval_val_select_val_best.json`.

## 11. Dashboard/report/script generation contract

- `docs/dashboard.html` is generated from `docs/PROJECT_STATE.md` plus result JSONs.
- Dashboard should use canonical metric keys from section 7 and should not use method-specific FLOPs aliases as primary fields.
- Dashboard must label eval role, split, checkpoint role, and `used_for_selection`; `test_monitor` and `test_final` must not be merged.
- Dashboard may display `test_monitor` and `test_final`, but report/final-claim tables must use `final/test_final/*`, not `test_monitor/*`.
- Dashboard must reflect the active frontier grid `{2,4,8,16,24}`.
- Dashboard must not read legacy Markdown as active protocol.
- If dashboard generation cannot safely read partial or actively written results, do not regenerate; report the exact safe command for later.
- 2026-07-09 eval/audit sync: dashboard regeneration was not requested and remains deferred after develop eval repair and audit v1. Safe command when Kiet asks: `python scripts/make_dashboard.py`.
- Report scripts must not edit historical result JSONs.
- After changing this file in a way that affects the dashboard, regenerate dashboard when safe and verify it does not conflict with this file.

## 12. Known risks and blockers

- Result mutation risk: do not overwrite active or recently completed result directories; `results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0/` and `results/idea1_failrate_ramp80_apgd_8255/s0/` remain preserved.
- Reactive Soft-T Full-10 checkpoint provenance risk: `ckpt/val_best.pt` is canonical for `checkpoint_role=val_best`; `ckpt/last.pt` is canonical for `checkpoint_role=last`; `ckpt/best.pt` is a byte-distinct legacy alias of `val_best.pt` and must not be used for eval/audit.
- Audit interpretation risk: audit v1 primary APGD union is APGD-CE 100-step union on the fixed audit subset and is not the grade-matched RAMP reproduce gate metric; RAMP audit reporting is allowed only because the decision APGD 20/20/100 reproduce gate passed.
- CARD-PB checkpoint risk: predictor state may not be fully checkpointed, including `pb_Lbar`, `pb_seen`, and miss EMA/floor state. Predictive continuation may not be trajectory-equivalent until fixed or deliberately accepted.
- Split/checkpoint implementation risk: before launching the next full batch, run a cheap smoke/dry-run check that code and dashboard paths enforce train49k/val1k selection, create `ckpt/val_best.pt`, and do not select on `test_monitor` or `test_final`.
- RAMP grade risk: `46.0` is APGD-only 100/100/100 n1000 over-report, not final full-AA. Do not compare it against final-grade full-AA numbers as if they were the same grade.
- Threshold lock blocker: threshold lock is blocked until RAMP train-size provenance, `val_best.pt` selection source, and test-not-selection are confirmed and critic gate #1 reviews.
- Legacy risk: archived docs/configs/scripts may contain stale protocols; active agents should use this file and active config paths only.

## 13. Decisions log

- Keep RAMP80 schedule: lr 0.05 until epoch 70, then 0.005 until epoch 80. No early LR drop.
- Keep APGD train 10/10/10 for all active paper-lane comparisons.
- Use APGD 20/20/100 restarts1 as decision-grade eval.
- Use full AutoAttack standard only for final claim/winner.
- Separate paper lane from exploration lane; exploration ideas may graduate only by explicit Kiet decision.
- Use accuracy-per-FLOP framing rather than leaderboard-only SOTA framing.
- RAMP 46.0 vs Arm A 44.7 is a grade/checkpoint/subset reconciliation issue, not evidence that Arm A failed; Arm A `46.1` is linf-only, not union.
- Literature/research insights are stored in section 9 as advisory context and never as run blockers.
- 2026-07-09: ACCEPT critic concern on reactive-first execution. The previous idea2-order blocker is `[STATE=RESOLVED][METHOD=ACCEPTED]`: reactive baseline runs first, and idea2/method-screen runs remain blocked until the deterministic gatekeeper exists and passes. Reactive Soft-T Full-10 is a single run, not a batch >1, so it does not trigger critic gate #2.
- 2026-07-09: Standalone `scripts/preflight_check.py` was added by Claude Code and accepted as a read-only deterministic gatekeeper utility. It is not wired into `scripts/train.py` yet. Wiring is deferred until after the current Reactive Soft-T Full-10 baseline completes.
- 2026-07-09: Reactive Soft-T Full-10 seed-0 sync policy during training was lightweight: early sanity/logging check was enough, with no epoch-by-epoch agent tracking to avoid wasting resources.
- 2026-07-09: Reactive Soft-T Full-10 checkpoint provenance inspection: `ckpt/val_best.pt` was selected at stored epoch 71 by `val_select/worst_union = 0.4490000009536743`; `ckpt/best.pt` is a byte-distinct legacy alias saved from the same event with identical loaded checkpoint content, not clean-best, not test-selected, not stale earlier val-best, and not unknown. Use `ckpt/val_best.pt` and `ckpt/last.pt` for upcoming eval/audit; no checkpoint mutation or repair was performed.
- 2026-07-09: Completed approved eval/audit sequence. Reactive Soft-T Full-10 develop eval recorded all four `val_select/test_monitor x val_best/last` events; Idea1 develop-eval repair recorded all four events using legacy `best.pt` as `checkpoint_role=val_best`; RAMP Arm A decision APGD 20/20/100 reproduce gate passed with union 0.4480000138282776 vs target 0.448; multinorm audit v1 completed for reactive val_best, reactive last, and RAMP Arm A. No training, train.json mutation, dashboard regeneration, test/audit checkpoint selection, threshold tuning, method tuning, or audit-suite tuning was performed.

## 14. Run-update checklist

After every meaningful train/eval/frontier update:

1. Confirm no active writer will be overwritten.
2. Add or update the row in section 6 with run name, seed, epoch/checkpoint, grade, clean, union, linf/l2/l1, FLOPs, and verdict.
3. Update section 2 current phase if the run queue or allowed next action changed.
4. Update section 10 run queue status.
5. After a completed training run, run or record the four develop eval events for `val_select/val_best`, `val_select/last`, `test_monitor/val_best`, and `test_monitor/last`.
6. Add any new method/literature insight to section 8 or 9, keeping section 9 advisory.
7. Regenerate `docs/dashboard.html` from this file plus result JSONs when safe.
8. Verify dashboard and this file do not conflict.
9. Keep result JSONs historical and immutable unless Kiet explicitly asks for repair.


## 15. Strategic Scope Lock — 2026-07-09

### Decision

The project direction is now locked, critic-approved at the pre-registration level, as:

**Predictive Allocation for Reliable Multi-Source Adversarial Training**

The previous framing:

> accuracy-per-FLOP against RAMP as the main comparator

is superseded as the main story, but the old assets are not discarded. Prior runs, configs, and RAMP-aligned machinery are retained and reinterpreted under the new broader framing.

The project is no longer primarily about beating RAMP. RAMP is now treated as:

- a strong multi-norm robustness reference;
- evidence that robustness over multiple perturbation sources matters;
- evidence that lp perturbation distributions interfere;
- a useful baseline/reference point for multi-source robust training.

RAMP must not be framed as invalid or as the enemy. The project builds on the motivation of RAMP while studying an orthogonal question: predictive compute allocation and reliability of evaluation.

### Owner Constraints

- User: final-year CS student at HCMUT.
- Goal: one serious research artifact suitable for US PhD applications.
- Deadline: soft December 2026 target.
- Resources: solo researcher, RTX 5070 Ti + Colab.
- Existing assets: CIFAR-10 / PreActResNet-18 / multi-norm AT pipeline with eps=(l1=12, l2=0.5, linf=8/255), RAMP-related configs, reactive/full allocation baselines, dashboard/state tooling.

### Definition of “Real”

The project interprets “real” along three axes:

1. **Threat realism:** adversaries are not naturally restricted to a single norm or one optimizer mechanism.
2. **Deployment realism:** robust training is expensive, difficult to reproduce, and hard to deploy routinely.
3. **Evidence realism:** robustness claims should survive stronger audit rather than depend on one narrow attack suite.

The paper should not claim to solve real-world robustness. It should claim a deployment-aware and evidence-aware step beyond standard fixed multi-norm training.

### Main Research Question

Can adversarial-training compute be allocated predictively across multiple perturbation sources so that we reduce training cost while preserving robustness under a stronger, more honest evaluation suite?

Equivalently:

> Instead of paying for every adversarial source reactively and choosing the worst after the fact, can we predict which source deserves expensive inner maximization before paying the full attack cost?

### Main Claim Shape

The intended claim is:

> Predictive allocation can make multi-source adversarial training more deployment-aware by reducing adversarial compute, and more reliable by explicitly auditing whether robustness survives attack-mechanism changes.

The intended claim is **not**:

- We beat RAMP absolutely.
- RAMP is invalid.
- Multi-attack multi-norm is automatically more real than multi-norm.
- We solve real-world physical robustness.
- We solve all multi-attack robustness.

## 16. Pre-registration v1 — Critic Approved — 2026-07-09

### Gate Status

Status: **ACTIVE / CRITIC PASS**

Claude/Critic passed Pre-registration v1 after metric-design and pass/fail revisions.

Approved hierarchy:

1. **Primary APGD Union** = comparability metric.
2. **Per-norm same-budget same-norm audit gap** = primary evidence of APGD-only overestimation.
3. **Full Audit Union** = strict secondary reliability metric.

Approved method/run logic:

- Primary success target = Deployment Success only.
- Strong Success and Reliability Improvement are secondary descriptive outcomes.
- Global Failure conditions use OR logic.
- Default result is null if Primary Deployment Success is not achieved.
- Audit-based improvement claims require 95% confidence intervals excluding zero.
- Prediction-error/audit-gap kill condition is global and independent of result class.
- Predictive allocation v1 is simplified to loss + binding only.
- B3/B4/v1b probe boundaries are explicit.
- Analytic compute model is mandatory before RQ2+ training runs.

### Working Title

**Predictive Allocation for Reliable Multi-Source Adversarial Training**

### Research Question

Can adversarial-training compute be allocated predictively across multiple perturbation sources so that training becomes cheaper while preserving multi-source robustness under a stronger, more honest evaluation suite?

### Core Hypothesis

Reactive/full multi-source adversarial training spends compute on all candidate sources before knowing which source is binding. Predictive allocation should be able to use cheap signals from recent training dynamics to decide which source deserves expensive inner maximization, reducing attack compute while preserving union robustness.

### Scope

The main experimental sandbox is fixed:

- dataset: CIFAR-10;
- architecture: PreActResNet-18;
- perturbation radii:
  - l1 = 12;
  - l2 = 0.5;
  - linf = 8/255;
- primary training sources:
  - APGD-linf-style source;
  - APGD-l2-style source;
  - APGD-l1-style source;
- primary comparison setting: multi-source adversarial training under the same CIFAR-10 / PreActResNet-18 / eps setting used by the RAMP-style multi-norm literature.

RAMP remains a reference and motivation point, not an enemy comparator. RAMP studies union robustness over B1 ∪ B2 ∪ Binf and motivates the importance of multi-norm source interference. This project studies the orthogonal question of compute allocation and reliability auditing.

### Baselines

#### B0 — Natural / clean reference

Sanity check clean training and architecture pipeline. Not a main comparator.

#### B1 — Full/reactive multi-source baseline

Primary high-compute control.

Definition:

- for each batch, generate all three training attacks:
  - APGD-linf;
  - APGD-l2;
  - APGD-l1;
- compute per-sample or batch-level worst-case adversarial loss;
- update on the worst-case/reactive source.

This is the main baseline for deployment-cost comparison.

#### B2 — Static allocation baseline

Fixed source schedule independent of model state. Default schedule: cycle linf → l2 → l1 by batch, or fixed equal-probability sampling.

#### B3 — Cheap-probe reactive baseline

A baseline, not a method variant.

Definition:

- every non-refresh batch runs a cheap probe for all three sources;
- selects the source with highest cheap-probe loss;
- runs full attack only for the selected source;
- uses no EMA loss/binding history;
- uses no predictive state except current cheap probe.

Purpose: isolate the value of current-batch cheap probing.

#### B4 — Predictive allocation v1

Main method.

Definition:

- uses recent normalized per-source loss and recent binding frequency;
- uses scheduled refresh;
- uses starvation floor;
- does **not** use cheap probe.

#### B4-v1b — Optional tie-break probe variant

Allowed variant only after v1. May use cheap probe only when top two predictive scores differ by less than fixed margin m. It is B4 + tie-break probe, not a replacement for B3.

#### B5 — RAMP-style reference

Use public or locally implemented RAMP-style recipe only if the harness can reproduce its primary evaluation within the reproduction gate. If not runnable, RAMP remains citation/reference only.

### Predictive Allocation v1

Allowed input signals only:

- recent per-source losses from previous refresh windows;
- recent per-source binding frequency;
- epoch number or training phase;
- moving averages of source difficulty.

Disallowed:

- full current-batch all-source attack results except on scheduled refresh batches;
- test-set performance;
- audit-suite results;
- post-hoc manual schedule changes after seeing outcomes.

State variables:

- EMA_loss_linf, EMA_loss_l2, EMA_loss_l1;
- EMA_bind_linf, EMA_bind_l2, EMA_bind_l1.

Default EMA decay: alpha = 0.9.

Score rule v1:

```text
score_p = normalized_EMA_loss_p + EMA_bind_p
selected_source = argmax_p score_p
```

No tunable score weights. No EMA_success in v1. Safety floor is separate from score.

Refresh rule:

- every R batches, run full all-source refresh;
- default R = 10;
- if analytic compute model predicts <50% saving, do not train; patch R and gate again.

Safety floor:

- each source must be selected at least once every S batches;
- default S = 30.

Uncertainty fallback:

- if top two scores differ by less than margin m, choose lower recent selection count;
- v1b may use cheap probe only for this tie-break.

### Training Compute Accounting

Primary compute metric:

```text
attack_step_units = sum_over_attacks(batch_size * attack_steps * restarts)
```

Required logs:

- full attack calls by source;
- cheap probe calls by source;
- attack steps by source;
- refresh frequency;
- estimated total attack-step count;
- wall-clock time when stable/reliable;
- hardware used.

Do not rely only on wall-clock because Colab/local hardware variability can confound results.

### Analytic Compute Model Requirement

Before launching RQ2-RQ4, compute theoretical attack_step_units for B1, B2, and B4 from config:

- number of sources attacked per batch;
- APGD steps per source;
- refresh frequency R;
- starvation floor S;
- cheap-probe cost if enabled;
- restarts.

Record:

- expected attack_step_units per epoch;
- theoretical maximum saving vs B1;
- expected saving vs B1 under planned R and S.

The 40% reduction target is valid only if the analytic model predicts B4 can exceed it with at least 10 pp margin. Therefore:

- if predicted B4 saving >=50% before runtime overhead: RQ2+ may proceed;
- if predicted B4 saving <50%: stop, patch R/refresh policy, and send to critic before training.

Do not self-modify R and launch.

## 17. Evaluation and Audit Protocol — Critic Approved

### Metric 1 — Clean Accuracy

Standard CIFAR-10 test clean accuracy.

### Metric 2 — Primary APGD Robustness

Per norm:

- APGD-linf robust accuracy;
- APGD-l2 robust accuracy;
- APGD-l1 robust accuracy.

Primary APGD Union:

- sample is robust iff it survives APGD-linf, APGD-l2, and APGD-l1.

Purpose: comparability with multi-norm literature.

### Metric 3 — Per-Norm Audit Gap

Primary evidence of APGD-only overestimation.

For p ∈ {linf, l2, l1}:

```text
audit_acc_p = robust accuracy under worst same-norm audit attack for p
per_norm_audit_gap_p = APGD_acc_p - audit_acc_p
```

Interpretation:

- a large same-norm audit gap is evidence that APGD-only evaluation is optimistic for that norm;
- this avoids the mechanical confound of comparing 3 attacks against 9 attacks across all norms.

### Metric 4 — Black-box Gap

For norms with black-box/query attacks:

```text
black_box_gap_p = APGD_acc_p - worst_black_box_acc_p
```

Interpretation:

- black_box_gap_p >= 2.0 pp is evidence consistent with gradient masking or white-box optimizer-specific failure;
- do not conflate this with minimum-norm or alternative white-box strength gaps.

### Metric 5 — Full Audit Union

Strict secondary reliability metric.

A sample is robust only if it survives every attack in the audit suite across all norms and mechanisms.

Full Audit Union must not be used alone as proof of APGD overestimation because it mechanically stacks more attacks than Primary APGD Union.

### Metric 6 — Audit Gap with Confidence Intervals

For each per-norm audit gap:

- report gap in percentage points;
- report 95% confidence interval or bootstrap interval;
- if the interval overlaps zero, describe the gap as inconclusive.

Any audit-based improvement claim must exceed estimated uncertainty.

### Audit Suite v1

Default audit subset:

- 1,000 CIFAR-10 test images;
- fixed seed;
- same subset for all models;
- class-balanced if convenient, otherwise fixed-seed random subset;
- record exact indices and subset construction method.

Attack suite:

linf:

- APGD-linf;
- FAB-linf;
- Square-linf.

l2:

- APGD-l2;
- FAB-l2;
- Square-l2 if available/stable;
- DDN-l2 or FMN-l2 if feasible.

l1:

- APGD-l1;
- FAB-l1 if available/stable;
- Sparse-RS-l1 or FMN-l1 if feasible.

If Square-l2 is not available, explicitly state that l2 audit tests alternative white-box/minimum-norm strength, not black-box masking behavior.

### Minimum-Norm Thresholding

FMN/DDN success counts under fixed-epsilon audit only if found perturbation norm <= epsilon_p.

Default thresholds:

- linf: 8/255;
- l2: 0.5;
- l1: 12.

Minimum-norm curves may be reported separately, but cannot be mixed into fixed-epsilon robust accuracy without thresholding.

### Query-Based Attack Config

For Square, Sparse-RS, or any query-based/random attack:

- fix attack seed;
- fix query budget;
- fix restarts;
- store all parameters in config;
- use same budget for all models in the same table.

Extended-budget audit is allowed only as a separate table.

### Baseline Reproduction Gate

Before audit:

- reproduce primary evaluation of each public/reference checkpoint.

For RAMP-style public/reference checkpoint:

- reproduce clean accuracy;
- reproduce per-norm APGD/AutoAttack-style robustness;
- reproduce Primary APGD Union.

Kill condition:

- if reproduced Primary APGD Union differs from expected reference by more than ±1.0 percentage point, stop;
- do not report audit;
- debug harness.

For internal checkpoints:

- reference is the last deterministic evaluation recorded in PROJECT_STATE.md.

## 18. Pass/Fail Criteria — Critic Approved

### Primary Success Criterion — Deployment Success

The method succeeds on the pre-registered main claim if and only if all hold:

1. Primary APGD Union is within 2.0 pp of B1 full/reactive baseline.
2. Training attack_step_units are reduced by at least 40% relative to B1.
3. Full Audit Union is within 2.0 pp of B1.
4. No per-norm audit gap is worse than B1 by more than 2.0 pp.
5. No black-box/query gap is worse than B1 by more than 2.0 pp.
6. Clean accuracy is within 2.0 pp of B1.

This is the only primary win condition.

### Secondary Outcome — Strong Success

If the method already satisfies Primary Deployment Success, it may additionally be described as a strong success if all hold:

1. Primary APGD Union is within 1.0 pp of B1.
2. Training attack_step_units are reduced by at least 30% relative to B1.
3. Full Audit Union is within 1.0 pp of B1.
4. No per-norm audit gap is worse than B1 by more than 1.0 pp.
5. Clean accuracy is within 1.0 pp of B1.

Strong Success is a bonus label only. It is not an alternative path to paper success.

### Secondary Outcome — Reliability Improvement

Reliability improvement may be reported only as a secondary result.

It requires:

1. Primary APGD Union is within 2.0 pp of B1.
2. The improvement exceeds uncertainty:
   - Full Audit Union is better than B1 with a 95% CI excluding zero; or
   - at least one per-norm audit gap is smaller than B1 by at least 2.0 pp and the 95% CI of the difference excludes zero.
3. No other per-norm audit gap worsens by more than 1.0 pp.

Reliability Improvement cannot rescue a failure of the Primary Deployment Success criterion.

### Global Failure / Kill Conditions

The method fails the pre-registered main claim if any of the following occurs:

1. Primary APGD Union drops by more than 2.0 pp relative to B1.
2. Full Audit Union drops by more than 2.0 pp relative to B1.
3. Any per-norm audit gap is worse than B1 by more than 2.0 pp.
4. Any black-box/query gap is worse than B1 by more than 2.0 pp.
5. Clean accuracy drops by more than 2.0 pp relative to B1.
6. Training attack_step_units are not reduced by at least 40% relative to B1.

These are OR conditions. Triggering any one means the method does not support the primary deployment-cost claim.

Default interpretation:

- if Primary Deployment Success is not achieved, the result is null for the main method claim;
- secondary reliability observations may still be reported, but not as a win.

Null results must be reported honestly and used to revise the method rather than post-hoc changing metrics.

### CI Reporting Note

When reporting “within 2 pp of B1”, include confidence intervals for both B1 and B4, especially on 1,000-image audit subsets.

Fixed margins remain the pre-registered decision rule, but CI must be shown so readers can judge statistical uncertainty.

## 19. Run Queue v1 — Critic Approved

### Launch Policy

RQ0 and RQ1 are allowed to run immediately with owner approval:

- RQ0: harness reproduction;
- RQ1: audit-lite baseline.

Restrictions:

- RQ0/RQ1 must not be used to tune method, thresholds, pass/fail criteria, or success definitions;
- RQ1 audit-gap results are findings to report, not grounds to modify thresholds;
- no training claim can be made from RQ0/RQ1.

RQ2 and later are unblocked only after:

1. analytic compute model is completed;
2. predicted B4 saving is at least 50% before runtime overhead;
3. if analytic saving is below 50%, stop and patch R / refresh policy before launching training.

### Active Writer Constraint

Reactive Soft-T Full-10 seed-0 completion and checkpoint provenance are confirmed. RQ0/RQ1 must not write into or mutate that completed run directory; use isolated audit/eval output paths.

### RQ0 — Harness Reproduction

Goal:

- verify evaluation harness before any new claim.

Runs:

- evaluate an existing checkpoint under Primary APGD Union;
- compare against recorded/reference value.

Exit:

- PASS if Primary APGD Union within ±1.0 pp of reference;
- FAIL if outside ±1.0 pp.

If FAIL:

- stop;
- debug harness;
- do not proceed to audit or training.

### RQ1 — Audit-Lite Baseline

Goal:

- measure per-norm audit gaps for at least one baseline checkpoint;
- verify audit harness correctness;
- estimate runtime of audit suite;
- report baseline audit behavior.

Input:

- checkpoint passing RQ0.

Evaluation:

- 1,000-image fixed subset;
- audit suite v1 where available.

Exit:

- record per-norm audit gaps;
- record Full Audit Union;
- record confidence intervals.

RQ1 must not be used to tune the predictive allocation method, thresholds, or success criteria.

### RQ2 — B1 Full/Reactive Control

Status:

- blocked until analytic compute model passes.

Goal:

- establish high-compute control under current code path.

Training:

- full/reactive multi-source baseline;
- no predictive allocation.

### RQ3 — B2 Static Allocation

Goal:

- establish low-compute non-predictive baseline.

Training:

- fixed schedule or equal-probability source sampling.

### RQ4 — B4 Predictive Allocation v1

Goal:

- test main method.

Training:

- predictive source selection using EMA loss/binding;
- refresh every R=10 batches unless analytic model forces a patch;
- starvation safety S=30 batches.

Output:

- source selection trace;
- refresh trace;
- binding prediction accuracy on refresh batches;
- same metrics as RQ2.

Exit:

- classify result against Primary Deployment Success only.

### RQ5 — Optional Cheap-Probe Baseline

Goal:

- test whether simple cheap-probe selection explains gains.

Only run if RQ4 shows promising result and compute allows.

### RQ6 — Final Audit

Goal:

- run final audit on selected checkpoints.

Candidates:

- B1 full/reactive;
- B2 static;
- B4 predictive;
- B5 RAMP-style reference if reproduced.

Evaluation:

- standard 1,000-image audit;
- full-test audit only if compute allows.

## 20. Task Contract — RQ0 Harness Reproduction — 2026-07-09

### Status

Approved to run under Pre-registration v1.

RQ0 is evaluation-only:

- no training;
- no method tuning;
- no threshold changes;
- no audit-threshold edits;
- no run queue changes.

### Objective

Verify that the evaluation harness reproduces the reference Primary APGD Union before any audit or training claim is made.

### Input

- Dataset: CIFAR-10 test set.
- Architecture: PreActResNet-18.
- Checkpoint: existing reference checkpoint from current project assets.
- Epsilon setting:
  - l1 = 12;
  - l2 = 0.5;
  - linf = 8/255.

### Required Evaluation

Run deterministic evaluation for:

- clean accuracy;
- APGD-linf robust accuracy;
- APGD-l2 robust accuracy;
- APGD-l1 robust accuracy;
- Primary APGD Union.

Primary APGD Union:

- sample is robust iff it survives APGD-linf, APGD-l2, and APGD-l1.

### Required Logging

Record:

- checkpoint path/hash;
- git commit hash;
- dataset split;
- normalization parameters;
- attack parameters;
- epsilon values;
- random seed;
- batch size;
- device;
- clean accuracy;
- per-norm APGD robust accuracies;
- Primary APGD Union;
- expected reference Primary APGD Union;
- absolute difference from reference.

### Exit Criteria

PASS:

- reproduced Primary APGD Union is within ±1.0 percentage point of the expected reference.

FAIL:

- reproduced Primary APGD Union differs from expected reference by more than ±1.0 percentage point.

If FAIL:

- stop;
- do not run RQ1;
- do not run training;
- debug harness.

### Forbidden During RQ0

- changing success thresholds;
- changing attack settings after seeing results;
- changing epsilon values;
- switching checkpoint after seeing results;
- tuning method parameters;
- reporting audit or training conclusions.

## 21. Codex/Claude Code Sync Prompt — 2026-07-09

Use the following prompt when handing this state to Codex/Claude Code:

```text
You are operating on the adversarial-robustness research repo.

TASK: Sync PROJECT_STATE.md only.

Read docs/PROJECT_STATE.md and update it to exactly match the provided PROJECT_STATE.md content from Kiet/ChatGPT. This is a state sync task, not a training/evaluation task.

Hard constraints:
- Do not train.
- Do not evaluate.
- Do not clean result folders.
- Do not regenerate dashboard while the active writer is incomplete.
- Do not edit train/eval/core training code.
- Do not modify result JSONs.
- Do not launch any run.
- Do not change thresholds, configs, or run queue beyond the provided PROJECT_STATE.md sync.

Repo facts to preserve:
- Current active writer may still be results/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k/s0/.
- Do not write into or mutate the active writer directory.
- docs/dashboard.html is generated from PROJECT_STATE.md + result JSONs; if dashboard is stale, leave it stale and mention regeneration is deferred until safe.

Required file target:
- docs/PROJECT_STATE.md

Expected exit condition:
- docs/PROJECT_STATE.md contains the new sections:
  - Strategic Scope Lock — 2026-07-09
  - Pre-registration v1 — Critic Approved — 2026-07-09
  - Evaluation and Audit Protocol — Critic Approved
  - Pass/Fail Criteria — Critic Approved
  - Run Queue v1 — Critic Approved
  - Task Contract — RQ0 Harness Reproduction — 2026-07-09
  - Codex/Claude Code Sync Prompt — 2026-07-09
- The top Last sync is updated to 2026-07-09 12:05 UTC or later.
- The state explicitly says RQ0/RQ1 are clear but RQ2+ requires analytic compute model >=50% saving before training.
- The state explicitly says analytic <50% saving means stop, patch R, and gate again.
- The state explicitly says RQ0/RQ1 must not be used to tune thresholds/method/success criteria.

After editing, print only:
1. git diff -- docs/PROJECT_STATE.md
2. a short summary of changed sections
3. confirmation that no training/eval/dashboard/result mutation was performed
```

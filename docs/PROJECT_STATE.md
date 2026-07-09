# PROJECT_STATE.md - single active repo contract

**Source-of-truth rule:** After every run/eval/frontier update, update this file first. The dashboard must be regenerated from this file plus result JSONs. If `docs/dashboard.html` and this file disagree, this file is the source of truth and the dashboard must be regenerated.

Last sync: 2026-07-09 UTC. Active branch family: `card-pb-fixes`; cleanup work may be on `cleanup/repo-contract`. Do not treat files under `docs/legacy/`, `docs/archive/`, or `configs/legacy/` as active protocol.

## Goal And Claim
- Goal: mechanism/method paper on difficulty-aware adversarial training for CIFAR-10 union robustness over linf, l2, and l1.
- Main comparator: RAMP under the same recipe.
- Main method: CARD-PB v2 predictive binding with confidence floor.
- Claim shape: match reactive/RAMP-recipe union robustness at lower attack-FLOPs, not a leaderboard-only claim.
- Current decision: keep the RAMP schedule. Do not propose an early LR drop as part of the paper protocol.

## Locked Protocol
- Dataset/model: CIFAR-10, PreActResNet-18, raw `[0,1]` pixels, `normalize: false`.
- Threat model: eps `(linf 8/255, l2 0.5, l1 12)`.
- Schedule: 80 epochs, SGD, lr `0.05`, milestone `[70]`, gamma `0.1` to lr `0.005`, momentum `0.9`, weight decay `5e-4`, `save_freq: 10`.
- Train attack: APGD train, steps `10/10/10`; reactive FLOPs denominator is `30`.
- Develop eval: APGD `20/20/100`, restarts `1` = DECISION grade for batch comparison and thresholds.
- Final claim eval: full AutoAttack via `--version standard` = FINAL grade, only for the selected winner/final claim.
- Frontier grid: `{2,4,8,16,24}`.
- Paper configs must be raw-run safe and must not rely on hidden CLI overrides.

## Current Results Table
| run | epoch | grade/source | clean | union | linf/l2/l1 | FLOPs | role |
|---|---:|---|---:|---:|---|---:|---|
| RAMP paper ep50 | 50 | APGD-only `100/100/100`, n1000 | 77.5 | 42.9 | 43.9/61.9/47.1 | 1.00 | reference |
| RAMP paper ep80 | 80 | APGD-only `100/100/100`, n1000, original paper ckpt `RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`, `results/ramp/eval_ramp_ep80_eps8255_apgd_n1000.json` | 81.2 | 46.0 | 47.3/65.8/49.6 | 1.00 | APGD-grade over-report; not decision/final grade |
| RAMP paper ep80 decision | 80 | APGD `20/20/100`, restarts1, n1000, same original paper ckpt, `results/ramp/eval_ramp_ep80_apgd_2020100_n1000.json` | 81.2 | 46.2 | 47.3/65.8/49.7 | 1.00 | DECISION grade anchor |
| RAMP paper ep80 standard | 80 | `--version standard`, n1000, same original paper ckpt, `results/ramp/eval_ramp_ep80_fullAA_n1000.json` | 81.2 | 46.2 | 47.3/65.8/49.7 | 1.00 | FINAL-command measurement; did not confirm expected 44.6 on n1000 |
| Arm A ep80 full-AA log | 80 | 5070ti reproduce log `results/exploration/ramp_armA_full_5070ti.log` | 80.9 | 44.7 | 46.1/65.9/48.8 | 1.00 | RAMP reproduce, final-grade log; matches paper lambda=5 target 44.6 +/- 0.6 |
| Arm A ep80 decision | 80 | APGD `20/20/100`, restarts1, n1000, `external/RAMP/trained_models/armA_rampfull_5070ti/ep_80_0.pth`, `results/ramp/eval_armA_rampfull_ep80_apgd_2020100_n1000.json` | 81.2 | 44.8 | 45.9/66.4/50.1 | 1.00 | reproduce cross-check, DECISION grade |
| reactive_apgd_8255 | 50 | APGD `20/20/100`, n1000 | 76.7 | 38.8 | 39.2/60.1/45.6 | 1.00 | reactive ref |
| predictive ks16 | 50 | APGD `20/20/100`, n1000 | 76.1 | 38.7 | 40.3/59.9/44.8 | 0.665 | l1 holds, FLOPs high |
| predictive ks2 | 50 | APGD `20/20/100`, n1000 | 77.2 | 39.4 | 41.4/59.9/44.2 | 0.584 | current frontier winner |
| predictive ks4 | 50 | APGD `20/20/100`, n1000 | 78.4 | 39.8 | 41.1/61.2/46.9 | 0.603 | l1 strong, just over target |
| predictive ks8 | 50 | APGD `20/20/100`, n1000 | 77.7 | 39.3 | 41.0/59.9/45.7 | 0.623 | holds, over target |
| predictive ks24 | 37/50 at sync | incomplete | pending | pending | pending | about 0.706 so far | active/incomplete |

## RAMP Grade Reconciliation
- T2 grade for `results/ramp/eval_ramp_ep80_eps8255_apgd_n1000.json`: original paper checkpoint `external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`, APGD-only `100/100/100`, restarts1, n1000. It is APGD-grade and can over-report; it is not DECISION grade `20/20/100` and not FINAL grade full AutoAttack.
- New T1a DECISION-grade re-eval of the same paper checkpoint at APGD `20/20/100` gives clean/linf/l2/l1/union = `81.2/47.3/65.8/49.7/46.2`.
- New T1b `--version standard` re-eval of the same paper checkpoint on n1000 gives the same `81.2/47.3/65.8/49.7/46.2`; this did not confirm the expected `~44.6-44.7` for that checkpoint/subset, so do not cite it as resolving paper-ckpt `46.0` vs `44.6`.
- Arm A reproduce full-AA log is resolved: clean/linf/l2/l1/union = `80.9/46.1/65.9/48.8/44.7`; the `46.1` number is linf-only, not union.
- Arm A DECISION-grade APGD `20/20/100` cross-check gives clean/linf/l2/l1/union = `81.2/45.9/66.4/50.1/44.8`, consistent with the Arm A full-AA log union within `0.1pp`.

## Current Next Action
- Kiet-requested next controlled comparison: seed-0 full-80 Colab/RAMP80 batch under the locked protocol. Run develop eval APGD `20/20/100` after each run; run full AutoAttack only for the batch winner after the batch is complete.
- Batch order:
  1. control fixed-10: `configs/paper/reactive_ramprecipe.yaml`
  2. flat-6: `configs/exploration/flat6_ramp80.yaml`
  3. predbind ks4: pin config plus kspan/floor before launch (`__TBD__`)
  4. curriculum-rescaled: `configs/exploration/idea3_curriculum_ramp80.yaml`
  5. idea1 fail-rate: `configs/exploration/idea1_failrate_ramp80.yaml`
- Active writer status: no `scripts/train.py` process at 2026-07-09 01:30 UTC. Run #5 idea1 fail-rate has completed to `results/idea1_failrate_ramp80_apgd_8255/s0/` with `ckpt/ep080.pt`, `ckpt/last.pt`, `train.json`, and `eval.json`; do not overwrite those files.
- Control fixed-10 eval is BLOCKED until `results/reactive_ramprecipe/s0/ckpt/best.pt` exists from the control training run. Requested command: `python scripts/evaluate.py --config configs/paper/base_ramp_apgd_8255.yaml --checkpoint results/reactive_ramprecipe/s0/ckpt/best.pt --version apgd --n-examples 1000 --bs 250`. Current `evaluate.py` would otherwise infer run name `best` from `best.pt`, so use `--run-name reactive_ramprecipe` or `--out results/reactive_ramprecipe/s0/eval.json` when actually running it.
- W&B online preflight passed on 2026-07-08 for run names `idea1_failrate_ramp80_apgd_8255_s0`, `idea2_kmin_recovery_ramp80_apgd_8255_s0`, and `idea3_curriculum_ramp80_apgd_8255_s0`.

### Pre-registration Placeholders
- Winner rule / union threshold: `__TBD__`
- Per-norm floors and l1 risk tolerance: `__TBD__`
- Predbind ks4 kspan/floor: `__TBD__`

## Method Mechanics
- Reactive reference: per-sample soft objective, temperature `T=0.25`, all three APGD source attacks at full `10/10/10`.
- CARD-PB v2: per-sample binding predictor gives full budget to the predicted norm and floor budget to the others.
- Confidence floor formula: `floor_g = clamp(k_min + round(kspan * miss_vol_g), k_min, full_g)`.
- Floor driver: `miss_vol_g = P(true_bind=g and b_hat!=g)`, measured on full-attack recalibration samples and EMA-smoothed.
- `miss_rate_g` is a conditional diagnostic log only; it must not drive the confidence floor.
- Adaptive probe guard is OFF in confidence mode.
- Predictor state update rule: `Lbar` is updated only from full-budget norms, not from under-attacked floored losses.
- FLOPs ratio: measured attack passes divided by reactive APGD budget `10+10+10 = 30`.
- Canonical train FLOPs key for all methods: `efficiency/attack_flops_ratio`; method-specific `exp/attack_flops_ratio` and `pb/attack_flops_ratio` may remain as aliases.
- Canonical eval keys are additive aliases in `eval.json` and W&B summary: `eval/checkpoint`, `eval/clean_acc`, `eval/robust_linf`, `eval/robust_l2`, `eval/robust_l1`, `eval/worst_union`, attack-step keys, and `eval/seed`.

## Paper Lane Configs And Runs
| config | role | result dir | tags |
|---|---|---|---|
| `configs/paper/base_ramp_apgd_8255.yaml` | canonical protocol base, APGD train `10/10/10`, eval `20/20/100` | `results/` | `lane:paper` |
| `configs/paper/reactive_ramprecipe.yaml` | reactive comparator, per-sample soft `T=0.25` | `results/` | `lane:paper` |
| `configs/paper/cardpb_v2.yaml` | CARD-PB v2 confidence floor, `pb_floor_kspan: 12`, `pb_k_floor: 3`, adaptive floor false | `results/` | `lane:paper` |

Canonical paper commands, only after confirming GPU/result writers are clear:

```bash
python scripts/train.py --config configs/paper/reactive_ramprecipe.yaml
python scripts/train.py --config configs/paper/cardpb_v2.yaml
python scripts/evaluate.py --config configs/paper/base_ramp_apgd_8255.yaml --checkpoint <ckpt> --version apgd
```

## Exploration Lane Status
- Default exploration/dev outputs stay separate from paper results and write under `results/exploration/`.
- Idea 1 `configs/exploration/idea1_failrate.yaml`: fail-rate early stop; promising signal, but FLOPs may rise.
- Idea 2 `configs/exploration/idea2_kmin_recovery.yaml`: most promising exploration finding; tests k-min recovery under CARD-PB.
- Idea 3 `configs/exploration/idea3_curriculum.yaml` plus `idea3_control_fixed10.yaml`: curriculum/control ablation only.
- Current RAMP80 idea-compare configs write to top-level `results/<run>/s0/` by explicit orchestration request; these are one-seed challenger screens, not final paper claims.
- RAMP generality scripts are exploration-only and must not contaminate paper configs or paper result claims.

## Known Risks And Blockers
- Active writer risk: do not edit, truncate, summarize as final, or regenerate from moving result files unless the parser is partial-safe.
- CARD-PB checkpoint risk: model, optimizer, scheduler, epoch, RNG, and DRO state are restored, but `pb_Lbar`, `pb_seen`, and confidence-floor EMA state may not be checkpointed. Predictive Phase-2 continuation may not be trajectory-equivalent until this is fixed or deliberately accepted.
- Dashboard risk: regenerate only when safe. If stale, this file overrides the dashboard.
- Legacy risk: old `0.03`, 50-epoch, and plain-PGD documents/configs are archived provenance only.

## Run-Update Checklist
- Check for active writers before editing summaries or regenerating dashboard.
- For each meaningful run/eval: record run name, seed, config path, git commit, epoch, union, per-norm robust acc, FLOPs ratio, schedule, and paper/exploration role.
- Update this file first.
- Regenerate `docs/dashboard.html` from this file plus result JSONs when safe.
- Keep paper-lane configs under `configs/paper/`; keep exploration under `configs/exploration/`; keep smoke/dev under `configs/dev/`.
- Do not launch training/eval from cleanup/review turns unless Kiet explicitly asks.

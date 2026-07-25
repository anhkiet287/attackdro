# Program A G2 evidence packet

Date: 2026-07-11 UTC  
Mode: G2-RUN, evaluation only; no training; no Program-A result evaluation; no `results/` mutation.

## 1. Provenance and execution boundary

- Branch: `feat/programA-g2-impl`
- HEAD: `f4f937005223aa928bc3e4a83b6c0414a1475fc6`
- Canonical audit/eval harness commit: `50b4e5cb7abb7e2366ef0d0627963d9bc54dc804`
- B3-R1 checkpoint: `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/ckpt/val_best.pt`
- Checkpoint SHA-256: `4676e75fe06b4a5751db45c1284a7f1c68a0628c18b0748b064b89e5cdcdb769` — exact locked match.
- No `scripts/train.py` writer was active at precheck. No training command was run. W&B was disabled for every calibration command.
- `git status --porcelain=v1 results` was empty after execution. Existing files under `results/` were read-only inputs.
- Pre-existing unrelated worktree state was preserved: modified `docs/dashboard.html` and pre-existing untracked files/directories. G2-RUN added only the six isolated calibration JSONs, three generated arm configs, one audit config, and this evidence packet.

Runtime script hashes:

| File | SHA-256 |
|---|---|
| `scripts/evaluate.py` | `28c119c908459837f9de9a729ec3faabc6c72c3e3a7de664deaba3c43772ac5f` |
| `scripts/eval_multinorm_audit.py` | `ce9cd3c592da1fbd37fa0744f85ee27422060addf93af93e05163a1c984d978c` |
| `scripts/dev/gen_program_a_configs.py` | `98d55a4c9907b78d2e844b93ed77587655fd0f2ddb6c0d3f1286a50527d666b8` |
| `scripts/dev/compute_preflight_program_a.py` | `1202dc4b6e64af0e082f166e35bcb9dea04de4ba3266033ca298927e4a185bec` |
| `scripts/dev/make_program_a_subsets.py` | `cbbcc4e6431a3ddd8b587fb2079d6376f272f366ac6f3d66c8a5a7d1ba71f3ef` |

## 2. Split and selection gate evidence

Canonical half-open ranges:

| Role | Source | Range | n |
|---|---|---:|---:|
| `train_core` | CIFAR-10 train | `[0,49000)` | 49,000 |
| `val_select` | CIFAR-10 train | `[49000,50000)` | 1,000 |
| `test_monitor` | CIFAR-10 test | `[0,1000)` | 1,000 |
| `test_final` | CIFAR-10 test | `[1000,9000)` | 8,000 |
| `cal` | CIFAR-10 test | `[9000,10000)` | 1,000 |

`assert_canonical_splits_disjoint()` passed. Direct set checks returned:

- `cal ∩ test_monitor = 0`
- `cal ∩ test_final = 0`
- new audit subset `∩ cal = 0`
- new audit subset `∩ test_monitor = 0`

Selection gate:

- `SELECTION_ELIGIBLE_SPLITS = {val_select}`.
- `cal`, `test_monitor`, and `test_final` each raised the fail-closed `assert_not_selection_split` error.
- Training selection key is exactly `val_select/worst_union`; the fit loop checks every non-selection split's `used_for_selection` flag before reading that key.
- All six calibration JSONs record `eval_split=cal`, `used_for_selection=false`, and `selected_by_test=false`.

Frozen split artifacts:

| Artifact | SHA-256 |
|---|---|
| `results/audit/subsets/cal_cifar10_test_9000_9999.json` | `fa0d949dc63a90ad0146d3aa857f5de1cdb253745b26a77a2941d9e0b083191d` |
| `results/audit/subsets/cifar10_testfinal_1000_seed20260709_v3A.json` | `af0b037aa964b438facaa911c9f6337e9c5233a72efa5453dc5eaa35fe8f8aef` |

## 3. Locked calibration

Protocol: B3-R1 `val_best.pt`; split `cal`; n=1000; APGD-CE + APGD-t; Linf/L2/L1 iterations 20/20/100; one restart; seed 0; batch size 250; Linf ε=8/255 and L2 ε=0.5 fixed. Grid was exactly `{16,20,24,28,32,40}` and was not extended.

Admissibility: `a_l1 <= a_linf - 0.03 AND a_l1 <= a_l2 - 0.03 AND a_l1 >= 0.10`.

| L1 ε | Linf racc | L2 racc | L1 racc | Admissible | JSON SHA-256 |
|---:|---:|---:|---:|---|---|
| 16 | 0.423 | 0.689 | 0.366 | yes | `1009d374de0017a155d8bf654c4c2b5f83d8385748bbd2c15c8414305377adaf` |
| 20 | 0.423 | 0.689 | 0.256 | yes | `162ab7ccb0cdee65c8e05298c219c46be9c622ce8340859b859e8268dfab9a05` |
| 24 | 0.423 | 0.689 | 0.175 | yes | `36861f15a4f4a512b956c58193e49d8c9c81dafce0dfefebd969f7e704de775f` |
| 28 | 0.423 | 0.689 | 0.129 | yes | `c92f839a536708033ad3d97fe1ba424f897a536d0bbd7b173b889e76833dc7b1` |
| 32 | 0.423 | 0.689 | 0.076 | no: non-collapse floor | `df8d64cd5f336dc2f760144afa002c72560512a7bbce388bf932071781a03bd0` |
| 40 | 0.423 | 0.689 | 0.025 | no: non-collapse floor | `2980adac3dbe2553b272179339becbe14ef491f93157728b370d3ad9cd46d0e8` |

Selected by the locked smallest-admissible rule: **L1 ε = 16**.

Output JSONs: `g2_evidence/programA/calibration/l1_eps{16,20,24,28,32,40}.json`.

Exact command pattern, instantiated once for every grid value:

```bash
.venv/bin/python scripts/evaluate.py --config configs/paper/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k.yaml --checkpoint results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/ckpt/val_best.pt --model-family robustdro --eval-split cal --checkpoint-role val_best --norms linf l2 l1 --version apgd --n-examples 1000 --bs 250 --seed 0 --set threat_model.l1.eps=<GRID_VALUE> --out g2_evidence/programA/calibration/l1_eps<GRID_VALUE>.json --no-wandb
```

All six commands exited 0.

## 4. Generated Program-A configs and checkpoint-save proof

Generator command (exit 0):

```bash
.venv/bin/python scripts/dev/gen_program_a_configs.py --l1-eps 16 --out-dir configs/paper
```

| Arm config | Allocation | SHA-256 |
|---|---|---|
| `configs/paper/programA_b4_adaptive_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml` | `predictive_refresh` | `6f2dd502a033ee0fd096fc59d94dda3db2f4e828bcb414ad8daea75d201eb914` |
| `configs/paper/programA_b3_mis_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml` | `static_cycle`, Linf-informed R1 schedule | `99b0c257f02ed450095f0041c1d9dfad1bb2d7a4171b5d8a90415d535190884a` |
| `configs/paper/programA_b3_bottleneck_informed_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml` | `static_cycle`, L1-informed schedule | `56baed33ae7d09d7ff8e3a27922c562370da87989887c85863a8fed6544d981e` |

All resolve to L1 ε=16 and `save_freq=10`. The 1-indexed periodic checkpoint expression is `save_checkpoint(f"ep{epoch + 1:03d}", ...)` when `(epoch+1) % save_freq == 0`; for 80 epochs this proves filenames `ep020.pt`, `ep040.pt`, `ep060.pt`, and `ep080.pt`. `last.pt` and selection-only `val_best.pt` remain separately saved.

## 5. Finite-run compute preflight

Command (exit 0):

```bash
.venv/bin/python scripts/dev/compute_preflight_program_a.py configs/paper/programA_b4_adaptive_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml configs/paper/programA_b3_mis_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml configs/paper/programA_b3_bottleneck_informed_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml
```

| Arm | Mode | B | Special | N_special | Total attack-step units |
|---|---|---:|---|---:|---:|
| B4-adaptive | predictive refresh | 30,560 | refresh | 3,056 | 366,720 |
| B3-mis | static cycle | 30,560 | extra | 6,112 | 366,720 |
| B3-bottleneck-informed-L1 | static cycle | 30,560 | extra | 6,112 | 366,720 |

Result: **`tau_C = max(total)-min(total) = 0`**, exact pass. B4 `predictive_use_cheap_probe=false` and `score_rule=loss_plus_bind`; no uncounted predictor attack is enabled.

## 6. Canonical audit dry-run freeze

Audit config: `configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml`  
Config SHA-256: `9162ce4497e77cde5ead9340340a32b17082ddc5e8cd20ab05baf55c10b20585`  
Subset SHA-256: `af0b037aa964b438facaa911c9f6337e9c5233a72efa5453dc5eaa35fe8f8aef`

Exact command (exit 0):

```bash
.venv/bin/python scripts/eval_multinorm_audit.py --config configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml --checkpoint results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/ckpt/val_best.pt --run-id programA_g2_dryrun_b3_r1 --checkpoint-role val_best --out g2_evidence/programA/audit_dryrun/audit_multinorm_v3A.json --device cuda --model-family robustdro --export-masks --validate-mask-sidecar --dry-run
```

Dry-run facts: checkpoint argument accepted but not loaded; dataset not loaded; attacks not run; no JSON or masks written. All components were import-available.

Pre-results matrix:

| Norm | APGD-CE | APGD-DLR | FAB-T | Square |
|---|---|---|---|---|
| Linf | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| L2 | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| L1 | REQUIRED | REQUIRED | REQUIRED | REQUIRED |

**EXCLUDED: none.** The locked two-identical-clean-runs incompatibility branch was not triggered because this was a non-executing dry-run and reported all components available. Any future qualifying exclusion must follow the locked pre-results mechanism; result values may not inform inclusion.

## 7. Gate outcome

G2 evidence conditions 1–13 executed here pass: split/hash proof, no-selection gate, exact checkpoint identity, locked calibration with ε₁=16 selected, generated configs and hashes, periodic checkpoint proof, `tau_C=0`, new test-final subset hash, frozen all-required audit dry-run, and commit/dirty-worktree provenance.

This packet does **not** authorize training. Program-A launch remains blocked pending ATLAS-CRITIC G2 Pass and a separate director launch authorization.

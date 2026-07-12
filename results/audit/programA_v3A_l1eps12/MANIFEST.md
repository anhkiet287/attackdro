# Audit manifest — Program A symmetric canonical Tier-1 audit

**Canonical evaluation radii (clarification — critic-authorized):** the canonical union is evaluated at the **standard CIFAR-10 triple (ℓ∞ 8/255, L2 0.5, L1 12)**; the L1 eps=16 used to train the arms is a **training stressor** (to shift the training bottleneck), NOT an evaluation radius. RE-P0/P1/P2 therefore concern **U(8/255, 0.5, 12)**.

## Provenance
- Audit config: `configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml` — SHA-256 `9162ce4497e77cde5ead9340340a32b17082ddc5e8cd20ab05baf55c10b20585`
- Audit subset: `results/audit/subsets/cifar10_testfinal_1000_seed20260709_v3A.json` — SHA-256 `af0b037aa964b438facaa911c9f6337e9c5233a72efa5453dc5eaa35fe8f8aef` (1000, class-balanced, test_final only)
- eps: ℓ∞ 8/255, L2 0.5, L1 12.0 (harness-locked standard triple)
- Components: 12 REQUIRED (APGD-CE, APGD-DLR, FAB-T, Square × linf/l2/l1), seed 20260709
- Frozen RE-P analysis script: `scripts/dev/analyze_endogenous_statics.py` — SHA-256 `aa8994f41b00e8e72af7ae4b4d630b7ef80636a2b5dc1888c3bbaae826c60873`
- Withdrawn: `audit_..._testfinal_l1eps16.yaml` (eps16 sibling superseded — canonical audit is the standard triple)

## Arms (val_best.pt)
| arm | dir | checkpoint SHA-256 |
|---|---|---|
| B4-adaptive | `b4_adaptive/` | `c2d708dd1b2da5379d26b0e9cc4318c76cb0d93790401b920824b718aed49439` |
| B3-static-L∞-weighted (b3_mis) | `b3_mis/` | `bdc4bd23221e97ecd1b043f551794ceb6d8db4aeda8383c5f72606b5f90a3cb1` |
| B3-static-L1-weighted (b3_bottleneck_informed) | `b3_bottleneck_informed/` | `09062f6f56d10c0c2eed64ff0d2c6b020298860437ba7cade33dd7cf04319f24` |

Per-arm outputs (`audit.json`, `masks_multinorm_v1.npz`) and derived union/per-norm masks + hashes + the REQUIRED/EXCLUDED matrix are appended on audit completion.

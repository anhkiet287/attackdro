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

## Results — COMPLETED 2026-07-12 (canonical, standard triple)

**Component execution:** all **12/12 REQUIRED components executed** per arm (APGD-CE, APGD-DLR, FAB-T, Square × ℓ∞/L2/L1), **0 skipped, 0 EXCLUDED**. No required component errored, so the mechanical exclusion branch was not triggered. Harness `--validate-mask-sidecar` = PASS for all arms.

**REQUIRED/EXCLUDED matrix** (all three arms identical):

| norm | APGD-CE | APGD-DLR | FAB-T | Square |
|---|---|---|---|---|
| ℓ∞ | REQUIRED·ran | REQUIRED·ran | REQUIRED·ran | REQUIRED·ran |
| L2 | REQUIRED·ran | REQUIRED·ran | REQUIRED·ran | REQUIRED·ran |
| L1 | REQUIRED·ran | REQUIRED·ran | REQUIRED·ran | REQUIRED·ran |

**Index alignment:** all 3 arms share subset SHA `af0b037…`, n=1000, same order → masks are paired/aligned on test_final.

**Per-arm canonical robust accuracy** (per-norm = AND of that norm's 4 components; union = AND over all 12; True=robust):

| Arm | clean | ℓ∞ | L2 | L1 | UNION | bottleneck (argmin) |
|---|---|---|---|---|---|---|
| B4-adaptive | 0.8170 | 0.4450 | 0.6710 | 0.5310 | **0.4340** | ℓ∞ |
| B3-static-L∞-weighted | 0.8300 | 0.4570 | 0.6670 | 0.5160 | **0.4420** | ℓ∞ |
| B3-static-L1-weighted | 0.8250 | 0.3710 | 0.6680 | 0.5830 | **0.3700** | ℓ∞ |

Derived union/per-norm means match `audit.json` to float32 precision (≤2e-8).

**Mask artifacts + SHA-256** (union + 3 per-norm `.npy` are the frozen RE-P script inputs):

**B4-adaptive** (`b4_adaptive/`)
  - audit.json `a2e6baa522c4acea499594a85438e5ca04c5b097bf77ad6f1b80bdb42e3099f8`
  - masks_multinorm_v1.npz `9a199614ef947a5621a06935ab6ae34becc40061352ed61d788f7266ffe11563`
  - mask_union.npy `ab9ce21dcbcd062aca01ac0c7c579ea909803854aef1ed2d012b2b6d7ff0b76f`
  - mask_linf.npy `cabd4306ae3469c0e4e671b9413315a87a9bc355d66027e8e6b86404b85a81ab`
  - mask_l2.npy `f175f6ad44edfbb37d69e531410c6bf6863d523e8d79ce5999a6e72b6c3e901c`
  - mask_l1.npy `13ce0337818bc466e86bda4be970b63fac81c5ed60d0a65db84675b375572593`
**B3-static-L∞-weighted** (`b3_mis/`)
  - audit.json `403bffabc7483271d97fd2749f24b3fc6cd2c85acf514f7c28e3da3c4a0b765b`
  - masks_multinorm_v1.npz `a5f4216ac59b6632da4272b508e4b91110f5f6b2b78449914feb72f81564ba60`
  - mask_union.npy `448841eb49a32b6b06aefc2d11aa85fdfb17e9b122d88782c0f4586689a5ba36`
  - mask_linf.npy `2818e6d525ce2f238c9279b528d55ea411d803f4fde32810e2ab311a5dc1a876`
  - mask_l2.npy `ef12e0152de737f0d151797560ba7d15479f93a070be882847cf640fd4800f9a`
  - mask_l1.npy `cc5001b8e87f7913f079e363201f5410495be6e64dcfb669373490a885f2b4d2`
**B3-static-L1-weighted** (`b3_bottleneck_informed/`)
  - audit.json `665c73939cecacb23a39d28a6e28fb281a1bd611d550d02a257ea87ad0207a82`
  - masks_multinorm_v1.npz `1189b67ded48964a9a98a842c6a1bebe76033f79d6195ebefe5529960460bd3b`
  - mask_union.npy `de59280e0fddfc1d03b885b3e3116e4c76cbd2c3b176abf3a4c6d3a747bdbb98`
  - mask_linf.npy `d767a98449a17edcb50e9e81bc182a2d1877bcb4eac05d212ec84343e1af2208`
  - mask_l2.npy `61d1e40d87aaf4330e9dad5d242a5ad7275c94a644ab020455e4667f34faa466`
  - mask_l1.npy `a8eddfd13b99c651b8bd1ef1f67eecabe2d1e1c47312d0ab2507ffbf2547defa`

_RE-P analysis NOT run (frozen script `aa8994f4…` awaits separate authorization)._

# RESULTS MASTER — CLAMP (CLean-Anchored Multi-norm Pull-push)

Generated 2026-07-21 · CIFAR-10 · PreActResNet-18 · triple (ℓ∞ 8/255, ℓ₂ 0.5, ℓ₁ 12)
Harness: frozen 12-AA `scripts/eval_multinorm_audit.py`, schema `audit_cifar10_preactrn18_multinorm_v1`, config sha `9162ce44`
Every number below is recomputed from per-example masks in `results/main/<arm>/<scale>/masks_multinorm_v1.npz`.
Verification at last sync: **union(eval.json) == union(masks) — 40 pass / 0 mismatch**.

## 0. Tier semantics (read before quoting any number)

| tier | components | driver |
|---|---|---|
| `full` | **12** (apgd_ce · apgd_dlr · fab_t · square) × (ℓ∞, ℓ₂, ℓ₁) | `eval_multinorm_audit.py` |
| `no_square` | **9** (12 − 3 square) | `union_bench_eval.py --skip-square` |

`union(no_square) ≥ union(full)` always. Measured drift 0…+0.0010 over 29 audits, non-zero in 7.
**Tiers are not interchangeable** — never mix them inside one column.
Paired bootstrap: sample-level, B=10000, `np.random.default_rng(0)`, 1-sided 95% LCB at q=0.05.

## 1. Headline paired results @10k (tier = full, 12 attacks)

| pair | A (term ON) | B (term OFF) | Δ | 95% CI | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ |
|---|---|---|---|---|---|---|---|---|
| **CLAIM A seed 0** | 0.4173 | 0.3886 | **+0.0287** | [+0.0234, +0.0339] | **+** | +0.0229 | +0.0015 | +0.0481 |
| **CLAIM A seed 2** | 0.4195 | 0.3943 | **+0.0252** | [+0.0202, +0.0301] | **+** | +0.0217 | +0.0108 | +0.0432 |
| **CLAIM A seed 3** | 0.4227 | 0.3970 | **+0.0257** | [+0.0212, +0.0303] | **+** | +0.0177 | +0.0084 | +0.0468 |
| M1b (clean negatives) | 0.4139 | 0.3886 | +0.0253 | [+0.0201, +0.0304] | + | +0.0180 | +0.0010 | +0.0483 |
| **REVERSAL seed 0** (full budget) | 0.4201 | 0.4297 | **−0.0096** | [−0.0142, −0.0048] | **−** | −0.0077 | +0.0075 | +0.0008 |
| **REVERSAL seed 1** (full budget) | 0.4230 | 0.4288 | **−0.0059** | [−0.0110, −0.0006] | **−** | −0.0026 | −0.0045 | −0.0056 |
| FT-∞ (fine-tune RobustBench) | 0.4026 | 0.3991 | +0.0035 | [−0.0012, +0.0083] | ns | +0.0074 | +0.0094 | +0.0109 |
| H3 — 1-view MSD glue | 0.4012 | 0.3886 | +0.0126 | [+0.0075, +0.0177] | + | +0.0122 | −0.0093 | +0.0111 |
| 3-view vs 1-view glue | 0.4173 | 0.4012 | +0.0161 | [+0.0111, +0.0211] | + | +0.0107 | +0.0108 | +0.0370 |

**CLAIM A holds on 3/3 seeds** (seed-min LCB +0.0202 > 0). Gain routes through **ℓ₁** (+4.3…+4.8 pp), then ℓ∞; ℓ₂ ≈ flat.

**REVERSAL replicates on 2/2 seeds.** At the Maini 50-epoch full budget with MSD-50 views the sign flips negative and is significant both times. This is a real, pre-registered negative result, not noise.

## 2. Comparison against RAMP (CLAIM B) @10k, tier = full

| pair | ours | RAMP | Δ | 95% CI | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ |
|---|---|---|---|---|---|---|---|---|
| M1a − R′ (RAMP from-scratch) | 0.4173 | 0.4461 | **−0.0288** | [−0.0342, −0.0234] | − | **−0.0312** | +0.0086 | **+0.0233** |
| M1a_full − B1 (RAMP full) | 0.4201 | 0.4456 | **−0.0255** | [−0.0308, −0.0201] | − | **−0.0246** | +0.0005 | −0.0059 |

**Structural reading:** RAMP is ℓ∞-strong / ℓ₁-weak; CLAMP is the mirror image. The union is ℓ∞-bound, so our ℓ₁ advantage does not convert into union. One mechanism explains both the RAMP gap and the full-budget reversal.

**Efficiency claim is DROPPED.** Measured wall-clock: RAMP 268 s/ep, 5.96 h, 3 adv/step vs M1a_full 365 s/ep, 8.11 h, 4 adv/step. RAMP is cheaper *and* better on union — no efficiency argument survives.

## 3. Base-generality 2×2 (@1k, tier = no_square, 9 attacks)

CE base varied; the CLAMP term is byte-identical across arms (3 per-norm APGD-10 views, α=β=0.5, τ=0.1, neg=adv, warmup 10).

| # | CE base | term ON | term OFF | Δ | 95% CI | sig |
|---|---|---|---|---|---|---|
| headline | MSD-10 | 0.4370 | 0.3920 | **+0.0451** | [+0.0280, +0.0620] | **+** |
| **#11** | MAX (worst-of-3 APGD-10) | 0.4300 | **0.4530** | **−0.0230** | [−0.0380, −0.0080] | **−** |
| **#12** | AVG (mean-of-3 APGD-10) | 0.4250 | *pending* | — | — | — |

Base effect with the term held ON: MAX−MSD −0.0071 (ns), AVG−MSD −0.0121 (ns).
Base effect with the term OFF: **MAX−MSD +0.0610 [+0.0440, +0.0780] SIGNIF +**.

**Interpretation (negative, report-regardless):** the CLAMP gain is **not base-general**. On a MAX base the term *hurts* significantly. The MAX control alone (0.4530) is the strongest 80-epoch from-scratch arm we have — stronger than MSD+CLAMP (0.4370). The gain we measure on the MSD base looks like it is repairing a weakness specific to MSD, not adding robustness on top of a strong base.

## 4. View-staleness probe (v50)

`M1a_full_v50` = full budget with 50-step attack views instead of 10-step, @1k no_square:

| pair | A | B | Δ | 95% CI | sig |
|---|---|---|---|---|---|
| M1a_full_v50 − M0_full | 0.4490 | 0.4350 | +0.0139 | [−0.0030, +0.0310] | ns |
| M1a_full_v50 − M1a_full | 0.4490 | 0.4340 | +0.0150 | [−0.0020, +0.0320] | ns |

Per-norm vs M0_full: Δℓ∞ +0.0120, Δℓ₂ +0.0040, **Δℓ₁ +0.0340**.
The sign flips back positive when views are strengthened, but @1k is underpowered (CI half-width ≈1.7 pp vs a ~1 pp effect). **Not a claim until run @10k.**

## 5. Full arm inventory

Legend: `atk` = attack components; **12** = full tier, **9** = no_square tier.

### CLAIM A — from-scratch, MSD base, 80 ep
| arm | scale | n | atk | clean | union | ℓ∞ | ℓ₂ | ℓ₁ |
|---|---|---|---|---|---|---|---|---|
| M1a | 1k | 1000 | 12 | 0.8180 | **0.4370** | 0.4520 | 0.6700 | 0.5190 |
| M1a | 10k | 10000 | 12 | 0.8164 | **0.4173** | 0.4269 | 0.6636 | 0.5148 |
| M0 | 1k | 1000 | 12 | 0.8450 | **0.3920** | 0.4120 | 0.6670 | 0.4590 |
| M0 | 10k | 10000 | 12 | 0.8379 | **0.3886** | 0.4040 | 0.6621 | 0.4667 |
| M1b | 10k | 10000 | 12 | 0.8157 | **0.4139** | 0.4220 | 0.6631 | 0.5150 |
| M1a_seed2 | 10k | 10000 | 12 | 0.8113 | **0.4195** | 0.4293 | 0.6658 | 0.5153 |
| M0_seed2 | 10k | 10000 | 12 | 0.8392 | **0.3943** | 0.4076 | 0.6550 | 0.4721 |
| M1a_seed3 | 10k | 10000 | 12 | 0.8171 | **0.4227** | 0.4299 | 0.6638 | 0.5209 |
| M0_seed3 | 10k | 10000 | 12 | 0.8300 | **0.3970** | 0.4122 | 0.6554 | 0.4741 |

### FULL BUDGET — Maini 50 ep, MSD-50 views
| arm | scale | n | atk | clean | union | ℓ∞ | ℓ₂ | ℓ₁ |
|---|---|---|---|---|---|---|---|---|
| M1a_full | 1k | 1000 | 12 | 0.8080 | **0.4340** | 0.4500 | 0.6620 | 0.5050 |
| M1a_full | 10k | 10000 | 12 | 0.8063 | **0.4201** | 0.4298 | 0.6563 | 0.5057 |
| M1a_full | last | 1000 | 12 | 0.8260 | **0.4320** | 0.4420 | 0.6700 | 0.5120 |
| M0_full | 1k | 1000 | 12 | 0.8170 | **0.4350** | 0.4450 | 0.6540 | 0.5140 |
| M0_full | 10k | 10000 | 12 | 0.8055 | **0.4297** | 0.4375 | 0.6488 | 0.5049 |
| M0_full | last | 1000 | 12 | 0.8370 | **0.4320** | 0.4380 | 0.6520 | 0.5040 |
| M1a_full_seed1 | 1k | 1000 | 9 | 0.7960 | **0.4330** | 0.4470 | 0.6430 | 0.5160 |
| M1a_full_seed1 | 10k | 10000 | 12 | 0.7851 | **0.4230** | 0.4327 | 0.6420 | 0.5042 |
| M0_full_seed1 | 1k | 1000 | 9 | 0.8290 | **0.4370** | 0.4490 | 0.6640 | 0.5070 |
| M0_full_seed1 | 10k | 10000 | 12 | 0.8046 | **0.4288** | 0.4353 | 0.6465 | 0.5098 |
| M1a_full_v50 | 1k | 1000 | 9 | 0.8120 | **0.4490** | 0.4570 | 0.6580 | 0.5470 |

Last-epoch vs val-best selection: Δ ≈ 0 (ns) @1k for the full-budget pair — the reversal sign is **not** an artefact of the selection rule; @1k is simply underpowered (CI ±1.5 pp vs a 0.96 pp effect). Headline stays val-best.

### ABLATION / EXTENSION
| arm | scale | n | atk | clean | union | ℓ∞ | ℓ₂ | ℓ₁ |
|---|---|---|---|---|---|---|---|---|
| M1a_msdglue (1-view) | 1k | 1000 | 9 | 0.8320 | **0.4140** | 0.4320 | 0.6650 | 0.4750 |
| M1a_msdglue | 10k | 10000 | 12 | 0.8215 | **0.4012** | 0.4162 | 0.6528 | 0.4778 |
| M1a_max (#11 term ON) | 1k | 1000 | 9 | 0.8240 | **0.4300** | 0.4490 | 0.6720 | 0.5060 |
| M0_max (#11 control) | 1k_nosq | 1000 | 9 | 0.8250 | **0.4530** | 0.4680 | 0.6630 | 0.4980 |
| M1a_avg (#12 term ON) | 1k_nosq | 1000 | 9 | 0.8300 | **0.4250** | 0.4330 | 0.6830 | 0.5340 |
| ft_clamp | 10k | 10000 | 12 | 0.8542 | **0.4026** | 0.4342 | 0.6876 | 0.4754 |
| ft_none | 10k | 10000 | 12 | 0.8662 | **0.3991** | 0.4268 | 0.6782 | 0.4645 |
| M1a_c100 (CIFAR-100) | 1k | 1000 | 9 | 0.5650 | **0.1960** | 0.1990 | 0.3750 | 0.2580 |
| M0_c100 (CIFAR-100) | 1k | 1000 | 9 | 0.5780 | **0.1890** | 0.1980 | 0.3710 | 0.2500 |

CIFAR-100 @1k no_square: Δunion **+0.0070 (ns)**, Δℓ₁ +0.0080 (thesis-consistent direction), Δclean −1.3 pp. Underpowered; @10k not yet run.

### RAMP / CLAIM B, public baselines, legacy arc
| arm | scale | n | atk | clean | union | ℓ∞ | ℓ₂ | ℓ₁ |
|---|---|---|---|---|---|---|---|---|
| Rprime (RAMP from-scratch) | 10k | 10000 | 12 | 0.8118 | **0.4461** | 0.4581 | 0.6550 | 0.4915 |
| B1 (RAMP full budget) | 10k | 10000 | 12 | 0.8010 | **0.4456** | 0.4544 | 0.6558 | 0.5116 |
| msd (public, Maini) | 1k | 1000 | 12 | 0.8240 | **0.4420** | 0.4580 | 0.6660 | 0.4990 |
| max (public, Maini) | 1k | 1000 | 12 | 0.8190 | **0.2800** | 0.3990 | 0.6400 | 0.2980 |
| avg (public, Maini) | 1k | 1000 | 12 | 0.8530 | **0.3880** | 0.4060 | 0.6970 | 0.5140 |
| b3_static_linf (legacy) | 10k | 10000 | 12 | 0.8334 | **0.4028** | 0.4221 | 0.6704 | 0.4881 |
| b4_adaptive (legacy) | 10k | 10000 | 12 | 0.8253 | **0.4114** | 0.4463 | 0.6572 | 0.4591 |

## 6. Training cost (measured, 5070 Ti / Colab A100 as noted)

| arm | s/epoch | epochs | wall-clock | adv/step |
|---|---|---|---|---|
| M1a (80 ep, MSD-10 base + 3 views) | ~423 | 80 | ~9.4 h | 4 |
| M1a_full (50 ep, MSD-50) | 365 | 50 | 8.11 h | 4 |
| M1a_full_seed1 | 911 | 50 | ~12.7 h | 4 |
| R′ / RAMP | 268 | 80 | 5.96 h | 3 |

## 7. What is still open

- `M0_avg` (#12 control) training → then `M1a_cleance` (necessity ablation, CE on clean images).
- Re-eval of 11 existing checkpoints onto one common `@1k tier=no_square` grid (`results/main/<arm>/1k_nosq/`).
- v50 @10k — needed before the staleness explanation can be claimed.
- CIFAR-100 12-AA @10k.
- Ablations #13/#14 (redundancy: msdglue / linfglue on MAX base).

## 8. Reproduce any number

```bash
python scripts/dev/sync_results_main.py          # rebuild results/main/ + INDEX.md, verify union==masks
python scripts/dev/eval_arm.py --arm <A> --ckpt <path> --scale {1k,10k} --tier {full,no_square}
```

Masks are per-example boolean survival vectors, one key per attack component; the union is the AND over the
components in the tier. `meta.json` in each arm folder carries the checkpoint sha256 and provenance.

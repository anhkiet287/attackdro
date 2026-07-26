# Retrain-at-8/255 plan (train==eval) — DRAFT for Kiet approval; DO NOT launch yet

*Paper-grade 8/255 numbers. The reeval_8255 rows are lower-bound estimates (0.03-trained ckpts) and are NOT these.*

## Retrain set (all: eps=8/255 train==eval, PreActResNet-18, raw [0,1], 3 seeds, W&B `attackdro-union`)

| # | run | config (inherits base=8/255) | new run-name (keeps 0.03 ckpts intact) | device | ETA/seed | 3-seed | status |
|---|---|---|---|---|---|---|---|
| 1 | **reactive** (per-sample soft T=0.25) — FIRST, claim rests on it | `bindaware_sample.yaml` T=0.25 | `reactive_T025_8255_s{0,1,2}` | **Colab** | ~3.4h | ~10.1h | ready to launch |
| 2 | **predictive** (CARD-PB) — the new method | *needs new `configs/cardpb.yaml` + trainer code* | `predictive_pb_8255_s{0,1,2}` | **5070ti** (custom attack alloc) | ~1.6–2.4h* | ~5–7h* | **BLOCKED: needs implementation + smoke** |
| 3 | **avg_frozen** — uniform anchor (F4) | `avg_frozen.yaml` | `avg_frozen_8255_s{0,1,2}` | Colab (queue) | ~3.4h | ~10.1h | ready |
| 4 | **3a binding-group** — signal axis | `bindaware.yaml` | `bindaware_grp_8255_s{0,1,2}` | Colab (queue) | ~3.4h | ~10.1h | **CONFIRM w/ Kiet: keep in main table?** |

\*predictive ETA is an estimate (fewer attack steps by design ≈ 40–50% of reactive attack-FLOPs, and attacks are ~88% of step time → ~0.5× step); **confirm with a real smoke + FLOPs-ratio readout before committing 3 seeds.**

## DO NOT retrain (per Kiet)
- **RAMP** — already train@8/255, union **46.1** (matched reference).
- **Official MSD/AVG/MAX** — reference tier; reeval@8/255 is valid as "public ckpt re-measured" (released MSD also train@0.03 → flagged reference, not competitor).

## OPTIONAL (only if F7 is a main-table headline, not just motivation) — **ask Kiet, flag cost**
- **msd_inhouse @8/255**: MSD is heavy (msd_steps=50) → **~6h/seed × 3 = ~18h**. Expensive.
- **max_inhouse @8/255**: ~3.4h/seed × 3 = ~10h.
- (Conj-1 hard-MAX variance is a SEPARATE, cheaper 0.03 test — s1/s2 only — already queued on the 5070ti.)

## Cross-device assignment (real parallelism = across devices, not concurrent on one GPU)
- **Colab Pro Edu** = the standard-trainer stream (disconnect-proof: mount Drive, cache dataset, per-unit skip-if-done, copy ckpt to Drive per unit, resume-capable). Runs: **reactive (1) first**, then avg_frozen (3) + 3a (4) queued. ~30h of Colab wall-clock for all three ×3 seeds (can trim seeds).
- **Local 5070ti** = custom-code stream: finish reeval → **Conj-1** (max s1/s2 @0.03, ~7h) → **implement + smoke CARD-PB** → **predictive (2)**. Predictive MUST be local (needs the CARD-PB attack-allocation code + `phi/misprediction_rate` logging).
- **Config identity:** both devices use `configs/base.yaml` (eps_inf=8/255, l2=0.5, l1=12) — Colab pulls the same file. **Checkpoint format identical** (`{model, cfg, dro}` dict) → cross-device eval works via `eval_standard_pack.load_model kind="ours"`. Same arch (preact_resnet18), same raw-[0,1] convention. Verify md5 of the dataset on Colab (PLAYBOOK §VI).

## W&B (both devices) — method-note-v2 keys
Project `attackdro-union`, entity `kietna`, run-name `{method}_8255_s{seed}`, tags method/seed/tier=in-house. Per-epoch (step=epoch): `train/loss`, per-norm `train/loss_{linf,l2,l1}`, per-norm `probe/robust_*`, `probe/clean_acc`, `q/*` or per-sample mean-W; **predictive also: `phi/misprediction_rate`, `attack_flops/epoch`, `floor/<norm>`**. Eval summary auto-attached to the run (evaluate.py).

## Pre-registration (record)
- **reactive-8/255 (3 seeds) = the headline reactive row AND the baseline the efficiency claim compares against.** Seed-std expected ~0.06–0.1pp (matches 0.03 T=0.25 stability). Comparator RAMP 46.1.
- **predictive-8/255 target:** union within noise (≈±0.2pp) of reactive **at ≤60% attack-FLOPs** = efficiency win (C2). Decision rule per CARD-PB §4.

## Critical path / blockers (what actually gates a launch)
1. **reactive/avg/3a**: launch-ready NOW (configs already 8/255). Only need NEW run-names (above) so 0.03 ckpts aren't overwritten. → **Colab, on Kiet's go.**
2. **predictive**: BLOCKED on CARD-PB implementation (φ EMA + batch-partition budget allocation + safety floor + p/FLOPs logging) — a real dev task, ~a few hours of coding + a smoke. I can implement it locally while Colab runs reactive. Launch only after: card approved + smoke passes + FLOPs-ratio ≤60% confirmed.
3. **3a inclusion** and **optional msd/max** are Kiet decisions (open questions below).

## Open questions for Kiet
- **Q1:** 3a binding-group — keep in the main table (→ retrain, +10h Colab) or motivation-only (→ skip)?
- **Q2:** F7 (recipe confound) — main-table headline (→ retrain msd_inhouse+max_inhouse @8/255, +~28h) or motivation-only with the reeval estimates + cited numbers (→ skip the retrains)?
- **Q3:** Seeds — 3 for exploratory now, escalate to 20+Wilcoxon only for the final claim rows? (default yes per PLAYBOOK §VI)

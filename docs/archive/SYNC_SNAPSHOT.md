# SYNC_SNAPSHOT — for strategy assistant
*Generated 2026-07-04 09:20 (Sat) by Claude Code. All numbers from disk. Read-only snapshot.*

## 1 · RUNS LIVE NOW (2 trainers — at the ≤2 cap)
| run | where | state | ETA |
|---|---|---|---|
| bindaware_s2 (3a seed 2) | tmux `finalpipe` | epoch 49/50 (slowed 239→330s/ep by GPU sharing) | ~09:00 + eval; then T05 s1,s2 → std-AA pack |
| **RAMP seed-0 (CARD-8, Kiet self-launched 08:25)** | `scripts/ramp/train_ramp_seed0.sh` | epoch 1/80, official code+recipe, W&B 4oek4cr7 | **~14h+ (tonight/early tomorrow)** |
| msd_inhouse s0 (CARD-7, now OPTIONAL) | tmux `p7msd` | gated: fires when finalpipe ends | after finalpipe (~+6h) |
Directive items 1–5 all executed into docs (POSITIONING / MEMORY / CARDS / PROGRESS).

## 2 · RESULTS ON DISK (APGD CE+T, n=1000, locked ε=(0.03, 0.5, 12))
| run | clean | ℓ∞ | ℓ2 | ℓ1 | worst-∪ | seeds |
|---|---|---|---|---|---|---|
| **max_inhouse (CARD-5)** | 78.9 | 45.5 | 61.6 | 47.2 | **43.9** | 1 |
| **T025 (3b)** | 79.7–81.2 | 43.7–44.1 | 62.8–64.7 | 48.3–49.9 | **43.03 ± 0.05** | **3** ✅ |
| 3a (bindaware) | 80.3 / 80.7 | 44.6 / 44.4 | 64.6 / 65.1 | 47.1 / 46.6 | **42.3 / 42.1** | 2 (s2 lands ~09:00) |
| T05 (3b) | 81.2 | 43.8 | 63.3 | 50.1 | **42.6** | 1 (s1,s2 queued) |
| T1 / T2 (3b sweep) | 82.4 / 82.3 | 42.2 / 42.8 | 65.3 / 63.9 | 51.5 / 50.5 | **41.5 / 42.1** | 1 each |
| avg_frozen (in-house AVG, Colab) | 82.2–83.0 | 40.3–42.0 | 64.7–66.3 | 50.0–50.6 | **40.6 ± 0.4** | **3** ✅ (files validated) |
| P1 attackdro_union | ~81 | 46–47 | 62.5–64.5 | 41.2–42.6 | **39.7 ± 0.7** | 3 |
| Official ckpts (tier-2): MSD **42.5** · AVG 38.7 · MAX 25.0 · LINF 7.0 · L2 17.5 · L1 0.0 | | | | | | |
‡ cited (5 seeds, ε∞=8/255): RAMP **44.6±0.6** · MAX 44.0±0.7 · MSD 43.9±0.8 · E-AT 42.4±0.6 · SAT 40.4 · AVG 40.1.

## 3 · WHAT CHANGED THIS SYNC (directive executed + Kiet's RAMP launch)
1. **Claim ladder REVISED (locked into POSITIONING §1):** "match/beat MSD" dropped entirely. C1 = mechanism F4–F6 (unique) · +F7 = eval corrections (MAX-ckpt artifact, ℓ1 overstatement, probe bias) · C2 = simple cheap adaptive weighting (~2–3× cheaper/iter) *reaching* the 43.9–44.6 SOTA cluster, not exceeding it. With 5-seed stds, RAMP 44.6±0.6 overlaps MAX 44.0±0.7 / MSD 43.9±0.8 within ~1σ → read as a CLUSTER, not an isolated outlier.
2. **Baseline methodology = RAMP's own:** final from-scratch table cites ‡ (C&H 2022 + RAMP 2024, 5 seeds, 8/255), no retraining; our in-house rows in a separate tier with explicit protocol caveat.
3. **F7 upgraded to 3 independent evidences:** AVG +1.9 · MAX +18.9 · **max_inhouse 43.9 @0.03 matches C&H MAX 44.0±0.7 ‡ within 1σ = external cross-validation of our recipe.** CARD-5 prediction miss (band 41.5–43.5) explained: hard-MAX genuinely ~44, not an eval error.
4. **CARD-7 msd_inhouse downgraded to OPTIONAL** (gates nothing now); still queued, cancel-without-regret if a slot is needed.
5. **RAMP Eq. 2 L_max note added (POSITIONING §2):** per-sample max is community-endorsed at NeurIPS'24; our T-dial analyzes/softens that exact term → complementary, not competing.
6. **P3-closed REVERSED by Kiet: CARD-8 ramp_verify RUNNING** — their code (upstream be4971f) + their official recipe, single 1-line patch (`import copy`, upstream bug in gp(), zero algorithmic change). Dual eval armed: apgd + standard AA × {0.03, 8/255}. Output row = "our reproduction of RAMP (1 seed)", separate tier — does NOT replace their ‡.

## 4 · STRATEGY-SIDE ITEMS (status 2026-07-04 09:00)
1. ✅ **CARD-8 bands REGISTERED** (into card, before landing): within 44.6±0.6 → repro OK · 43.3–44.0 → mild under-repro, keep bar + caveat · **<43.3 → investigate bug FIRST (E9), never lower RAMP from 1-seed** · >45.2 → suspicious edge. **INVARIANT: 1-seed repro never replaces ‡cited 44.6±0.6; row always "our 1-seed reproduction of RAMP".** Purpose = validate we run their code correctly (gate for CARD-8b composition) + one @0.03 point.
2. ✅ **T025 43.0 framing CONFIRMED = "within the SOTA cluster", NOT a beat claim.** Locked across POSITIONING/MEMORY.
3. ✅ **avg_frozen s0/s1/s2 checkpoints RESOLVED** — Kiet shared a Drive folder; pulled all 3 via gdown into `checkpoints/` (2026-07-04 09:00). Load-integrity verified (3 distinct md5, all load as `{'model','cfg'}`, compatible with `kind=ours` loader). **std-AA pack now includes the avg_frozen_s0 row** (no longer footnoted-missing). This was the last load-bearing tier-1 dependency.

## 5 · RISK REGISTER (Aug-29 submission)
1. **GPU contention:** finalpipe + RAMP sharing slows both (~38% observed); T05 seeds + std-AA finish pushed ~+4h; RAMP ~14h+. No stall — just serialization.
2. **CARD-8 outcome cuts both ways:** if reproduction lands ≪44.6, we must report it honestly with 1-seed + our-hardware caveats (not as "RAMP debunked").
3. **std-AA deltas unknown:** all iteration numbers are APGD CE+T; standard AA may shave a few tenths — the C2 "reaching the cluster" wording must survive that (cluster numbers are full-AA already).

## 6 · NEXT 48H
- **Automatic:** T05 s1,s2 → **standard-AA pack + APGD→std delta table** (now includes avg_frozen_s0) → p7msd msd_inhouse (optional) → RAMP seed-0 lands → dual-protocol RAMP eval scripts ready.
- **I will report:** T05 3-seed mean vs the revised ladder, the std-vs-apgd delta table, and RAMP reproduction vs the pre-registered bands.

## 7 · DIRECTIVE EXECUTED 2026-07-04 (consolidation + fair-comparison + table + PIVOT; no GPU)
- **EXPERIMENT_TABLE.md is now the single results table** — `python scripts/make_experiment_table.py` regenerates it from `results/*.json` (can't drift; `--check` for CI). 4 tiers: in-house / locuslab re-eval `*` / ‡cited 5-seed / pending. Fresh disk means: **3a 42.4±0.4 (3 seeds)** · T025 43.0±0.1 · P1 39.7±0.9 · AVG 40.6±0.5 · MAXih 43.9.
- **PIVOT to METHOD-FORWARD recorded** (MEMORY §1): target = beat/approach 43.9–44.6 cluster via composition **or** the argued gap (all current aggregations static + sample-agnostic; F5/F6 = the gap argument, NOT discarded). **Gate G2′ = Aug 2:** CARD-8b Δ>repro+0.5 → method paper; else mechanism framing.
- **CARD-8b written** (RAMP + our per-sample T-softmax replacing/softening L_max; Opt A recommended; gated on CARD-8 passing band; **needs your pre-registration + Kiet approval before launch**).
- **CARD-9 written**: E-AT in-house is FEASIBLE (their code runs, 3-epoch fine-tune cheap) — the most direct fixed-geometric-vs-adaptive comparison; gated, card-first, **needs your pre-registration**.
- **Trait-vs-state diagnostic ran** (population STATE-like, supports F6 + 8b Opt A); ⚠ honest limit: shuffled first-batch dumps can't do true per-sample stability — that needs fixed-probe-batch instrumentation (gated).
- **Repo consolidated** (zero behavior change, pipeline verified alive): scripts/ = paper entry points only, rest → scripts/dev/; toy MLP deleted; docs/configs archived; README 29-line; WORK_FLOWS rules-only.
- **⚠ Needs you (planner):** pre-register bands for **CARD-8b** and **CARD-9** before their (gated) launches — same E1 discipline as CARD-8.

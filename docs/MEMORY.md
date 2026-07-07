# MEMORY.md - AttackDRO compact state
Last-updated stamp: 2026-07-05 UTC (protocol re-locked to eps_inf=8/255 + W&B standardized; 8/255 re-eval queued behind live gate_bind). Update this file in place; append history only to `docs/LOG.md`.

## Identity/Goal
- Owner: Kiet, final-year CS at HCMUT, building a rigorous mechanism/method paper for US PhD applications.
- Project: AttackDRO / robustdro studies difficulty-aware training for CIFAR-10 robustness over the union of `linf`, `l2`, and `l1` threat models.
- Current goal: produce one honest workshop-ready paper plus arXiv/code. Method-forward path requires a controlled positive CARD-8b delta over local RAMP reproduction; otherwise write the mechanism paper with F1-F7.
- Paper spine: F1-F7 frame a gap: current aggregations are static/sample-agnostic, while binding norms are measurable and state-dependent.
- Operating rule: honesty > pretty numbers; training probes are not final claims.

## Locked decisions
- Dataset/model: CIFAR-10, PreActResNet-18 family, raw pixel space `[0,1]`, no dataset normalization.
- Threats: **`linf eps=8/255` (RE-LOCKED 2026-07-05** — subfield standard: RAMP/E-AT/C&H/RobustBench, enables direct SOTA comparison; was 0.03); `l2 eps=0.5`; `l1 eps=12`. Old 0.03 protocol → `configs/archive/base_eps003.yaml`; 0.03 eval JSONs → `results/archive/`.
- Primary metric: worst-case union robust accuracy. A sample is robust only if `linf_mask & l2_mask & l1_mask` is true.
- Evaluation source of truth: independent union harness, currently `scripts/evaluate.py` -> `src/robustdro/eval/eval_union.py`.
- Iteration metric: APGD CE+T; final metric: full AutoAttack `standard` when feasible.
- Do not mix per-norm averages with worst-union; do not claim SOTA from local rows. (Single protocol now = 8/255; the old "don't compare 0.03 vs 8/255" rule is retired.)

## Protocol notes (eps_inf=8/255 relock, 2026-07-05) — train/eval mismatch DECISION
1. **reeval_8255 of our in-house ckpts = QUICK ESTIMATE / LOWER BOUND only.** Those models were TRAINED at 0.03 and eval'd at the harsher 8/255 (train/eval mismatch) → numbers artificially low, NOT fairly comparable to train-8/255 models (RAMP repro 46.1). `evaluate.py` auto-records the mismatch into each JSON; table/dashboard flag such rows `train@0.03/eval@8255 — lower bound`. **These are NOT paper numbers for our methods.**
2. **Paper-grade 8/255 for our methods REQUIRES RETRAINING at 8/255 (train==eval) — now REQUIRED, not optional.** But do NOT retrain old finalists just for this (wasteful).
3. **Fold it into the NEXT real run (post-Gate-α):** ONE 8/255 clean run carries BOTH the paper protocol AND the method the Gate-α verdict selects. Draft plan in Open items; DO NOT launch (gate on the verdict).
4. **Official MSD/AVG/MAX 8/255 re-eval is a VALID REFERENCE** (Tier-2, re-eval refs — we don't claim they trained at our eps). Note the released MSD ckpt was also trained at 0.03 → same lower-bound flag applies to it *as a reference, not a competitor*.

## W&B logging schema (CANONICAL — every in-house run logs these; confirmed by Kiet 2026-07-06)
> One project **`attackdro-union`**, entity **null** → default `kietna-ho-chi-minh-city-university-of-technology` (`kietna` alone is INVALID → silent fallback to stdout). Prefer **offline+`wandb sync`** for unattended runs (online silently disables on any init failure = lost data). Same key set for reactive & predictive → directly comparable in the W&B runs table / parallel-coords. Source of truth stays the eval JSON on disk; W&B is a view. Impl: `groupdro.py` (train/probe) + `wandb_log.py` (`log_eval_summary`).
- **Identity:** run name=id=`{method}_{variant}_s{seed}` (deterministic id → eval resumes SAME run). **Tags:** `method:X`, `run:X`, `seed:N`, `tier:in-house`. **Config:** full resolved cfg.
- **Run summary (once):** `train/protocol` ("8/255" or `eps_inf=…`), `train/eps_inf`, `train/eps_mismatch` (0/1: train==eval vs 0.03-ckpt-re-eval'd lower-bound); `best/probe_worst_union`.
- **Per-epoch (x=epoch):** `train/loss`, `train/epoch_time_s`, `lr`, `epoch`; per-norm `train/adv_acc_{linf,l2,l1}`, `train/loss_{norm}`, `q/{norm}`; loss-dist `dist/{norm}/{std,p10,p50,p90}`; probe `probe/robust_{linf,l2,l1}`, `probe/worst_union`, `probe/clean_acc`.
- **Predictive-only (CARD-PB):** `pb/attack_flops_ratio` (MEASURED), `pb/attack_passes`, `pb/beta` (EMA horizon), `phi/misprediction_rate`, `floor/{linf,l2,l1}`.
- **Eval summary (attached to same run, post-APGD):** `eval/union`, `eval/clean`, `eval/avg`, `eval/{linf,l2,l1}`, `eval/version` (apgd), `eval/n` (1000), `eval/tier`.
- **Manuscript axes:** headline = `eval/union`; efficiency scatter = `eval/union` vs `pb/attack_flops_ratio`; all vs RAMP repro 46.1.

## Results table
> ⚠ **PROTOCOL RE-LOCKED to eps_inf=8/255 on 2026-07-05.** The numbers below are the PRIOR **eps=0.03** values, kept until the 8/255 re-eval lands. Re-eval (reuse checkpoints, NO retrain) is QUEUED behind the live gate_bind run: `scripts/reeval_8255.py --phase apgd` (then `--phase std`), then rebuild views. RAMP repro already at 8/255 (union 46.1). Treat every 0.03 row below as STALE pending re-eval.

Generated from `results/*.json` on 2026-07-05 UTC. Values are robust accuracy %, n=1000 unless noted; multi-seed rows are mean+/-sample-std from synced eval JSONs.

### Tier 1 - in-house recipe-controlled, APGD, eps=(0.03,0.5,12)
| row | seeds/files | clean | linf | l2 | l1 | worst-union | source |
|---|---:|---:|---:|---:|---:|---:|---|
| msd_inhouse | 1 | 77.1 | 44.6 | 63.2 | 50.0 | 44.1 | `results/eval_msd_inhouse_s0.json` |
| max_inhouse | 3 | 79.4 | 45.7 | 63.1 | 47.7 | 44.1+/-0.6 | `results/eval_max_inhouse_s{0,1,2}.json` (@0.03; Conj-1) |
| bindaware_sample_T025 | 3 | 80.6+/-0.8 | 43.9+/-0.2 | 63.8+/-1.0 | 49.0+/-0.8 | 43.0+/-0.1 | `results/eval_bindaware_sample_T025*.json` |
| bindaware_sample_T05 | 3 | 81.6+/-0.4 | 43.7+/-0.6 | 64.5+/-1.0 | 50.5+/-0.4 | 42.7+/-0.3 | `results/eval_bindaware_sample_T05*.json` |
| bindaware | 3 | 80.8+/-0.5 | 44.8+/-0.5 | 64.7+/-0.4 | 47.4+/-1.0 | 42.4+/-0.4 | `results/eval_bindaware_s*.json` |
| bindaware_sample_T2 | 1 | 82.3 | 42.8 | 63.9 | 50.5 | 42.1 | `results/eval_bindaware_sample_T2.json` |
| bindaware_sample_T1 | 1 | 82.4 | 42.2 | 65.3 | 51.5 | 41.5 | `results/eval_bindaware_sample_T1.json` |
| bindaware_v2 | 1 | 81.4 | 44.0 | 64.6 | 47.1 | 41.5 | `results/eval_bindaware_v2_s0.json` |
| avg_frozen | 3 | 82.5+/-0.4 | 41.2+/-0.9 | 65.5+/-0.8 | 50.4+/-0.3 | 40.6+/-0.5 | `results/eval_avg_frozen_s*.json` |
| attackdro_union | 3 | 80.9+/-0.5 | 46.4+/-0.5 | 63.4+/-1.0 | 42.0+/-0.7 | 39.7+/-0.9 | `results/eval_attackdro_union_s*.json` |

### Tier 1 - standard-AA spot checks, eps=(0.03,0.5,12)
| row | seed | clean | linf | l2 | l1 | worst-union | APGD->std delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| bindaware_sample_T025 | 0 | 79.7 | 43.8 | 62.8 | 48.3 | 43.0 | +0.0 |
| bindaware_sample_T05 | 0 | 81.2 | 43.8 | 63.3 | 50.1 | 42.6 | +0.0 |
| bindaware | 0 | 80.3 | 44.6 | 64.6 | 47.1 | 42.3 | +0.0 |
| avg_frozen | 0 | 82.2 | 41.3 | 64.7 | 50.6 | 40.7 | +0.0 |

### Tier 2 - external checkpoints/reproductions under our harness
| row | eps | attack | clean | linf | l2 | l1 | worst-union | source |
|---|---|---|---:|---:|---:|---:|---:|---|
| RAMP repro, local seed 0 | (0.03,0.5,12) | APGD | 81.2 | 48.4 | 65.8 | 49.7 | 47.2 | `results/eval_ramp_repro_eps003_apgd_n1000.json` |
| RAMP repro, local seed 0 | (8/255,0.5,12) | APGD | 81.2 | 47.2 | 65.8 | 49.7 | 46.1 | `results/eval_ramp_repro_eps8255_apgd_n1000.json` |
| MSD official ckpt | (0.03,0.5,12) | standard | 82.1 | 44.6 | 64.5 | 47.1 | 42.8 | `results/eval_std_MSD_official.json` |
| MSD official ckpt | (0.03,0.5,12) | APGD | 82.1 | 44.6 | 64.5 | 46.7 | 42.5 | `results/baseline_table.json` |
| AVG official ckpt | (0.03,0.5,12) | APGD | 84.6 | 40.7 | 65.5 | 47.7 | 38.7 | `results/baseline_table.json` |
| MAX official ckpt | (0.03,0.5,12) | APGD | 81.7 | 39.2 | 62.1 | 26.2 | 25.0 | `results/baseline_table.json` |
| single-norm ckpts | (0.03,0.5,12) | APGD | see JSON | see JSON | see JSON | see JSON | 0.0-17.5 | `results/baseline_table.json`, `results/eval_pgd_at_linf_best.json` |

### Tier 3 - cited anchors only
- Literature-only anchors remain separate: RAMP 44.6+/-0.6, C&H MAX 44.0+/-0.7, C&H MSD 43.9+/-0.8, E-AT 42.4+/-0.6 at eps=(8/255,0.5,12). These are not regenerated from local JSON and never replace Tier 1/2 rows.

## Findings F1-F7
- F1: Published or weak-attack `l1` robustness can be overstated; APGD-`l1` changes both absolute numbers and diagnosis.
- F2: Strong union models are usually `linf`-bound; RAMP local reproduction is especially `linf`-strong under both eps settings.
- F3: `l2` robustness is largely free/spillover in this geometry; starving `l2` did not make it the binding norm.
- F4: Loss-signal GroupDRO is neutral-to-slightly harmful under recipe control: `attackdro_union` 39.7+/-0.9 trails `avg_frozen` 40.6+/-0.5.
- F5: Weak probes can misrank norm difficulty; train-time signals can point at the wrong binding norm.
- F6: Binding norm is state-dependent at population level (migration l1 3.7%→~10% over training), AND per-sample it is a **TRAIT** (Gate-α decision-grade, 2026-07-05): on 512 fixed indexed samples over 10 dumps of the binding-aware T=0.25 run, a sample's binding persists far above the drifting base rate — **l1 EXCESS +53.8pp** (retention 63.4% vs 9.6% chance, κ=0.60), **linf κ=0.52** — both norms sticky beyond chance → **predictive per-sample weighting is viable (Gate-α PASS, Conj 2)**. Caveat: TRAIT-dominant WITH a state overlay (durable full-run stability modest: only 11/135 ever-l1 samples stay l1 >80% of dumps; most l1 samples spend the majority linf-bound) → the predictor should be SHORT-HORIZON/EMA, not a fixed lifelong label. Source: `results/trait_state_v3.md`.
- F7: Recipe/source confounds are large: AVG official 38.7 vs avg_frozen 40.6+/-0.5, MAX official 25.0 vs max_inhouse 43.9, MSD official 42.5 vs msd_inhouse 44.1. Keep tiers separate.
- **F9 (predictive-binding s0 NEGATIVE, 2026-07-06)**: CARD-PB (EMA φ budget allocation, cold=5, β=0.5, floor guard 3pp) delivers the FLOPs cut but NOT the union — **paper-grade @8/255 s0: predictive union 40.6 vs reactive 41.7 (Δ −1.1pp), l1 collapses 47.5→44.3 (−3.2pp)** at **measured attack-FLOPs 0.522** (≤0.60 ✓). The predictor mispredicts ~29%, and the adaptive floor raised l2 (3→5) but **failed to protect l1** — i.e. the compute it removes is exactly what l1 (the TRAIT-heavy minority norm, F6 EXCESS +53.8pp) needs. **Pre-registered kill fired (Δunion<−0.3 AND l1<−2pp) → predictive halted at s0, NOT escalated; reactive s1/s2 continue as baseline.** Read: efficiency-at-equal-robustness claim does NOT hold for CARD-PB as configured; the failure is l1-specific (a per-norm/l1-priority floor is the obvious next lever, but that's a NEW design — Kiet-gated). Context: BOTH in-house methods trail RAMP 46.1 (reactive −4.4pp), so the "≤60% FLOPs at RAMP-parity" framing is not currently supported. Artifact: `results/reactive_vs_predictive_s0.md`.
- F8 (Conj-1, 2026-07-06): hard per-sample MAX is far more SEED-VARIABLE than soft T-softmax — max_inhouse @0.03 3-seed std 0.62pp (44.1+/-0.6; s0 43.9/s1 44.8/s2 43.6) vs T=0.25 std 0.06pp (~10x). Hard argmax routes each sample's whole gradient to one seed-sensitive norm; soft-T averages it → stabler. Formulation argument for soft over hard, eps-independent. Pre-registration CONFIRMED.

## Refuted ideas
- Spatial masking / union-of-masks: perturbations hit salient regions; the issue is feature semantics, not easy spatial separation.
- Per-norm adapter or logit-selection MoE: inference-time routing is vulnerable to adaptive attacks and does not compose specialists safely.
- q-floor for `l2`: gate did not open because `l2` stayed high despite near-zero learned weight.
- "Hard MAX fails in principle": refuted by `max_inhouse`; the weak official MAX checkpoint is a provenance/recipe issue.
- "Cold softmax exceeds hard max": refuted by `max_inhouse` 43.9 vs T025 43.0+/-0.1 under the same in-house tier.

## Phase gates
- Current phase: P2/P3 bridge. Finalist seeds, avg_frozen sync, max_inhouse, msd_inhouse, RAMP repro Phase-A, and standard-AA pack are synced.
- CARD-8 `ramp_verify`: DONE/PASS for local reproduction; source-of-truth controlled rows are the two RAMP eval JSONs above.
- CARD-8b `ramp_bindaware`: no synced pilot eval JSON found in `results/*.json`; delta_8255/delta_003 and Linf-gate readout are therefore missing, not inferred.
- Standard-AA pack: DONE for seed-0 finalists, avg_frozen_s0, and MSD official; see `results/standard_vs_apgd.md`; no model dropped more than 1.5pp.
- Gate alpha `probe_binding`: **PASS → TRAIT (2026-07-05).** Decision-grade run `gate_alpha_bindaware_T025_s0` (bindaware per-sample-soft T=0.25, 50 ep, 512 fixed probe, 10 dumps ep0–45 spanning both LR drops). Headline EXCESS +10.1pp; **l1 minority EXCESS +53.8pp (κ0.60)**, linf κ0.52 → both norms trait-sticky → TRAIT. Contrast `gate_alpha_avgfrozen_s0` (uniform) = MIXED (l1 +44pp κ0.47, linf κ0.38 — linf too saturated to clear the bar), which is the motivation figure: per-sample structure is only MEASURABLE once the recipe balances the norms. Verdict/numbers: `results/trait_state_v3.md`; diagnostic `scripts/dev/binding_trait_vs_state_v3.py`.
- G2 prime pivot: if CARD-8b later shows a controlled positive delta over matched local RAMP reproduction and survives standard-AA, write the method-forward paper; otherwise write the mechanism paper with F1-F7.

## Positioning
- Final paper table has three tiers: in-house recipe-controlled rows, external checkpoint/reproduction rows under our harness, and cited literature rows.
- RAMP remains separate in two eps groups: local seed-0 reproduction is 47.2 at eps=0.03 and 46.1 at eps=8/255, both APGD n=1000; cited 44.6+/-0.6 stays literature-only.
- Our distinctive contribution is mechanism plus cheap adaptive weighting, not leaderboard wording.
- E-AT is the direct fixed-geometry comparison; RAMP is the direct loss-shaping/composition comparison.
- Archived detailed positioning: `docs/archive/POSITIONING.md`.

## Sources
- Internal eval rows: `results/eval_*.json` and `results/baseline_table.json`.
- Standard-AA delta table: `results/standard_vs_apgd.md`.
- Trait/state report: `results/trait_state_v3.md` (decision-grade, TRAIT verdict; v2 superseded) plus fixed-probe dumps under `dumps/probe_binding/`.
- Official checkpoint provenance: locuslab `robust_union`; RAMP upstream clone at `external/RAMP`.
- Core eval: Croce & Hein AutoAttack; Croce & Hein `l1` APGD.
- Multi-norm robustness: Tramèr & Boneh AVG/MAX; Maini/Wong/Kolter MSD; Croce & Hein E-AT; Jiang & Singh RAMP.
- Verify before paper submission: author lists, venue details, workshop CFP dates, and any source still uncertain in archived source docs.

## Open items
- **8/255 PAPER RUNS (Kiet-APPROVED plan, `docs/RETRAIN_8255_PLAN.md`; retrains NOT launched — Kiet gates launch):** train==eval@8/255, 3 seeds, NEW run-names (0.03 ckpts intact). Order: **reactive (soft T=0.25) FIRST** (headline reactive row + efficiency baseline) → **predictive (CARD-PB)** → **avg_frozen** → **3a (kept, Q1)**. **F7 motivation-only (Q2)**: NO msd/max@8/255 retrain. **Q3**: 3 exploratory seeds; escalate ONLY the reactive-vs-predictive PAIR to 20 seeds+Wilcoxon+BH-FDR for the efficiency claim. Devices: reactive/avg/3a → Colab; predictive+Conj-1 → 5070ti. Matched comparator RAMP 46.1.
- **CARD-PB RAN + s0 KILLED (F9, 2026-07-06):** paper-grade @8/255 s0 = union 40.6 (−1.1pp vs reactive 41.7), l1 −3.2pp, at FLOPs 0.522. FLOPs cut real, robustness cost real (l1-specific). Pre-registered kill fired → predictive halted at s0, not escalated. **Next lever = CARD-PB v2 (Kiet-DECIDED 2026-07-07, `docs/CARD-PB_v2_CONFIDENCE_FLOOR.md`, DRAFT — launch gated):** predictability-aware floor `floor_g=clamp(k_min+round(k_span·miss_g),…)`, miss_g=1−recall_g measured unbiased on the recal subset (NOT the probe — the v1 probe-driven guard is blind to l1 because the l1 probe is the weakest attack, F1: it raised l2 3→5 but never l1). **Option B PRIMARY**, sweep k_span∈{8,12,16} each mapped to measured FLOPs; A (fixed l1 floor 8) = fallback/ablation. Pre-reg winning arm: union within ±0.3pp of reactive AND l1 drop <2pp AND FLOPs≤0.60; if no arm → FINDING "l1 predictability floors efficiency". Needs new code (per-norm miss + `phi/miss_rate_{norm}` keys) + `configs/cardpb_v2.yaml`. **Launch gate: reactive 3-seed baseline (~06:30 UTC 2026-07-07) → write code+config → show card+sweep → Kiet GO.** Do NOT relaunch predictive unattended. v1 artifact `results/reactive_vs_predictive_s0.md`.
- **Reactive (soft T=0.25) @8/255 paper row:** s0 = **union 41.7** (clean 79.9, linf 42.4/l2 64.1/l1 47.5), train==eval, W&B `reactive_T025_8255_s0`. s1/s2 running (3-seed mean pending → `results/reactive_vs_predictive_3seed.md`). Trails RAMP 46.1 by 4.4pp.
- **reeval@8/255 (LOWER-BOUND estimates, not paper #):** 9 in-house rows done (T025 41.4, T05 40.9, 3a 41.1) then **HUNG on max_inhouse_s0 (killed)**; msd/avg/attackdro/official still 0.03 (idempotent resume available). RAMP 46.1 (matched).
- **Conj-1 (hard-MAX variance, cheap 0.03 test):** baseline T=0.25 std ≈ **0.06pp**; need max_inhouse s1/s2 @0.03 (5070ti, GPU free). Pre-reg: hard-MAX std > 0.06.
- Gate alpha: **DONE → TRAIT (PASS)**, see Phase gates + `results/trait_state_v3.md`.
- CARD-9 `eat_inhouse`: gated, card-first, cheap fine-tune row for fixed-geometry comparison.
- Dashboard regeneration: `.venv/bin/python scripts/make_dashboard.py` or `python3 scripts/make_dashboard.py`.

## Last-updated stamp
- 2026-07-06 UTC (late) by Claude Code: PAPER 8/255 chain v2 ran (W&B online, schema Kiet-confirmed +pb/beta +train/protocol/eps_mismatch). **Reactive s0 = union 41.7** (paper-grade). **Predictive s0 = 40.6 → pre-registered KILL (F9): FLOPs cut real (0.522) but l1 −3.2pp / union −1.1pp; predictive halted, NOT escalated.** Reactive s1/s2 continuing as baseline. Both trail RAMP 46.1. Next predictive lever (l1-priority floor) is Kiet-gated.
- 2026-07-06 UTC by Claude Code: Gate-α TRAIT verdict; protocol re-locked 8/255 + W&B standardized; CARD-PB implemented+smoked (FLOPs 0.502); 8/255 retrain plan Kiet-approved; reeval@8/255 partial (9 rows LB, hung on max_inhouse). Numbers on disk; retrains + predictive NOT launched (Kiet-gated).
- 2026-07-05 UTC by Codex after full results sync from `results/*.json`, standard-AA delta refresh, and Gate-alpha status update.
- 2026-07-05 UTC re-verified by Claude Code executor: all 18 `eval_*.json` re-parsed, every Tier-1/Tier-2 row and every std-AA delta reproduced identically to disk; no malformed JSON; CARD-8b pilot eval JSON still absent; trait/state still smoke-only. No numbers changed, no findings changed; stamp refreshed, one LOG re-sync block appended, dashboard regenerated from disk.

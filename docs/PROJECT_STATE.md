# PROJECT_STATE.md — AttackDRO single sync surface
*The one file planner (Claude) ↔ executor (CC) sync on. Compact form, full substance. Update in place. History → git log. Kiet's read surface = `docs/dashboard.html` (regenerated from this file + `results/*.json`).*
*Stamp: 2026-07-07 (recipe reset → single-recipe RAMP plan; v2 miss-volume fix committed 1c061d5).*

## 1 · GOAL
- **Owner:** Kiet, final-year CS @ HCMUT → building a rigorous mechanism/method paper for **US PhD applications**. Honesty > pretty numbers.
- **Problem:** difficulty-aware adversarial training for CIFAR-10 robustness over the **union of linf (8/255), l2 (0.5), l1 (12)**. Model: PreActResNet-18, raw [0,1] pixels. Metric: **worst-case union** robust acc (robust only if linf∧l2∧l1 all hold).
- **Central claim (paper spine):** *predictive* binding-aware AT matches *reactive* union robustness at **lower attack-FLOPs** — i.e. **accuracy-per-FLOP**, not leaderboard-beating. Mechanism story F1–F9 stands even if the efficiency win doesn't.
- **Honest positioning:** RN-18 union SOTA is a **CLUSTER ~42–44** (MAX 44.0 · MSD 43.9 · RAMP@50 42.9 · E-AT 42.4), **not** the single 46.1 point (RAMP's 80-epoch peak). We target the cluster and compete on **efficiency** (a separate axis) — *in* the cluster, not 4pp behind a point.

## 2 · CURRENT PHASE + TODO  (owners: [Kiet] human · [Claude] planner · [CC] executor)
**Phase:** Phase-1 launched → reactive ep50 exposed an **l1-under-attack bug** (train plain-PGD / eval APGD mismatch, +41.7pp l1 gap) → **fixed: ported APGD training (P-10), verified** (APGD-10 +11.6pp > PGD-10; FLOPs counter OK; ~1.22× cost). **Ready to re-run reactive → predictive with APGD — awaiting Kiet GO.** Branch `card-pb-fixes`.
- [Kiet] **GO on the new plan** (`docs/EXPERIMENT_TABLE.md`): PHASE-1 develop @ep50 1-seed → PHASE-2 winner @ep80 3-seed. **Infra ready — this is the only remaining gate before Phase-1.**
- [Kiet ✓] **v2 frontier arm range = {8, 16, 24, 32}** (LOCKED) — maps the l1-recovery-vs-FLOPs frontier: (a) l1 recovers at FLOPs≤0.60 → conditional efficiency win; (b) recovers only >0.60 → characterized "l1 predictability floors efficiency at ~X%" (the curve IS the result).
- [**P-02 ✓ RESOLVED**] **λ=5 IS RAMP's canonical from-scratch RN-18 value** — their own `scripts/cifar10/RAMP_scratch_cifar10.sh` runs `--lbd 5 --fname RAMP_beta_0.5_lbd_5` (exactly our model family); paper's λ=2 is the WideResNet/TRADES variant (`--lbd 2` only on the WRN line). Our re-run is recipe-matched **by construction**; no λ=2 re-run needed.
- [CC ✓] **Blockers DONE + verified:** `save_freq` (every-10 ckpts) + `resume` (model+optimizer+scheduler+epoch+RNG) — **continuity check PASS** (`scripts/dev/resume_continuity_check.py`: resumed run reproduces the uninterrupted lr schedule incl. the drop, epoch seq 0..N). Per-run results layout live (`results/<run>/s<seed>/{train,eval,eval_fullAA}.json + ckpt/ + run_meta.json`).
- [CC ✓] **Eval convergence curve MEASURED (P-04, RAMP ep80, n=1000):** l∞ plateaus @20, l2 @10-20, l1 @100 (l1@100=49.6=l1@200 49.5; l1@50=50.1 is 0.5pp short). *(My earlier n=500 "l1@100=46.8=200" was a subsample artifact — RETRACTED; first-500 images are harder.)* **Screening budget SET = `20/20/100`** (base.yaml eval_attack) — reproduces 100/100/100 EXACTLY (union 46.0, Δ0.00) at ~53% fewer steps; l1-heavy = F1-safe. Curve in §6. **On first reactive run:** re-confirm plateaus (model-dependent) + l1 TRAINING step_size 0.10 strength.
- [Kiet/Colab] avg_frozen + 3a @our-recipe = **SUPERSEDED** (stopping); archive JSONs if they land.
- [CC ✓ / Kiet] **PHASE-1 LAUNCHED** (tmux `phase1`, RAMP recipe ep50 1-seed): reactive → predictive v2 (kspan=16) → **DE-RISK STOP** (does l1 hold after R1/R3/v2, or F9 again?). Verdict → `results/phase1_derisk.md`; first-run checks → `results/phase1_firstrun_checks.md`.
- [Kiet — PARKED, gated] **Cross-attack consistency ablation** (`docs/CONSISTENCY_ABLATION_PARKED.md`): `+consistency` row on the winning kspan (3-norm, APGD-l2 fixed-eps, FLOPs-accounted). Orthogonal lever (representation) to CARD-PB (allocation); do NOT implement. **Gate: after frontier + seeds lock the efficiency claim.**
- [Kiet — conditional] **Early-stop-attack idea scanned** (`docs/EARLYSTOP_FEASIBILITY.md`): avoids the predictor (no F9 misprediction) BUT mechanism not novel (FAT/ATES, single-norm) and inherits the weak-example → l1-under-attack risk (our F1/F9 by another name); FLOPs saving decays late-training + batch-limited. **Rec: stay with CARD-PB if de-risk holds; cheap pilot only if F9 recurs.**

## 3 · KEY DECISIONS LOCKED  (each = one-line reason)
- **eps = (8/255, 0.5, 12)** — subfield standard (RAMP/E-AT/C&H), enables direct SOTA comparison. (Was 0.03; archived.)
- **Single recipe = RAMP throughout** — lr 0.05→0.005 (×10 drop at ep70), 80ep, save/10 — confound-free comparison with RAMP at *every* epoch.
- **Training inner-max = APGD, 10 steps all norms** (P-10, LOCKED — supersedes 10/10/10 & 10/10/20 plain-PGD) — RAMP trains with **APGD** (paper states it 3×, code `autopgd_train`/`--attack apgd`), incl. **APGD-l1** (adaptive top-k + L1-proj). We were training plain fixed-step top-k PGD but *evaluating* with APGD → that **train-weak/eval-strong attack-class mismatch was the l1 root cause** (train-vs-eval l1 gap +41.7pp; RAMP l1 47.1 vs ours 30.7 at 10 steps). Ported `apgd_train` (`src/robustdro/attacks/apgd_train.py`), `attack: apgd` in attackdro.yaml. **Verified:** unit ε-balls exact; **APGD-10 is +11.6pp stronger than PGD-10** and within 7pp of APGD-100 → closes the mismatch. Reactive AND predictive craft with APGD; **FLOPs denominator = 30** (uniform-10 APGD, since APGD is strong at 10 — no need for more steps). Cost ~1.22× PGD (~223 vs 182 s/ep). *(RAMP's paper is transparent about APGD — the mismatch was entirely on our side.)*
- **Eval screening = APGD `20/20/100` (l∞/l2/l1), n=1000** — the measured per-norm convergence plateaus (P-04); l1-heavy so l1 is never under-attacked (our own F1); = full-budget 100/100/100 exactly, ~53% faster. **Final claim rows = full AutoAttack** (matches RAMP; APGD reads ~1.5pp high). Never uniform-few.
- **1-seed develop → winner to 3-seed → 20-seed for the final efficiency pair** — cheap screening, seeds only where a claim rides. FLOPs ratio reliable at 1 seed; Δunion needs seeds.
- **Metric = worst-case union**; iterate on APGD, claim on full-AA. Never mix per-norm avg with union; never claim SOTA from local rows.
- **W&B** project `attackdro-union`, entity null (→ default), run `{method}_{variant}_s{seed}`; offline+sync more reliable unattended. Eval JSON on disk = source of truth.

## 4 · FINDINGS F1–F9  (conclusion + key evidence; the reasoning chain is the research-maturity story — do not flatten)
- **F1** — weak/uniform l1 attacks *overstate* l1 robustness; APGD-l1 changes numbers AND diagnosis. (Drives the "eval l1-heavy" rule.)
- **F2** — strong union models are usually linf-bound; RAMP repro especially linf-strong.
- **F3** — l2 robustness is largely free/spillover; starving l2 didn't make it binding.
- **F4** — loss-signal GroupDRO is neutral-to-harmful under recipe control (`attackdro_union` 39.7 < `avg_frozen` 40.6). Signal choice is first-order.
- **F5** — weak probes can *invert* norm-difficulty ranking; train-time signals can point at the wrong binding norm → calibrate any decision-driving signal against strong eval.
- **F6 (Gate-α, decision-grade)** — per-sample binding is a **short-horizon TRAIT with a state overlay**: on 512 fixed probes, l1 **EXCESS +53.8pp** (retention 63.4% vs 9.6% chance, κ0.60), linf κ0.52 → **predictive weighting is viable** (Conj-2 PASS). Caveat: most l1 samples spend the majority linf-bound → predictor must be **short-horizon/EMA**, not a fixed label. (`results/trait_state_v3.md`)
- **F7** — recipe/source confounds are huge (AVG official 38.7 vs in-house 40.6; MAX official 25.0 vs 43.9; MSD 42.5 vs 44.1) → keep tiers separate; **this motivates the single-recipe reset**. Corroborated: RAMP@50 42.9 vs RAMP@80 46.1 = +3.2pp from *epochs alone* (our own measurement); RAMP Table-25 (44.6 needs 10-15 ft epochs, 43.6 at 5); RAMP Table-5 lists 42.9 (fine-tune RN-18) = our RAMP@50 — suggestive, NOT a validation (fine-tune ≠ from-scratch; see P-02).
- **F8 (Conj-1)** — hard per-sample MAX is ~10× more seed-variable than soft-T (max_inhouse std 0.62pp vs T=0.25 0.06pp). Hard argmax routes each gradient to one seed-sensitive norm; soft-T averages → **soft > hard as a formulation**, eps-independent.
- **F9 (the reasoning chain — CENTRAL) —** predictive CARD-PB s0 was NEGATIVE (union 40.6 < reactive 41.7, l1 −3.2pp) at FLOPs 0.522. **Then code review found the read was corrupted by bugs, not the hypothesis:**
  - **R1 (blocker):** the safety floor was **INERT** — a "raised" floor changed the FLOPs counter + log but never the actual attack (floor attacks fixed at 3 steps). So "floor self-corrected" was false, FLOPs over-counted, and any floor sweep was a no-op. **Fixed:** `_rebuild_floor` rebuilds the attack from the current step count. Unit + re-smoke proof: floor 3→5 raised FLOPs 0.525→0.614 and epoch-time +12% (real steps).
  - **R3:** φ's EMA `Lbar` was updated from **floored (under-attacked) losses** → once φ dropped a norm it got floored → its loss decayed → **self-reinforcing misprediction** (likely the real F9 driver). **Fixed:** update `Lbar` only for full-budget norms (predicted + recal).
  - **R2:** the card's β↔memory labels were inverted vs the code EMA (high β = short memory). **Fixed** in the card.
  - **v2 sweep-1 confound:** the confidence floor drove on **miss = 1−recall** (population-blind) → free l2 (rare-but-mispredicted) got miss=1.0 → floored to full; probe-blind adaptive guard pushed linf full → FLOPs 0.80 (ks8: union 41.5, l1 45.6 — l1 still short). **Fixed:** miss = **P(true=g ∧ b̂≠g)** (volume) + adaptive guard OFF. Re-smoke: l2 floor stays 3, FLOPs 0.53–0.59.
  - **Open frontier:** volume-miss gives LOW l1 floors (~5) — below sweep-1's 7 which already failed → l1 likely recovers only at higher FLOPs → the **"l1 predictability floors the efficiency"** finding, which the extended {8,16,24,32} sweep would *characterize as a frontier* rather than just confirm.

## 5 · METHOD STATE — CARD-PB v2 (the fixed method)
Reactive predecessor = per-sample soft-T=0.25 (crafts all 3 norms every step **with APGD** post-P-10, softmax-weights losses). CARD-PB adds: **(a)** EMA predictor φ of each sample's binding norm; **(b)** full attack budget on the predicted norm, **floor** budget on the others; **(c)** a safety floor. Both reactive and predictive now craft with APGD (predictive allocates APGD-steps); efficiency = predictive APGD-passes vs reactive APGD-passes. Claim = match reactive union at ≤60% attack-FLOPs.
- **v2 confidence floor:** `floor_g = clamp(k_min + round(kspan·miss_vol_g), k_min, full_g)`, **miss_vol_g = P(true=g ∧ b̂≠g)** measured unbiased on the recal subset, EMA-smoothed. Norms both frequently-bound AND often-missed (l1) earn a high floor; rare free l2 stays at k_min. **Adaptive probe-guard OFF** in confidence mode.
- **What R1/R2/R3 fixed** — see F9 chain (floor now physical; φ no longer self-poisons; β labels correct).
- **⚠ Binding-remeasure FLAG (P-10):** Gate-α (F6) and φ measured per-norm binding on the *old plain-PGD* training loss. Under APGD the per-norm loss ranking may shift (APGD finds stronger adversarials) → the binding pattern (and φ's targets) could move. **Not re-run now** — but if the APGD de-risk behaves oddly, a Gate-α re-check on APGD-loss is the first suspect. φ auto-adapts (it reads whatever the training attack produces); only the *decision-grade F6 numbers* are on old-PGD loss.
- **Frontier plan (Kiet-gated):** sweep kspan, each mapped to *measured* FLOPs; pre-reg SUCCESS = union∈[reactive±0.3] ∧ l1 drop <2pp ∧ FLOPs≤0.60, lowest-FLOPs winner; else the l1-floors-efficiency finding. Arm range = **{8, 16, 24, 32}** (LOCKED — lets l1's floor climb toward full to map the frontier). Design detail: `docs/CARD-PB_v2_CONFIDENCE_FLOOR.md`.
- **Logged (per-epoch):** `pb/attack_flops_ratio`(measured), `phi/miss_vol_{norm}`(driver), `phi/miss_rate_{norm}`(1−recall diag), `floor/{norm}`, `pb/beta`, `pb/floor_{mode,kspan}`.

## 6 · TASK OUTPUTS (measured numbers)
- **RAMP epoch curve (ACTIVE reference, n=1000 APGD @8/255, `results/ramp_epoch_curve.md`):** ep 10→80 union = 33.3 / 40.7 / 42.4 / 43.6 / **42.9** / 43.6 / 43.1 / **46.0**. Provenance: upstream **be4971f**, pristine (`bindaware_variant=none`), `static`=70+10 schedule, **λ=5 = RAMP's own from-scratch RN-18 value (P-02 resolved)**.
  - **Shape (changes how we report):** NON-MONOTONIC — **plateaus ~43 across ep30–70, jumps to 46 ONLY after the ep70 lr-drop.** So the **lr-drop (not raw epochs) is what lifts RAMP past ~43.** Recipe contribution = 46.0−42.9 = **+3.1pp from the drop**; matched-budget gap (RAMP@50 − our archived reactive@50) = ~1.0pp.
  - **ep_50 (42.9) is a LOCAL TROUGH** (43.6/42.9/43.6 at ep40/50/60) — valid matched-budget comparator but RAMP's weakest point there. **Paper framing:** report the CURVE, not the single point (avoids looking like we cherry-picked RAMP's low point): "matched-budget gap ~1pp; RAMP plateaus 43–44 pre-drop, reaches 46 only after the final lr-drop." **Phase-2 implication:** our method@ep50 likely low pre-drop; the decisive number is **union@80 post-drop** (expect a similar jump) → reinforces the top-2 carry rule (pre-drop ranking unreliable for post-drop order).
  - Cited: RAMP paper 44.6 (full-AA), C&H MAX 44.0 / MSD 43.9 / E-AT 42.4.
- **APGD eval convergence curve (P-04, RAMP ep80, n=1000)** — sets the screening budget:

  | APGD steps | 10 | 20 | 50 | 100 | plateau |
  |---|---|---|---|---|---|
  | l∞ | 47.4 | 47.3 | 47.3 | 47.3 | **@20** |
  | l2 | 65.8 | 65.8 | 65.8 | 65.8 | **@10-20** |
  | l1 | 55.6 | 51.3 | 50.1 | 49.6 | **@100** (=l1@200 49.5) |
  | union | 47.3 | 46.8 | 46.4 | 46.0 | |

  → **screening = 20/20/100** (base.yaml): = 100/100/100 exactly (union 46.0, Δ0.00), ~53% faster, l1-heavy/F1-safe. l1 genuinely needs 100 (still −0.5pp at 50); l∞/l2 converge by 20. Re-confirm per model class on the first reactive run.
- **ARCHIVED (provenance only, `results/archive/our_recipe_50ep_drop25/`, our-recipe 50ep):** reactive 3-seed **41.90±0.22** (l1 48.10) — RETRACTED as active; predictive v1 (F9) 40.6/l1 44.3/FLOPs 0.522; predictive v2 sweep-1 ks8 41.5/l1 45.6/FLOPs 0.80.
- Official ckpt re-evals (reference): MSD* 42.5, AVG* 38.7, MAX* 25.0 (@0.03, our harness).

## 7 · AGENT WORKFLOW
- **Claude = planner** (strategy, verdicts, pre-registration, prompts). **CC = reviewer + executor** (code, runs, file ops, reviews Codex/its own diffs). **Kiet = human** (decides, owns claims, GO gates).
- **Compute:** **5070ti is default** (serial; long runs in tmux; powercfg sleep off). No Colab unless Kiet mentions it. Predictive/FLOPs runs must be on the 5070ti (hardware-measured).
- **One-agent-one-file:** don't edit a file another agent is actively writing; check running jobs/recent commits first.
- **Sync surface:** this file (PROJECT_STATE.md) for agents; **`docs/dashboard.html` is Kiet's only read surface** (regenerated by `scripts/make_dashboard.py` from PROJECT_STATE + `results/*.json` + `results/run_status.json`). `docs/EXPERIMENT_TABLE.md` = the results-table plan. `PLAYBOOK.md` = how-we-work methodology.
- **Discipline:** pre-register before running; evidence-gate cheap kills first; validate the instrument (harness vs known refs); tiered comparison (in-house / re-eval / cited), never mixed; negative result → research question; report outcomes honestly.
- **Legacy:** retired docs in `docs/legacy/` (MEMORY, LOG, old cards, retrain plan); older still in `docs/archive/`. `results/archive/` = retired numbers, **untouched provenance**.

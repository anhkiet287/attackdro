# BUILDER WORKLOG — ATTACKDRO  (planner ↔ Claude Code shared context)

**This is the single, reused file for builder tasks.** READ IT FULLY before any task. Do NOT create per-task prompt files anymore — append here.

## How to use — round-trip protocol
This file is passed back and forth: **planner writes tasks → user sends file to Claude Code → Claude Code writes results INTO this file → user sends file back to planner → planner updates → repeat.**
- **Claude Code: write results directly into this file, do NOT just print to console.** Fill the `RESULT:` block under each task you run (what you did, files produced with paths, hashes, numbers, verdict). Set each task's `STATUS:` to TODO / RUNNING / DONE.
- **Claude Code: fill the CHECKPOINT REGISTRY** — for every model you load or produce, write its exact hash/path into the table, so future tasks tra bảng instead of re-searching.
- Planner: after reading results, move finished tasks to **LEDGER (done)** and keep REGISTRY/PATHS current.
- Grades: *decision/probe* (exploratory, fast) vs *canonical Tier-1 audit* (frozen, 12-AA). Never touch locked/canonical artifacts in a probe task.
- SYNC: repo `ATTACKDRO` (PC) mirrors `paper1` (MacBook). Loss/config authority = `docs/preregistrations/`.

## Context (one line)
Multi-norm union AT, CIFAR-10, PreActResNet-18. Paper 1 = allocation-ceiling characterization (done-ish). Paper C = pull-push representation loss, two co-primary goals (mechanism vs compute-matched MSD; SOTA vs RAMP).

## Paths & conventions
- Eval harness: `scripts/eval_multinorm_audit.py` · **canonical config `9162ce44`** (standard triple ℓ∞8/255 / ℓ2 0.5 / ℓ1 12) · 12-component AutoAttack (APGD-CE, APGD-DLR, FAB, Square × 3 norms), strict union = per-example AND.
- Paired bootstrap script: `aa8994f4` (B=10000, seed 0, one-sided 95% LCB).
- Results: `results/audit/` · `results/probe/` · `results/eval/union_bench/`.
- State authority: `docs/STATE.md` (append-only — preserve other sessions' uncommitted edits).

## CHECKPOINT REGISTRY  ← source of truth for "which model". Fill/confirm hashes when you touch them.
| id | role | path / hash | regime | notes |
|---|---|---|---|---|
| MSD-80 (public) | benchmark baseline + **mechanism-track backbone** | `external/robust_union/CIFAR10/Selected/MSD.pt` (repo @ ef34194) · sha256 `482bf2876572cdcadb1edd6de51e173ac50de79bee0351ba5cf96c6e3c6384bc` | standard | re-eval union **0.442** (1k) + C0 fine-tune source; clean-acc gate PASS 81.7% (paper 81.1) |
| MAX (public) | benchmark baseline | `…/Selected/MAX.pt` · sha256 `9f05f649e8baf75b…` | standard | re-eval union 0.280 (ℓ1-collapse); gate PASS 81.0% |
| AVG (public) | benchmark baseline | `…/Selected/AVG.pt` · sha256 `987b38f5b3afb963…` | standard | re-eval union 0.388 (1k); gate PASS 84.6% |
| **RAMP-80 (ours)** | **SOTA bar (R) + SOTA-track backbone** | `external/RAMP/trained_models/armA_rampfull_5070ti/ep_80_0.pth` · sha256 `aec84129cb424c96…` | scratch 80ep, standard | **loads via `ramp` family (fast_models softplus1); clean-gate PASS 81.2% (log 80.9%)**; own-eval union 44.7%. Under-our-12-AA number MISSING → Task A (EVAL-ONLY) |
| B4-adaptive (native) | our arm | `results/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k/s0/ckpt/val_best.pt` · sha256 `7ceb02ae76ac9cec8ab46716988dacf21c28b6c447453bd87fa8fea5fb05daf2` | standard | benchmark "our-model" union 0.420 (1k) |
| B3-static-ℓ∞w (native) | our arm | `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/ckpt/val_best.pt` · sha256 `4676e75fe06b4a5751db45c1284a7f1c68a0628c18b0748b064b89e5cdcdb769` | standard | benchmark "our-model" union 0.410 (1k) |
| eps16 arms | stressor regime — NOT for native benchmark | b4 `c2d708dd` · b3-mis `bdc4bd23` · L1w `09062f6f` | eps16 | Step-0 probe used these |

## Data splits (leakage-safe)
train_core 49k · val_select 1k / train-holdout (tuning, C0–C5) · **test 10k** (confirmatory only; never tuned/selected on) · monitoring subsets observation-only · `af0b037` = test_final 1k subset.

---

## LEDGER (done)
- **Step-0 DDN probe (P1/P2)** — margin-signal method **DEAD** on CIFAR (loss-rank = margin-rank = ℓ∞ everywhere); DDN-ℓ2 ≈ APGD-ℓ2. Margin ratios ℓ∞≈1.15 / ℓ1≈1.65 / ℓ2≈2.65. → `results/probe/step0_ddn/`.
- **Step-0b CW + libDDN** — self-DDN == adv_lib DDN (validated); CW finds mild ℓ2 masking ~1.3–1.5pp (bottleneck unchanged). → `results/probe/step0b_cw_ddn/`.
- **Stage-1 benchmark 1k (12-AA strict union)** — MSD **0.442** > B4-adaptive 0.420 > B3-static-ℓ∞ 0.410 > AVG 0.388 > MAX 0.280 (MAX ℓ1-collapse 0.298). Load-gate passed. → `results/eval/union_bench/`.
- **10k full sweep** — **PARTIAL → PAUSE after B3 for C5 (director 2026-07-14).** Full CIFAR-10 test (leak-safe). **B4 done (10k union 0.4114, clean 0.825)**; **B3 finishing (last one before pause)**; **MSD/MAX/AVG at 10k DEFERRED** (not run now — resume after the C5 GPU window). **E-AT skipped** (no released multi-norm ckpt). The moment B3 lands → stop sweep → free GPU → launch **C5 (Task F, main method)**. Regenerate `union_table.csv`+`SUMMARY.md`+STATE once the deferred 10k models eventually run.

---

## ACTIVE TASKS

### Task A — RAMP-80 eval under 12-AA (eval-only, decision grade)
STATUS: TODO
Register our own **RAMP-80** checkpoint in the registry above (hash + path from STATE). Run the canonical 12-AA strict union on **full 10k** (1k first to catch bugs) via the standard harness. **Clean-acc gate:** confirm clean acc matches our RAMP training log before trusting the union number. Output `results/eval/union_bench/RAMP/eval.json` + add the row to `union_table.csv`. This RAMP union number = the **SOTA bar** (H1-SOTA target) and RAMP-80 = the SOTA-track backbone. No training, no locked-artifact change.
RESULT: ⟨builder fills⟩

### Task B — C0 kill-test, Paper C mechanism gate (fine-tune, exploratory)
STATUS: DONE (2026-07-13) — **GO over placebo** (A2−A1 LCB +0.021); **A2 ties A0** → C1 launched.
Loss authority = `docs/preregistrations/preregistration_paperC_pullpush.md` §3. Anchor fixed to **variant A (own clean, stop-grad)** — A/B/C comparison is later (C3), do NOT sweep anchors here.
- **Backbone:** fine-tune the **MSD-80** checkpoint (registry) 3 epochs. Add projection head g(·); L2-normalized embeddings; head discarded at eval. Per sample generate x∞, x2, x1 (backbone APGD per norm).
- **Loss** `L = L_CE-AT + λ·L_pp`; `L_pp` = NT-Xent (τ=0.1): anchor = clean z̄ᶜˡᵉᵃⁿ (STOP-GRAD); positives = own {z∞,z2,z1}; negatives = other-class adversarial embeddings.
- **Arms (≥2 seeds each):** **A0** continued-MSD (no head, no L_pp) · **A1** placebo (head + L_pp with shuffled/random negatives) · **A2** pull-push (real loss).
- **Procedure:** (1) λ triage — A2 at 3 λ, 1 seed, 1k val → pick λ. (2) GO decision — {A0, A1, A2@best-λ}, ≥2 seeds, **≥5k validation** (train-holdout, NOT test; print provenance). Eval = 12-AA strict union + per-norm (esp ℓ∞) + **masking diag** (APGD-vs-Square gap, few-vs-many-step PGD gap) + **collapse diag** (alignment/uniformity, embedding-norm).
- **GO / NO-GO:** GO iff **A2 beats A1 (placebo) on union by > noise band** (report val SE + seed SD) **and** shows ℓ∞ lift **and** no masking flag. Beating only A0 / the MSD checkpoint is NOT sufficient.
- **Output:** `results/probe/C0_killtest/{A0,A1,A2_l<λ>}/eval.json` (per seed) + `SUMMARY.md` (A0/A1/A2 table mean±sd, noise band, explicit GO/NO-GO with A2−A1 vs noise). Append a C0 entry to `docs/STATE.md`. No from-scratch, no locked-artifact change.
RESULT (in-progress, 2026-07-13):
- **Impl:** `scripts/dev/c0_killtest.py` (new probe code; NO change to locked trainer/config/artifacts). Backbone = public MSD-80 (sha `482bf28…`) + SimCLR head (512→512→128, L2-norm, discarded at eval). Loss `L_CE-AT + λ·L_pp`; L_CE-AT = worst-case max-CE over 3 per-norm APGD adversarials (constant across arms); L_pp = NT-Xent τ=0.1, anchor A (clean, **stop-grad**), positives = own {x∞,x₂,x₁} (`apgd_train` per norm, standard triple), negatives = other-class adv (A2) / randomized (A1 placebo). Fine-tune SGD lr0.005 mom0.9 wd5e-4, 3ep, train-APGD n_iter=10.
- **Data:** fine-tune train[0:44000]; val = train-holdout train[44000:49000] (5k; 1k=train[44000:45000] for triage); val_select[49000:50000] reserved; TEST never used. Caveat: public backbone saw all train → val robustness optimistic, but kill-test is RELATIVE (A2 vs A1), bias cancels.
- **Validation:** CPU unit tests PASS (loss shapes; embed‖·‖=1; placebo≠real; grad flow; real-L_pp with no other-class negatives = exactly 0). GPU end-to-end smoke PASS (full 12-AA path, JSON+masks written).
- **Eval:** train-holdout 12-AA strict union + per-norm + masking diag (APGD−Square gap, few−many-step ℓ∞) + collapse diag (alignment/uniformity/embed-norm); per-example union & ℓ∞ masks stored for paired A2−A1 bootstrap.
- **Verdict aggregator ready:** `c0_verdict.py` — mean±sd union/ℓ∞ over seeds, paired A2−A1 union bootstrap (B=10000, LCB), noise band = valSE+seedSD, GO iff (gap>band ∧ ℓ∞-lift>0 ∧ no A2 mask-flag).
- **λ triage (1k val):** λ=0.5 union 0.504 > λ=2.0 0.498 > λ=1.0 0.497 (all masking-clean) → **λ=0.5**.
- **✅ C0 VERDICT = GO (over placebo), 2026-07-13.** GO stage (5k train-holdout, 2 seeds): **A0 continued-MSD 0.5068**, **A1 placebo 0.4774**, **A2 pull-push 0.5027**. **A2−A1 = +0.0253 union** (paired bootstrap B=10000 **LCB +0.0210 > 0**, noise band 0.0112), **ℓ∞ lift +0.0281**, no masking → all pre-registered criteria met. Class-geometry signal is REAL (beats compute/head-matched placebo).
- **⚠️ Caveat:** **A2 ≈ A0** (−0.0041) — pull-push does NOT yet beat plain continued-MSD at fine-tune scale (the win is only over the placebo). → motivates **C1 (harder negatives)**.
- Outputs: `results/probe/C0_killtest/{A0,A1,A2_l0.5}/seed{0,1}/eval.json` + `verdict.json` + `SUMMARY.md`. Masking-diagnostic direction bug found+fixed (APGD-vs-Square: white-box stronger = healthy).

### Task C — C1 negative-selection sweep (extend C0 module, fine-tune, exploratory)
STATUS: DONE (2026-07-13) — **NO-WIN**: no mining beats A0; C0 tie holds (see RESULT).
- **Impl:** extended `scripts/dev/c0_killtest.py` with `--neg-mode {all,hardtopk,semihard}` (+`--k`,`--delta`). `all`=full other-class pool (C0 A2 control); `hardtopk`=top-k nearest negatives (k=8); `semihard`=FaceNet band (posSim−δ<sim<posSim, δ=0.2, fallback nearest). Selection uses detached sims; loss keeps grad at kept negatives. Config FROZEN from C0: λ=0.5, τ=0.1, anchor A, positives=own{ℓ∞,ℓ2,ℓ1}. Verified: `all` unchanged, hardtopk≤k, semihard band+fallback (CPU unit + GPU smoke PASS).
- **Arms (5k val, 2 seeds):** A0 + A2-all **reused from C0** (identical config, masks present); NEW = **A2-semihard**, **A2-hardtopk(k=8)** — 4 GPU runs (~10h).
- **Verdict (`scripts/dev/c1_verdict.py`):** per variant A2*−A0 paired bootstrap (B=10000, LCB) union & ℓ∞, noise band, masking + **collapse** flags. WIN iff LCB(union)>0 ∧ ℓ∞ lift ∧ no masking ∧ no collapse.
- Output: `results/probe/C1_negmine/{A0,A2-all,A2-semihard,A2-hardtopk}/seed{0,1}/eval.json` + `SUMMARY.md`. ⟨verdict to fill on completion.⟩
RESULT (2026-07-13): **NO-WIN — pull-push ties continued-MSD regardless of negative selection.** 5k train-holdout, 2 seeds, paired bootstrap B=10000 LCB. Bar **A0=0.5068**. **A2-all 0.5027** (ΔvsA0 −0.0041, LCB −0.0072), **A2-semihard 0.5046** (−0.0022, LCB −0.0051), **A2-hardtopk 0.5036** (−0.0032, LCB −0.0062). All LCB<0 → none beats A0; **no collapse, no masking** in any arm (hard-negative collapse risk did not materialise; uniformity ~−3.6). Hard/semi-hard mining does NOT rescue the C0 tie → the "easy negatives" hypothesis is rejected at fine-tune scale. (Session restarted mid-sweep; resumed idempotently, no data lost.) → `results/probe/C1_negmine/{verdict.json,SUMMARY.md}`.

### Task D — C1b RAMP-pairing POSITIVE CONTROL (extend same module, fine-tune, exploratory)
STATUS: DONE (2026-07-13) — **probe UNDER-POWERED**: RAMP (known MSD-beater) also ties A0 → pull-push tie INCONCLUSIVE (see RESULT).
- **Impl:** added `--aux {none,pullpush,ramp}` (+`--lam-ramp`) to `scripts/dev/c0_killtest.py`. **A3 = MSD + RAMP KL logit-pairing** (Jiang & Singh NeurIPS'24), taken verbatim from `external/RAMP/RAMP.py:296-332`: source=ℓ∞ adv, target=ℓ1 adv; on the source(ℓ∞)-correct subset, `KLDivLoss(sum)(log_softmax(f(x₁)+1e-12), softmax(f(x∞)))/N_sel`; **λ_ramp = 1.5 (RAMP default `lbd`)**. `L = L_CE-AT + λ_ramp·L_KL`. **Logit-pairing ONLY — no RAMP gradient-projection** (probe limitation, recorded). A0/A2 paths byte-identical (unit-verified). CPU unit + GPU smoke PASS.
- **Arms (5k val, 2 seeds):** A0 + A2* reused; NEW = **A3-ramp** ×2 (queued to start after the C1 sweep — single GPU, no contention).
- **Verdict (positive-control):** A3−A0 and A2*−A0 paired bootstrap LCB (union & ℓ∞). Reads: A3>A0 while A2≈A0 → probe sensitive, pull-push weak; A3≈A0 → probe UNDER-POWERED → pull-push tie inconclusive → motivates from-scratch.
- Output: `results/probe/C1_negmine/A3-ramp/seed{0,1}/eval.json`; A3 row + verdict added to `C1_negmine/SUMMARY.md`.
RESULT (2026-07-13): **A3-ramp = 0.5082 ± 0.0003** (ΔvsA0 **+0.0014**, paired LCB **−0.0017**, ℓ∞ Δ +0.0057), no masking, no collapse. **RAMP logit-pairing — a KNOWN from-scratch MSD-beater — does NOT beat A0 at 3-epoch fine-tune** (barely-positive point estimate, LCB<0; only a faint ℓ∞ nudge). **⇒ Positive control FAILS to fire → the fine-tune probe is UNDER-POWERED**: it cannot resolve *any* mechanism in 3 epochs. Therefore the C0/C1 pull-push≈A0 tie is **INCONCLUSIVE, NOT a kill**. **Recommendation:** do not abandon pull-push on probe evidence; the honest next test is **from-scratch (C5+)**, where mechanisms have room to act. Caveat: RAMP here is logit-pairing ONLY (grad-projection omitted) — its full from-scratch strength is understated, which only reinforces "probe under-powered." → `results/probe/C1_negmine/{verdict.json,SUMMARY.md}`.

### Task E — C1c: negative-SOURCE + decoupled scaffold/glue + epoch-sweep + embedding-trajectory diagnostic
STATUS: DEFERRED (2026-07-14) — director chose straight-to-from-scratch (Task F / C5) over the epoch-sweep gate. The **decoupled α/β loss + `--neg-source {adv,clean}` + embedding-trajectory dump are folded into Task F** (`scripts/dev/c5_fromscratch.py`). Kept as an optional cheap diagnostic.
RESULT: ⟨superseded by Task F⟩

### Task F — C5 from-scratch mechanism test (CONFIRMATORY grade, pre-registered, NEW training — authorized 2026-07-14)
STATUS: RUNNING (2026-07-15) — **✅ M1a DONE + 12-AA HIT CONFIRMED.** M0/attribution/seed-2 in the recovered queue; M1b on Colab.
- **✅ M1a (from-scratch MSD-base steps=10 + decoupled pull-push, neg=adv, seed0, 80ep):** best val-worst-union 0.473 (peak ep70), no collapse. **12-AA: 1k union 0.437; 10k union 0.4173** [0.408,0.427] · ℓ∞0.427/ℓ20.664/**ℓ1 0.515**/clean 0.816 · **no masking**. val_best sha `f02924cb230b`. → `results/eval/union_bench/M1a/{eval.json,10k/eval.json}`, embed_dump saved.
- **Paired @10k (union masks):** M1a−B3 **+0.0145 LCB +0.0094** (signif); M1a−B4 **+0.0059 LCB +0.0008** (signif but marginal). → from-scratch pull-push beats both same-recipe no-pull-push peers; **1-seed caveat** (M1a−B4 seed-fragile → seed-2 queued). Clean isolation = **M1a−M0** (pending).
- **M0 = matched control** (`--variant M0`: pure MSD-AT, no head/positives/scaffold/glue, same recipe/seed0) TRAINING; then M0 audit → **attribution M1a−M0** (`c5_attribution.py`) → eval-batch @10k (RAMP 1k+10k; MSD/MAX/AVG 10k) → **M1a seed-2**.
- **M1b (clean-neg, same seed0) on Colab** via `notebooks/M1b_colab.ipynb` (mounts Drive CIFAR, resume-safe). On return → 12-AA audit for paired adv-vs-clean.
- **⚠ Restart recovery (2026-07-15):** a session restart wiped the ephemeral scratchpad (killed the overnight queue + deleted the baseline driver). Fixed by moving persistent runners into the repo: **`scripts/dev/union_bench_eval.py`** (baseline eval driver) + **`scripts/dev/run_c5_queue.sh`** (skip-if-done queue). M1a audits survived; M0 relaunched clean.
- **Scope:** TRAIN ONLY pull-push (M1a neg-source=adv, M1b neg-source=clean; 1 seed each, SAME seed). MSD/RAMP/MAX/AVG/B3/B4/RAMP-pairing are EVAL-ONLY under the 12-AA harness — not retrained/re-implemented. Recipe = B3/B4 RAMP-80 for comparability.
PREP RESULT (2026-07-15): **`scripts/dev/c5_fromscratch.py`** built (new training code; NO change to canonical audit config / locked probe artifacts / training code). Reuses `apgd_train` + the C0 pull-push design — loss not rewritten.
- **Model:** robustdro **PreActResNet-18 (ReLU, final-BN) + SimCLR head (512→512→128, L2-norm, discarded at eval)** — matches B3/B4 arch (NOT robust_union MSD arch) so M1-vs-B3/B4 is comparable. val_best saved in robustdro `{cfg,model}` (backbone only) → direct 12-AA audit via the frozen harness.
- **Decoupled loss** `L = L_CE-AT + α·L_scaffold + β·L_glue`: glue = pull own {ℓ∞,ℓ2,ℓ1} adv → own **DETACHED clean anchor (anchor A)**; scaffold = logsumexp-push over other-class negatives; **`--neg-source {adv,clean}`** (M1a=adv, M1b=clean). α=β recovers decoupled-contrastive ≈ C0 NT-Xent up to neg-source swap. α,β warm up 0→target over epochs[0,10]; CE-AT from ep0. Defaults α=β=0.5, τ=0.1.
- **Recipe (== B3/B4):** 80ep, bs128, SGD lr0.05 mom0.9 wd5e-4, MultiStepLR@70 γ0.1, RandomCrop(32,pad4)+HFlip, standard triple, train_core[0:49000], **val_select[49000:50000] worst-union selection** (== B3/B4 val_best).
- **Base CE = TRUE MSD (`msd_v0`, multi-steepest-descent) — CONFIRMED (director 2026-07-14).** M1 = MSD + pull-push: MSD adversarial drives the CE term; the 3 per-norm APGD adversarials still feed the pull-push positives (≈4 attacks/step). `--base {msd,apgd}` (default msd; **apgd = smoke-gated fallback** — loop raises on non-finite loss); `--msd-steps` default 10 (recipe budget; msd_v0 default is 50). CPU unit PASS (msd_v0 valid multi-norm adv, CE finite, full loss+grad OK).
- **Embedding-trajectory dump** (folded from C1c): K=8 × 3 classes (24 tracked, ids logged), epochs {0,20,40,80}, clean+3-norm-adv+class-prototype, **PCA-2D fit once on ep0 clean** → `embed_dump.json` per `docs/embedding_diagnostic.html` schema. Checkpoints ep{20,40,60,80}+val_best+last.
- **Validation:** CPU unit PASS — model shapes (feat 512 / logits 10 / embed 128, ‖z‖=1), decoupled adv≠clean, grad→backbone+head, α/β decoupling, PCA dump + prototypes.
- **⚠ REVISABLE config (flagged):** base=TRUE MSD but **msd_steps=10** (recipe budget, not msd_v0's 50 → weaker than public MSD-80); per-norm positives via APGD (B3/B4 used pgd_*); α=β=0.5. Documented in `train.json`.
- **Next:** on B3-done → stop sweep → **GPU pre-flight `--smoke` (gates MSD stability; NaN/blow-up ⇒ fall back to `--base apgd` + flag)** → launch **M1a on 5070 Ti**, hand off **M1b on Colab (same seed)**. Output `results/fromscratch/C5/{M1a_advneg,M1b_cleanneg}/seed0/{ckpt,eval.json,embed_dump.json,train.json}`; then 12-AA audit each val_best vs bars (MSD-80, RAMP-80, MAX, AVG, B3/B4) → `SUMMARY.md` + `verdict.json`.
RESULT: ⟨builder fills on completion — after GPU launch⟩

### Task G — Inventory & gap analysis (READ-ONLY, no train, no heavy eval)
STATUS: DONE (2026-07-15) — inventory + gap table written; RAMP-80 registered + load-gated.
RESULT: → full artifact `results/eval/union_bench/INVENTORY.md` (+ STATE entry). Headlines:
- **Checkpoints:** all load. **RAMP-80 = `armA_rampfull_5070ti/ep_80_0.pth` sha `aec84129…`, clean-gate PASS 81.2% (log 80.9%)** via `ramp` family (fast_models softplus1) — the R bar is usable eval-only. M1a in-progress (ep48/80, valWU 0.43); M1b on Colab.
- **Eval cells under `9162ce44`:** HAVE — MSD/MAX/AVG @1k (0.442/0.280/0.388), B3 @1k+10k (0.410/**0.4028**), B4 @1k+10k (0.420/**0.4114**). **All eval.json keep per-norm ⇒ worst-norm(min) + average(mean) DERIVABLE, no re-eval.** MISSING — MSD/MAX/AVG @10k, and **RAMP under our 12-AA entirely** (=R).
- **Compute:** RAMP ~241s/ep→~6.0h; B3/B4 per-ep-time in history; M1a ~365s/ep→~8h. FLOPs not logged (estimate if needed).
- **Gap → 4 target numbers:** **R** (RAMP@12-AA)=EVAL-ONLY (ckpt ready, Task A); **M1**=NEEDS-TRAINING (M1a finishing/M1b Colab); **m0**=NOT PLANNED (no matched-MSD retrain → B3/B4 are the same-recipe no-pull-push peers); **R′** (RAMP+pull-push)=NEEDS-TRAINING (future). MSD/MAX/AVG @10k = EVAL-ONLY (deferred).
- **Cheapest complete-table path:** only M1 + R′ need training; everything else is a single 12-AA pass on an existing ckpt. **Reading frame confirmed:** M1 read primarily vs B3 0.410 / B4 0.420 (same 10-step recipe); MSD-80 0.442 / RAMP recipe-&-step-mismatched → ballpark.

### Task H — Claim-B: RAMP + representation term (2 arms, Colab RTX PRO 6000, eval on PC)
STATUS: RUNNING (2026-07-16) — **B1 TRAINED** (Colab, 80ep, no collapse; val_best ep78) → audit-pending on Colab-A. **B2 (RAMP+SupCon) TRAINED** (5070 Ti, 80ep; val_best ep78, proxy valWU 0.615). **R′ (matched control, `--claimB none`)** training Colab-C — the honest Claim-B baseline (see BASELINE PIVOT below). **M0 DONE → Claim A CONFIRMED** (M1a−M0 10k +2.87pp LCB +0.0234). Method/loss code PUSHED (commit `89bc0e83`, branch feat/programA-g2-impl). Subtractions: **B1−R′ / B2−R′** (NOT −R).
- **⚠ RE-PRIORITY (director, 2026-07-16, ~08:45): B2 → extension material, NOT urgent; 5070 Ti reassigned to seed-2 immediately.** Local B2 10k audit STOPPED; B2 audit moves to Colab (upload `results/fromscratch/ClaimB/B2_supcon_seed0/val_best.pth` → Drive → `run_audit(B2,'ramp','B2')` after R′ 10k, same Colab-C queue) → **B2−R′ paired**. **B2 proxy 0.615 recorded but NOT read** — same proxy-inflation pattern as B1 0.592 / R′ 0.601 (APGD-20 val-selection metric ≫ true 12-AA); only the frozen 12-AA is trusted.
- **Arms (RAMP recipe, seed 0, full RAMP loss + rep term reusing RAMP's own ℓ∞/ℓ₁ adversarials):** **B1 = RAMP + pull-push** (α·scaffold+β·glue, anchor=clean stop-grad, positives={ℓ∞,ℓ₁} adv, negs=other-class adv, τ=0.1, α=β=0.5); **B2 = RAMP + worst-case SupCon** (γ=0.2 on ℓ∞ adv embeddings). Head discarded at eval; backbone saved as RAMP-family state_dict.
- **`scripts/dev/claim_b_rep.py`** — rep terms; CPU unit PASS (B1 finite+grad+scaffold→0 no-negs; B2 SupCon **NaN bug fixed** [-inf×0], lower on clustered, grad ok). Head = pooled-512→128, L2-norm.
- **`scripts/dev/patch_ramp_claimb.py`** — injects head + head-optimizer + rep term + per-epoch worst-union **val_best** into RAMP.py → `RAMP_claimB.py` (6 exact-anchor patches; validated against real RAMP.py, AST-parses; base RAMP loss/pred-alloc/GP UNTOUCHED; external/ left pristine — generated on Colab).
- **`notebooks/ClaimB_colab.ipynb`** — clone RAMP (public) → apply patch → **1-epoch smoke** → full B1 + B2 (80ep). val_best+logs → Drive. `attackdro_code.zip` rebuilt (carries claim_b_rep + patch).
- **On PC:** audit each `val_best.pth` under frozen 12-AA (`--model-family ramp`) → **B1−R, B2−R paired bootstrap** vs R (RAMP ≈0.456).
- **⚠ Caveats:** RAMP integration untested on PC (env deps) → Colab smoke gates it; **LBD/config must match the R bar (armA_rampfull)** for a clean B−R read (notebook default lbd=5 = scratch recipe). No changes to the M0 queue / locked artifacts.
**✅ B1 REAL 12-AA (2026-07-16, Colab, frozen 9162ce44, ramp family):** **1k union 0.4540 · 10k union 0.4456.** Proxy 0.592 confirmed inflated (real ≈0.446 @10k). **Collapse dump (backbone pooled-512, head discarded) — NO collapse:** clean-vs-adv alignment worst/mean 0.9055/0.9415 (per-norm ℓ∞ 0.9055 / ℓ2 0.9820 / ℓ1 0.9371) = strong attack-INVARIANCE (glue working, desirable); **uniformity −3.1582** (clearly <0 → features spread, not collapsed); **embed_norm 12.55** (healthy, not degenerate). Degeneracy independently ruled out by the audit itself — a collapsed rep can't score 0.446 real union. **Claim B (B1−R′) PENDING R′ audit** (do NOT read B1 in isolation; the ballpark RAMP bar ≈0.456 is not the matched control). → Drive `audit_out/B1/{eval.json,10k/eval.json,masks,collapse.json}`.

RESULT (2026-07-16, filling): **B1 (RAMP+pull-push, lbd5/seed0/at_iter10/80ep/static-lr) TRAINED on Colab, full 80ep, no collapse** (train loss went negative as rep term dominated late — expected for the glue/scaffold term, backbone still learning: acc_t 0.39→0.62). **val_best = epoch 78, val-worst-union 0.5920** (val_select 1k, APGD-20 3-norm, ramp-family) → `Bet1_out/B1_pullpush_seed0/val_best.pth` on Drive. RAMP-internal quick-eval ep80: test union 43.5% / clean 84.5% (2-attack, NOT the 12-AA — decision # is the frozen audit). **Pending: audit B1 + R under frozen 9162ce44 (ramp family) → B1−R paired bootstrap.** ⟨append B1−R + B2−R + M1a−M0 LCB⟩

### Task J — Inventory + gated audit notebook (2026-07-16, read-only + 1 new notebook)
STATUS: DONE — inventory reported; `notebooks/audit_gate_colab.ipynb` built (run-all idempotent, outputs → Drive `union_bench/`).
- **Inventory (local/repo):** union_bench eval+masks — M1a 1k 0.4370/10k 0.4173 (**10k masks PRESENT**), b3 1k 0.4100/10k 0.4028, b4 1k 0.4200/10k 0.4114, msd/max/avg **@1k only** (0.4420/0.2800/0.3880); **B2 local audit NOT done** (stopped). Ckpts local: M1a s0 val_best `f02924cb230b` (43M), B2 val_best `ddaaeb5c653a` (43M); **M1b/M0 s0 = NOT local** (Colab/Drive). Code zip 135KB/37 files = harness+collapse_dump+M1a masks(1k+10k)+eval+configs ✓.
- **⚠ SEED DISCREPANCY:** director text says "seed-1"; 5070 Ti is training **`--seed 2`** (per earlier explicit "seed 2"), outdirs `results/fromscratch/C5/{M1a_advneg,M0_matched}/seed2`; NO seed-1 pair. Notebook uses `REP_SEED` (default 2) — CONFIRM. (Residual risk: seed2 wrapper skip-guard keys on val_best.pt which exists from ep1 → on restart it would skip an incomplete M1a; live run unaffected, flagged.)
- **E3 extraction (no GPU, local, done):** M1a alignment (head-space) **0.7846**, uniformity −3.446, embed_norm 3.042 (no collapse). **Proxy/masking VERIFIED:** primary-APGD proxy union 0.4413 ≥ full 12-AA union 0.4173 (FAB+Square tighten); union ≤ min-component 0.4320 (ℓ∞ binding); **no masking** — all norms Square ≥ worst white-box, black_box_gap ℓ∞/ℓ2/ℓ1 = −0.077/−0.053/−0.107, `masking_caveats` all null.
- **Notebook (FINAL, `RESULTS_DIR=union_bench/` CANONICAL, writes Drive-direct; REP_SEED=2 locked):** 22 cells, all compile. **One-time MERGE cell** `cp -rn audit_out/* → union_bench/` (no-clobber) runs FIRST → skip-if-done reuses existing M0(1k+10k)/B1/Rprime(1k) → **M0 re-audit auto-skips (~2h GPU saved)**. PHASE 0 CPU: **E1** M0 per-norm/clean (1k+10k), **E2** copy M1a from zip + per-component 12×2 + masking check (Square≥worst-wb all norms → no-mask) → `_phase0/m1a_vs_m0.json`, **CONCORDANCE** per-norm survival overlap + union-AND M1a-vs-M0 → `_phase0/concordance.json` (the "gain routes through ℓ₁" number; per-norm AND verified == harness per_norm_audit: ℓ∞0.4269/ℓ20.6636/ℓ10.5148). PHASE 1 GPU gated: **M0** (skips post-merge), **M1b** (→paired M1a−M1b @10k), **M1a/M0_seed2** with **done-gate** `_completion_ok` (train.json history epoch≥70 or val_best_meta epoch≥70 — NOT bare val_best.pt, since c5 saves an ep1 snapshot; upload folder `seed2/{M1a,M0}/{val_best.pt,train.json}`), **MSD/MAX/AVG@10k** optional. EXTENSION disabled: **Rprime 10k** → paired(B1,R′), **B2** → paired(B2,R′). Auto-paired M1a−M0 (existing masks, reproduction check), seed-min LCB, summary table. NOT committed.

### Task I — Audit-on-Colab + R-config confirmation (2026-07-16)
STATUS: SET UP — AutoAttack audits move to Colab (5070 Ti stays on B2); paired bootstrap = CPU (PC).
- **✅ R config CONFIRMED = B1/B2** (`scripts/ramp/run_generality_5070ti.sh` BASE: `--lr-max 0.05 --lr-schedule=static --at_iter 10 --kl --max --gp --lbd 5 --seed 0 --epochs 80`; STATE_LOG "matches paper lambda=5", union 0.448). B1/B2 use the SAME (lbd5/seed0/at_iter10/80ep/static-lr) → **R is a valid subtraction baseline, NO R′ needed.**
- **`notebooks/audit_colab.ipynb`** — runs the frozen harness `9162ce44` on Colab (autoattack install; reuses uploaded CIFAR tarball; harness+configs+subsets+M1a-masks baked into `attackdro_code.zip`, rebuilt 133 KB). `run_audit(ckpt,family,name)` → 1k+10k eval.json+masks to Drive; `paired(a,b)` → LCB.
- **Flow:** M0 done → audit on Colab-A → `paired('M1a','M0')` = **M1a−M0** (Claim A). B1 done → audit R once + B1 on Colab-C → **B1−R**. B2 done local (~16:00) → audit locally (PC free) → **B2−R**.
RESULT (2026-07-16): **✅ CLAIM A CONFIRMED — from-scratch pull-push beats the compute/seed-matched pure-MSD control at both scales.** M0 (pure MSD-AT, no head/positives/pull-push, same recipe/seed0) audited on Colab under frozen 9162ce44 (robustdro family): **M0 1k union 0.3920, M0 10k union 0.3886**. Paired sample-level bootstrap (B=10000, seed0, 1-sided 95% LCB):
  - **1k: M1a−M0 = +0.0451, LCB95 = +0.0280 → SIGNIF** (M1a 0.4370 vs M0 0.3920)
  - **10k: M1a−M0 = +0.0287, LCB95 = +0.0234 → SIGNIF** (M1a 0.4173 vs M0 0.3886)
  The 10k Δ (+2.87pp) is the decision number: LCB +0.0234 ≫ 0, no seed-fragility caveat (matched seed). Larger than M1a−B3/B4 (+1.45/+0.59pp) because M0 is the clean same-base/same-seed control isolating the pull-push mechanism, not just the recipe. **The C0/C1 fine-tune "tie" is now resolved as UNDER-POWERED (Task E positive-control), not a kill: at from-scratch scale the mechanism fires.**
- **▶ SEED-2 CONFIRMATION LAUNCHED (priority #1, director 2026-07-16 08:45): M1a seed-2 → M0 seed-2** on 5070 Ti (paired, same recipe/base=msd, `--seed 2`, W&B offline; `results/fromscratch/C5/{M1a_advneg,M0_matched}/seed2`). ~13h total, resume-safe chain; must audit before 18/7. Closes the 1-seed caveat — a 2nd matched M1a−M0 delta hardens Claim A beyond the seed-0 point. (Claim A@seed0 already SIGNIF; seed-2 is the robustness replicate.)

**⚠ CLAIM-B BASELINE PIVOT (director, 2026-07-16): the honest subtraction is B1 − R′, NOT B1 − R.** R (`armA_rampfull_5070ti`) was trained by a *different script* (`run_generality_5070ti.sh`) with a *different val-selection*, on a different machine → confounds the B−R read. **R′ = matched RAMP control via the SAME patched harness `RAMP_claimB.py --claimB none`** (same lbd5/seed0/at_iter10/80ep/static-lr, same Colab env, same per-epoch worst-union val_best selection) — differs from B1/B2 *only* by the rep term. R′ training on Colab-C now. The earlier "no R′ needed" (Task I) is SUPERSEDED for Claim B.
- **B1's 0.592 proxy is DISCARDED as overfit** (val_select worst-union under APGD-20 — the selection metric, not an independent test). B1's real number = the frozen 12-AA audit.
- **Flow (Colab-A free now):** `run_audit(B1_val_best,'ramp','B1')` (1k→10k) + **`collapse_dump.py`** (backbone pooled-512 space, head discarded at eval): alignment(per-norm)/uniformity/embed_norm — guards against the pull-push term cheating the proxy via representation collapse. When R′ done: `run_audit(Rprime_val_best,'ramp','Rprime')` → **`paired('B1','Rprime')` (Claim B) + `paired('B2','Rprime')`**. `scripts/dev/collapse_dump.py` built + added to `attackdro_code.zip` (37 files). ⟨append B1 12-AA + collapse + B1−R′ + B2−R′⟩

---
**SYNC 2026-07-19 — extensions + full-budget close + paper fillables**
- **✅ FULL-BUDGET paired (honest denominator):** M1a_full−M0_full 10k union **−0.0096 [−0.0142,−0.0048]** → M0_full SIGNIF > M1a_full (CLAMP slightly harms at full budget; ℓ∞-driven Δℓ∞ −0.0077). **Δclean +0.0008** (no clean cost; M1a_full 0.8063/M0_full 0.8055). Selection-robustness @1k (last-epoch & val-best) both ~0 ns → 1k underpowered vs a −0.96pp effect; headline stays val-best.
- **✅ CIFAR-100 matched pair (local):** `c5_fromscratch --dataset cifar100` wired (no fork; nc=100, val 2k). Harness C100 support (`load_subset`/`validate_config` branch + C100 configs, backward-compat, CPU-validated). M1a_c100 valWU 0.2310 / M0_c100 0.2185. Probe apgd_ce_linf @1k Δℓ∞ +0.0040 ns; **no-Square@1k union Δ +0.0070 ns@1k, ℓ1-driven +0.0080**, clean −0.013. Sign consistent across proxy/probe/union. 12-AA@10k = candidate.
- **✅ FT-∞ (RobustBench Sehwag base):** B2 reproduce PASSED (clean@10k 0.8459 exact); B3 fine-tune (lr0.005/warmup2/15ep, Colab). Local 12-AA@10k: **ft_clamp union 0.4026 DONE**, ft_none running → paired. robustbench installed `--no-deps` (torch untouched).
- **✅ H3 ablation paired @10k:** M1a_msdglue−M0 **+0.0126 LCB +0.0075**; M1a−M1a_msdglue **+0.0161 LCB +0.0111** (both SIGNIF) → 3-view structure necessary (§3.3). Order M1a 0.4173 > msdglue 0.4012 > M0 0.3886.
- **✅ Paper fillables (CPU):** B1−R′ 10k Δ−0.0005 CI **[−0.0050,+0.0038]** (width 0.88pp, incl 0); **359-verify (M9): both counts = 359 REAL but different sets** (overlap 317/359 — don't claim same examples); **proxy = APGD-CE 3-norm 20-iter single-restart worst-union** on train[49000:50000]; **M0_full 680 s/ep × 50 = 9.44h**.
- **▷ Chained:** seed-3 12-AA@10k (ckpts local, ep80) auto after FT-∞ → paired. Phased pull-push ×2 (push_then_pull@40 ±grad-surgery) gated no-Square@1k section wired in `eval_colab`. Tooling: `union_bench_eval` +`robustdro`/`--skip-square` builder (bit-identical to harness, max|Δ|=0); zip sha `ad84a561`.

---
**SYNC 2026-07-19 (19:15 ICT) — replicates + audits + de-risk builds**
- **Claim A #3 (seed-3):** M1a_seed3 12-AA@10k union **0.4227** done; M0_seed3 running → paired imminent.
- **CONFIG AUDIT (§4.4):** full-budget recipe from ckpt/train.json — ε∞=**8/255** (not 0.03), base **msd_v0 50-step** (not 40/50/50), **50 ep**, one-cycle peak 0.1, views 3×APGD-10, ε₂/ε₁ 0.5/12. M1a_full msd_steps/n_iter not in local artifacts → Kiet W&B confirm.
- **WALL-CLOCK:** R (RAMP) 268 s/ep / 5.96h / 3 adv/step **< M1a** 365 s/ep / 8.11h / 4 adv/step → drop "cheaper-than-RAMP training-cost" claim (M1a adds the MSD base adversarial). Matched-control gain claim unaffected.
- **Fillables:** §5.3 ablation (both SIGNIF), Table-3 ft per-norm (Δℓ∞+0.74/ℓ₂+0.94/ℓ₁+1.09), B1−R′ CI [−0.0050,+0.0038], Δclean full-budget +0.0008, 359 (counts match/sets differ), proxy=APGD-CE 3×20-iter, M0_full 9.44h.
- **De-risk builds:** reversal seed-1 pair (VERIFY passed, chained after seed-3); staleness M1a_full_v50 (`--apgd-view-steps` flag, byte-identical, run-name `_v50` to avoid W&B collision, Colab). Zip sha `8620970e`.

---
**SYNC 2026-07-20 (00:50 ICT) — seed-3 lands + FAT-CLAMP/logging builds**
- **Claim A #3 (seed-3):** M1a_seed3 0.4227 − M0_seed3 0.3970 = +0.0257 LCB +0.0212 SIGNIF. 3-seed min LCB +0.0202>0 (robust).
- **Seed-1 reversal:** M0_full_seed1 ep15/50 training → M1a → no-Sq@1k gate → @10k iff Δ<0.
- **FAT-CLAMP fully built** (zip aded192b): FAT attack + resume-safe + W&B online + std train-logging + FAT logging. Live Colab: valWU declining, nf↑0.64, steps_fooled~1-2 → friendly views too weak → likely no gain (pre-registered). mean_steps_fooled validates ℓ₁≫ℓ∞.
- **Part A logging:** pernorm_train_metrics (RNG-isolated, verified) wired into c5 + finetune; git_commit Colab fallback baked in zip. train_full_msd/RAMP deferred (protect seed-1/v50). backfill_logged.py GATED.
- **RAMP code read:** base=APGD-10 on source+target PAIR (not MSD/worst-of-3); edge = cross-norm KL transfer (weak←strong) + per-epoch clean-model weight fusion (gp=fusion not surgery); 2-3 adv/step (cheaper than our 4). Research Hy A = norm-dependent rep-reg (exploit our reversal), differentiate from RAMP.

---
**SYNC 2026-07-20 (01:20 ICT) — ablations #11–14 built (no new results)**
- seed-1 still training M0_full_seed1 ep17/50 (train.json = per-epoch snapshot, not done-sentinel). No result since 00:50.
- **Ablations #11–14 BUILT** (c5_fromscratch): --base max/avg + --glue-view linf; loop refactored (xs_ce ≠ xs_adv for #13); byte-identical for existing arms (6 smoke paths clean, adv/step 3/3/3/3/4/2). Zip 6a991d47. Runners: run_ablations_11_14.sh (local gated) + train_colab loop + eval_colab paired section. Gated after seed-1/v50.

---
**SYNC 2026-07-20 (10:25 ICT) — seed-1 M0 arm complete**
- **M0_full_seed1 DONE** 50/50, best valWU 0.4920 (seed-0 M0_full 0.481 → consistent). Config verified == seed-0 (msd_steps 50, lr_peak 0.1, ε∞ 8/255).
- M1a_full_seed1 ep11/50 (911 s/ep) → ~20:15 ICT, then no-Sq@1k gate → @10k iff Δ<0.
- Eval queue (v50 staleness + m1a_max) armed behind seed-1; M0 @1k masks+eval.json now local (union 0.3920) → paired(M1a_max,M0) with real LCB (base-confounded; clean #11 needs M0_max).
- Background chains survived session teardown; monitoring waiters re-armed.

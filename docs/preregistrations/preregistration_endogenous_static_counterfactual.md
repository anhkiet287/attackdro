# PRE-REGISTRATION — Endogenous-bottleneck static-counterfactual v3

v2→v3 correction (ATLAS-CRITIC G4 v2 REJECTED-narrow, 3 linked stat fixes): added **RE-P0** (direct static-vs-static union test), made the mechanism claim **conjunctive RE-P0 ∧ RE-P1 ∧ RE-P2**, made **RE-P3a/b descriptive non-gating**, and updated the interpretation matrix to include RE-P0. Everything else was ACCEPTED at v2 (chronology, hybrid disclosure, terminology, RE-P1/RE-P2, B4 reference lock, case-study framing).
Supersedes v1/v2.
Status: **✅ G4 PASSED (ATLAS-CRITIC, 2026-07-12). Research direction LOCKED. Two clerical corrections applied (matrix RE-P3b-union row + execution step-4 RE-P0) — no G4 re-review needed. Remaining: G2 launch evidence packet + director authorization. No new training before that.**
Design type (mandatory disclosure): **HYBRID RETROSPECTIVE–PROSPECTIVE.** A fixed, already-observed exploratory B4 reference + two prospectively pre-registered static comparators. The three-way narrative is NOT purely confirmatory.
Reuses locked infra from `preregistrations/` (splits, canonical suite, compute τ_C=0, grade, bootstrap) — hashes to be frozen in the G2 packet.

---

## 0. Chronology (preserved, per critic §1)
1. Static-model calibration (fixed B3-R1) predicted **L1** as the shifted bottleneck at eps16.
2. B4@R2 did **not** satisfy the pre-registered L1-dominance prediction (f_L1=0.28 < 0.50).
3. → original controlled-shift mechanism claim **FALSIFIED** (preserved negative result).
4. The observed outcome **generated** a new hypothesis (post-hoc).
5. That hypothesis is tested only via **subsequently unrun** static arms.

## 1. Terminology (policy-specific — no "true" bottleneck; critic §3)
- **Calibration bottleneck** = L1 (argmin per-norm on the frozen B3-R1 model at eps16). Verified.
- **B4 terminal bottleneck** = L∞ (argmin on the already-trained B4@R2 model; observed, **exploratory**).
- **Static-arm terminal bottleneck** = to be measured **prospectively** on each new static arm.
No norm is labeled "true", "correct", or "misleading" in run names, arm labels, or results.

## 2. Exploratory vs confirmatory (narrowed; critic §2,4,5)
- **EXPLORATORY (observed, descriptive, NOT re-predicted):** B4's allocation (f_∞0.720/f_2 0/f_1 0.280), B4 per-checkpoint terminal bottleneck (L∞), and the endogenous-bottleneck *hypothesis*. Statements that B4 "tracked/discovered/ignored/was robust to" a bottleneck remain **exploratory only** — they require an independent adaptive replicate (new seed/regime registered before its trajectory is seen), which is **future work**.
- **CONFIRMATORY (pre-registered here; the 2 statics are unrun):** the **conditional static-counterfactual** — given the frozen B4 reference, whether an L∞-weighted static differs from an L1-weighted static, and where B4 falls relative to them.

## 3. Confirmatory question (narrowed; critic §2)
> After observing that the fixed B4@R2 run remained L∞-limited, do two newly pre-registered static runs show that an L1-weighted static (trusting the pre-training L1 calibration) performs worse than an L∞-weighted static, and does the fixed B4 checkpoint lie near/above the L∞-weighted static?

## 4. Arms (renamed; critic §3)
| Arm | Policy (unchanged schedules, hashed) | Config | Status |
|---|---|---|---|
| **B4 (fixed reference)** | predictive_refresh | `programA_b4_adaptive_l1eps16…` | DONE (exploratory reference; canonical checkpoint = **val_best.pt**) |
| **B3-static-L∞-weighted** | `[ℓ∞,ℓ∞,L2,L1]`+extra-ℓ∞/5 | `programA_b3_mis_l1eps16…` | to run (prospective) |
| **B3-static-L1-weighted** | `[L1,L1,L2,ℓ∞]`+extra-L1/5 | `programA_b3_bottleneck_informed_l1eps16…` | to run (prospective) |

## 5. Confirmatory predictions + statistics (critic v2 §4,5,6,7,8)
Machinery = the locked paired sample-level bootstrap on canonical union-survival masks (test_final 8k, B=10,000, seed 0, one-sided 95%), seed-s0 conditional. **B4's comparison checkpoint = val_best.pt (frozen before the statics run).**

**The mechanism claim is CONJUNCTIVE and requires ALL of RE-P0 ∧ RE-P1 ∧ RE-P2** (each one-sided 95%, seed-s0 conditional; individual components not promoted beyond the registered family without caution).
- **RE-P0 (direct static-vs-static — the actual "L1-weighting is worse" test):** `LCB(U_{static-L∞-weighted} − U_{static-L1-weighted}) > 0`. This establishes the L∞-weighted static beats the L1-weighted static on union. **Required** — RE-P1/RE-P2 alone do not test this (both can pass while the statics are indistinguishable).
- **RE-P1 (B4 vs L1-weighted):** `LCB(U_B4 − U_{static-L1-weighted}) > 0`. Limit: *B4 outperforms the L1-weighted static under the locked regime* — NOT "because it discovered the bottleneck".
- **RE-P2 (B4 non-inferiority to L∞-weighted):** `LCB(U_B4 − U_{static-L∞-weighted}) > −0.02`. Limit: *fixed B4 is non-inferior to the L∞-weighted static by 2pp* — NOT equality/oracle/discovery.

**RE-P3 — DESCRIPTIVE mechanism evidence only (NON-gating; does NOT enter the conjunction):**
- **RE-P3a (arm-level terminal ordering):** for each of the 3 arms, `b_j = argmin_k a_{j,k}`; L∞ uniquely limiting iff `a_{j,∞} ≤ min(a_{j,2},a_{j,1}) − 0.03`. **Arm-by-arm descriptive reporting.** If static arms have different terminal bottlenecks → **no shared regime-level bottleneck claim permitted**.
- **RE-P3b (per-norm tradeoff explaining RE-P0):** report point estimates + paired bootstrap CIs (descriptive, NOT gating) for `a_{L1w,L1} − a_{L∞w,L1}` (expect ≥ +0.03: L1-weighted defends L1 more) and `a_{L1w,∞} − a_{L∞w,∞}` (expect ≤ −0.03: defends L∞ less). The union direction is covered inferentially by RE-P0; RE-P3b only explains it per-norm.

**Falsification:** RE-P0 fail OR RE-P1 fail OR RE-P2 fail → the conjunctive mechanism claim is unsupported (negative, Rule 6). RE-P3 never rescues a failed conjunction.

## 6. Interpretation matrix (critic v2-supplied; adopted — includes RE-P0)
| RE-P0 (L∞w vs L1w) | RE-P1 (B4 vs L1w) | RE-P2 (B4 vs L∞w) | RE-P3b | Permitted conclusion |
|---|---|---|---|---|
| pass | pass | pass | supports tradeoff | Full conditional static-counterfactual pattern; **consistent with** (not proof of) the post-hoc endogenous hypothesis |
| pass | pass | fail | any | L∞ weighting beats L1 weighting and B4 beats L1 weighting, but B4 does not match L∞ weighting |
| pass | fail | pass | any | Static weighting contrast exists, but B4 not shown superior to L1-weighted static |
| fail (U_L∞w ≤ U_L1w) | any | any | any | **No evidence L1 weighting is worse** → harmful-calibration claim not supported (if U_L1w > U_L∞w, observed ordering favors L1 → hypothesis rejected) |
| any | any | any | arm terminal bottlenecks differ (RE-P3a) | No shared regime-level bottleneck statement |
"Consistent with" is required because the B4 side was generated post-hoc. RE-P3a/b are descriptive (non-gating).

## 7. B4 fixed-reference verification bundle (critic §7) — to attach
Complete run dir `results/programA_b4_adaptive_l1eps16_ramp80_apgd_8255_t49k_v1k/s0/`; **full** SHA-256 of train.json = `0f495606eb9a15a7b27eb4d7d647b136178e6e1a8201be97bf06f118139b7039`; checkpoint hashes ep020 `e5d3139fa7b8616f655df167057aaab5354d08074a6456b88fe558e81d6c2671`, ep040 `223d8df9daa64ead88e4a426ea3d05a00b7439452b8d53b507a478f24a9eb0f6`, ep060 `82b08588cefdfb2cfa6dd5cedd3946a86adf084a1c8eb159744e3697cb8d1ab9`, ep080 `198f64953383fd5f78a160430414425b12274fd752cc495ca7818cbca126e5d5`; trace manifest `6e4462fc9343ec296a0d3fa1004481b31183c0d9d65671bf54aeac171b4500bd`; allocation-fraction extraction = sum `selected_count_*` over non-refresh batches / total (per v3-A A3 method).
**Comparison checkpoint = `ckpt/val_best.pt`, SHA-256 `c2d708dd1b2da5379d26b0e9cc4318c76cb0d93790401b920824b718aed49439`, selected via `val_select/worst_union` (used_for_selection=True), saved epoch idx 70, worst_union 0.47200000286102295** (cross-checked: train.json.best_val_select_worst_union = train.json.history[70].val_select/worst_union = 0.47200000286102295). This checkpoint is **frozen now, before the two static runs exist**, and is the fixed B4 reference for RE-P1/RE-P2/RE-P3.

## 8. Reuse-infra + launch evidence for G2 (critic §8)
The G2 packet must include: **director disposition of the prior unauthorized calibration (RATIFIED 2026-07-11)**; final implementation commit; **exact hashes of the 2 static configs**; **compute preflight on those exact configs (τ_C=0)**; result-dir absence checks; gatekeeper PASS; no active trainer/evaluator; new test_final audit subset SHA `af0b037…`.

## 9. Top-venue framing (critic §10)
This is a **controlled CIFAR-10 eps-16 case study / mechanism evidence**, NOT a stand-alone general "robust to bottleneck misidentification" claim. A top-tier central claim additionally needs a natural dataset-level bottleneck change (the separate ImageNet/cross-dataset program) and/or a fresh seed/regime registered before results.

## 10. Execution after G4 Pass + director gate
1. Run **B3-static-L∞-weighted** + **B3-static-L1-weighted** (2×80ep, s0, eps16; existing configs).
2. Component-execution check (12 components) → REQUIRED else mechanical exclusion branch.
3. Canonical audit on **test_final** for all 3 arms (B4 val_best + 2 statics val_best), masks exported.
4. Paired bootstrap **RE-P0, RE-P1, AND RE-P2**; evaluate the conjunctive **RE-P0 ∧ RE-P1 ∧ RE-P2** criterion; report descriptive RE-P3a/b (non-gating).
5. Director interprets per §6 matrix → authorized claim (with hybrid + case-study disclosures).

## 11. Lock status
NEW-LOCKED: hybrid design disclosure, policy-specific terminology, narrowed conditional claim, **RE-P0 (LCB(U_L∞w − U_L1w) > 0)**, **conjunctive RE-P0 ∧ RE-P1 ∧ RE-P2**, RE-P3a/b **descriptive non-gating** (thresholds m_b=0.03, δ=0.03 as point-estimate references + reported paired CIs), B4 comparison checkpoint=val_best (frozen, SHA c2d708dd…), RE-P0 interpretation matrix, case-study framing.
val_best.pt SHA + selection record: **FILLED (§7)** — bundle complete.
`[in G2 packet]`: 2 static config hashes + preflight + gatekeeper + provenance + governance disposition.
**No new training before ATLAS-CRITIC G4 Pass + director gate.**
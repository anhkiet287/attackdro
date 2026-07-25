# STATE — compact current state

Document ID: `PROJECT-STATE` · Version: `compact-1.0` · Governed by `RESEARCH_CHARTER.md`
This is the routine-read authority. Full history and transcripts → `STATE_LOG.md` (do not read in full).
Fields marked `[CONFIRM]` need director confirmation — drafted from prior state, not yet verified this session.

---

**PROJECT_OBJECTIVE**
Efficiency (accuracy-per-FLOP) of **predictive vs static** allocation for multi-norm adversarial training on CIFAR-10 / PreActResNet-18, eps = (ℓ∞ 8/255, ℓ2 0.5, ℓ1 12), built on RAMP. Claim is efficiency, **not** beating SOTA union accuracy.

**CURRENT_RESEARCH_QUESTION**
Given ℓ∞ is a static bottleneck, can predictive allocation earn its cost over static — and does that failure open the door to a loss-mechanism fix (Claim 2)?

**CURRENT_PHASE**
Allocation arc closing. B4 branch diagnostic complete → registered conclusion `ADAPTIVE_DIRECTION: PAUSE`. Next: decide whether current evidence closes Claim 1 (allocation ceiling) and pivot to Claim 2 (pull-push loss) — pending critic gate.

**ACTIVE_HYPOTHESES**
- Claim 1 — allocation hits a hard ceiling due to the static ℓ∞ bottleneck (evidence: B1–B4).
- Claim 2 — a pull-push loss (FaceNet triplet + RAMP logit pairing) resolves the bottleneck RAMP's logit pairing misses.

**LOCKED_PROTOCOLS**
- RAMP 80-epoch recipe; APGD training for all 3 norms.
- Decision grade: APGD 20/20/100, n=1000. Full AutoAttack for winner only.
- Checkpoint selection: `val_best.pt` via `val_select/worst_union`. `best.pt` aliases not canonical-safe.

**ACCEPTED_EVIDENCE** (APGD 20/20/100, n=1000)

| Arm | Union | Compute ratio |
|---|---|---|
| B1 reactive control | 0.423 | 1.000 |
| B2 static equal cycle | 0.399 | 0.333 |
| B3 static weighted-ℓ∞ | 0.422 | 0.400 |

Key finding: **B3 near-B1 parity at 40% compute without predictive.** ℓ∞ static bottleneck across 80 epochs (switch count 0, margin ~12pp) in B2 and B3.

**B4 BRANCH DIAGNOSTIC** (`branch_diag_b4_small_v1`, registered `ADAPTIVE_DIRECTION: PAUSE`)
- Anchor resolution: ep10 spread 0.003 < noise 0.010 → unresolved; ep40 spread 0.008 < noise 0.018 → unresolved; ep75 L1 0.429 / ℓ∞ 0.425 / L2 0.423, spread 0.006 > noise 0.001 → L1 nominally best. **Only 1/3 anchors resolved.**
- B4 non-refresh allocation leans strongly ℓ∞, does not pick L2. Mechanism does **not** demonstrate adaptive benefit over static.
- Reading: supports B4's role as **diagnostic for the allocation ceiling, not a winner** — but the diagnostic is partially under-resolved (2/3 anchors noise-limited); strength of a universal-ceiling claim is the open question.

**NEGATIVE / GUARDS**
- FAB-T non-deterministic (restart RNG not seed-controllable) → old Tier-1 JSONs canonical; future audits export masks at first run.
- Bootstrap CIs exclude zero (mild overestimation) but FAB-T restart variance not captured → acknowledge in claims.

**OPEN_QUESTIONS**
- Does current evidence (B1–B4 diagnostic) suffice to **close Claim 1** (allocation ceiling), or is it under-resolved (2/3 anchors noise-limited) and needs strengthening?
- B4-vs-B2-vs-B3 criterion **not locked** — B4 role = diagnostic per evidence, not competitor; a formal "beat B3 at equal compute" criterion is likely moot given `PAUSE`. Defer to the Claim-1-closure critic gate.
- Pull-push loss design (Claim 2).

**ACTIVE_TASKS**
- Phase 1 repo restructure — **done** (merged `e5dfea92`).
- Tier-1 AA-component audit on B3 + B4 val_best — **Director-authorized, running** (canonical per-norm + CIs; masks exported at first run).
- Path A shift-experiment — **Director-chosen**; in FRAME/design (below). Goal: show adaptive B4 tracks a shifted bottleneck while static-oracle B3 (fixed weights) is mis-specified and lags.

**DIRECTOR_DECISIONS** (verbatim)
- Claim-1 framing (director): B2/B3 static; B3 weights human-set with intent knowing ℓ∞ is bottleneck. But ℓ∞ is not guaranteed to always be the bottleneck. B4's value = adaptive discovery of the bottleneck (the thing B3 needs a human for), at ~equal union to B3 but higher per-norm ℓ∞. Contribution = removing the oracle assumption, not raw performance.
- Adaptive-value evidence: **Path A** — run a shift-setting to demonstrate adaptivity earns its keep.
- Shift mechanism: **eps-shift** (cross-regime static), new bottleneck = **L1**. Refinement: keep ℓ∞ 8/255 standard, **raise L1 eps** (not lower ℓ∞) to avoid a toy threat model. 3 arms (B4 / B3-mis / B3-oracle-R2), seed s0, falsification δ = 2pp union (CI excludes 0).
- Venue = **top-tier** → cross-dataset generality nearly required. Plan: **A first (controlled proof of adaptivity), B (cross-dataset) planned early** as the external-validity amplifier. Compute budget: comfortable; prioritize valuable results.

**SITUATE — cross-dataset (findings, citation-verify at gate)**
- Bottleneck norm **does shift** and is citable: L∞ hardest on CIFAR-10; **L1 hardest on ImageNet** (Croce & Hein, E-AT, ICML 2022) due to eps calibration + dimension. → justifies A's raise-L1 regime as a cheap CIFAR proxy of a real phenomenon; neutralizes "contrived" critique.
- **L2 rarely the bottleneck** — largely implied by L1∪L∞ union (E-AT Prop 3.1); real tension is L1↔L∞. → B4's L2=0% allocation is theory-predicted correct, not a collapse.
- CIFAR-10 union triple (ℓ∞ 8/255, L2 0.5, L1 12) is de-facto standard (MSD ICML 2020 origin; E-AT, RAMP inherit). **RAMP tested only CIFAR-10 + ImageNet** → CIFAR-100/SVHN union AT is uncovered (novelty).
- Cross-dataset pick: **CIFAR-100 primary** (same triple/cost, standard, expected), **SVHN secondary** (cheap, may show natural shift, but AT-unstable + eps less standard), **avoid TinyImageNet** (3–4× cost, non-standard eps, noisy). Sources: MSD ICML2020, E-AT ICML2022, RAMP NeurIPS2024.

**DEV-EVAL PER-NORM** (grade: develop-eval, 1 restart, n=1000 — NOT canonical)
B3: ℓ∞ 0.451 / L2 0.679 / L1 0.493 / union 0.422. B4: ℓ∞ 0.466 / L2 0.672 / L1 0.463 / union 0.423. Same compute 0.3997. Read: B4 sharpens ℓ∞ (+1.5pp) by reallocating from L1 (−3.0pp); L2 easy (ignoring it is correct). Margins thin → await Tier-1 audit before claiming.

**CRITIC_GATE_LOG — G1/G2 shift pre-reg v1 (2026-07-11)**
- ATLAS-CRITIC verdict: **BLOCKER** (G1 REJECTED, G2 REJECTED). 11 items + citation audit.
- Chief/Director disposition: **all items ACCEPTED** → pre-reg v2 required. Key: (1) split 3 estimands (identification/allocation/outcome); (2) executable calibration rule + split map; (3) P1 superiority LCB>0 + 2pp as practical point estimate, seed-s0-conditional; (4) P2 quantitative thresholds; (5) P3 non-inferiority test w/ margin; (6) rename oracle→**bottleneck-informed** static; (7) per-arm compute accounting; (8) **defer Program B** (separate pre-reg after A); (9) explicit data-role split map (distinct calibration/selection/decision splits, no reuse); (10) symmetric canonical audit all arms; (11) outcome→conclusion interpretation matrix.
- Citation corrections ACCEPTED: **remove "L2 rarely binding / Prop 3.1"** (Chief overclaim — Prop 3.1 gives only ~L2 0.22 < 0.5; RAMP shows L2 can be bottleneck at larger eps2); narrow "standard triple" to CIFAR-10 only; drop cross-dataset novelty claim (needs broader search); internal JSONs = contextual only until artifact-level (hashes/configs) exported.

**CALIBRATION FEASIBILITY** (Claude Code, val_select probe — NOT the locked value): argmin flips to L1 at eps≥16; admissible {16,20,24,28} (m=0.03, non-collapse 0.10); smallest=16 (L1 racc 0.375). Program A **feasible**. Config extracted: B3 cycle `[ℓ∞,ℓ∞,L2,L1]`+extra-ℓ∞/5; splits 49k/1k/1k/10k (no cal); FLOPs denom n_batches×30, R1 ratio 0.39965.

**CRITIC_GATE_LOG — G1/G2 v2 (2026-07-11)**
- Verdict: **BLOCKER** (G1/G2 REJECTED), but strong convergence. Full PASS: P1, P3, comparator, all citations. Accepted-with-condition: E_id/estimands, calibration, P2, compute, splits, matrix. **3 hard blockers:** (8) Program B not an executable multi-run contract; (10) canonical audit not parametrically frozen; (1) E_id signal not operationally defined.
- Critic guidance: a **narrow v3 addendum** suffices; "accepted foundation" locked (3 estimands, calibration grid/margins, P1/P2/P3 thresholds, s0 scope, A arms/permutation, split 49k/1k/1k/1k/8k, no novelty, no L2-rarely-binding).
- v3 required (9): executable E_id + unresolved-label rule · one primary P2 window · full CIFAR-100/SVHN data/train/split contract · informed schedules for all b_D branches · branch-specific B run table · exact norm×component audit table · corrected cal-role wording · artifact-verify R1 trace used by P2 · "oracle-equivalence"→comparator-specific non-inferiority.

**GATE SPLIT (director): Program A now, Program B separate later.**
v3-A addendum **complete + all specs FILLED** from Codex extraction (E_id signal, checkpoint SHA, attack config, f_L1^R1=0.13907 verified, attack-step formulas, canonical audit table). Ready to relay to ATLAS-CRITIC (G1, Program A only).

**Implement-before-launch (Builder, after G1 Pass):** (1) `cal` split + frozen index (test 9000–9999) + `test_final`=8k in `datasets.py`; (2) ep20/40/60/80 checkpointing on R2 run; (3) re-draw canonical audit subset from `test_final`; (4) compute-enforcement preflight; (5) locked calibration on `cal` (predicts eps=16).

**CRITIC_GATE_LOG — v3-A (2026-07-11):** BLOCKER-narrow. A4/A6/A7 + Program-B-removal = PASS. 3 narrow fixes needed: checkpoint semantics, audit optionality, extraction bundle. **All 3 closed this round** (checkpoint table frozen; all-4-components-REQUIRED + named mask + error-contingency; verification bundle = extraction report). Critic: "G1 eligible for Pass" once these are frozen+supplied.

**✅ G1 PASS — PRE-REGISTRATION LOCKED (2026-07-11).** ATLAS-CRITIC locked Program A: 3 arms (B4-adaptive / B3-mis / B3-bottleneck-informed-L1), E_id/P2/P1/P3 estimands, calibration rule, split roles, symmetric audit, interpretation matrix, seed-s0 scope. All repo-facts VERIFIED from extraction bundle. Two locked-text fixes applied (A1 starvation-override wording; A4 exact finite-run compute formulas). **Design frozen — no reopening thresholds/arms/splits/predictions.**

**PROTOCOL_DESIGN_STATUS: LOCKED. Program B: separate contract (out of scope).**

**CANONICAL R1 AUDIT (baseline context — OLD subset overlaps cal, seed s0, NOT the Program-A claim):** B3 ℓ∞0.420/L20.668/L10.483/union**0.401**; B4 ℓ∞0.456/L20.665/L10.473/union**0.421**. Under canonical (full-AA) grade the B4–B3 gap widens vs dev-eval: ℓ∞ +3.6pp, **union +2pp** (dev-eval was tied). Marginal union CIs overlap (B3 [0.370,0.433], B4 [0.391,0.451]) → paired-diff needed for significance. Masks exported (B3 JSON `476fc76b…`, B4 JSON `3c73d1b1…`). G2-BUILD complete on branch `feat/programA-g2-impl` (splits/preflight τ_C=0/subsets hashed); **core machinery still uncommitted (provenance risk)**; v2 not in repo.

**✅ G2 IMPLEMENTATION PACKET COMPLETE (2026-07-11).** Commit hygiene done (5 commits, all harnesses tracked, provenance clean). Calibration on `cal` → **L1 eps = 16 LOCKED** (band {16,20,24,28}; eps32+ collapse). Preflight τ_C=0 (366,720 units/arm). Audit dry-run: all 12 components REQUIRED, 0 excluded. 3 configs @ eps16 + frozen audit config generated. Packet `g2_evidence/programA/G2_EVIDENCE_PACKET.md` SHA `2889d58c…`. Only condition 14 (director launch authorization) remains, after critic G2.

**✅ G2 REVIEW: Technical PASS + Scientific PASS. Only governance closure required (calibration ran before G2 auth; prior gate's instructions were internally contradictory — required calibration as G2 evidence yet marked it unauthorized).**

**GOVERNANCE DISPOSITION (Director — RATIFIED 2026-07-11):** The locked calibration (R2 L1-eps selection) executed during G2 packet assembly, before ATLAS-CRITIC G2 authorization, contrary to the prior gate's `CALIBRATION_OPERATION_AUTHORIZED: NO` — while the same gate required a completed calibration as G2 evidence (internal contradiction). Acknowledged + recorded. Because the calibration (a) followed the G1-locked rule, (b) used NO Program-A arm results, (c) made NO post-hoc change to any threshold/grid/split/attack/arm, (d) deterministically selected **L1 eps=16** by smallest-admissible — the result is **RETAINED AS BINDING**, not repeated/replaced/discarded. **Corrective control:** when a gate's instructions are internally contradictory (an operation's output required as evidence yet the operation marked unauthorized), CHIEF must obtain explicit director/critic clarification before executing — not resolve unilaterally; protocol-affecting ops get an explicit phase-authorization checkpoint distinct from packet assembly.

**RETAINED (non-blocking) audit condition:** dry-run verified only config-parse + component import, NOT execution. Before any Program-A checkpoint is audited: either all 12 components execute → stay REQUIRED, or the 2-identical-run unsupported-op branch applies on the frozen B3 ckpt+subset (no acc/mask may affect exclusion; infra failures don't). Happens at audit time (post-training), not now.

**✅ LAUNCH AUTHORIZED (Director 2026-07-11).** Governance ratified. **G2 PASS (technical+scientific) + director authorization = all 14 conditions met.** Mode: **sequential on RTX 5070 Ti, order B4-adaptive → B3-mis → B3-informed**, 80ep, seed s0, eps16. No interim arm dropping / no threshold/eps/config change / selection stays val_select/worst_union.

**B4@R2 eps16 COMPLETE (80 ep, s0). P2 FAIL (registered falsification).** eps16 = calibration-locked (cal). Full-run allocation f_ℓ∞0.720 / f_L2 0.000 / f_L1 0.280 (shift +0.141<0.20 FAIL; dominance 0.28<0.50 FAIL). Per-norm ep80 ℓ∞0.467 < L10.501 < L20.690 → **ℓ∞ bottleneck at ALL checkpoints (ep20/40/60/80); E_id label b*=ℓ∞ everywhere.** val_best worst_union 0.472 (ep70). train.json SHA `0f495606…`.
**FINDING — endogenous bottleneck:** setting L1 eps to make L1 the *static-calibration* bottleneck does NOT make it the *trained* bottleneck; adaptive B4 defends L1 up (0.501) and ℓ∞ re-emerges hardest (0.467). B4 responded (L1 alloc 0.14→0.28) but correctly kept ℓ∞ dominant → **B4 tracks the true endogenous bottleneck, not the static calibration.** eps-shift on CIFAR-10 is fundamentally too weak (natural L1-bottleneck needs dimension → ImageNet, per SITUATE). This **strengthens Claim 1** (allocation ceiling: ℓ∞ bottleneck is endogenous+stubborn) and motivates Claim 2. Registered negative result — preserved (Rule 6).
**DIRECTOR CHOSE (c) — reframe on CIFAR + existing arms.** New pre-reg drafted: `preregistration_endogenous_v1.md` + `critic_packet_endogenous_v1.md`. Reframe = *"bottleneck endogenous; B4 tracks true (ℓ∞); static-trusting-calibration (B3-static-L1) mis-allocates; B4 matches correct static (B3-static-ℓ∞) without a prior."* Arms re-labeled (same configs): b3_mis→**B3-static-ℓ∞ (correct)**, b3_bottleneck_informed→**B3-static-L1 (calibration-trusting, wrong)**. B4 = exploratory/observed; 2 statics = confirmatory/not-yet-run. RE-P1 (B4>B3-static-L1 LCB>0), RE-P2 (B4 non-inf B3-static-ℓ∞, δ0.02), RE-P3 (ℓ∞ endogenous). Reuses ALL locked infra. **Cross-dataset natural-shift = separate program (b) later.**

**CRITIC_GATE_LOG — endogenous G4 v1 (2026-07-12):** REJECTED-narrow, direction accepted. 6 fixes: (1) narrow confirmatory claim — B4 "discovered/tracked/robust" stays EXPLORATORY, only a *conditional static-counterfactual* is confirmatory; (2) no "true bottleneck"/"correct"/"misleading" — policy-specific terms, rename arms to **L∞-weighted / L1-weighted**; (3) operationalize RE-P3 → RE-P3a (terminal ordering m_b=0.03) + RE-P3b (tradeoff δ_1=δ_∞=0.03); (4) conjunctive **RE-P1∧RE-P2** multiplicity; (5) B4 verification bundle (full hashes, val_best=comparison ckpt, frozen-before-statics); (6) frame as **controlled case study / mechanism evidence**, not standalone top-tier claim; + hybrid retrospective-prospective disclosure. **All fixed in `preregistration_endogenous_v2.md`.** Original P2 negative result PRESERVED.

**CRITIC_GATE_LOG — endogenous G4 v2 (2026-07-12):** REJECTED-narrow, all conceptual items PASS (chronology, hybrid, terminology, RE-P1, RE-P2, B4 lock, framing). **3 linked stat fixes:** add RE-P0 (static-vs-static), conjunctive RE-P0∧RE-P1∧RE-P2, RE-P3 descriptive non-gating + matrix. **All applied** → pre-reg now **v3** (in `preregistration_endogenous_v2.md`; B4 bundle complete, val_best SHA `c2d708dd…`).

**✅ G4 PASS (2026-07-12) — endogenous static-counterfactual reframe LOCKED.** Primary pattern = **RE-P0 ∧ RE-P1 ∧ RE-P2** (conjunctive); RE-P3a/b descriptive non-gating; hybrid retrospective-prospective; B4 val_best frozen (SHA c2d708dd…). 2 clerical corrections applied. Original shift-P2 preserved as falsified.

**✅ G2 PASS (2026-07-12) for the 2 static arms.** Technical + scientific readiness PASS; governance ratified; τ_C=0; B4 frozen; RE-P0∧P1∧P2 contract locked. **Conditions:** (1) post-G4 director launch decision (July-11 auth covered old framing only); (2) immediate pre-launch checks + frozen commands; (3) analysis script built + dummy-tested + frozen BEFORE inspecting real canonical masks; (4) both statics = one inseparable set, no outcome-dependent dropping.
**AUTHORIZED CLAIM BOUNDARY (locked):** strongest permitted = "in this hybrid CIFAR-10 eps16 case study, L∞-weighted static > L1-weighted static; frozen B4 > L1-weighted and non-inferior to L∞-weighted; consistent with (not proof of) the endogenous hypothesis." NOT: prospective discovery / general robustness / multi-seed / cross-dataset / optimal-static superiority.

**✅ POST-G4 LAUNCH DECISION (Director, 2026-07-12):** authorizes B3-static-L∞-weighted + B3-static-L1-weighted as ONE inseparable prospective static-counterfactual set under the G4-locked endogenous protocol; analysis RE-P0∧RE-P1∧RE-P2; RE-P3a/b descriptive; original shift-P2 failure preserved; NO outcome-dependent stopping or arm removal (first static result must not be used to reconsider the second); frozen commands, no overrides; analysis script built+dummy-tested+frozen BEFORE any real canonical mask is inspected. Analysis script: build in parallel during training.

**⚠️→✅ AUDIT CONFIG CORRECTNESS FIX (2026-07-12):** frozen test_final audit config had L1 eps=**12** (standard-triple leftover) — inconsistent with the eps16 regime (arms trained L1=16; pre-reg is eps16). Auditing at 12 = wrong threat model → would corrupt RE-P. Runner caught it, refused to edit frozen config. New regime-matched config `audit_…_v3A_testfinal_l1eps16.yaml` SHA `3cd64efb01b3c1296364e813f91dc2d1886329478577ca3816d71caacf9145af`, commit `9eba859d`; diff vs old (9162ce44) = ONLY `l1 12→16`; subset/seed/12-components/ℓ∞/L2 identical; pre-results (no audit run). **✅ CRITIC CONFIRMED CORRECTION + 🚀 CANONICAL AUDIT LAUNCHED (2026-07-12) at STANDARD TRIPLE `9162ce44` (L1=12).** Critic: canonical eval = standard triple (8/255,0.5,12); eps16 = training stressor; RE-P0/P1/P2 concern U(8/255,0.5,12); eps16 config `3cd64efb` WITHDRAWN (git rm, commit 3cccba99). Audit running (3 arm sequential, monitor b79atcbwk, isolated results/audit/programA_v3A_l1eps12/). Clarification sentence added to pre-reg §5+§9 (canonical=standard triple; eps16=stress) — critic-authorized as clarification, NOT design change; **to commit into repo locked pre-reg + audit manifest**. Claim boundary: ordering holds under standard canonical union, NOT eps16 eval.

**↩️ (superseded) CORRECTION note — canonical audit STANDARD TRIPLE L1=12 (config `9162ce44`).** Chief's "L1=12 leftover" flag was WRONG. Hard evidence (runner): audit harness CODE-LOCKS canonical eps to standard triple (`eval_multinorm_audit.py:154`, L1=12); ALL prior B1–B4 canonical audits used L1=12; pre-reg §5 "reuses locked canonical suite" = this standard-triple harness. So canonical eval at standard triple is DELIBERATE + registered; eps16 is a TRAINING-time stressor, arms evaluated on the fixed standard union for comparability with B1–B4/literature. eps16 sibling config `3cd64efb…` WITHDRAWN. Correction note sent to critic to withdraw the eps16 re-auth + restore `9162ce44…` as canonical. RE-P0/P1/P2 compare STANDARD-triple canonical union.

**✅ RE-P ANALYSIS SCRIPT FROZEN (pre-analysis lock met).** `scripts/dev/analyze_endogenous_statics.py` SHA `aa8994f41b00e8e72af7ae4b4d630b7ef80636a2b5dc1888c3bbaae826c60873`, commit `d99aaa7c`. Paired bootstrap B=10000/seed0/1-sided-95%; RE-P0/P1/P2 + conjunctive verdict; RE-P3a/b descriptive; selftest PASSED on synthetic masks (no real data). Locked before any canonical mask opened. → canonical audit authorized.

**✅ 2 STATICS COMPLETE (2026-07-12, launched HEAD c60d2be8).** 80 ep each, s0, eps16, no override; flops 0.39965; val_best via val_select/worst_union. Artifacts: b3_mis (L∞-w) train.json `b88c3e8e…` val_best `bdc4bd23…`; b3_bottleneck_informed (L1-w) train.json `c0bb2540…` val_best `09062f6f…` (+ ep020/040/060/080 hashes recorded).
**IN-TRAINING PROBE (val_select grade — NOT canonical, NOT the registered test):** val_best worst_union — B4 0.472 / L∞-weighted 0.448 / L1-weighted 0.395. ep80 per-norm: L∞-w ℓ∞0.434/L20.686/L10.496 (argmin ℓ∞); L1-w ℓ∞0.372/L20.677/L10.554 (argmin ℓ∞). Probe direction consistent with RE-P0 (L∞-w > L1-w), RE-P1 (B4 > L1-w), RE-P2 (B4 ≳ L∞-w), and RE-P3b over-defence (L1-w: higher L1, lower ℓ∞, lower union). **HOLD — registered RE-P0/P1/P2 run on canonical test_final audit, not these probes; orderings can shift under canonical (cf. R1 dev-eval→canonical).** W&B backfill fix committed 96fbec0a (logging only, train.json canonical throughout, no data lost).

**G2 LAUNCH PACKET READY (provenance-complete).** Commit `c60d2be8` (branch feat/programA-g2-impl): 3 arm configs + test_final audit config + STATE + locked endogenous pre-reg (`docs/preregistrations/preregistration_endogenous_static_counterfactual.md`, unedited) all tracked. Packet `g2_evidence/programA_statics/G2_STATIC_LAUNCH_PACKET.md` HEAD=c60d2be8. Preflight τ_C=0 (both statics 366,720), result-dirs absent, gatekeeper PASS, B4 bundle re-confirmed. **Submit to ATLAS-CRITIC G2; on Pass → director authorize → launch.**

--- (prior) assemble G2 LAUNCH EVIDENCE PACKET for the 2 statics (9 items): governance disposition (ratified) · impl provenance (branch/HEAD/hashes) · 2 static config SHAs (programA_b3_mis…, programA_b3_bottleneck_informed…) · finite-run preflight τ_C=0 on both · result-dir absence · gatekeeper PASS + no active trainer + no test/cal selection · B4 verification bundle · analysis contract (RE-P0/P1/P2 + subset/config hashes + bootstrap script + masks + matrix) · director launch authorization. → critic G2 review → director authorize → run 2 statics (80ep s0) as one locked set → component check → canonical audit 3 arms test_final → RE-P0∧P1∧P2 + RE-P3a/b → interpret. No new training before G2 packet + authorization.**

**NEXT_DECISION_GATE — post-training pipeline (after all 3 runs):** (1) develop-eval per arm; (2) **component-execution check** (all 12 run → REQUIRED, else mechanical exclusion branch) BEFORE audit; (3) **symmetric canonical audit** on test_final (new subset) for all 3 arms, masks exported; (4) compute **E_id / P2 / P1 / P3** per the locked interpretation matrix; (5) director interprets → authorized claim. No post-hoc protocol change.

--- (superseded) 14-condition checklist below all now satisfied except director authorization ---
**Prior — G2 implementation packet (14 conditions).**
Build + verify before launch: cal + 8k test_final split & frozen hashed index · ep020/040/060/080 checkpointing · new class-balanced audit subset from test_final (hashed) · locked calibration run (→admissible eps or STOP) · 3 concrete configs at calibrated eps · finite-run compute preflight τ_C=0 · audit dry-run frozen+recorded · provenance commit/diff · gatekeeper no-leak · **then director launch authorization** for the 3 arms. Critic G2 reviews this evidence packet. No calibration/training authorized yet.

**━━━ PAPER C (pull-push representation loss) — exploratory probe arc, 2026-07-13 ━━━**
Round-trip worklog = `docs/BUILDER_LOG.md` (registry + task RESULTs). All below: fine-tune only, no from-scratch, no locked-artifact/canonical-config/training-code change, no git commits. Backbone = public MSD-80 (`external/robust_union/…/Selected/MSD.pt`, sha `482bf28…`). Probe = 3-epoch fine-tune, 5k train-holdout val (train[44000:49000]; TEST never used), 12-AA strict union (config `9162ce44`), ≥2 seeds, paired bootstrap B=10000. Impl `scripts/dev/c0_killtest.py` (+`c0_verdict`/`c1_verdict`).

**✅ C0 kill-test = GO (over placebo).** A0 continued-MSD 0.5068 · A1 placebo 0.4774 · A2 pull-push 0.5027. **A2−A1 = +0.0253 union, paired LCB +0.0210 > 0** (noise band 0.0112), ℓ∞ lift +0.0281, no masking → class-geometry signal is REAL. **But A2 ≈ A0** (−0.0041): pull-push does not beat plain continued-MSD at fine-tune scale. Anchor fixed = A (clean, stop-grad); λ=0.5 (triage), τ=0.1.

**✅ C1 negative-mining = NO-WIN.** hard-topk (k8) 0.5036 · semi-hard (δ0.2) 0.5046 · full-pool 0.5027 — all LCB<0 vs A0, no collapse. Harder negatives do NOT rescue the tie.

**✅ C1b RAMP-pairing positive control = probe UNDER-POWERED.** A3 = MSD + RAMP KL logit-pairing (λ_ramp=1.5, RAMP default; logit-pairing only, no grad-projection). A3-ramp 0.5082 (ΔvsA0 +0.0014, LCB −0.0017). **A known from-scratch MSD-beater ALSO ties A0 at 3ep** → the fine-tune probe cannot resolve any mechanism. **⇒ the pull-push tie is INCONCLUSIVE, not a kill.** **NEXT (director decision):** test pull-push **from-scratch (C5+)** rather than abandon on probe evidence.

**⏸ Union benchmark (Paper-1 baselines).** Under our 12-AA (standard triple): MSD 0.442 > B4 0.420 > B3 0.410 > AVG 0.388 > MAX 0.280 (1k); **10k: B4 0.4114 / B3 0.4028 done, MSD/MAX/AVG @10k DEFERRED** (director paused sweep after B3 for the C5 GPU window). **E-AT skipped**. → `results/eval/union_bench/`.

**▶ C5 FROM-SCRATCH pull-push RUNNING (Task F, 2026-07-15).** `scripts/dev/c5_fromscratch.py` — **M1 = TRUE MSD (`msd_v0`, steps=10) + decoupled pull-push** (`L=L_CE(MSD)+α·scaffold+β·glue`, anchor A stop-grad, α=β=0.5 warmup 0→10, τ=0.1); recipe = B3/B4 RAMP-80 (PreActRN18, 80ep, lr0.05@70, crop+flip); positives=3 per-norm APGD; head discarded at eval; resume-safe; **W&B online** (project `attackdro-union`). **M1a (neg=adv, seed0) DONE** — from-scratch best valWU 0.473 (peak ep70), no collapse. **12-AA HIT: 1k union 0.437; 10k union 0.4173** [0.408,0.427] (ℓ∞0.427/ℓ20.664/ℓ1**0.515**, clean 0.816), no masking. **Paired @10k: M1a−B3 +0.0145 LCB +0.0094 (signif); M1a−B4 +0.0059 LCB +0.0008 (marginal).** Caveat: 1 seed (M1a−B4 seed-fragile → seed-2 queued); clean isolation = M1a−M0 (M0 matched-control training). MSD/RAMP@10k pending. **M1b (neg=clean, same seed)** → Colab. Read M1 primarily vs B3 0.410 / B4 0.420 (same 10-step recipe). On done → 12-AA audit val_best vs all bars.

**ℹ️ RAMP-vs-M1 training-attack budget (read-only check, 2026-07-15).** RAMP-80 (`RAMP.py`, from `RAMP_scratch_cifar10.sh`: `--at_iter 10 --epochs 80`): per step **2 APGD attacks** = the ℓ∞↔ℓ1 tradeoff pair (init source=ℓ∞, target=ℓ1; L2 in `l_norms` but NOT attacked in train), **n_iter=10 each**, restarts 1, standard-triple eps; `--max` CE on per-sample max-loss of the 2 + KL logit-pairing (lbd). ~240s/ep. **M1a** (`c5_fromscratch.py`): per step **1 MSD (`msd_v0` steps=10; 1 grad+3 candidate-fwd/step) + 3 per-norm APGD (n_iter=10 each, ℓ∞/ℓ2/ℓ1 positives)** = **4 attack-generations**; ~365s/ep (incl. per-epoch val worst-union eval RAMP skips). **Verdict: per-attack step count MATCHES (both 10)** — our `steps=10` is RAMP's exact per-attack budget. **Attacks/step: RAMP 2 vs ours 4** (the 3 per-norm positives are the pull-push tax); grad-computations ~20 vs ~40 (~2×), wall-clock only ~1.5× because MSD's 3 candidates are forward-only + fixed val overhead. So M1 pays ~2× RAMP's attack volume at the same per-attack depth.

**✅ Task G INVENTORY + gap analysis (2026-07-15) → `results/eval/union_bench/INVENTORY.md`.** All ckpts load; **RAMP-80 registered** (`armA_rampfull_5070ti/ep_80_0.pth` sha `aec84129…`, clean-gate PASS 81.2%/log 80.9%, own-eval union 44.7% = SOTA bar R). All eval.json keep per-norm ⇒ worst-norm/average DERIVABLE. Gap: **R** (RAMP@12-AA) + MSD/MAX/AVG@10k = EVAL-ONLY (ckpts ready); **M1** = training; **m0** (matched-MSD) NOT planned (B3/B4 = same-recipe peers); **R′** = future.

**✅ CLAIM A CONFIRMED (2026-07-16) — from-scratch pull-push beats the compute/seed-matched pure-MSD control.** M0 = pure MSD-AT (no head/positives/pull-push, same recipe, seed0), audited under frozen `9162ce44` (robustdro): **1k union 0.3920 · 10k 0.3886**. Paired sample-bootstrap (B=10000, seed0, 1-sided 95% LCB): **1k M1a−M0 +0.0451 LCB +0.0280 SIGNIF; 10k M1a−M0 +0.0287 LCB +0.0234 SIGNIF.** 10k Δ +2.87pp is the decision number — matched-seed, no fragility caveat; larger than M1a−B3/B4 (+1.45/+0.59) because M0 isolates the pull-push mechanism, not just the recipe. Resolves the C0/C1 fine-tune tie as UNDER-POWERED, not a kill.

**▶ CLAIM B (Task H) — RAMP + rep term, baseline pivoted R→R′ (director, 2026-07-16).** **B1 = RAMP + pull-push** (α·scaffold+β·glue on RAMP's own ℓ∞/ℓ₁ adv, τ0.1, α=β0.5) TRAINED on Colab (80ep, no collapse); **0.592 proxy DISCARDED as overfit** — real number = frozen 12-AA. **B2 = RAMP + worst-case SupCon** (γ0.2, ℓ∞ adv) training local (5070 Ti). **Honest subtraction = B1−R′ / B2−R′, NOT B1−R:** R′ = matched control via SAME patched `RAMP_claimB.py --claimB none` (same lbd5/seed0/at_iter10/80ep/static-lr + same worst-union val_best), differing from B1/B2 *only* by the rep term (R `armA_rampfull` used a different script/val-selection/machine → confounded). R′ training Colab-C. **Flow:** audit B1 (Colab-A, 1k→10k) + `collapse_dump.py` (backbone pooled-512: alignment/uniformity/embed_norm, head discarded) → on R′ done `paired('B1','Rprime')` + `paired('B2','Rprime')`. Task I "no R′ needed" SUPERSEDED for Claim B.

**▶ SEED-2 REPLICATE RUNNING (priority #1, director 2026-07-16 08:45).** **M1a seed-2 → M0 seed-2** chain on 5070 Ti (paired, same recipe/base=msd, `--seed 2`, W&B offline; resume-safe; ~13h; audit before 18/7) → 2nd matched M1a−M0 delta to harden Claim A past the seed-0 point. **B2 RE-PRIORITIZED to extension material:** trained (val_best ep78, proxy 0.615) but local 10k audit STOPPED to free the GPU; B2 audit moves to Colab (`run_audit(B2,'ramp','B2')` after R′) → **B2−R′ paired**. **Proxy inflation noted across ramp arms (B1 0.592 / R′ 0.601 / B2 0.615 = APGD-20 val-selection ≫ true 12-AA) — only the frozen 12-AA audit is read.**

**━━━ SYNC 2026-07-20 ~10:25 ICT — seed-1 M0 arm done, M1a training ━━━**

**✅ M0_full_seed1 COMPLETE** — 50/50 ep, best valWU **0.4920** (seed-0 M0_full was 0.481 → consistent). Config re-verified from train.json: msd_steps=50, lr_peak=0.1, ε∞=0.03137 (8/255) — matches seed-0 exactly.
**🔄 M1a_full_seed1 ep11/50** (911 s/ep; lr ramping to peak 0.1 @ep20) → ~9.9 h left → ~20:15 ICT. Then seed-1 no-Sq@1k gate → 12-AA@10k iff Δ<0.
**▷ Eval queue armed behind it** (survived session teardown, nohup): v50 staleness + m1a_max, no-Sq@1k → paired. **M0 @1k masks + eval.json now local** (union 0.3920, same subset) → `paired(M1a_max, M0)` gets a real LCB. ⚠ that pair is BASE-CONFOUNDED (term+MAX vs no-term+MSD); the clean #11 is `M1a_max − M0_max` (M0_max not trained).
**Note:** both background chains (`seed1_chain.sh`, `eval_queue.sh`) survived the session teardown; the *monitoring waiters* were orphaned and have been re-armed.

**━━━ SYNC 2026-07-20 ~01:20 ICT — ablations #11–14 built (no new results) ━━━**

**🔄 seed-1 still training** M0_full_seed1 **ep17/50** (~700 s/ep; train.json is a per-epoch snapshot, NOT a done-sentinel — done at epochs_completed=50). → M1a_full_seed1 → gate. No result landed since 00:50.

**✅ BUILD — base-generality + redundancy ablations #11–14 (c5_fromscratch, gated after seed-1/v50).** New `--base max`(=worst-of-3 APGD)/`avg`(=mean-of-3) + `--glue-view linf`; loop refactored to separate CE-views (`xs_ce`) from glue-views (`xs_adv`) → handles #13 (base=max + msd-glue). **Byte-identical for existing arms** (RNG order MSD→APGD preserved; verified 6 smoke paths clean, adv/step matches table 3/3/3/3/4/2). Arms: M0_max/M1a_max (#11), M0_avg/M1a_avg (#12), M1a_max_msdglue (#13), M1a_linfglue (#14). Zip `6a991d47`. Runners: `scripts/dev/run_ablations_11_14.sh` (local, gated) + train_colab section (6-arm loop, resume-safe) + eval_colab section (no-Sq@1k + paired + per-norm ℓ₁). Pre-reg: #11/#12 signif+ℓ₁ → base-generality; #13/#14 ≈ msdglue → single-view culprit else redundancy.

**━━━ SYNC 2026-07-20 ~00:50 ICT — seed-3 lands + FAT-CLAMP/logging builds ━━━**

**✅ CLAIM A #3 (seed-3) DONE — SIGNIF.** M1a_seed3 **0.4227** − M0_seed3 **0.3970** = **Δ +0.0257, LCB95 +0.0212**. 3-seed: seed0 +0.0287 / seed2 +0.0252 / seed3 +0.0257; **seed-min LCB +0.0202 > 0** → from-scratch pull-push beats matched control **robustly across 3 seeds**.

**🔄 Reversal seed-1 pair RUNNING.** M0_full_seed1 ep15/50 (680s/ep) → then M1a_full_seed1 → no-Sq@1k gate → 12-AA@10k iff Δ<0. VERIFY passed (== seed-0 except seed+term). Waiter on the @1k gate.

**✅ FAT-CLAMP fully built (finetune_msd_clamp, extension, gated on v50).** `fat_view_attack` (per-sample freeze @ margin≤κ, cap K_max) + `--view-attack fat/--fat-kappa/--fat-max-steps` + regime-b `--base-ckpt` (M0_full robustdro). Resume-safe (ckpt_latest/epoch, restore model/opt/sched/rng, per-epoch loader re-seed) + **W&B online** + standard train-logging + FAT logging. Zip `aded192b`. **Live Colab read (M1_fatclamp): valWU DECLINING (0.442→0.415), nf↑ (0.64), steps_fooled ~1-2 → friendly FAT views too weak → likely NO gain (report pre-registered).** `mean_steps_fooled` validates ℓ₁≫ℓ∞ (l1 2.6 > linf 1.4 > l2 0.7).

**✅ Part A — standardized per-norm TRAIN logging + git_commit.** `pernorm_train_metrics` helper (RNG-isolated, `fork_rng` verified: M0-gen leaves RNG unchanged → 0.3886 preserved; invariants asserted). WIRED into `c5_fromscratch` + `finetune_msd_clamp` (`train/acc_{clean,linf,l2,l1}, loss_*, worst_union`; `--no-log-train-pernorm`). `git_commit()` Colab fallback (`.git_commit` baked in zip). **DEFERRED (protects in-flight seed-1/v50):** train_full_msd + RAMP wiring. `backfill_logged.py` GATED (waits seed-3+seed-1+v50 + `--go`; reads orig config from disk; paper numbers inviolable).

**▷ RESEARCH DIRECTION (discussion, no run).** RAMP read from code: base = **APGD-10 on source+target norm PAIR** (not MSD, not worst-of-3), CE on worst-of-pair; edge = **cross-norm KL transfer (weak←strong) + per-epoch clean-model WEIGHT FUSION** (`gp()` = fusion, not surgery). Wall-clock: RAMP 268 s/ep (2-3 adv/step) < M1a 365 (4). Proposed **Hy A** (differentiate from RAMP): exploit OUR reversal — representation regularization is **norm-dependent** (helps sparse ℓ₁, harms dense ℓ∞); norm-adaptive constraint (`--glue-weighting bindworst` half-built). Phenomenon-paper, not method-clone.

**━━━ SYNC 2026-07-19 ~19:15 ICT — replicates + audits + de-risk builds ━━━**

**▶ CLAIM A replicate #3 (seed-3) 12-AA@10k.** M1a_seed3 union **0.4227** DONE; M0_seed3 12-AA@10k RUNNING (started 11:28 UTC) → `paired(M1a_seed3,M0_seed3)` imminent (3rd matched Δ for Claim A). Chained after it: reversal seed-1 pair + (below) C100@10k.

**✅ CONFIG AUDIT — full-budget recipe CONFIRMED from ckpt/train.json (resolves §4.4).** ε∞ = **8/255 (0.03137), NOT 0.03** (train.json `upstream_linf_not_used:0.03` + "changed 0.03→8/255"); base = **msd_v0 50-step (NOT 40/50/50)**; **50 epochs**; **one-cycle np.interp peak 0.1** (not constant@70); term views = **3×APGD 10-step**; ε₂/ε₁ = 0.5/12. Source: `C5_full/{M0_full,M1a_full}/val_best['cfg']` + `M0_full/train.json`. ⚠ M1a_full's `msd_steps`/`n_iter` NOT in local artifacts (Colab-trained, no local train.json) → verified-identical from M1a_full's OWN ckpt = eps/epochs/lr-peak; **Kiet to confirm msd_steps=50 via W&B** to fully close the confound.

**⚠ WALL-CLOCK finding — RAMP trains CHEAPER than M1a.** R (`armA_rampfull`, 80ep, 5070Ti) **268 s/ep, 5.96h**, **3 adv/step** (3×APGD-10); M1a (from-scratch seed0, 80ep) **365 s/ep, 8.11h**, **4 adv/step** (1 MSD-10 + 3 APGD-10). Source: `ramp_armA_full_5070ti.log` + `C5/M1a_advneg/seed0/train.json`. → **drop any "CLAMP cheaper than RAMP (training cost)" claim** — M1a is ~1.36× slower (extra MSD base adversarial). (R′ not trained; R is the cost proxy, same recipe.) Matched-control gain (M1a−M0) claim unaffected — different axis.

**✅ Paper fillables filled (CPU, read-only):** §5.3 ablation `M1a_msdglue−M0 +0.0126 LCB+0.0075` / `M1a−M1a_msdglue +0.0161 LCB+0.0111` (both SIGNIF → 3-view necessary); Table-3 **ft row** per-norm Δℓ∞ +0.74 / Δℓ₂ +0.94 / Δℓ₁ +1.09 (cross-check OK); B1−R′ CI [−0.0050,+0.0038]; Δclean full-budget +0.0008; 359-verify (both counts 359 REAL, sets differ overlap 317); proxy=APGD-CE 3-norm 20-iter worst-union; M0_full 680 s/ep ×50 = 9.44h.

**▷ DE-RISK builds (RQ2 reversal).** (a) **Reversal seed-1 pair** — `train_full_msd.py --variant {M0,M1a} --seed 1`; VERIFY gate PASSED (== seed-0 except seed+term); chained (wait seed-3 → train M0→M1a → no-Sq@1k → gate → 12-AA@10k iff Δ<0). (b) **Staleness test M1a_full_v50** — new flag `--apgd-view-steps` (default→n_iter=10, byte-identical; controls 3 glue views only, NOT base msd_steps); VERIFY == seed-0 except views 10→50; run-name gets `_v{steps}` suffix so it does NOT collide with the canonical seed-0 W&B run; trains on Colab (Kiet), eval local vs M0_full after. Code zip sha **`8620970e`**.

**━━━ SYNC 2026-07-19 ~11:30 ICT — extensions + full-budget close ━━━**

**✅ FULL-BUDGET paired DONE (the honest denominator).** M1a_full−M0_full 10k union **−0.0096, 95% [−0.0142, −0.0048]** → M0_full SIGNIF > M1a_full (CLAMP slightly HARMS at full budget; does NOT hold the pre-reg band). Per-norm Δ: ℓ∞ −0.0077, ℓ2 +0.0075, ℓ1 +0.0008 (ℓ∞-driven). **Δclean full-budget +0.0008 (≈0, no clean cost)** — M1a_full 0.8063 / M0_full 0.8055. Selection-robustness @1k: last-epoch & val-best both ~0 ns (1k underpowered vs the −0.96pp effect; headline stays val-best). Masks `union_bench/{M1a_full,M0_full}/{1k,10k,last}/`.

**✅ CIFAR-100 matched pair (NEW dataset, local 5070Ti).** `c5_fromscratch --dataset cifar100` (no fork; num_classes 100, val_select 2k, recipe == C10 headline). Harness C100 support: `load_subset`+`validate_config` dataset branch (backward-compat) + `configs/eval/audit_cifar100_*_{test1k,test10k}.yaml`. M1a_c100 valWU **0.2310** / M0_c100 **0.2185**. **Probe apgd_ce_linf @1k** Δℓ∞ +0.0040 ns, clean −0.013. **no-Square@1k union paired** Δ **+0.0070 ns@1k**, ℓ1-driven **+0.0080** (thesis-consistent), clean −0.013. Three sources agree in sign (proxy +1.25 / probe +0.40 / union +0.70). 12-AA@10k for power = candidate (awaiting go).

**✅ FT-∞ (RobustBench Sehwag2021Proxy_R18 base, extension).** B2 reproduce gate PASSED (clean@10k 0.8459 exact). B3 full fine-tune (lr0.005/warmup2/15ep) on Colab. Local 12-AA@10k **paired(ft_clamp−ft_none) = +0.0035, 95%[−0.0012,+0.0083] ns** (ft_clamp 0.4026 / ft_none 0.3991), ℓ1-driven Δℓ1 +0.0109, clean −0.012 — CLAMP adds a small ℓ1-routed gain on a strong ℓ∞-AT base but ns@10k (pre-registered, report regardless). robustbench installed local (`--no-deps`, torch untouched).

**✅ H3 ABLATION (single-MSD-view glue) paired @10k.** M1a_msdglue−M0 **+0.0126 LCB +0.0075 SIGNIF**; M1a−M1a_msdglue **+0.0161 LCB +0.0111 SIGNIF** → **3-view multi-norm structure is necessary** (§3.3). Ordering M1a 0.4173 > msdglue 0.4012 > M0 0.3886.

**✅ Paper fillable numbers (CPU).** §5.3 ablation (above); **B1−R′** 10k Δ−0.0005 CI **[−0.0050,+0.0038]** width 0.88pp (includes 0 — replace prose); **359-verify (M9): count match REAL** (union-only-M0 & ℓ∞-only-M0 both 359) BUT **different sets** (overlap 317/359 — do NOT claim "same examples"); **proxy (App A) = APGD-CE, 3-norm, 20-iter, single-restart, worst-union** on `train[49000:50000]`; **M0_full wall-clock 680 s/ep × 50 = 9.44h** (5070Ti).

**▷ QUEUED/CHAINED.** seed-3 12-AA@10k (ckpts local, ep80 done) auto after FT-∞ → paired(M1a_seed3,M0_seed3). Phased pull-push ×2 (push_then_pull@40 ± grad-surgery) — gated no-Square@1k section wired in `eval_colab` (Colab). Tooling: `union_bench_eval` +`robustdro`/`--skip-square` builder (bit-identical to frozen harness, verified max|Δ|=0); code zip sha `ad84a561`.

**━━━ SYNC 2026-07-18 10:30 ICT — current CLAMP-arc status ━━━**

**✅ CLAIM A (from-scratch pull-push beats matched pure-MSD control) — holds across seeds.** seed-0 M1a−M0 10k **+0.0287 LCB +0.0234**; seed-2 10k **+0.0252 LCB +0.0202**; **seed-min LCB @10k = +0.0202 > 0**. seed-3 pair TRAINED (proxies M1a 0.461 / M0 0.433) → 10k audit pending on Colab (REP_SEED=3). Per-norm/concordance: **gain routes through ℓ₁** (Δℓ₁ +0.0481 vs Δℓ∞ +0.0229 / Δℓ₂ +0.0015; per-class sign test ℓ₁ 9+/1− p=0.0215; bottom-3 lift +0.048). No masking (black_box_gap<0 all norms), proxy consistent, alignment 0.785 (no collapse).

**▶ FULL-BUDGET arm (Maini MSD recipe: 50ep, np.interp one-cycle peak 0.1, msd_v0 50-step; ONE triple 8/255 everywhere; term == M1a).** **M1a_full** trained on Colab (proxy 0.477) → 12-AA **10k union 0.4201, apgd_ce_linf 0.4608**. **M0_full** (matched control, no term) trained local (50/50, proxy 0.481) → **1k union 0.4350**, **10k audit RUNNING** (authoritative, ~13:00 ICT). Then **paired(M1a_full, M0_full)** = the honest full-budget denominator (pre-registered, reported regardless). `scripts/dev/train_full_msd.py` (imports M1a term verbatim; atomic-save resume verified 9/9; W&B).

**▶ CLAIM B (RAMP + rep term).** B1 real 12-AA **10k 0.4456, no collapse** (unif −3.16). B2 (SupCon) trained. **Subtraction = B1−R′ / B2−R′** (R′ = matched `--claimB none`). Awaiting R′ audit on Colab.

**⏸ Baselines under our 12-AA:** msd @1k 0.442 (10k pending Kiet, driver now Colab-portable, ckpt sha `482bf287…` = public locuslab/robust_union MSD). MAX/AVG @10k deferred.

**✅ REVIEWER ARTIFACT repo (`../pullpush-artifact`, method renamed CLAMP = CLean-Anchored Multi-norm Pull–push).** 1 commit, Anonymous, no remote; **full verification gate PASSED** — every paper number reproduces from shipped masks on CPU (<1min: union table, M1a−M0/M1b−M0/seed-2 paired, per-norm concordance, per-class, per-component) + `train.py` 1-epoch smoke PASS. Push-ready; awaiting anon GitHub account + 4open.science link (≥ Oct 2026). Dev README carries the CLAMP official-name line.

**✅ W&B:** 5 main CLAMP runs synced online → project `attackdro-union` (M1a/M0 seed-2, seed-3, M0_full). ⚠ entity names institution — keep W&B links OUT of the anonymous artifact/paper.

**▷ EXPLORES (Track B, background, do NOT slow Track A; pre-registered, extension only).** #1 fine-tune matched pair from MSD.pt (CLAMP term vs none, union+clean) — `finetune_msd_clamp.py` built, waiter ARMED (launches after M0_full audit). #2 hybrid CLAMP+GP — spec `docs/drafts/gp_clamp_spec.md`, **NOT launched**; RAMP's `gp()` is weight-space fusion, not the described gradient-surgery — awaiting design decision (project g_at off g_pp? fine-tune vs scratch? per-tensor vs global).

**⚙ Notebooks pushed (branch feat/programA-g2-impl):** `audit_gate_colab` (merge→PHASE0 extraction→gated queue + fast-review PROBE + PROBE_LAST diagnostic + msd baseline), `train_full_colab` (resume-safe, W&B), `perclass_colab`. Code zip sha `679299d6…` (40 files). `scripts/dev/{probe_attack,collapse_dump,perclass_breakdown,export_sanitize,finetune_msd_clamp}.py` in-repo.

**POINTERS**
Detail/history → `STATE_LOG.md` · process → `docs/governance/` · runs → `results/MANIFEST.md` · Paper-C worklog → `docs/BUILDER_LOG.md`
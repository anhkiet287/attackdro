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

**NEXT_DECISION_GATE — G2 implementation packet (14 conditions).**
Build + verify before launch: cal + 8k test_final split & frozen hashed index · ep020/040/060/080 checkpointing · new class-balanced audit subset from test_final (hashed) · locked calibration run (→admissible eps or STOP) · 3 concrete configs at calibrated eps · finite-run compute preflight τ_C=0 · audit dry-run frozen+recorded · provenance commit/diff · gatekeeper no-leak · **then director launch authorization** for the 3 arms. Critic G2 reviews this evidence packet. No calibration/training authorized yet.

**POINTERS**
Detail/history → `STATE_LOG.md` · process → `docs/governance/` · runs → `results/MANIFEST.md`

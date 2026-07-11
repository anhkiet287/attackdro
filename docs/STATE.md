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
Allocation study done (B1–B3 + B4 branch diagnostic complete). Deciding B4's role and starting Claim 2. `[CONFIRM: branch diagnostic outcome not yet folded into state]`

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

**NEGATIVE / GUARDS**
- FAB-T non-deterministic (restart RNG not seed-controllable) → old Tier-1 JSONs canonical; future audits export masks at first run.
- Bootstrap CIs exclude zero (mild overestimation) but FAB-T restart variance not captured → acknowledge in claims.

**OPEN_QUESTIONS**
- B4 role = evidence (allocation trace shows intelligent allocation still dumps ℓ∞), **not** competition. `[CONFIRM criteria B4-vs-B2-vs-B3]`
- Pull-push loss design (Claim 2).

**ACTIVE_TASKS**
- Phase 1 repo restructure (this workstream). `[in progress]`
- Fold B4 branch diagnostic result into state. `[CONFIRM]`

**NEXT_DECISION_GATE**
G1 — B4-vs-B2-vs-B3 pre-registration → requires ATLAS-CRITIC Pass before any B4 lock. Currently **blocked pending critic pass / documented protocol decision.**

**POINTERS**
Detail/history → `STATE_LOG.md` · process → `docs/governance/` · runs → `results/MANIFEST.md`
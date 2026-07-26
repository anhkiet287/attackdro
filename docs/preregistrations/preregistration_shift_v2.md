# PRE-REGISTRATION v2 — Adaptive bottleneck discovery (shift + cross-dataset)

Supersedes v1 (BLOCKER at ATLAS-CRITIC G1/G2, 2026-07-11 — all 11 items + citation audit ACCEPTED).
Status: **DRAFT for re-submission to ATLAS-CRITIC (G1) + G2. No training run before Pass + director decision gate.**
`[extract-and-lock]` = exact value to be pulled from repo config/probe and frozen before submission (not fabricated).

---

## 0. What changed from v1
Three estimands separated; executable calibration rule + frozen split map; quantitative P1/P2/P3 with formal tests; "oracle" renamed **B3-bottleneck-informed** with exact schedules; per-arm compute/information accounting; Program B fully specified with a prior-independent bottleneck-label procedure and no-shift/no-unique fallbacks; symmetric canonical audit; outcome→conclusion interpretation matrix; citations corrected (removed the L2/Prop-3.1 overclaim, narrowed triple + novelty).

## 1. Claim & three estimands (item 1)
Contribution = oracle removal + robustness to bottleneck mis-identification (not union SOTA). Claimed only via three **separately measured** estimands:

- **E_id — identification:** agreement between B4's internal predicted bottleneck and an **independent** bottleneck label (per-norm robustness measured on the `cal` split, §9), at checkpoints {ep20, ep40, ep60, ep80}. Metric: fraction of checkpoints with `argmin(predicted)==argmin(independent)`.
- **E_alloc — allocation response:** change in B4's per-norm selected-source fraction R1→R2 (event defined in §P2).
- **E_out — outcome:** union differences vs B3-mis (superiority) and vs B3-bottleneck-informed (non-inferiority).

"Oracle removal" is claimable **only if** E_id and E_alloc hold **and** B4's policy never consumed the evaluation/`cal` labels to update itself.

## 2. Executable calibration rule for R2 L1 eps (item 2)
- **Source model:** B3 R1 `val_best.pt` (epoch 72), fixed.
- **Split:** `cal` = **1k held out from the test set** (§8), disjoint from the reported `test_final` (9k) and from `val_select`; never trained on, never used for selection or reporting. Fixed indices (seed 0). (Chosen over a train-carved cal because the R1 reference model already trained on all train examples.)
- **Attack:** APGD, 20/20/100, 1 restart, seed 0, fixed batch size, ℓ∞=8/255 & L2=0.5 fixed.
- **Grid (pre-declared, full):** L1 eps ∈ {16, 20, 24, 28, 32, 40}.
- **Admissible eps:** L1 is unique min iff `a_1 ≤ a_∞ − m` AND `a_1 ≤ a_2 − m`, with margin **m = 0.03**; and non-collapse `a_1 ≥ 0.10`.
- **Selection:** among admissible eps, pick the **smallest** (least deviation from standard). Tie-break: smallest eps.
- **No-admissible outcome:** if no eps in the full grid is admissible → **Program A eps-shift is infeasible; STOP and report; do not extend the grid post-hoc.**
- **Separation:** `cal` is used ONLY here + for bottleneck labeling (§6); never for `val_best` selection or decision eval.
- **Feasibility (val_select probe, NOT the locked value):** on B3 R1 val_best, argmin flips to L1 at eps≥16; admissible band {16,20,24,28} (eps28 L1 racc 0.109 near the 0.10 floor; eps≥32 collapses). **Smallest admissible = 16** (L1 racc 0.375, margin 0.076). Sources: `results/calibration_probe_l1_b3_v1/l1_eps{16,20,24,28,32,40}.json`.
- **LOCK step (before launch):** re-run this exact rule on the `cal` test-subset (not val_select) to fix the eps; feasibility strongly predicts **eps=16**, to be confirmed leakage-safe.

## 3. Program A — arms & exact schedules (item 6)
CIFAR-10, PreActResNet-18. Regime R2 = (ℓ∞ 8/255, L2 0.5, L1 = calibrated). Seed **s0 (LOCKED)**.

| Arm | Policy | Information received |
|---|---|---|
| **B4-adaptive** | predictive allocation (its own signal) | none (no bottleneck prior) |
| **B3-mis** | R1 static allocation, unchanged: primary cycle `[ℓ∞, ℓ∞, L2, L1]` (ℓ∞ 0.50 / L2 0.25 / L1 0.25) + extra ℓ∞ every 5th batch (aggregate_loss, primary/source weight 0.5/0.5) | R1 prior (ℓ∞) — now wrong under R2 |
| **B3-bottleneck-informed-R2** | same scheme **re-pointed to L1**: primary cycle `[L1, L1, L2, ℓ∞]` (L1 0.50 / L2 0.25 / ℓ∞ 0.25) + extra **L1** every 5th batch, same aggregate_loss weights; only norm labels permuted, counts/extra-freq/aggregation preserved | only the bottleneck **identity** (L1), nothing tuned on R2 outcomes |

Not called "oracle": it is a **bottleneck-informed static comparator**, not proven optimal within a static class.

## 4. Program A — predictions & formal tests (items 3,4,5)
All union comparisons use the **paired sample-level bootstrap** on identical `test` examples via exported union-survival masks; unit = example; **B=10,000** resamples, seed 0; 95%; one-sided. Scope: **seed-s0 conditional** evidence (not method-level) unless independent seeds added later.

- **P1 (E_out superiority):** primary criterion `LCB(union(B4) − union(B3-mis)) > 0`. Secondary practical-effect: point estimate ≥ **0.02** (reported, not gating).
- **P2 (E_alloc):** event = per-batch selected attack-source norm on **non-refresh selected-source batches**; denominator = total such batches; refresh/all-source excluded (report counts). Windows: full-run + early(0–9)/mid(35–44)/late(70–79). Ties/no-unique-source: excluded, counted. Criteria: **shift** `f_L1^R2 − f_L1^R1 ≥ δ_f=0.20`; **dominance** `f_L1^R2 ≥ 0.50` AND `f_L1^R2 − max(f_∞^R2, f_2^R2) ≥ δ_d=0.10`. (Justification: R1 L1≈14%; dominance = clear majority flip, not marginal.)
- **P3 (E_out non-inferiority vs informed):** margin **δ_oracle = 0.02**; criterion `LCB(union(B4) − union(B3-informed)) > −δ_oracle`. Failure to reject a difference is NOT equivalence — this is a proper non-inferiority test.
- **E_id:** `argmin` agreement ≥ 3/4 checkpoints for a positive identification result.

## 5. Compute & information-budget accounting (item 7)
Held fixed across arms: **total attack-step budget**. FLOPs denominator (from config, `groupdro.py:419,1157,1325-1332`) = `reference_attack_step_units = n_batches × Σ(full_attack_steps)` where `full_attack_steps = [10,10,10]` (ℓ∞/L2/L1) → **n_batches × 30**; ratio = measured `total_attack_step_units / reference`. R1 measured ratio (ep79) = **0.39965** for both B3 and B4. Reported per arm (pre-registered table to fill): optimizer steps · attack calls by norm · attack iters×restarts · forward/backward counts · B4 signal/prediction/refresh overhead · measured wall-clock + hardware. Because L1 vs ℓ∞ per-step attack cost can differ, fairness is on **attack-step budget**, not raw optimizer steps. Any residual B4 overhead is reported as a **tradeoff**, not folded into adaptive benefit.

## 6. Program B — cross-dataset, fully specified (item 8)
For each D ∈ {CIFAR-100, SVHN}: PreActResNet-18 (D-appropriate head), standard preprocessing/aug, standard splits (SVHN excludes the extra unlabeled set). Threat radii adopted = (ℓ∞ 8/255, L2 0.5, L1 12) — **adopted from the CIFAR-10 convention with justification; NOT claimed as an established standard for D** (item-8 citation fix). Full training schedule = RAMP 80-epoch, APGD all norms.

**Prior-independent bottleneck label (per D):** train ONE **reference equal-budget** union model (equal allocation across the 3 norms, no bottleneck bias), eval per-norm robust acc on D's `cal` split (APGD 20/20/100, n=1000, seed 0). Label = `argmin` with the same margin m=0.03. Branches:
- **Unique bottleneck b_D:** informed arm targets b_D; mis arm targets ℓ∞.
- **b_D = ℓ∞ (no shift):** B3-mis ≡ B3-informed → D is a **generality-only** point.
- **No unique min within m (no clear bottleneck):** informed = equal-budget reference; D is **generality-only**; no adaptivity/shift claim on D.

**Arms per D:** B4-adaptive · B3-bottleneck-informed(b_D) · B3-mis(ℓ∞) [dropped if b_D=ℓ∞].
**Predictions per D:** generality `LCB(union(B4) − union(B3-informed)) > −δ_oracle`; if shift (b_D≠ℓ∞): superiority `LCB(union(B4) − union(B3-mis)) > 0` + B4 allocation tracks b_D (P2-style, thresholds as §4).

## 7. Run & compute budget
Program A: 3 runs. Program B: per D {1 reference + 1 adaptive + 1 informed + (1 mis if shift)} → 3–4 each. **Total ≈ 3 + 2×(3→4) = 9–11 runs × 80 epochs, s0.** Reuse R1 CIFAR-10 (B3, B4) as no-shift reference. No early stop of the pre-registered set on interim results. Audit only after all arms of a program complete.

## 8. Frozen split map (item 9)
From the training pool, disjoint splits, indices frozen (seed 0, explicit index files):

| Split | Indices (CIFAR-10) | Size | Role — used ONLY for |
|---|---|---|---|
| `train_core` | train 0–48 999 | 49k | training (unchanged) |
| `val_select` | train 49 000–49 999 | 1k | `val_best.pt` selection via `worst_union` |
| `cal` | **test 9 000–9 999** | 1k | eps calibration (§2) + bottleneck labeling (§6) |
| `test_monitor` | test 0–999 | 1k | interim monitoring only |
| `test_final` | **test 1 000–8 999** | 8k | decision grade + canonical audit reporting |

`cal` is carved from **test** (never trained on), disjoint from `val_select`, `test_monitor`, and `test_final`. `test_final` is tightened to exclude both `test_monitor` and `cal` (was "full 10k" — requires a small `datasets.py` change to add the `cal` split and exclude those indices; `[Builder task before launch]`). No `test_*`/`cal` split may determine checkpoint or claim wording; `cal` never used for selection or reporting.

## 9. Symmetric canonical audit (item 10)
Canonical **Tier-1 AA-component audit** on **all Program-A arms symmetrically** (B4, B3-mis, B3-informed): APGD-CE, APGD-DLR, FAB-T, Square per norm; identical steps/restarts/seeds; **masks exported at first run** (FAB-T). Union-robust = point survives the logical AND of all per-norm attack survival masks. If APGD decision-grade ordering and canonical ordering disagree, **canonical governs the claim**; both reported. Same protocol applied per-arm on Program B.

## 10. Interpretation matrix (item 11)
| calibration | E_id | P2 (alloc) | P1 (vs mis) | P3 (vs informed) | Permitted conclusion |
|---|---|---|---|---|---|
| no admissible eps | — | — | — | — | Program A infeasible; report; no A adaptivity claim |
| ok | pass | pass | pass | pass | Full Claim 1: correct identification + tracking + utility + oracle-equivalence |
| ok | pass | pass | pass | fail | Tracking + utility, but **oracle-equivalence rejected** |
| ok | pass | pass | fail | — | Tracking without utility (allocation shifts, no union benefit) |
| ok | pass | fail | pass | — | Outcome difference **not attributable** to the tracking mechanism |
| ok | fail | (any) | (any) | (any) | Allocation shift **not** attributable to correct identification; discovery claim downgraded |
| ok | — | — | — | B3-informed < B3-mis | Comparator anomaly; investigate before any claim |

Every cell pre-committed; no post-hoc reinterpretation.

## 11. Citation register (corrected — item citations)
- E-AT (Croce & Hein, ICML 2022): VERIFIED (paper). "ℓ∞ hardest on CIFAR-10" — VERIFIED **within studied setups**, not universal. "L1 hardest on ImageNet" — PARTIALLY_VERIFIED, setup-dependent → **motivation only**.
- ~~"L2 rarely binding, implied by Prop 3.1"~~ — **REMOVED** (REJECTED: Prop 3.1 yields only ~L2 0.22 < 0.5; RAMP shows a config where L2 is the bottleneck at larger ε₂). B4's R1 L2=0% is stated only as an empirical fact **at this eps**, not a general property.
- MSD (ICML 2020): triple (8/255,0.5,12) VERIFIED **for CIFAR-10 only**; adoption on C100/SVHN is our choice with justification, not "standard".
- Novelty ("C100/SVHN union AT uncovered"): **DROPPED** pending a proper literature search; not asserted as novelty.
- Internal JSONs (B3/B4 dev-eval, allocation traces, branch diag, calibration): **contextual only** until artifact-level export (configs, commands, hashes, sample IDs) is supplied for independent verification `[Codex export queued]`.

## 12. Lock status
LOCKED: δ_P1 criterion (LCB>0; 2pp practical) · seed s0 · P2 δ_f=0.20/δ_d=0.10 · P3 δ_oracle=0.02 · E_id ≥3/4 · calibration rule (m=0.03, non-collapse 0.10, smallest-admissible) · split roles+indices · symmetric audit · interpretation matrix.
FILLED (from Claude Code extraction 2026-07-11): B3 allocation vector (cycle `[ℓ∞,ℓ∞,L2,L1]` + extra-ℓ∞/5) · split indices (49k/1k/1k/8k + cal=test 9000–9999) · FLOPs denominator (n_batches×30, R1 ratio 0.39965) · feasibility band {16,20,24,28}, smallest=16.
`[before launch]`: (1) implement `cal` split + tighten `test_final` in `datasets.py` (Builder); (2) run the locked calibration on `cal` to fix L1 eps (feasibility predicts 16); (3) ATLAS-CRITIC Pass (G1) + director decision gate.
**No training before those three.**

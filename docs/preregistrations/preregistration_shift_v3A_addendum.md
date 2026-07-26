# PRE-REG v3-A ADDENDUM — Program A only (narrow fixes on v2)

Scope: **Program A (CIFAR-10 eps-shift) ONLY.** Program B is **removed from this gate** and will be pre-registered as a separate sequential contract (reference → bottleneck label → branch configs), per ATLAS-CRITIC's own structural note.
This addendum modifies only the items ATLAS-CRITIC left open on v2; all "accepted foundation" items in `preregistration_shift_v2.md` remain locked and unchanged.
`[EXTRACT]` = exact value pulled read-only from repo code/artifacts (not fabricated), then frozen.

---

## A1 — E_id operational definition (v2 §1; critic item 1) — FILLED
- **Predictor signal (logged):** `H_{k,t}` = trace field `scores.<k>` = `z(ema_loss_k) + ema_bind_k` (rule `loss_plus_bind`); `ema_loss` = EMA of refresh-batch mean per-example CE; `ema_bind` = EMA of fraction of examples where norm k has the largest CE; α=0.9, refresh interval 10, starvation floor 30 (`groupdro.py:142-152,784-805`; `.../ep000.json:8-14`). **Higher = harder.** `argmax(scores)` is the **internal predicted bottleneck**; the actual trained `selected_source` may instead be the registered **starvation-floor override**. E_id intentionally measures the **internal score prediction** (`argmax scores`), NOT the actual allocation — so the override does not affect E_id.
- **Predicted bottleneck** at epoch e: `b̂_e = argmax_k ( mean over non-refresh rows in ep{e}.json of scores[k] )` (caller-computable `E_id_nonrefresh`; on non-refresh rows post-state = pre-selection state, `groupdro.py:957-975`). Extraction script frozen with the packet.
- **Checkpoints — frozen semantics** (checkpoint named 1-indexed by completed epochs, matching existing `ckpt/ep080.pt`; trace 0-indexed per epoch, `ep000..079.json`):

  | Completed epochs | 0-based epoch idx | Checkpoint file | Trace file | E_id score source |
  |---|---|---|---|---|
  | 20 | 19 | `ckpt/ep020.pt` | `ep019.json` | mean of non-refresh `scores` in `ep019.json` |
  | 40 | 39 | `ckpt/ep040.pt` | `ep039.json` | mean of non-refresh `scores` in `ep039.json` |
  | 60 | 59 | `ckpt/ep060.pt` | `ep059.json` | mean of non-refresh `scores` in `ep059.json` |
  | 80 | 79 | `ckpt/ep080.pt` | `ep079.json` | mean of non-refresh `scores` in `ep079.json` |

  **Temporal alignment (accurate):** the predictor label for checkpoint N is an **epoch-level controller summary** = the **unweighted mean over B4's logged non-refresh `scores` rows observed throughout training epoch index N−1** (i.e. `b̂_N = argmax_k mean_{t∈epoch N−1, non-refresh} scores[k,t]`, unweighted row mean — not batch- or example-weighted). It is **compared with** the independent bottleneck of the **model saved after completing that epoch** (`b*_N = argmin_k a_k(θ_N)` on `cal`). The predictor score is therefore a summary of the training interval that produced the checkpoint; **it is NOT recomputed from the saved checkpoint**. This is a valid lagged/epoch-aggregate identification test. `[Builder: enable saving `ep020/040/060.pt` (only `ep080/last/val_best` currently saved) + confirm exact filenames.]`
- **Independent label** at e: `b*_e = argmax_k (1 − a_{k,e})` i.e. argmin per-norm robust acc on `cal`. Not unique by margin **m=0.03** → **UNRESOLVED**.
- **E_id metric:** fraction of 4 checkpoints with `b̂_e == b*_e`. **Unresolved/tied checkpoints count as non-agreement.** Positive identification ≥ **3/4**.

## A2 — Calibration attack + artifact freeze (v2 §2; critic item 2) — FILLED
- **Source model:** `results/b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k/s0/ckpt/val_best.pt`, **SHA-256 `4676e75fe06b4a5751db45c1284a7f1c68a0628c18b0748b064b89e5cdcdb769`**, selected by `val_select/worst_union`. Epoch 72 descriptive/**non-binding**; expected eps=16 **non-binding**.
- **Per-norm decision-grade attack (frozen):** `version=apgd` = **APGD-CE + APGD-t (targeted DLR, 9 target classes)**; iterations **ℓ∞ 20 / L2 20 / L1 100** each component; **1 restart**; random start (`use_rs`); **seed 0**; batch size **250**; implementation **AutoAttack 0.1, commit `a39220048b3c9f2cca9a4d3a54604793c68eca7e`** (`attacks_aa.py:29-52`, `evaluate.py:199-201`).
- **`cal` sample-index file:** no persisted calibration index exists (val_select is just "last 1000, sliced"). **Create a frozen `cal` index JSON** (test 9000–9999) mirroring the audit-subset pattern (`results/audit/subsets/*.json`), committed before the locked calibration. `[Builder]`
- Rule/grid/margins/smallest-admissible/no-admissible-STOP: unchanged from v2 §2.

## A3 — P2 primary window (v2 §4; critic item 4)
- **Primary P2 = FULL-RUN only.** P2 passes iff **both** hold on the full run: `f_L1^R2 − f_L1^R1 ≥ 0.20` **AND** `f_L1^R2 ≥ 0.50` **AND** `f_L1^R2 − max(f_∞^R2, f_2^R2) ≥ 0.10`.
- Early/mid/late windows are **secondary diagnostics only** — reported, no authority to rescue or overturn the primary decision.
- **R1 reference fraction** `f_L1^R1` = **3825 / 27504 = 0.13907** — **VERIFIED** from the 80 frozen trace artifacts (per-file SHA-256 recorded in `pre_reg_v3_A_spec_extraction_REPORT.md`; extraction: concatenate `ep000..079.json` `trace`, keep `refresh==false`, count `selected_source==l1` / total). Locked as the R1 baseline.

## A4 — Compute-budget enforcement (v2 §5; critic item 7) — FILLED
- **Attack-step-unit formulas** (`s_g` = full_attack_steps[g] = 10 for all norms; `groupdro.py:927-975,1028-1060,1157-1158`):
  - static_extra (B3-mis, B3-informed): `U_g = s_g·Σ_b(1[c(b)=g] + 1[extra fires ∧ x=g])`; extra fires when `global_batch % 5 == 0`. Per-batch total = 10 (primary) + 10 every 5th ⇒ **12 units/batch**.
  - predictive_refresh (B4): `U_g = s_g·[Σ_b 1[refresh] + Σ_b 1[¬refresh ∧ p(b)=g]]`; refresh every 10 ⇒ per 10 batches = 30 (refresh, all norms) + 9·10 ⇒ **12 units/batch**.
  - reference = `n_batches·Σ_g s_g = n_batches·30` ⇒ ratio 12/30 = 0.40 (measured 0.39965, boundary effect).
- **Exact per-batch cost (12/batch is average shorthand, not literal):** ordinary static = 10; static-extra firing batch = 20; ordinary predictive = 10; predictive refresh = 30. Finite-run totals: **`C_static = 10·B + 10·N_extra`**, **`C_B4 = 10·B + 20·N_refresh`** (B = global batches, N_extra = extra-firing batches, N_refresh = refresh batches).
- **Enforcement preflight (fail-closed):** compute the **finite-run scheduled totals directly** (above) for each of the 3 configs and assert **exact equality, τ_C = 0** — do NOT rely on 12·B. Measured wall-clock/FLOPs (0.39965 vs analytic 0.40) reported as boundary effect, not gated.
- Preflight FAILS if: scheduled steps differ · extra/refresh frequency differs from registered policy · any norm gets extra restarts · B4 issues **uncounted** attacks for its signal (its signal uses only refresh-batch attacks, already counted — verify no extra).
- Report **both** normalized attack-step units **and** measured wall-clock/FLOPs.

## A5 — Canonical audit table (v2 §9; critic item 10) — FILLED
Verified against `eval_multinorm_audit.py` + `configs/eval/audit_cifar10_preactrn18_multinorm_v1.yaml`. All components: seed **20260709**, 1 restart, AutoAttack 0.1 commit `a39220048b3c9f2cca9a4d3a54604793c68eca7e`. Order per norm: APGD-CE → APGD-DLR → FAB-T → Square.

| Norm | APGD-CE | APGD-DLR (untargeted) | FAB-T (targeted, 9 cls) | Square (margin) |
|---|---|---|---|---|
| ℓ∞ | 100 it, **required** | 100 it, **required** | 100 it, **required** | 5000 q, **required** |
| L2 | 100 it, **required** | 100 it, **required** | 100 it, **required** | 5000 q, optional |
| L1 | 100 it, **required** | 100 it, optional | 100 it, optional | 5000 q, optional |

- **No optionality (locked):** the config's `optional` flags are **superseded** — all four components (APGD-CE, APGD-DLR, FAB-T, Square) are **REQUIRED for all three norms**. **Canonical per-norm mask** `M_norm = AND(APGD-CE, APGD-DLR, FAB-T, Square)`; **canonical union** `= AND(M_ℓ∞, M_L2, M_L1)`. Any REQUIRED component error → audit **fails closed** (fix + rerun). All three arms get the identical suite.
- **Pre-results exclusion branch — mechanically frozen (not result-contingent):**
  1. **Dry-run model:** `results/b3_static_weighted_linf_.../s0/ckpt/val_best.pt`, SHA-256 `4676e75fe06b4a5751db45c1284a7f1c68a0628c18b0748b064b89e5cdcdb769` (single model).
  2. **Dry-run subset:** the newly frozen `test_final`-only audit subset (its recorded SHA-256), all its samples.
  3. **Dry-run command:** the audit harness with the frozen audit config (config SHA recorded); exact command logged.
  4. **Deterministic incompatibility (only trigger):** the *same* component/norm combination exits with the *same implementation-level unsupported-operation error* in **two identical clean runs**.
  5. **Non-qualifying failures (NO exclusion):** OOM, timeout, machine interruption, missing file, storage error, transient CUDA failure → infrastructure fix + rerun, never exclusion.
  6. **No source modification** before the rerun.
  7. **Global + pre-results:** any exclusion applies **identically to all three arms**, is **logged before any Program-A checkpoint is audited**, and **no robust-accuracy or survival-mask value from the dry run may inform inclusion**.
  On exclusion, that norm's canonical mask = AND of the remaining REQUIRED components; otherwise all four remain REQUIRED.
- **Sample set:** current audit subset (1000 class-balanced **full-test**, seed 20260709, `results/audit/subsets/cifar10_test_1000_seed20260709.json`, **SHA-256 `0ad0c27b9c91aec8f87398de6224d15431a70c96e47fe9c5e1d8e8e36f3137d3`**) **overlaps `cal`/`test_monitor`** → **superseded**. New subset **re-drawn from `test_final` only** (test 1000–8999), class-balanced, frozen, persisted, **hashed**, disjoint from `cal`/`monitor`. `[Builder: new audit subset config + record its SHA-256]`
- **Masks:** `True`=robust; per-norm mask = AND over that norm's components; union = AND over all per-norm masks (`:391-397,868-911`). Each component starts from the full 1000 and attacks only clean-correct survivors within its own AutoAttack call; components do **not** chain survivors.
- **FAB-T non-determinism:** seeded (CPU+CUDA) but no retry/tolerance → per STATE, first-run masks exported and treated as canonical; audit records repo commit at execution.
- Canonical governs over decision-grade on disagreement (v2 §9).

## A6 — cal role wording (v2 §8; critic item 9)
Replace the contradictory rule with: **`cal` may determine ONLY the preregistered eps, the bottleneck labels, and the E_id outcome. It may NOT determine checkpoints, thresholds, unregistered arm choices, reported final accuracies, or post-hoc claim definitions.** `cal` is a **permanently held-out development/calibration subset, disjoint from final reporting** (do not call it "leakage-safe" unqualified). **Final robustness reported on `test_final` = 8k for ALL arms**; the full official 10k test is never reported after 1k is used for calibration.

## A7 — Interpretation matrix corrections (v2 §10; critic item 11)
- Replace every "oracle-equivalence" → **"non-inferior to the preregistered bottleneck-informed comparator"** (also in v2 §4/§10 and the packet).
- **P2 = single pass/fail** = (shift AND dominance) on the primary full-run window.
- Add E_id **unresolved** handling: any checkpoint with no unique independent bottleneck (< m) or a prediction tie counts as **E_id failure** toward the 3/4 criterion.

## Verification bundle for the critic
Supply with re-submission: **`docs/drafts/pre_reg_v3_A_spec_extraction_REPORT.md`** (Codex, read-only) — contains checkpoint path + full SHA-256, per-file trace hashes + recompute of f_L1^R1, predictor `scores` formula w/ `groupdro.py` lines, attack-step formulas, decision-grade attack config, and the canonical-audit code/config lines. This is the compact bundle the critic needs to mark the repo-derived facts VERIFIED (no whole-repo dump required).

## Lock status
FIXED (final round): (1) **E_id temporal alignment corrected** — epoch-level controller summary (unweighted non-refresh row mean over epoch N−1) vs model-after-epoch-N checkpoint; not recomputed from the checkpoint; (2) **audit exclusion branch mechanically frozen** — dry-run model+SHA, subset, command, 2-run determinism, qualifying vs non-qualifying failure classes, no source-mod, global + pre-results, no result inspection.
ACTION REQUIRED (director): **paste the FULL CONTENT** of `docs/drafts/pre_reg_v3_A_spec_extraction_REPORT.md` to the critic — a path reference is not verification.
FIXED (prior round): checkpoint-semantics table; audit optionality removed (all-4 REQUIRED, named mask); full subset SHA.
LOCKED + FILLED (prior): Program-A-only scope · E_id signal + aggregate + unresolved rule · checkpoint SHA `4676e75f…` · decision-grade attack (APGD-CE+APGD-t, 20/20/100, seed0, bs250, AA 0.1 `a392200`) · P2 primary=full-run · f_L1^R1=0.13907 · attack-step formulas + τ_C=0 scheduled · cal wording + 8k final · matrix fixes.
`[implement before launch, after G1 Pass]` — all Builder tasks: (1) `cal` split + frozen `cal` index (test 9000–9999) + `test_final`=8k in `datasets.py`; (2) ep20/40/60/80 checkpointing on the R2 predictive run; (3) re-drawn canonical audit subset from `test_final`; (4) compute-enforcement preflight; (5) locked calibration on `cal`.
**Program B:** deferred to a separate pre-registration. **No training/calibration before critic Pass on v3-A + director decision gate.**

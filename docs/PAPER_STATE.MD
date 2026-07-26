# PAPER_STATE — CLAMP / AISec '26 — SINGLE SOURCE OF TRUTH
Updated 2026-07-22 (after M1a_avg @10k + F2/v50 @10k landed). Supersedes PROJECT_SYNC.md.
Merges the syncs of CLAUDEUS · GPT · REVIEWER-2. One file all agents read; Kiet locks it, then Kiet writes prose.

**Truth hierarchy:** eval.json/masks (artifact) > IMPLEMENTATION_DOSSIER.md + claim_evidence_ledger.md (operational) > this snapshot.
If this doc conflicts with an artifact → artifact wins, fix the doc.
**Status labels:** 🔒 LOCKED · ✅ EVIDENCE (number in, provenance clean) · ⏳ PENDING · ❌ DELETE (never revive) · ◻️ DECIDE (Kiet).
**Process rule (Jul 22):** Kiet writes 100% of prose. Agents = evidence / mechanical edits / checks / audits. No agent prose. RQ3 approved text is inviolable (append-only, thầy sign-off).

---

## A. VERIFICATION NOTES (this pass — cross-checked vs TONY artifacts)
- 2×2 arithmetic ✅: MSD +0.0287, AVG +0.0203, MAX −0.0224 all reproduce from off/on unions.
- AVG route ✅: Δℓ∞ +0.0211 > Δℓ₁ +0.0138 → via ℓ∞ (predicted; sign held @1k→@10k).
- F2 ✅: v50−MSD50 = −0.0030 ns; v50−views10 = +0.0066 sig; recovery all ℓ₁ (+0.0232), Δℓ∞ stays −0.0060.
- Δclean AVG ✅: 0.8417→0.8331 = −0.0086.
- Cross-verify (TONY): M1a_avg + v50 both 12/12 attacks, union(eval)=union(masks), ckpt sha matched. Clean.
- CONFIRM 1–4 (TONY, re-computed from masks, not re-read from reports): 2×2 ✅ · MAX per-norm ✅ · F2 ✅ · Δclean ✅. All PASS.
- 🔒 **REPO SEPARATION (locked fact).** PAPER artefacts (`main.tex`, `figs/*.pdf`, `docs/*.md` incl. `claim_evidence_ledger.md`) live in the **paper1 workspace**. CODE/RESULTS artefacts (`eval.json`, `masks_multinorm_v1.npz`, `train.json`) live in **ATTACKDRO**. Neither tree is "missing" the other's files. This is why TONY's G5 returned FAIL — no `.tex` and no fig PDFs exist in ATTACKDRO by design. **G5 is CLOSED by CLAUDEUS** (fig PDFs present; CFP = 10p + 2p).
- Cosmetic FIXED (TONY): ROUTE label no longer asserts whose prediction a row matches; it now names the larger channel only. Numbers were always correct.
- ⚠️ Discrepancy to resolve: TITLE (see ◻️D7).
- Fig count RECONCILED (CLAUDEUS): 5 PDFs exist, 4 `\include`d, `union_failure_norms.pdf` built but commented out pending decision. ⚠ cleared.
- 🔒 **L13 in force**: "ns" is never written as neutral / no effect / unchanged — see §B L13. Affects RAMP, FT, F2-matched-views, clean-negatives rows in §D.

---

## B. 🔒 LOCKED DECISIONS
| # | Decision |
|---|---|
| L1 | Identity: a controlled empirical study of a representation **intervention** for multi-norm union robustness. CLAMP = vehicle. NOT SOTA. |
| L2 | RQ arc: RQ1 help / RQ2 when / RQ3 why / RQ4 reveal. |
| L3 | Sweep rule: "geometry"/"alignment" appear ONLY as hypothesis-under-test, never as established cause. |
| L4 | Lexicon: "the evidence does not support" (never "falsify"); "demonstrates/consistent with" (never "proof") for observational. |
| L5 | Reversal demoted: abstract/contribs/conclusion say "does not persist at full budget"; the number lives in RQ2 with confound + seed-count adjacent. |
| L6 | Framing: ℓ₁ = largest per-norm effect; **union is bought by ℓ∞** (binding). Never "gain routes through ℓ₁". |
| L7 | Scope rider on every anatomy/union-structure claim: "for these two models, at this ε-triple" — incl. abstract. |
| L8 | Tier discipline: @1k / screening NEVER cited for ordering or effect size. |
| L9 | One-harness rule; public checkpoints = context only, labeled "@1k, CI≈±3pp, budget-mismatched". |
| L10 | RQ3 prose = approved verbatim; append-only after Interpretation; thầy sign-off. |
| L11 | Absolute union = upper bound (APGD-DLR untargeted); matched-pair deltas unaffected. |
| L12 | No new runs except F1 (optional, cutoff). M5 = camera-ready. *(stale "conditional B2" clause dropped — B2 was never the RAMP arm.)* |
| L13b | **L13 extends to per-norm deltas.** Δℓ∞/Δℓ₂/Δℓ₁ in `tab:bases` are point estimates with no CI in the .tex. (i) INTERIM RIDER: per-norm smallness must be **magnitude-only** ("moves by less than one point"), never a null claim ("unchanged", "no effect on ℓ₂"). (ii) CIs computed — see §D companion table; drop the rider only where a CI is cited. |
| L13 | **ns ≠ no effect.** Write "not distinguishable from zero at this power" with n and CI adjacent; never "neutral" / "no effect" / "unchanged". Applies to RAMP, FT, F2-matched-views, clean-negatives. |

---

## C. SKELETON (structure frozen)
Intro → Related (short) → Method → Experiments [Setup×4 → RQ1(+per-norm) → RQ2 → RQ3 → RQ4 → Ablation → Sanity] → Discussion (3 beats) → Limitations → Conclusion → Ethics → Refs → App A (training) / App B (per-class) / App C (harness) → GenAI decl.

**Write order (Kiet writes; R2+GPT agreed):**
1 Method → 2 Setup → 3 RQ1+per-norm → 4 RQ4 → 5 Ablation → 6 Sanity → 7 Related → 8 Limitations(indep)
→ **[both numbers now IN]** 9 Intro framing → 10 Contributions → 11 **RQ2** → 12 Discussion → 13 Limitations final → 14 Conclusion → 15 Abstract (last).
Only F1 (optional) gates the abstract's mechanism claim-level.

---

## D. EVIDENCE LEDGER (numbers final unless marked)

### RQ1 — headline pair (CLAMP vs MSD-10, CIFAR-10, PARN-18) ✅
Union @10k 3-seed: **0.420 vs 0.393 = +2.65 ±0.19**, seed-min LCB **+2.02**. Clean 0.815 vs 0.836 (−2.1).
Per-seed: s0 +2.87 (LCB+2.34) · s2 +2.52 (+2.02) · s3 +2.57 (+2.12). Bootstrap B=10k on union masks.
Per-norm 3-seed: ℓ∞ +2.1 · ℓ₂ +0.7 · ℓ₁ +4.6 (per-seed ℓ₁ +4.8/+4.3/+4.7).
Decomposition: +2.65 = ℓ∞ ceiling (+2.1) + reduced non-ℓ∞ leakage among ℓ∞-pass (+0.6); ℓ∞→union gap MSD 1.5pp→CLAMP 0.9pp.
Union-flip (s0, App C): 646 fixes (40.9% ℓ∞, 41.2% ℓ∞+ℓ₁, 14.9% ℓ₁-only, 3.0% ℓ₂); 359 new fails (48.8% ℓ∞).
Per-class (s0, App B, exploratory B=2k): ℓ₁+ 9/10 (p=0.02); union ns (7+/3−); hard classes cat/deer/bird +4.8; dog −3.1, horse −1.8.

### RQ2 — base-generality 2×2 @10k ✅ **COMPLETE (both gating numbers in)**
| Base | off | on | Δunion | 95% CI | LCB | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ | Δclean | route | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSD-10 | 0.3886 | 0.4173 | **+0.0287** | [+.0234,+.0339] | +.0234 | + | +.0229 | +.0015 | +.0481 | −.0215 | **ℓ₁** | 3 |
| AVG | 0.3846 | 0.4049 | **+0.0203** | [+.0155,+.0250] | +.0155 | + | +.0211 | +.0031 | +.0138 | −.0086 | **ℓ∞** | 1 |
| MAX | 0.4349 | 0.4125 | **−0.0224** | [−.0274,−.0175] | −.0274 | − | −.0204 | +.0112 | +.0039 | +.0035 | ℓ∞ | 1 |
| MSD-50 (views-10) | 0.4297 | 0.4201 | **−0.0096** | [−.0142,−.0048] | −.0142 | − | −.0077 | +.0075 | +.0008 | +.0008 | ℓ∞ | 2 (s1 −.0059) |
| FT-ℓ∞ | 0.3991 | 0.4026 | +0.0035 | [−.0012,+.0083] | −.0012 | ns | +.0074 | +.0094 | +.0109 | −.0120 | ℓ₁ | 1 |
| RAMP (R′→B1) | 0.4461 | 0.4456 | −0.0005 | [−.0050,+.0038] | −.0050 | ns | −.0037 | +.0008 | +.0201 | −.0108 | ℓ₁ | 1 |

**RQ2 story (writable now, calibrated):** the term **helps 2 of 3 aggregation bases** (MSD +2.87 via ℓ₁; AVG +2.03 via ℓ∞ — both sig), **reverses on the strongest aggregation base MAX** (−2.24, sig), and on two strong single-base recipes the effect is **not distinguishable from zero at this power** — RAMP −0.05 (CI [−0.50, +0.38]), FT-ℓ∞ +0.35 (CI [−0.12, +0.83]), both n=1, CI half-width ≈0.44 / 0.48 pp: **absence of a detectable effect, not evidence of none** (L13). Mechanism is base-dependent: the term supplies the norm the base under-serves (MSD lacks ℓ₁; AVG lacks ℓ∞; MAX lacks neither → only ℓ∞ interference remains). Union tracks ℓ∞ throughout.


#### Companion: per-norm paired-bootstrap CIs (L13b) — NOT for `tab:bases` in the .tex
Same machinery as the union deltas: per-example masks, B=10,000, `rng(0)`, percentile CI, LCB = q05.
Per-norm block = AND over that norm's 4 attacks at tier=full. **The .tex keeps point estimates; cite a
CI in prose only where a smallness or directional claim is made.**
Source: `scripts/dev/pernorm_ci.py` → `results/analysis/pernorm_ci/pernorm_ci.json`.

| row | Δℓ∞ [95% CI] | Δℓ₂ [95% CI] | Δℓ₁ [95% CI] |
|---|---|---|---|
| MSD-10 s0 (= RQ1 headline) | +0.0229 [+.0178,+.0278] **+** | +0.0015 [−.0037,+.0068] **ns** | +0.0480 [+.0425,+.0537] **+** |
| MSD-10 s2 | +0.0217 [+.0169,+.0266] **+** | +0.0109 [+.0056,+.0161] **+** | +0.0432 [+.0378,+.0486] **+** |
| MSD-10 s3 | +0.0177 [+.0133,+.0223] **+** | +0.0084 [+.0037,+.0131] **+** | +0.0468 [+.0416,+.0519] **+** |
| **MSD-10 aggregate** — see the two-column block below | mean **+0.0207** · seed-min LCB **+0.0133** | mean **+0.0069** · seed-min LCB **−0.0037** | mean **+0.0460** · seed-min LCB **+0.0378** |

**MSD-10 3-seed aggregate — MEAN and SEED-MIN LCB are different numbers; never mix them.**
Same convention as the union headline (mean +2.65 pp, seed-min LCB +2.02 pp).

| quantity | per-seed Δ (s0, s2, s3) | **MEAN Δ** | per-seed LCB (s0, s2, s3) | **SEED-MIN LCB** | all-seed + ? |
|---|---|---|---|---|---|
| union | +0.0287, +0.0252, +0.0257 | **+0.0265** | +0.0234, +0.0202, +0.0212 | **+0.0202** | **YES** |
| Δℓ∞ | +0.0229, +0.0217, +0.0177 | **+0.0207** | +0.0178, +0.0169, +0.0133 | **+0.0133** | **YES** |
| Δℓ₂ | +0.0015, +0.0109, +0.0084 | **+0.0069** | −0.0037, +0.0056, +0.0037 | **−0.0037** | **NO** (s0 ns) |
| Δℓ₁ | +0.0480, +0.0432, +0.0468 | **+0.0460** | +0.0425, +0.0378, +0.0416 | **+0.0378** | **YES** |

Rounded for prose: means ℓ∞ **+2.1**, ℓ₂ **+0.7**, ℓ₁ **+4.6** pp; seed-min LCBs ℓ∞ **+1.33**, ℓ₂ **−0.37**, ℓ₁ **+3.78** pp.
The ℓ₂ row is the one that breaks: **mean is positive (+0.7) while the seed-min LCB is negative (−0.37)** — a sentence quoting the mean without the LCB would imply an all-seed ℓ₂ gain that the data does not support.
| AVG | +0.0211 [+.0164,+.0258] **+** | +0.0030 [−.0018,+.0079] **ns** | +0.0138 [+.0086,+.0192] **+** |
| MAX | −0.0204 [−.0253,−.0155] **−** | +0.0112 [+.0063,+.0161] **+** | +0.0039 [−.0013,+.0091] **ns** |
| MSD-50 (views-10) s0 | −0.0077 [−.0125,−.0030] **−** | +0.0075 [+.0026,+.0123] **+** | +0.0007 [−.0042,+.0059] **ns** |
| MSD-50 (views-10) s1 | −0.0026 [−.0078,+.0025] **ns** | −0.0045 [−.0095,+.0005] **ns** | −0.0056 [−.0109,−.0002] **−** |
| RAMP (B1−R′) | −0.0037 [−.0081,+.0007] **ns** | +0.0008 [−.0036,+.0052] **ns** | +0.0201 [+.0155,+.0248] **+** |
| FT-ℓ∞ | +0.0074 [+.0027,+.0120] **+** | +0.0094 [+.0048,+.0141] **+** | +0.0109 [+.0057,+.0162] **+** |
| **F2 v50 − MSD-50** | **−0.0060 [−.0111,−.0009] −** | +0.0010 [−.0040,+.0059] **ns** | **+0.0239 [+.0186,+.0293] +** |
| **F2 v50 − views-10** | +0.0017 [−.0033,+.0067] **ns** | −0.0065 [−.0115,−.0016] **−** | **+0.0232 [+.0178,+.0287] +** |

**What the CIs change (TONY, numeric only — story is CLAUDEUS's):**
- **F2 both parts now supported.** "Recovery is entirely ℓ₁" — Δℓ₁ +0.0232 [+.0178,+.0287] **significant**, while Δℓ∞ +0.0017 [−.0033,+.0067] is **ns** (recovery does not reach ℓ∞). "Δℓ∞ stays negative" — v50−MSD-50 Δℓ∞ −0.0060 [−.0111,−.0009] is **significantly negative**, so the directional claim holds. Rider droppable for both.
- **ℓ₂ is NOT null anywhere it was assumed to be.** MSD-10 s0 Δℓ₂ is ns, but s2 (+0.0109) and s3 (+0.0084) are **significantly positive**, and the seed-min LCB is **−0.0037 → ℓ₂ does NOT hold across all three seeds**. MAX Δℓ₂ +0.0112 and FT Δℓ₂ +0.0094 are also **significant**. Any prose reading ℓ₂ as "flat/unaffected" is unsupported.
- **MAX Δℓ₁ +0.0039 is ns** [−.0013,+.0091] — L13b rider applies (magnitude-only).
- **RAMP Δℓ∞ −0.0037 is ns** — "RAMP loses ℓ∞" is not supported; only Δℓ₁ +0.0201 is significant.
- No pooled multi-seed bootstrap exists (dossier §7); the aggregate row uses the project's existing seed-min-LCB convention, not a new estimator.

**Naming trap + provenance caveat (RAMP row).** `Rprime` = RAMP **base** (`Rprime_out/R_prime_ramponly_seed0`); `B1` = RAMP **+ term** (`Bet1_out/B1_pullpush_seed0`). Any comparison written as "ours vs RAMP" that used **B1** is **mislabelled** — it is "ours vs RAMP+CLAMP"; the RAMP baseline is `Rprime`. Provenance: `Rprime`/`B1` machine **UNTRACED** (ckpts arrived via Drive); `Rprime`/1k has **no masks** and is excluded from every bootstrap — the **@10k** pair does have masks, so this row is valid.

### F2 — staleness vs redundancy (matched 50-step views) ✅ DONE
| contrast @10k | Δunion | 95% CI | sig | Δℓ∞ | Δℓ₁ |
|---|---|---|---|---|---|
| views-50 − MSD-50 | −0.0030 | [−.0081,+.0021] | **ns** | −.0060 | +.0240 |
| views-50 − views-10 (term ON both) | +0.0066 | [+.0015,+.0118] | + | +.0017 | +.0232 |
**Reading (writable, with caveats):** strengthening views 10→50 recovers +0.66 (≈⅔ of the −0.96), **entirely ℓ₁** (+0.0232); Δℓ∞ stays negative (−0.0060). So staleness is a **major but not sole** contributor; the ℓ∞ interference survives strength-matching, and since the union is ℓ∞-bound, fixing ℓ₁ does not flip the union. Under matched views the reversal **loses significance** (−0.30 ns) but the estimate stays negative.
**Caveats (mandatory):** n=1 seed at views-50 vs 2 at views-10 → do NOT write "staleness ruled out"; Δclean −0.0013 (no clean trade); proxy mis-ranked v50 (0.4710<0.4810) yet @10k higher — weak-proxy artifact.

### RQ3 — mechanism ✅ DONE (approved verbatim)
Encoder clean–adv distance ↑ 6/6 (CIs excl 0, cosine+Euclidean); head worse than encoder & random; in-sample cos 0.92 no transfer. "evidence does not support the motivating alignment explanation … requires an alternative explanation, untested."

### RQ4 — anatomy ✅ (masks @10k exist for both audited models)
| | MSD-50 | R′ (RAMP control) |
|---|---|---|
| single-norm-blocked (ℓ∞/ℓ₂/ℓ₁) | 14.6% (13.2/0.0/1.4) | 10.4% (8.2/0.0/2.2) |
| multi-norm-blocked | 85.4% | 89.6% |
| clean margin \|B\|=0→3 | +4.57/+1.84/+1.39/−0.25 | +7.17/+3.25/+2.51/−0.24 |
| AUC margin→multi | 0.95 | 0.93 |
**AVG asset (CLAMP-free) ✅:** AVG control ℓ₁ 0.5209 (highest, > CLAMP 0.5148, +5.4 vs MSD) but union flat (−0.40 ns) because ℓ∞ lowest → union ℓ∞-bound, independent of the intervention. Lexicon "demonstrates".

### Ablation ✅ (seed-0)
CLAMP full 0.4173 · single view **(MSD)** 0.4012 (−1.61; vs control LCB +0.75) · clean negatives 0.4139 (−0.34 ns).
❌ clean-CE, α/β={0.25,1.0}: NOT RUN → rows removed.

### Sanity ✅
No masking (Square never beats white-box); no collapse (uniformity −3.45); selection parity 2×2 @1k (C50 .4340/.4320, M50 .4350/.4320) → write by magnitude ("rule can't fabricate a ~1pp effect"), not by @1k sign; proxy over-read symmetric (CLAMP −5.6, MSD −5.1). Hardware: no blanket claim; defense = paired design + 3-seed.

---

## E. CLAIMS TABLE (claim → evidence → status)
| Claim | Evidence | Status |
|---|---|---|
| Intervention improves union under matched control | tab:main +2.65/3-seed | 🔒 LOCKED |
| Largest per-norm effect = ℓ₁ | tab:pernorm | 🔒 |
| Union improvement is bought by ℓ∞ | per-norm + AVG control | 🔒 |
| Helps 2/3 bases, reverses on strongest (MAX) | tab:bases 2×2 | 🔒 (now complete) |
| Alignment explanation not supported | mechanism (RQ3) | 🔒 |
| Union ℓ∞-dominated, ℓ₂ never sole blocker | anatomy + AVG asset | 🔒 (scope L7) |
| Multi-view necessary | single-view(MSD) ablation | 🔒 |
| Full-budget reversal: staleness major-not-sole | F2 | 🔒 (single-seed caveat) |

---

## F. ❌ CORRECTIONS LOG (dead — never revive)
| Dead claim | Replaced by |
|---|---|
| "RAMP+CLAMP never ran" / "RAMP-base tie is VOID" | **ITSELF DEAD.** It ran, as **B1**. RAMP row reinstated: R′ 0.4461 → B1 0.4456, −0.0005 ns. Stray empty `union_bench/B2/` (static-cycle experiment's misplaced dir) **deleted 2026-07-22**. |
| "Gain routes through ℓ₁" | "largest effect on under-served norm; union bought by ℓ∞" |
| "APGD-T" label | APGD-DLR untargeted + disclosure |
| "Frontier slide" | REJECTED by Δclean (§ no control dominates CLAMP-10) |
| "Helps 2/3 weak bases" (as strength heuristic) | mechanism-specific: helps when base under-serves the term's norm |
| any @1k number for ordering | forbidden (L8) |
| "12-attack AutoAttack union" | "12-component AutoAttack-based union" |

---

## G. DATA-INTEGRITY / HARNESS / RECIPES (App A/C) ✅
Harness: AutoAttack commit **a3922004**; APGD-CE & **APGD-DLR (untargeted)** 100it/1restart; FAB-T 100it/1restart targeted; Square 5000q/1restart; ε (8/255, 0.5, 12); ℓ₁ = Croce–Hein sparse ✓; eval seed 20260709; config hash **4465ab20** (9162ce44 = @1k, not headline). Masks released; exceptions: M1a_full per_attack={} → per-component table = headline pair only.
Recipes: 10-step family SGD 80ep lr .05→.005@70; α=β=0.5 τ=0.1 warmup10; head 512→128 eval-discarded; val-best worst-union proxy. 50-step family: 50ep one-cycle lr .1, MSD 40–50it, **views kept 10-step (= the staleness F2 tests)**. v50 provenance CLOSED (apgd_view_steps=50, 1.80× slowdown, sha triple-match).
Δclean @10k: MSD-10 −2.15 · MSD-50 +0.08 · MAX +0.35 · FT −1.20 · **AVG −0.86**.

---

## H. FIGURES & TABLES
Figs: geometry ✅ · pipeline ✅ · mechanism(3-panel) ✅ · perclass(App B, label exploratory) ✅ · union_failure_norms(RQ4, ◻️D2 include/drop) · Fig5 2×2 (**now buildable — data complete**).
Tables: models(RAMP→context) · main · pernorm · **bases (now fully fillable: MAX+AVG rows in; RAMP/clean-base removed)** · ablation(cleaned) · anatomy · hparams · percomp(DLR).

---

## I. MECHANICAL EDIT STATUS (main.tex) — mostly DONE this session
✅ APGD-T→DLR (5 spots) · ✅ tab:bases clean col + Δclean(MSD rows) · ✅ removed NOT-RUN rows · ✅ App C harness pin · ✅ single-view=MSD label · ✅ RAMP→context (tab:models/limitations) · ✅ abstract/contrib/conclusion comments de-hardcoded −0.96 · ✅ sanity selection-parity numbers.
◻️ TO DO: fill tab:bases MAX + AVG rows (data now in) + Fig5 · revert Method prose→notes (◻️D1) · fill FT footnote (Sehwag 84.59/55.54).

---

## J. RUNS / GATES (all gating CLEARED)
| Run | Status |
|---|---|
| M1a_avg @10k | ✅ DONE (union 0.4049) |
| F2 = v50 @10k | ✅ DONE (−0.0030 ns) |
| F1 = cev1.6 (exposure control) | optional; NOT launched per decision → future work (◻️D5 if re-considered) |
| RAMP+CLAMP (= **B1**, not "B2") | ✅ DONE — union 0.4456, Δ −0.0005 ns, locally traced (ckpt path + negative train loss + sha match). Cloud-W&B confirm = Kiet. |
| M5 pull/push-only | future work |
Thầy sign-off: (a) title, (b) exposure-rival append, (c) heads-up APGD-DLR + RAMP removal — ONE message.

---

## K. GAPS (this doc does NOT hold — close before final)
- **G2 ✅ RESOLVED** — RAMP+CLAMP found: it is **B1**. "B2" in ATTACKDRO is a different experiment (`b2_static_cycle_ramp80_apgd_8255_t49k_v1k`, allocation-arc static cycle, own results dir + 5 offline W&B runs). Local evidence only (ckpt path `B1_pullpush_seed0`, negative train loss, sha match); cloud-W&B confirm = Kiet.
- **G3 ✅ PASS** — F1 (`cev1.6`) NOT running: `procctl --script c5_fromscratch.py` → NO MATCH; `results/fromscratch/C5_R2/` empty. Deferred = default holds.
- **G4 — honesty fix, adopted VERBATIM, no backdating:**
  > registration is mtime-ordered only; no ledger or commit timestamps exist, and the AVG @1k screening (07-21 12:00) preceded the @10k preregistrations.

  Mitigations retained, both independently checkable: the T2/T2b preregistrations concern **encoder/head geometry** and make no prediction about the AVG arm; and **w = 1.6 belongs to F1, which never ran**, so its timing affects no reported number. Detail + mtime table: `IMPLEMENTATION_DOSSIER.md` §6.4.
- **G5 ✅ CLOSED by CLAUDEUS** — CFP = 10p + 2p. **5 fig PDFs exist; 4 are `\include`d; `union_failure_norms.pdf` is built but commented out pending decision.** 4-vs-5 count RECONCILED. Not verifiable from ATTACKDRO by design (repo separation, §A).
- G6 current clock vs any remaining ETA (**none gating now**).

---

## L. ◻️ OPEN DECISIONS (Kiet)
1. Method prose → revert to notes (recommend yes; process rule Kiet-writes-all).
2. fig:union_failure_norms → include or drop (page budget).
3. (clean,union) plane → text+table or mini-scatter.
4. Matrices = branch-meaning only, Kiet writes the sentence (ratify).
5. F1 → keep deferred (future work) or re-launch on Colab (fits Jul 23 AM). Default: deferred.
6. RQ4 optional 1-sentence "ℓ₂-slack-by-calibration" → add or drop.
7. **Title**: "Understanding When…" (current file) vs "When Do…" (GPT/CLAUDEUS rec) → pick + thầy sign-off.
8. Appendix overflow cut order (compress tab:percomp first, shrink fig:perclass next).
# CARD-PB — Predictive-Binding Union AT (DRAFT — do not launch; Kiet review first)

*Pre-registered per PLAYBOOK §I.1. The post-Gate-α method run. Gated on Kiet approval.*

## 0. One-line
Gate-α said per-sample binding is a **short-horizon TRAIT** (l1 next-dump retention 63.4% vs 9.6% chance, +53.8pp; κ0.60). So we can **predict** each sample's binding norm cheaply and **spend attack budget where it binds** instead of fully attacking all three norms every step — the claim is **EFFICIENCY** (match reactive robustness at ≤60% attack-FLOPs), not higher robustness.

## 1. Hypothesis
A cheap EMA predictor φ of per-sample binding, used to allocate attack budget, **matches** the worst-case union robustness of the reactive all-norms-every-step baseline **within seed noise**, while spending **≤60% of the attack-FLOPs**. (Conj 2 = predictive weighting is *viable*; the honest paper claim is the efficiency win, C2 in the ladder.)

## 2. The ONE change vs predecessor
Predecessor = **per-sample soft T=0.25** (reactive: crafts linf(10)+l2(10)+l1(20) = 40 attack-steps/sample every step, then soft-max weights the losses). CARD-PB keeps the recipe/protocol/loss identical and adds exactly three things: **(a) EMA binding predictor φ, (b) budget allocation to the predicted norm, (c) a safety floor.** Nothing else changes.

## 3. Design

### (a) Short-horizon EMA binding predictor φ  — *free (reuses training signals)*
- Per sample i, maintain an EMA of per-norm attack losses computed **during training** (no extra forward):
  `Lbar[i,g] ← β · L[i,g](this epoch) + (1−β) · Lbar[i,g]`.
- Short-horizon because Gate-α showed binding is a ~5-epoch trait that drifts over the full run → **β ≈ 0.5** (≈2-epoch memory) as the working default.
- **β is a FIRST-CLASS ABLATION {0.3, 0.5, 0.8}** (the cheap validator Proposition 3 / method-note-v2 demands, NOT a hardcode): the β\* that maximizes φ’s 1-epoch-ahead hit-rate should land at a memory length consistent with the **measured ~5-epoch binding horizon** (Gate-α). If β\*≈0.5 wins, the judgment call becomes a **finding** — "optimal predictor memory matches the empirically measured trait timescale" (trait horizon ⇒ predictor memory, a mechanism link). Report β\* + the hit-rate-vs-β curve; β=0.8 (long memory) is the "ignore drift" arm that should LOSE if the state overlay is real.
- Predicted soft binding weight: `w_hat[i,g] = softmax(Lbar[i,g] / T)`, T=0.25 (same T as the reactive soft, for continuity). Hard predicted norm `b_hat[i] = argmax_g Lbar[i,g]`.
- φ is validated a priori by Gate-α: next-dump binding retention 63.4% ≫ 9.6% chance. Start with EMA (simple-first); upgrade to a tiny logistic head on loss-history features ONLY if EMA’s 1-epoch-ahead hit-rate < ~0.6 measured online.
- **Cold start:** epochs 0–4 run FULL reactive (no prediction) to seed Lbar and let binding form before trusting φ.

### (b) Budget allocation — *the FLOPs saver*
- Total per-sample attack budget target ≈ **16 steps** vs reactive 40 (=40% → 60% FLOPs cut).
- For each sample: **full attack in the predicted norm** (linf 10 / l2 10 / l1 20 steps as per that norm’s reactive spec) + **floor attack in the other two** (K_floor steps each, see (c)).
- Implementation (batched, GPU-friendly): partition each minibatch by `b_hat[i]` into ≤3 sub-batches; attack each sub-batch **fully in its predicted norm**, and run a **shared floor attack** of the other norms across the whole batch. Training loss = the same per-sample soft-max over whichever crafted examples exist (predicted-full + floors), so the loss shape matches the reactive baseline (one-variable change is the *budget*, not the *objective*).
- FLOPs bookkeeping: count attack forward+backward passes; log `attack_flops/epoch` and the ratio vs a reactive control run. This ratio IS the deliverable.

### (c) SAFETY FLOOR — *no robustness holes if φ is wrong*
- **K_floor ≥ 3 steps per non-predicted norm** (never zero): a mispredicted norm is still attacked enough to (i) generate gradient and (ii) surface vulnerability. Prevents the failure mode where φ locks onto linf and the model silently loses l1 robustness.
- **Recalibration pass:** a random **ρ=10%** of each minibatch (or a full-reactive epoch every 5) is attacked FULLY in all three norms regardless of φ — bounds predictor drift and provides an unbiased binding signal to refresh Lbar. This unbiased full-attack subset is ALSO how we **MEASURE the misprediction rate p** (φ’s predicted binding vs the true argmax on that subset) → **log `phi/misprediction_rate` per epoch**. This makes **Proposition 5’s safety-floor bound EMPIRICAL**: the floor budget can be set from the measured p (and its worst-norm conditional) instead of an assumed constant, and the >2pp guard below is checked against the p-implied bound.
- **Adaptive floor (safety valve):** the cheap per-epoch union probe already tracks per-norm robust acc. If any norm’s probe robustness falls > **3pp** below the reactive control at the same epoch, **raise that norm’s K_floor** (or force it full) until it recovers. Logged as `floor/<norm>`.
- Pre-registered guard: **no single norm’s final robust acc may drop > 2pp vs the reactive baseline** — if it does, the floor was too low (that’s a tuning failure, not a method refutation), raise floors and re-run.

### (d) Protocol & comparison — *matched, paper-grade*
- **train == eval at eps_inf=8/255** (base.yaml, l2=0.5, l1=12), PreActResNet-18, raw [0,1]. This is the first paper-grade 8/255 run (no 0.03 mismatch).
- **Matched reactive baseline, RETRAINED at 8/255** (not the lower-bound 0.03-ckpt estimates): per-sample soft T=0.25 at 8/255, same recipe/seeds. Also report MSD-8/255 if cheap.
- **External comparator:** RAMP repro **union 46.1** (train@8/255, matched, APGD n=1000) — the fair reference.
- Seeds: 3 exploratory → 20 + Wilcoxon + BH-FDR only if a claim leaves the repo (PLAYBOOK §VI).
- Metrics: worst-union APGD (iteration) + full standard-AA on the claim rows; **attack-FLOPs ratio** the co-primary metric.
- **RUN ORDER (confirmed):** **(1) reactive-8/255 RETRAIN** (per-sample soft T=0.25, train==eval@8/255) — REQUIRED, and it is the paper’s **headline reactive row** AND the **first paper-grade 8/255 run**; **(2) predictive-8/255** (this card); **(3) compare** Δunion + FLOPs. All rows vs **RAMP repro 46.1**. Predictive is meaningless without (1) as its matched control, so **(1) goes first** — the reactive retrain is not optional prep, it is the baseline the whole claim rests on.

## 4. Pre-registered decision rule  (the real claim = EFFICIENCY)
Let Δunion = predictive − reactive (both 8/255, matched, 3 seeds), noise = 2× pooled seed-std (T=0.25 seed-std ≈ 0.06–0.1pp → noise band ≈ ±0.2pp; use the larger of the two recipes’ std).
- **|Δunion| ≤ noise AND FLOPs ≤ 60% of reactive → EFFICIENCY WIN** → C2 claim: "a cheap short-horizon binding predictor matches reactive union robustness at ~1.7× fewer attack-FLOPs." **This is the target.**
- **Δunion < −noise (robustness lost)** → predictive weighting hurts → try raising floors (more FLOPs); if it still loses at ≤60% FLOPs → **NEGATIVE result**: "binding is a trait but exploiting it for budget doesn’t pay under matched robustness" — publishable, sharpens F5/F6 (the trait is real but not economically actionable).
- **FLOPs > 60% at matched robustness** → no efficiency win → report the FLOPs floor at which robustness is matched (still a mechanism result).
- **Δunion > +noise** → bonus (unexpected; Conj 2 only claimed viability). Verify before any "beats" wording (suspicious-good rule).

## 5. Cost & gating
- ≈ 3 seeds × (predictive + reactive control) at 8/255. Per-run wall-clock: measure with smoke_timing before commit (predictive should be *cheaper* per epoch than reactive by construction). Rough: ≤ reactive’s ~3.4h/seed → ~20 GPU-h for the 3+3 seeds. Gate on a 2–3 epoch smoke + a FLOPs-ratio readout first (do NOT commit all seeds before the FLOPs cut is confirmed on real hardware).
- **Escalation:** > 2 GPU-h and it alters a paper claim → Kiet sign-off (this card). Show Kiet before launch.
- **Scheduling (cross-device, not concurrent-on-one-GPU):** "parallel" = across devices (local 5070ti + Colab Pro); on a single GPU concurrent runs are compute-bound → no throughput gain, so serialize. The **reactive-8/255 retrain (step 1) is Colab-eligible** (standard resume-capable trainer, disconnect-proof pattern per PLAYBOOK §VI) → run it on **Colab** while the 5070ti finishes reeval/Conj-1, collapsing the critical path so the reactive control is ready when predictive-8/255 launches. Predictive-8/255 runs on whichever device is free after its control lands.

## 6. Why this is the right shape
- Rides on Gate-α: uses the *measured* short-horizon trait, doesn’t assume a fixed label (β EMA = short-horizon by design, matching the state overlay).
- Leverage lives in the objective/training budget, not an inference-time trick → robust to adaptive eval (PLAYBOOK §V).
- Every decision-rule branch is publishable (efficiency win / negative-but-sharpening / mechanism-only) → claim-ladder floor holds.
- Distinct from RAMP (hard L_max + fixed λ), E-AT (fixed geometry), MSD (per-step all-norm): those are static/sample-agnostic; CARD-PB is the *adaptive, measured-binding-driven, budget-aware* point — the gap F5/F6 named.

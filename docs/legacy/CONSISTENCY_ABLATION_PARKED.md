# Cross-attack consistency — PARKED future ablation lane

**Status:** PARKED. Do NOT implement. **Gate: after frontier + seeds lock the efficiency claim.** Not before.
**Owner:** Kiet (decides un-park). Origin: P-12/P-13 (2026-07-08). Source code reviewed = the linf/l2
cross-attack-consistency training core (mixed_batch baseline / both_attacks + feat_l2/feat_cos/pred_kl).

## WHAT
A consistency regularizer that aligns **features across norm-typed attacks** (linf-adv vs l2-adv vs
l1-adv views of the same sample), added as a **controlled `+consistency` ROW on the winning kspan** —
**NOT a redesign, NOT a new method, NOT now.** It is a second lever bolted onto the frozen operating point.

## WHY PARKED
- **Orthogonal lever.** CARD-PB's lever is *allocation* (where to spend per-sample attack budget by
  binding). Consistency's lever is *representation quality* (make the norm-typed adversarial views agree).
  Different axis — one decides *where to spend*, the other *what the shared representation learns*.
- **Opposite cost profile.** `both_attacks` crafts 2 (→3) attacks per sample + extra forwards ≈ **2×+ the
  attack-FLOPs**. That directly attacks our efficiency headline (predictive matches reactive at 0.665×).
- **Confound risk.** Folding it into the just-de-risked pipeline muddies the clean single-mechanism story:
  reviewers can't tell if a win came from *allocation* or *consistency*. **Clarity > completeness** for a
  mechanism paper. Add it only as a controlled, FLOPs-accounted ablation once the core claim is locked.

## PORT CONDITIONS (ALL required before it is valid in our pipeline)
1. **3-norm, not 2.** Extend consistency to **(linf, l2, l1)**. Form: **centroid-pull** (align all three to
   their mean feature) OR **pairwise** — decide + justify. l1 is the hard norm (F9 root cause) and MUST be in.
2. **Fixed-eps, not min-norm.** Replace **DDN-l2** (variable-norm) with **APGD-l2 @ eps=0.5** — our threat
   model. All three views at our fixed eps, crafted with **APGD** (matches the ported training attack).
3. **Attack outside autocast.** Craft the inner-loop in **fp32**; the reviewed code runs attacks under
   `autocast` (fp16) → gradient noise in the perturbation. Move generation out of the autocast block.
4. **Stop-grad vs collapse.** `feat_l2_norm` on L2-normalized features with gradient on **both** branches
   risks **feature collapse** (BYOL-style). Add **stop-grad on one branch**, OR watch collapse explicitly
   via the alignment diagnostic (align_l2 → 0 while clean acc drops = collapse).
5. **FLOPs accounting.** The 2×(→3×) attack cost **MUST be in the denominator.** The only honest comparison
   is **predictive vs predictive+consistency at ACCOUNTED FLOPs**: does union rise ≥ noise *for the extra cost*?

## READ RULE
- **union rises ≥ noise at accounted FLOPs** → a genuine **`+consistency` row** (efficiency + representation).
- **else** → **allocation is the primary lever** (also a publishable result — the regularizer doesn't pay
  for its compute here).

## SEPARATE DIAGNOSTIC VALUE (independent of the loss — usable even if consistency doesn't help)
The **feature-alignment measurement** (`evaluate_feature_alignment`: cross-attack feat L2 / cosine / pred-KL)
is analysis, not a loss. Cross it with **Gate-α** to ask a mechanism question:
> *Do samples with more aligned cross-norm features bind more stably (lower binding-flip / miss_vol)?*

If yes, cross-norm feature alignment is a **binding-stability signal** — feeds CARD-PB's predictor story
directly, with zero commitment to the consistency loss. Run this as cheap analysis anytime after the
efficiency claim; it does not require the regularizer.

## GATE (repeat)
**After frontier + seeds lock the efficiency claim.** Not before. Frontier keeps running untouched.

# Early-stop attack (union AT) — novelty + feasibility memo
*Non-blocking scan (P-05, 2026-07-07). Kiet is considering: instead of a per-sample binding
PREDICTOR (CARD-PB), stop each norm's inner attack once it flips the sample — avoiding the
predictor entirely (no binding-norm guess → no F9-style misprediction). Do NOT build yet.*

## Idea
For each training sample, run the inner PGD in each norm but **terminate that norm's attack at
the first step that misclassifies** (first adversarial example found), instead of the full step
budget. No predictor → no misprediction. Efficiency comes from easy samples flipping early.

## Novelty — the MECHANISM is NOT new (single-norm); the UNION application is a modest extension
- **Per-sample early-stop-on-misclassification is established in single-norm AT:**
  - **Friendly Adversarial Training (FAT, Zhang et al. 2020)** — "early-stopped PGD": stop when the
    sample is first misclassified; the canonical form of this idea.
  - **ATES / Curriculum AT** — data-adaptive PGD iterations, ramp attack strength over training.
  - **Dynamic/gradient-magnitude-guided efficient AT** — related per-sample budget ideas.
  - **Documented trade-off (directly relevant to us):** early-stop **under-attacks hard samples →
    weaker adversarial supervision → lower robustness ceiling; PGD-10 looks fine but AutoAttack
    degrades.** This is the strong-eval gap.
- **Union / multi-norm efficiency work does NOT use per-norm attack early-stop:** MSD (all 3 norms,
  50 steps, no early-stop), **E-AT** (efficiency via fixed geometry linf+l1, l2 free + fine-tuning),
  **RAMP** (logit pairing + hard L_max + grad projection). None terminate the inner attack per-sample
  per-norm. So **"per-norm early-stop inside the union attack" is a plausibly-novel *extension*** —
  but the underlying mechanism is old, so it's a small-delta contribution, not a new primitive.

## Feasibility notes (the three the user flagged)
- **(a) "fail" definition for union — saving is asymmetric and shrinks over training.** Union =
  worst-case over 3 norms, so you must still **probe all 3 norms per sample** (can't skip a norm you'd
  early-stop — the sample might be robust to it and that norm sets the union). Early-stop only saves
  **steps WITHIN each norm's attack**, on samples that flip early. **Robust/hard samples (never flip)
  cost the FULL budget in all 3 norms — no saving exactly where it's expensive.** As the model gets
  more robust (late training), fewer early flips → **saving decays over epochs.** This is the *inverse*
  of CARD-PB (which saves on predicted-easy norms regardless of flip).
- **(b) weak-adversarial-example risk — overlaps our l1 problem.** Stopping at first misclassification
  yields a **boundary-minimal** perturbation, weaker than running to max. FAT/ATES confirm this lowers
  robustness under strong eval. **For l1 (our fragile norm, F1: weak l1 attacks overstate l1
  robustness), early-stop could under-attack l1 → overstate l1 → the SAME failure surface as F9, via a
  different route (weak examples instead of misprediction).** So it does **not cleanly escape the l1
  issue** — it swaps misprediction-risk for weak-example-risk. Must be MEASURED, not assumed away.
- **(c) batch efficiency — real saving << on-paper step saving.** Samples flip at different steps. On a
  GPU, a batch runs until the last active sample stops (or you mask finished samples and continue only
  active ones → the tensor stays full-width unless you **compact/repack**, which is fiddly). Without
  repacking, wall-clock ≈ max-over-batch, not mean → the measured FLOPs cut is modest. Repacking adds
  engineering + sync overhead.

## Verdict / recommendation for Kiet
- **One genuine advantage:** avoids the predictor → structurally cannot have F9-style *misprediction*.
- **But:** (1) mechanism not novel (cite FAT/ATES — a small-delta union extension); (2) it inherits the
  **weak-example → l1-under-attack** risk, which is our *same* l1 vulnerability by another name; (3) its
  FLOPs saving is data-dependent, decays late-training, and is batch-limited.
- **So:** early-stop is **not obviously better than CARD-PB** for the *efficiency + l1-robustness* goal —
  it trades one l1 failure mode for another.
- **Decision rule:** if CARD-PB **survives** the Phase-1 de-risk (l1 holds), **stay with CARD-PB** (novel
  measured-binding mechanism, cleaner story). If the de-risk shows **F9 recurs**, early-stop is worth a
  **cheap 1-seed pilot** — but framed as "curriculum/early-stop efficiency in the union setting" (honest
  extension, cite FAT/ATES) and it MUST report the **l1-robustness-vs-savings trade-off** (weak-example
  risk) + full-AutoAttack l1, or it repeats F1/F9 silently.

Sources: FAT/early-stopped-PGD (Zhang et al. 2020), Bag-of-Tricks AT (Pang et al. 2020), Dynamic
Efficient AT (grad-magnitude), MSD (Maini et al. 2020), E-AT (Croce & Hein 2022), RAMP (Jiang et al.
NeurIPS 2024).

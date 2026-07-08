# Composition map — a principled framework for combining methods

**Status:** STRATEGIC FUTURE WORK. Do NOT implement now. **Gate: after frontier + seeds lock the
efficiency claim.** Frontier keeps running untouched. Origin: P-14 (2026-07-08). Owner: Kiet.

The point of this doc is to decide *what can be combined and what cannot*, so future work is a
deliberate single pick — not an ad-hoc pile-up. The organizing idea: **every method touches one of
three axes; same-axis methods are mutually exclusive, different-axis methods compose.**

## FRAMEWORK — classify by AXIS (the axis determines what combines)

| Axis | Question it answers | Methods on this axis |
|---|---|---|
| **A · ALLOCATION** | *where* to spend attack budget | reactive · **predictive / CARD-PB** |
| **B · LOSS-SHAPE** | *what* to learn (the objective) | soft-weighting · hard `L_max` · **consistency** (feature-align) · **RAMP logit-pairing** (logit-align) |
| **C · GRADIENT** | *how* to update | **RAMP gradient-projection** (natural+adversarial mixing) |

**RULE:**
- **Same axis = mutually exclusive** — pick ONE (e.g. soft-weighting *vs* `L_max` *vs* consistency *vs*
  logit-pairing all live on Axis B; you don't stack two B's, you swap them).
- **Different axis = composable** — orthogonal levers, can co-exist (an A method + a C method).

Our contribution (**predictive allocation**) lives on **Axis A**. RAMP's strength is actually **two**
pieces: logit-pairing on **B** and gradient-projection on **C**. That separation is what makes the
compositions below legible.

## PRINCIPLED COMPOSITIONS (ranked by promise — ALL gated after efficiency-lock)

### 1. predictive (A) + gradient-projection (C) — **rec pick**
Efficiency allocation + ℓ∞-preservation. **Cleanest beat-accuracy candidate.** GP (not logit-pairing) is
what actually gives RAMP its ℓ∞ strength, and it is **orthogonal to predictive allocation** (A×C), with
**none of the 2× cost of consistency**. Likely **> full CARD-8b** because it is simpler and isolates the
one mechanism that adds ℓ∞ robustness. *If we combine anything, this is it.*

### 2. predictive (A) + RAMP-full (B-logit + C-projection) = **CARD-8b**
The composition already scoped: keep *all* of RAMP (both its B and C pieces) and add predictive allocation
on top. Strictly more machinery than #1. **~15–30% beat chance.** Heavier, harder to attribute the win.

### 3. feature-consistency (B) **REPLACES** logit-pairing (B) in RAMP — *same-axis swap*
Not a stack — a **swap** on Axis B. Tests the mechanism question: **"feature-align vs logit-align for
union robustness?"** Cheap (**1–2 runs**), no allocation change. A clean mechanism probe, not a headline.

### 4. predictive (A) + consistency (B) — the "+consistency row"
Efficiency + representation. Composable (A×B), **but consistency's 2× attack cost attacks the efficiency
headline.** Only worth it if union rises enough to justify **accounted FLOPs**. This is the
`+consistency` row on the winning kspan — see `docs/CONSISTENCY_ABLATION_PARKED.md`. Lowest priority *for
the efficiency paper* precisely because it fights the FLOPs story.

## STRATEGIC CAVEAT — composable ≠ do-now
Each composition ≈ **a separate paper** (implement + seeds + ablation = months). The efficiency claim
(**predictive vs reactive**) *alone* is a sharp single-mechanism contribution — **sufficient for the first
paper.** Compositions dilute it. **Pick ONE** as the follow-on (**rec: #1 predictive + GP**); do NOT
attempt all. The map exists so the choice is principled, not so everything gets built.

## GATE
After frontier + seeds lock the efficiency claim. Not before. Pointer lives in PROJECT_STATE §2 (parked).

# CARD-PB v2 — predictability-aware safety floor (l1 recovery)

> **CURRENT design + status live in `docs/PROJECT_STATE.md` §5.** This doc is the detailed
> rationale/FLOPs-math. UPDATED since drafting: the miss driver is now **P(true=g ∧ b̂≠g)
> (population-aware volume)**, not 1−recall (sweep-1 confound fix); the adaptive guard is
> **OFF** in confidence mode. Frontier arm range = **{8,16,24,32}** (LOCKED — maps the l1-recovery-vs-FLOPs frontier).

**Status:** DRAFT / pre-registered. **DO NOT LAUNCH** until reactive 3-seed baseline
lands (~06:30 UTC 2026-07-07). Supersedes the floor mechanism of CARD-PB v1
(`docs/CARD-PB_PREDICTIVE_BINDING.md`), which was killed at s0 (F9).

## 1. Why v1 failed (grounded, seed-0 @8/255 train==eval)

| norm | reactive | predictive v1 | Δ |
|---|---|---|---|
| linf | 42.4 | **44.1** | **+1.7** |
| l2 | 64.1 | 63.1 | −1.0 |
| l1 | **47.5** | **44.3** | **−3.2** |
| union | 41.7 | 40.6 | −1.1 |
| attack-FLOPs | 1.00 | **0.522** | — |

φ shifts budget **from l1 → linf**: it over-predicts the sticky majority norm (linf)
and mispredicts the minority norm (l1). This is exactly what Gate-α (F6) predicts — l1
is the **short-horizon-TRAIT-with-state-overlay** norm, i.e. the *least predictable*, so
φ misses it most; missed l1-binders fall to the l1 **floor** (only 3 steps of a 20-step
l1 attack) → l1 under-attacked in training → robustness overstated → collapses under
APGD eval.

**Two compounding v1 defects, both isolated in the trajectory:**
1. **Floor too low for the least-predictable norm.** l1 floor stayed at 3/20 the whole
   run.
2. **The adaptive guard is probe-driven, and the l1 probe is the weakest attack (F1).**
   The guard raises a norm's floor when its *cheap-probe* robust-acc dips below its own
   running max. l2 tripped it (floor 3→5); **l1 never did** — the weak l1 probe never
   revealed the erosion that strong APGD later exposed. So v1's safety valve is
   structurally blind to exactly the norm that needs it. → **v2 must drive the l1 floor
   by a signal that is unbiased on l1: φ predictability, measured on the full-attack
   recalibration subset — NOT the probe.**

## 2. Design

**LOCKED (Kiet, 2026-07-07): Option B is PRIMARY. Option A is fallback/ablation only**
(shows a fixed l1 floor works, but B self-tunes and unifies Gate-α + F9 into one story).
Sweep **B `k_span ∈ {8, 12, 16}`**, each arm mapped to measured `attack_flops_ratio`.

Base allocation unchanged (predict norm → full budget there; floor in the other two).
Only the **floor per norm** changes from a uniform constant to a predictability-scaled
value. `full_steps=[linf 10, l2 10, l1 20]`, reactive = 40 steps/sample.

### Option A — fixed l1-priority floor (simple, the sanity baseline)
- `l1` floor = 8 (of 20); `linf`, `l2` floor = 3. Keep l2 starved (Gate-α F3: l2 is
  free/spillover, ~0% binding → cheap to leave thin).
- One knob, no new measurement. Good first point on the sweep.

### Option B — confidence-aware per-norm floor (PREFERRED, principled)
- Measure **per-norm miss rate** on the recal subset (already full-attacked → unbiased
  ground truth): `miss_g = P(b_hat ≠ g | true_bind = g)` (= 1 − recall_g). EMA-smooth.
- **`floor_g = clamp(k_min + round(k_span · miss_g), k_min, full_steps_g)`**, with
  `k_min=3`, `k_span=12`. Norms φ predicts poorly (l1) get a high floor *automatically*;
  norms it nails (linf) stay minimal; l2 stays thin because few samples bind it → its
  miss rate is ~0. The floor is tied to *predictability* and logs *why*.
- This subsumes Option A's intent (l1 gets the most floor) without hand-picking l1, and
  self-tunes if the binding geometry shifts.

### Required code changes (not yet written — draft only)
1. `_pb_step`: on the recal subset, accumulate per-norm hits/misses
   (`true_bind==g` vs `b_hat`). → `self._pb_miss_ema[g]`.
2. Replace uniform `pb_k_floor + _pb_extra_floor[g]` with the Option-A/B `floor_g`.
3. **Log `phi/miss_rate_{linf,l2,l1}`** per epoch (new keys — add to the canonical W&B
   schema on merge) alongside existing `floor/{norm}`, `pb/attack_flops_ratio`.
4. v1 probe-driven `_pb_adaptive_floor` → demote to secondary/OFF (probe-blind on l1).

## 3. FLOPs accounting — the tension is real, quantify it

l1 is floored for the ~85–90% of samples that don't predict l1, so **raising the l1
floor costs FLOPs broadly**. Each +1 step of l1 floor ≈ **+0.022** on the ratio
(≈0.87·B extra passes / 40). From the v1 base (~0.45 with all floors=3):

| l1 floor | est. FLOPs ratio | note |
|---|---|---|
| 3 (v1) | ~0.45–0.52 | l1 −3.2pp (fails) |
| 5 | ~0.49 | |
| 8 (Opt A) | ~0.56 | under 0.60 |
| 10 | ~0.60 | at the ceiling |
| 12+ | >0.62 | **breaks the ≤60% claim** |

So there is a **narrow window (l1 floor ≈ 8–10)** where l1 *might* recover while FLOPs
stay ≤0.60 — but the l2 adaptive raise already pushed v1 to 0.556, so headroom is thin.
The sweep finds the point empirically. **Log Δunion (esp. l1) AND measured FLOPs at
every setting.**

## 4. Pre-registration

**v2 SUCCESS (all three, vs the reactive 3-seed baseline):**
1. union within noise of reactive: `|union_v2 − union_reactive| ≤ 0.3pp`;
2. l1 recovers: `l1_reactive − l1_v2 < 2.0pp`;
3. efficiency holds: measured `attack_flops_ratio ≤ 0.60`.

**NEGATIVE / FINDING (report, not failure):** if the l1-floor sweep shows **no** setting
satisfies (2) and (3) simultaneously — l1 recovers only when FLOPs > 0.60 — then
**l1 predictability sets a hard efficiency floor**: the least-predictable minority norm
cannot be cheaply skipped, and predictive union-AT cannot beat reactive at ≤60% FLOPs on
this geometry. That is a clean, publishable result about *when* predictive budgeting
works (predictable binding) and when it can't (trait-heavy minority norms).

**Sweep (LOCKED): Option B `k_span ∈ {8, 12, 16}`**, each arm mapped to its measured
`attack_flops_ratio`. Winning arm = min-FLOPs arm meeting (2) AND (3). Per arm, log the
**chosen per-norm floor** `floor/{linf,l2,l1}` and **`phi/miss_rate_{linf,l2,l1}`** so the
FLOPs-vs-l1-recovery tradeoff is visible and the story (l1 miss high → l1 floor high →
FLOPs up) is legible. Validate-first (s0), then gated s1/s2 (same harness as v1). Option A
`l1_floor=8` = a single fallback/ablation point, run only if B is inconclusive.

**Sweep bookkeeping:** produce `results/cardpb_v2_sweep.md` — one row per `k_span` arm with
{measured FLOPs, union, l1, Δl1 vs reactive, per-norm floor, phi/miss_rate per norm,
verdict pass/fail on (1)(2)(3)}.

## 5. Launch gate (STOP — do not cross without Kiet + baseline)

- [ ] **Reactive 3-seed complete** (~06:30 UTC) → baseline locked: `union_reactive`
      mean±std and `l1_reactive` mean (from `results/reactive_vs_predictive_3seed.md`).
- [ ] Code changes 1–4 written + smoke-verified (new `phi/miss_rate_{norm}` keys emit;
      floor responds to a synthetic high-miss norm).
- [ ] Config `configs/cardpb_v2.yaml` (inherits base 8/255; `pb_floor_mode: confidence`,
      `pb_floor_kmin: 3`, `pb_floor_kspan: 12`, `pb_floor_ema: 0.5`, probe-guard OFF).
- [ ] **Kiet GO** on Option A-vs-B and the sweep grid.
- Then: v2 s0 → eval → decide (same s0 wake-up gate as v1) → gated seeds. 20-seed
  escalation stays Kiet-only.

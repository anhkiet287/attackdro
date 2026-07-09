# Exploration lane — idea testing (Colab, NOT paper-grade)

**Purpose:** cheaply probe 3 allocation/schedule ideas Kiet raised, to see if any is worth a
proper port later. **Exploration signal, NOT paper numbers.** The 5070ti paper-grade
frontier/seeds/Phase-2 is the source of paper claims and is untouched by this lane.

**Framing:** these are NOT confirmed better than CARD-PB (per-norm predictive allocation,
FLOPs 0.665, l1 holds). They are alternatives/variants. Each answers ONE decisive question
with ONE cheap short run (ep20–30, 1 seed). Kiet reads the SUMMARY and decides.

## Hard rules (baked into the configs)
- **Colab (RTX PRO 6000) only.** Do NOT run on the 5070ti; do NOT touch any paper `results/` file.
- Writes ONLY to `results/exploration/` (gitignored). Tagged `exploration:true device:colab`.
- Same threat model as the paper lane: CIFAR-10, PreActResNet-18, eps=(8/255, 0.5, 12), APGD.
  Eval APGD 20/20/100. RAMP recipe (lr 0.05 flat), but SHORT diagnostic runs.

## The gating (why the paper lane is safe)
All three ideas are **config-gated additions, OFF by default**:
- Idea 1 & 3 add code to `GroupDROTrainer` that fires ONLY when `groupdro.failrate_threshold`
  / `groupdro.curriculum` is set (reactive path only). Verified: the paper path leaks **zero**
  `exp/*` metrics and behaves byte-identically.
- Idea 2 needs no code — it is CARD-PB with `pb_floor_kspan=4, pb_k_floor=1`.
- `apgd_train(..., stop_frac=...)` is a backward-compatible optional arg (returns `x_best` only
  when unset).

## The three probes
| idea | config | decisive question | kill signal |
|---|---|---|---|
| **1 fail-rate** | `configs/exploration/idea1_failrate.yaml` | does stopping each norm at 50% batch-fail keep l1 alive AND beat 0.665 FLOPs? | l1 train-vs-eval gap blows up, or FLOPs ≥ 0.665 |
| **2 k_min recovery** | `idea2_kmin_recovery.yaml` | with kspan=4/k_min=1, does floor_l1 dip then climb (miss_vol feedback self-corrects)? | floor stuck low + l1 eval tanks |
| **3 curriculum** | `idea3_curriculum.yaml` (+ `idea3_control_fixed10.yaml`) | does ramping steps (3→6→10) match fixed-10 at ep30 with lower cumulative FLOPs? | curriculum l1 lags control, or no FLOPs saving |

## How to run
Open `notebooks/exploration_ideas_colab.ipynb` on Colab (RTX PRO 6000), Run All. It clones this
branch @ the pinned commit, installs, runs the 4 short runs, evals each (20/20/100), then
`scripts/dev/exploration_read.py` writes `results/exploration/idea{1,2,3}_*.md` + `SUMMARY.md`
(to Drive). The reads produce a KILL / POSITIVE / MIXED verdict per idea + a recommendation.

## New logged metrics (exploration only)
- Idea 1: `exp/steps_{linf,l2,l1}` (steps the threshold self-selected), `exp/attack_flops_ratio`.
- Idea 3: `exp/curriculum_steps` (the epoch's step count).
- Idea 2: `floor/{norm}` per epoch (already logged) — the trajectory IS the result.

## After the probes
STOP. No proper ports, no further runs, until Kiet reads `results/exploration/SUMMARY.md` and
decides (port properly / drop / needs a different test). Idea 1's per-sample repack is future
work (the batch-level version here is enough to read the mechanism).

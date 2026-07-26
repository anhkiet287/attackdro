# RESEARCH PLAYBOOK — v1.0
*Personal research operating system. Distilled from the AttackDRO / union-robustness project (Jun–Jul 2026).*
*Generic core (Parts I–V, VII–VIII) + ML-specific module (Part VI). Hand this file + MEMORY.md to any AI agent at session start.*

---

## 0 · How to use
- New project: copy this file in, create empty MEMORY.md + LOG.md, start at Part IV.
- New session: agent reads PLAYBOOK → MEMORY → last LOG entries, then states the current world in 3 lines before taking any task.
- After each milestone: run Part VIII retrospective; bump playbook version if a rule changed.

## I · Core principles (the non-negotiables)
1. **Pre-register before running.** Every experiment gets a card BEFORE launch: hypothesis · config · the ONE change vs predecessor · expected outcome · decision rule (if X→A, if Y→B) · cost. Predictions are on record; being wrong is data, hiding it is failure.
2. **Evidence-gate everything.** Before any expensive commitment, find the cheapest diagnostic that could kill the idea (0-GPU if possible). An idea that dies cheap was never alive.
3. **One variable at a time.** Compound changes are declared explicitly and justified, or not done.
4. **Validate the instrument first.** Before measuring anything, prove the harness against known references (official checkpoints, published numbers). Untrusted meter ⇒ every downstream number is noise.
5. **Tiered comparison.** Three tiers, never silently mixed: (a) in-house recipe-controlled, (b) re-evaluated released artifacts, (c) ‡cited numbers. Every table row states its tier.
6. **Claim ladder with a floor.** Design the project so EVERY branch of the decision tree ends in a publishable output. Ambition rides on top of a floor, never instead of it.
7. **Simple-first.** Don't fix unconfirmed problems; complexity must earn its place with evidence (a hyperparameter added without a measured need is debt).
8. **Honest verdict reading.** Results are read against the pre-registration, in a fixed format, immediately — not batched, not reframed. Suspiciously good numbers are treated as bugs until verified (config audit, independent slice, curve inspection).
9. **Refuted-ideas ledger.** Every killed idea is logged with the reason and the meta-lesson. Do not reopen without new evidence.

## II · Operating system — 3 files, 3 speeds
| File | Speed | Content | Discipline |
|---|---|---|---|
| **PLAYBOOK.md** | slow | how I work (this file) | changes only via retrospective |
| **MEMORY.md** | medium | canonical project state: identity/goal · locked decisions · validated numbers · findings · refuted ideas · phase gates · assets · open items | update-in-place at every milestone; ≤150 lines; stale info is DELETED (history lives in LOG); "Last updated" stamped |
| **LOG.md** | fast | append-only history: experiment cards, result blocks, decisions, failures | never edited retroactively |
- **Sync = send MEMORY.md only** (LOG on request). One source of truth; dashboards/HTML regenerate FROM these files, never hold facts of their own.
- Compaction rule: when a section of MEMORY outgrows its value, compress to one line + pointer to LOG.

## III · Session protocol with AI agents
**Division of labor:** PLANNER agent (reasoning, verdicts, prompts, strategy) ↔ EXECUTOR agent (code, runs, file ops) ↔ HUMAN (bridge, decider, owner of claims). Files are the sync channel; agents never assume shared memory.
**Task intake (planner):** restate task in 1–2 lines → context scan (cite MEMORY/LOG) → ≤3 questions ONLY if answers change the approach, else state assumptions → 2–3 options with effort/risk/quality → recommend one. Small reversible tasks: proceed on stated assumptions.
**Prompts to executor** are self-contained, framed as:
```
[CONTEXT] [TASK] [CONSTRAINTS] [PRE-REGISTER] [OUTPUT] [STOP]
```
**Escalation — human sign-off required for:** protocol/metric changes · runs > 2 GPU-hours · declaring any "win" · deletions of results/checkpoints · anything altering a paper claim · new dependencies.
**Session end (executor):** LOG updated, MEMORY synced if a milestone closed, return brief (Runs / Results vs predictions / Done / Needs-human / Next).

## IV · Experiment lifecycle
```
idea → cheapest killing diagnostic → card (pre-registered) → smoke test
     → timing projection (measure, don't guess wall-clock) → run (named, logged)
     → result block → verdict vs card → MEMORY update → next
```
- **Smoke before burn:** mandatory tiny run (data→model→backward→finite loss) before anything >15 min; sanity check at ~10 min of long runs.
- **Naming:** `{method}_{variant}_s{seed}` identical across run logs, sessions, checkpoints, result files.
- **Result block (fixed):** numbers vs expected · which decision-rule branch fired · surprises (honest) · mechanism note · the single next experiment.
- **Failures are logged before relaunching** (cause hypothesis included). Two crashes ⇒ stop and escalate.

## V · Research strategy patterns (earned this project)
- **Negative result → research question.** "Method X didn't improve Y" becomes "why does X move A but not Y, and what would?" — the question is the asset.
- **Validate-first opening (P0 pattern):** lock protocol → validate harness → fixed baseline table. Only then touch methods.
- **SOTA is a live claim.** Verify current SOTA by search before writing any "match/beat" wording; a 2-year-old SOTA is a trap. Follow the field's own comparison practice (e.g., cite the standard retrained numbers rather than retraining everything).
- **Recipe confound is real and quantifiable.** Same method, different recipe can differ by tens of points. Either control it in-house or cite the community-standard numbers — and say which.
- **Weak-signal hazard:** cheap probes can INVERT true rankings, not just add noise. Any signal that drives decisions must be calibrated against the strong evaluation at least once.
- **Leverage lives in the objective.** (Domain lesson, generalizes:) defenses/gains bolted on at inference get routed around by adaptive adversaries/evaluation; durable gains come from what the training optimizes.
- **Two-lane ambition:** Lane A = safe deliverable on schedule; Lane B = gated stretch (opens only while A is on time). Pre-register the bar for calling Lane B a success (outside noise, strongest evaluation), else it reports as the honest lesser claim.
- **Reframe binary bets into 3-branch questions** (trait/state/both pattern): design the ambitious question so every answer is publishable.

## VI · ML/DL module (project-class specific)
- **Compute reality:** local RTX 5070 Ti (Blackwell sm_120 ⇒ torch cu128 only) + Colab Pro Education. ≤2 trainers per GPU; long runs in dedicated tmux; PC sleep OFF (`powercfg` standby+hibernate = 0) before any overnight run.
- **Colab disconnect-proof pattern:** mount Drive first · dataset cached on Drive (verify md5) · per-unit skip-if-done guards · copy results to Drive IMMEDIATELY per unit · resume-capable trainer (save model/optim/sched/epoch/rng).
- **Eval discipline:** evaluation parity (same attack/version/n for every row of a table); cheap eval for iteration, strongest eval (e.g. full AutoAttack) for any number leaving the repo; seeds 3 exploratory / 20 + Wilcoxon + BH-FDR for final claims; mean±std, never best-seed.
- **Bookkeeping:** W&B for curves; results as JSON per run; MANIFEST maps every paper claim/figure → source file → generating script; figures regenerate from disk (no hand-entered numbers), colorblind/grayscale-safe.

## VII · Writing & communication
- **Write-as-you-go:** draft skeleton exists from week 1 with `[PENDING]` slots; numbers pour in, structure doesn't wait.
- **Sources discipline:** every external claim has an anchor; confidence-marked (✅ verified / ⚠ verify-before-cite); never invent an ID; internal numbers cite their file.
- **Mentor/advisor cadence:** biweekly 5-line updates (status vs gates + the one blocker); proposals carry numbered questions per audience so feedback is extractable.
- **Claims are scoped** (dataset, architecture, protocol, tier) and negative results are reported as results.

## VIII · Retrospective protocol (how this file improves)
After each milestone/project, answer in ≤10 lines and append below with a date:
1. What worked (keep) · 2. What cost time without value (drop) · 3. What was missing (add) · 4. One rule to change in this playbook → bump version.

### Lessons ledger
- **2026-07 (AttackDRO):** pre-registration repeatedly prevented post-hoc rationalization (CARD-5: prediction wrong, finding right). Evidence-gating killed 3 detours for <1 GPU-hour total. Instrument validation surfaced two field-level evaluation artifacts (F1, F7) — validation is not overhead, it's a finding generator. Weak probes inverted a difficulty ranking (F5) — calibrate signals. Two agents (planner/executor) + human bridge works when files are the only sync channel and prompts are self-contained.
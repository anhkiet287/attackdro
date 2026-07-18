# Scripts cleanup v2 — CLAMP-only (approved 2026-07-18)

**Scope:** repo keeps CLAMP core method only. Allocation / reweighting / program-A /
RAMP-generality / non-paper probes → archived **outside** the repo. No new per-run scripts;
shared computations get one reused implementation.

## Archive location (outside the repo)
`/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO_archive/` (sibling). Working copies
move there; tracked-file deletions are committed, so git history keeps them recoverable too.

## Import-check (gate for Phase 1) — PASSED
- 0 KEEP scripts import any ARCHIVE module.
- 0 kept notebooks reference any ARCHIVE script.
- 0 ARCHIVE scripts in `attackdro_code.zip`.
- Path-level "refs" from KEEP → ARCHIVE were all false positives (upstream `train.py`,
  `src/train.py`, docstrings for the regenerable `RAMP_claimB.py`).

## Phase 1 — CLAMP-only repo (executed)
**KEEP `scripts/` (2):** `eval_multinorm_audit.py`, `make_dashboard.py`.

**KEEP `scripts/dev/` (17):**
- trainers: `c5_fromscratch.py`, `train_full_msd.py`, `finetune_msd_clamp.py`, `c0_killtest.py`
- eval/diag: `union_bench_eval.py`, `probe_attack.py`, `collapse_dump.py`
- stats: `c5_attribution.py`, `perclass_breakdown.py`
- Claim-B: `claim_b_rep.py`, `patch_ramp_claimb.py`
- artifact: `export_sanitize.py`
- util: `check_gpu.py`, `healthcheck.py`(+`.sh`), `backfill_wandb_from_trainjson.py`,
  `resume_continuity_check.py`, `smoke_ramp_loader.py`, `setup_wsl.sh`

**KEEP notebooks (3):** `train_colab`, `eval_colab`, `analysis_perclass_colab`.

**ARCHIVE OUT (~37 scripts + subdirs + 10 notebooks):** all program-A/allocation/reweighting/
predictive-reactive/B2-B3-B4/RAMP-generality tooling, one-off verdicts and config-guards, the
regenerable `RAMP_claimB.py`, `scripts/{legacy,figures,ramp}/`, top-level `train.py`/`evaluate.py`,
`notebooks/legacy/*` + `explore_colab.ipynb`.

## Phase 2 — de-duplicate for fairness (deferred until Colab tasks finish; verify-reproduce first)
One shared library, imported by scripts AND notebooks (not a new per-run script):
```
scripts/dev/clamp_stats.py
  union_mask(masks)                       # AND over 12
  per_norm(masks, norm)                   # AND over 4
  paired_bootstrap(a, b, B=10000, seed=0) # -> (delta, lcb95)   ONE definition
  per_class_breakdown(...)                # rows + sign test + bottom-3 lift
```
Replaces the 4× paired-bootstrap and 6× union-AND reimplementations so every run (local/Colab,
any arm/seed) uses identical stats. `c5_attribution.py` + `perclass_breakdown.py` become thin
CLIs over it; `eval_colab`/`analysis_perclass_colab` import it (shipped in the zip). **Before
switching, reproduce the current numbers through clamp_stats and diff — must match exactly.**
Rebuilds `attackdro_code.zip` + edits notebooks, so it is done between Colab runs, not during.

Optional same-pass merges: `probe_attack.py`+`collapse_dump.py` → `diagnostics.py`; move the
MSD-backbone loader out of `c0_killtest.py` so the finetune script stops depending on it.

## Untouched
`../pullpush-artifact` (verified reviewer repo) — not modified. `src/` package — out of scope.

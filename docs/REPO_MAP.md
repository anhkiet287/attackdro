# REPO_MAP — read this first

Document ID: `REPO-MAP` · Version: `1.0` · Governed by `RESEARCH_CHARTER.md`
Purpose: the entry index for every agent. **Consult this instead of scanning the tree.** It says what to read, what is generated, what is archived, and who owns each artifact — so handoffs cost few tokens.

---

## Read order (routine handoff)

1. `docs/STATE.md` — compact current state. **Read every handoff.** ~1 screen.
2. `docs/REPO_MAP.md` — this file. Where things live.
3. `docs/governance/` — only when a role/gate/process question comes up.
4. `docs/STATE_LOG.md` — **only** when you need a specific past fact (append-only history; large — do not read whole).

Do **not** read whole-repo, `results/`, `checkpoints/`, `wandb/`, or `STATE_LOG.md` in full for a routine task.

---

## Layer map

| Layer | Path | Status | Agent rule |
|---|---|---|---|
| Governance | `docs/governance/RESEARCH_CHARTER.md`, `AI_LAB_ROLES.md`, `AI_LAB_FLOW.html` | authority (process) | read on process questions only |
| Current state | `docs/STATE.md` | authority (science, compact) | read every handoff |
| History | `docs/STATE_LOG.md` | append-only record | read on demand, never in full |
| Source | `src/robustdro/` | active code | edit under contract only |
| Configs | `configs/paper/` | canonical lane | only canonical configs here — no DRAFT |
| Configs | `configs/exploration/` | parked ideas | not part of central claim |
| Configs | `configs/legacy/` | archived | **never import from here** |
| Scripts | `scripts/*.py` (root) | operational | train/eval/audit/gate entry points |
| Scripts | `scripts/dev/`, `scripts/figures/` | tools | analysis/plots, not protocol |
| Scripts | `scripts/legacy/` | archived | superseded — do not run |
| Prompts | `prompts/` | role prompts (4 roles) | how to task each agent |
| Results | `results/` | protected outputs | see `results/MANIFEST.md`; never mutate completed dirs |
| Artifacts | `checkpoints/`, `wandb/`, `dumps/`, `logs/` | protected/local | gitignored; retention per MANIFEST |
| External | `external/` | provenance (RAMP etc.) | read-only reference |
| Generated | `docs/dashboard.html`, `*.html` | GENERATED | **never treat as truth** |
| Archive | `docs/archive/`, `results/archive/` | historical | provenance only |

---

## Artifact ownership

- `results/MANIFEST.md` is the single index of runs: for each run — canonical? retention class? checkpoint role? reproducibility grade? safe-to-archive? **CODEX-RUNNER maintains it; nobody deletes a run without checking it.**
- Completed run directories are immutable. Audits/evals write to isolated output paths, never into a completed run dir.
- `val_best.pt` selected by `val_select/worst_union` is canonical; `best.pt` aliases are **not** safe for canonical use.

---

## One authority per concern (no competing truth)

| Concern | Single authority |
|---|---|
| Process / roles / gates | `docs/governance/` |
| Current scientific state | `docs/STATE.md` |
| Full history | `docs/STATE_LOG.md` |
| Run provenance | `results/MANIFEST.md` |
| Codex repo rules | `AGENTS.md` (thin pointer) |
| Claude Code rules | `CLAUDE.md` (thin pointer) |

If two files claim the same authority, the one listed here wins; the other must be archived.
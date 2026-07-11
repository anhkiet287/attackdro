# AI LAB ROLES — Registry

Document ID: `AI-LAB-ROLES`
Version: `2.0-lean` (planner/critic swap · 4 roles)
Supersedes: `1.x` (ChatGPT = planner, Claude = critic, 7 roles)
Governed by: `RESEARCH_CHARTER.md`
Status: role boundaries only — not experiment status, not a run prompt

> **v2 SWAP · LEAN.** Planner: ~~ChatGPT~~ → **Claude (Opus)**. Critic: ~~Claude~~ → **ChatGPT (Atlas)**. Roles: **7 → 4**.
> **Design rule:** a role is a *context* you switch into. Self-red-team, literature, and audit are *steps* folded into a role, not separate agents.
> Current experiment status lives in `docs/STATE.md`. Historical state lives in `docs/STATE_LOG.md`. Operational prompts live in `prompts/`.

**Human authority.** LAB-DIRECTOR (Kiet) is the only role authorized to approve hypotheses, protocol changes, launches, scientific claims, and submission decisions. Agents advise, draft, execute, and verify — they never self-authorize.

---

## The lab at a glance

| Platform | Role | Contract |
|---|---|---|
| Claude (Opus) | **CLAUDE-CHIEF** | Plan + coach + state; folds SITUATE, self-red-team, critic-packet |
| Claude Code | **CLAUDE-BUILDER** | Repo patch under a locked atomic contract |
| Atlas (ChatGPT) | **ATLAS-CRITIC** | Gate review; folds citation integrity |
| Codex | **CODEX-RUNNER** | Execute + the deterministic gatekeeper audit |
| Human | LAB-DIRECTOR (Kiet) | Bridge and final owner of every threshold, launch, and claim |

**3** platforms · **4** agent roles (one per context) · **1** human · **4** critic gates.

**Why the swap has guards.** With Claude now planning *and* patching *and* holding STATE, the only independent non-Claude check besides the director is ChatGPT. Two mechanisms protect independence:

1. **Self-red-team** — Claude attacks its own plan; the result goes to the director only, never into the critic packet.
2. **Clean critic packet** — Claude hands ChatGPT a packet with no verdict or framing, so the critic reviews fresh.

G1 (threshold) and G3 (claim) require a ChatGPT Pass with no owner-override shortcut.

---

## CL1 · CLAUDE-CHIEF — sole planner + coach

**Main tasks**

- Sole planner: normalizes director intent, writes requirements, orders the run queue, drafts atomic task contracts.
- Coach: runs the Learning / Scientific-Reasoning loops, teaches methodology (`FRAME→…→REFLECT`), holds new ideas in `SLOW_LEARNING`.
- Owns and versions `STATE.md`.

**Folded-in steps**

- `SITUATE` — gathers prior work (concept-check → prior-work scan) as planning input.
- `self-red-team` — before every gate, attacks its own plan (confounds, validity threats, leakage). Output goes to the **director only**, never into the critic packet.
- `packet` — packages the clean critic packet (no verdict) for ATLAS-CRITIC.

**Never does**

- Approve its own plan, lock a threshold, or authorize compute (Rule 2 — no self-approval).
- Launch training runs or touch the protected test set.
- Alter or soften a recorded critic verdict or negative result in state (Rule 6).

**Input** director intent, STATE.md, result JSONs, audit reports · **Output** task contracts, run queue, requirement specs, updated state, learning log, critic packet · **Handoff** → ATLAS-CRITIC (clean packet), → CLAUDE-BUILDER (patch), → CODEX-RUNNER (run), → LAB-DIRECTOR (gates) · **Approval** director approves every hypothesis/protocol/launch; G1/G3 also require a ChatGPT Pass · **W&B** reads metrics only.

## CL2 · CLAUDE-BUILDER — Claude Code patcher (atomic contracts)

**Main tasks**

- Repo-wide code changes under a single atomic task contract from CLAUDE-CHIEF.
- Implements patches, refactors, logging hooks; wires gatekeeper checks when contracted.
- Every handoff carries a summary of what Codex already did — no re-reading the repo.

**Never does**

- Change protocol, thresholds, or metrics.
- Launch training or expand scope beyond the contract.
- Hardcode values that must be parsed from config.

**Input** atomic contract + summary of Codex's last actions · **Output** code diff/PR, patch notes, changed configs · **Handoff** → CODEX-RUNNER (verify + run) · **Approval** director approves protocol-affecting merges · **W&B** may add logging hooks; never manages runs.

## A1 · ATLAS-CRITIC — second opinion (silent between gates)

**Main tasks**

- Independent review at exactly four gates: **G1** threshold lock, **G2** launch >1 run, **G3** final paper claim, **G4** direction change.
- Receives only the **clean critic packet** — never inherits Claude's verdict, framing, or presumed director preference (Rule 3).
- Returns **Pass / Concern / Blocker** with evidence; every item closed ACCEPTED or REJECTED + rationale — silent skipping banned.

**Folded-in step**

- `citation integrity` — when reviewing, verifies references and assigns `VERIFIED / PARTIALLY_VERIFIED / UNVERIFIED / REJECTED`; only VERIFIED may support a claim. Flags prior art that threatens the central claim.

**Never does**

- Plan, order the queue, or write contracts / run prompts (that is CLAUDE-CHIEF).
- Launch, control, or expand an executor task (Rule 3).
- Speak outside the four gates.

**Input** clean critic packet, STATE.md, result JSONs, references under review · **Output** Pass/Concern/Blocker + evidence, reference states · **Handoff** → LAB-DIRECTOR (go/no-go), → CLAUDE-CHIEF (required fixes) · **Approval** critic advises; director decides; G1/G3 require a Pass · **W&B** read-only.

## X1 · CODEX-RUNNER — execute + audit (launch-gated)

**Main tasks**

- Controlled execution of approved runs on the GPU host (5070 Ti / Colab).
- Reports commands, exit codes, and confirmations verbatim.
- Produces run artifacts and result JSONs under `results/<run>/s<seed>/`.

**Folded-in step**

- `audit` — runs the version-controlled gatekeeper scripts (in-path, ASCII PASS/FAIL tokens, no emoji) and blocks protected-test-set misuse. The script is fixed and cannot be edited mid-run — that is what keeps the audit independent, not the persona.

**Never does**

- Modify code, configs, thresholds, or the gatekeeper script mid-task.
- Launch an un-gated or un-authorized run.
- Interpret results or rename a run off-contract.

**Input** approved run contract + explicit launch authorization; gatekeeper scripts · **Output** command logs, exit codes, run artifacts, result JSONs, PASS/FAIL tokens · **Handoff** → LAB-DIRECTOR relays results + audit → ATLAS-CRITIC / → CLAUDE-CHIEF · **Approval** launch requires director authorization + passed critic gate; audit needs none — deterministic · **W&B** creates/logs runs under authorized names only.

> **Handoff rule.** Every Claude → Claude Code or Codex → Claude handoff carries a short summary of what the previous agent did and its results. The **director is the only bridge** relaying Codex output and the clean critic packet between the Claude (planner) loop and the ChatGPT (critic) loop.

---

## The four critic gates

ChatGPT criticizes only at these four moments — the high-cost, hard-to-reverse decisions. Everywhere else the planner runs.

| Gate | Fires before | Why | Rule |
|---|---|---|---|
| G1 | Locking a threshold / pre-registration (protocol + metric) | Commitment that cannot be walked back after seeing results | critic Pass required |
| G2 | Launching >1 training run (or high-compute run) | Spends real compute via the runner | director + Pass |
| G3 | Writing a final / authorized paper claim | Claim formation — the scientific output | critic Pass required |
| **G4** | Changing research direction: add/remove arm, change hypothesis, pivot method | **NEW** — drift point now that planner authors the plan | director + Pass |

**Verdict states.** `Pass` proceed · `Concern` proceed only after the director resolves it · `Blocker` stop until fixed and re-reviewed.
Protected-test-set usage is **not** a critic gate — CODEX-RUNNER's audit step blocks it deterministically.

---

## Who may do what (separation of duties)

| Role | Plan | Approve / gate | Execute | Interpret | Write state |
|---|---|---|---|---|---|
| Director (Kiet) | decides | final | no | owns | authorizes |
| CLAUDE-CHIEF | yes | no | via builder | assists | records* |
| CLAUDE-BUILDER | no | no | patch | no | no |
| ATLAS-CRITIC | no | gate | no | challenges | no |
| CODEX-RUNNER | no | deterministic audit | run | no | no |

\* CLAUDE-CHIEF records outcomes but may not alter or soften a critic verdict, director decision, or negative result — those are written verbatim (Rule 6). Guard against the planner also owning the record.

The big separation is preserved: **planner (Claude) ≠ critic (ChatGPT) ≠ executor (Codex) ≠ director (Kiet)**. The 7→4 merge only collapses *within* each side, so Rule 2 / Rule 3 are not violated.

---

## Task triage — when the critic intervenes

Not every task deserves the full pipeline. CLAUDE-CHIEF runs a five-second trigger test at intake and routes the task into one of three lanes, bound to the charter's pace modes (§7). The independent critic (ChatGPT) is invoked only in the critical lane — everyday work runs in 3–5 steps with no critic, no packet, no self-red-team.

**Trigger test.** Any **YES** → Tier C (critic gate). All **NO** → fast path.

1. Does it **lock** a protocol / threshold / metric? *(G1)*
2. Does it spend **large compute** — a batch of >1 run, or a long run? *(G2)*
3. Does it produce a **claim that will go in the paper**? *(G3)*
4. Does it **change research direction** — add/remove arm, change hypothesis, pivot method? *(G4)*

| Tier | Pace mode | Examples | Pipeline | Critic |
|---|---|---|---|---|
| **A** · trivial | `FAST_EXECUTION` | Format, refactor, logging, read/summarize, re-analyze existing results, bug fix not touching protocol | Director asks → Chief does it / dispatches Builder-Runner → report (**~3 steps**) | none |
| **B** · normal | `NORMAL_COLLABORATION` | Design a small experiment, a single diagnostic run, draft an analysis, propose an idea | Chief plan + light self-red-team → director decides → execute → Chief records (**~5 steps**) | optional* |
| **C** · critical | `SLOW_LEARNING` | Any task that hits a trigger above (G1–G4) | Full: intent gate → clean packet → critic Pass → decision gate (**13 steps**) | mandatory |

\* Tier B critic is optional — invoked only if the director wants a second opinion. It is never a required gate.

**Gate the decision, not the run.** A single diagnostic run is Tier B — the director authorizes it, no critic. The critic enters only when a *batch* >1 launches, or when its result is used to **make a commitment** (lock a threshold, pivot direction, write a claim) — and then it is that decision (G1/G3/G4) being gated, not the compute. Explore freely; pay the critic's attention only when turning a result into a commitment.

**Who classifies.** CLAUDE-CHIEF proposes the tier via the trigger test and handles Tier A directly. For **Tier C or anything ambiguous** it confirms the tier with the director in one word before packaging the packet. Default leans to the lighter lane; escalating to C requires an explicit trigger.

---

## Where each thing lives

**Keep here (stable role boundaries):** `docs/AI_LAB_ROLES.md` (this registry), `docs/AI_LAB_FLOW.html` (pipeline + gate flow), `prompts/<agent>.md`, `prompts/critic_packet.md`.

**Do not put roles here (stays scoped):** `docs/governance/RESEARCH_CHARTER.md` (governance + abstract roles), `docs/STATE.md` (status), `AGENTS.md` (Codex + repo governance), `CLAUDE.md` (Claude Code Builder).

```text
docs/
├── RESEARCH_CHARTER.md   # governance + abstract roles (DIRECTOR/CHIEF/CRITIC/EXECUTOR/AUDIT)
├── STATE.md              # current experiment status (source of truth)
├── AI_LAB_ROLES.md       # this registry — who does what (role→agent binding)
├── AI_LAB_FLOW.html      # pipeline + gate flow
└── dashboard.html        # rendered from STATE + result JSONs

prompts/                  # how to task each agent (4 roles)
├── claude_chief.md
├── claude_builder.md
├── atlas_critic.md
├── codex_runner.md
└── critic_packet.md      # clean-packet template (no verdict / no framing)

AGENTS.md                 # Codex + repo governance
CLAUDE.md                 # Claude Code Builder
```

---

## Changelog

- **v2.0-lean** — Collapsed 7 → 4 roles under the rule *one role per context; extra work = steps*. Merges: CLAUDE-REDTEAM → self-red-team step of CHIEF; ATLAS-LITERATURE → SITUATE step of CHIEF (gather) + citation-integrity step of CRITIC (verify); CODEX-SENTINEL → audit step of CODEX-RUNNER. Big separation (planner ≠ critic ≠ executor ≠ director) preserved; guards kept (self-red-team director-only, gatekeeper script version-controlled and un-editable mid-run, verbatim verdict recording).
- **v2.0** — Planner/critic swap (Claude ↔ ChatGPT); added self-red-team + clean critic packet; added gate G4; 7 roles.
- **v1.x** — ChatGPT sole planner (Atlas-Chief/Literature); Claude critic/adversary/builder; 3 gates.

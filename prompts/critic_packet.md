# TEMPLATE — CLEAN CRITIC PACKET

Packaged by CLAUDE-CHIEF, relayed by the director to ATLAS-CRITIC. This is the **only** thing the critic receives.
Rule 3: it must contain **no verdict, no framing, no confidence label, no self-red-team, no presumed director preference.** State facts and options; let the critic judge.

```
CRITIC PACKET · gate: G1 / G2 / G3 / G4 · date:

DIRECTOR_INTENT:
  <what the director wants, 1–3 lines>

RESEARCH_QUESTION:
  <the specific question this decision serves>

PROPOSED_ACTION:
  <exact action to be gated — e.g. lock threshold X; launch arms Y; claim Z>

RATIONALE:
  <why this action, neutrally stated>

ASSUMPTIONS:
  <explicit assumptions the action rests on>

SUPPORTING_EVIDENCE:
  <result JSONs, metrics, references — with locations; references carry no state yet>

ALTERNATIVES_CONSIDERED:
  <other options and why not chosen — no verdict>

SCOPE:
  <what is in scope>

PROHIBITED_ACTIONS:
  <what must not happen>
```

Critic returns: `Pass / Concern / Blocker` + per-item evidence + required fix. Director resolves every item before the decision gate.
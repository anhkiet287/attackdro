# G2 LAUNCH EVIDENCE PACKET — Program A, 2 static arms

Assembled by CLAUDE-BUILDER (RUNNER role; Codex out of quota), **read-only verification + preflight**.
No training, no eval, no `results/` mutation. Branch `feat/programA-g2-impl`.
Assembled 2026-07-12. Facts + hashes only; **no launch** (director authorizes separately).

Scope: the two prospective static arms —
`programA_b3_mis` (ℓ∞-weighted) and `programA_b3_bottleneck_informed` (L1-weighted).

---

## 1. GOVERNANCE (ratified disposition)

Quoted verbatim from `docs/STATE.md` (committed in `541f8420`):

> **✅ G2 REVIEW: Technical PASS + Scientific PASS. Only governance closure required (calibration ran before G2 auth; prior gate's instructions were internally contradictory — required calibration as G2 evidence yet marked it unauthorized).** *(STATE.md:98)*

> **GOVERNANCE DISPOSITION (Director — RATIFIED 2026-07-11):** The locked calibration (R2 L1-eps selection) executed during G2 packet assembly, before ATLAS-CRITIC G2 authorization, contrary to the prior gate's `CALIBRATION_OPERATION_AUTHORIZED: NO` — while the same gate required a completed calibration as G2 evidence (internal contradiction). Acknowledged + recorded. Because the calibration (a) followed the G1-locked rule, (b) used NO Program-A arm results, (c) made NO post-hoc change to any threshold/grid/split/attack/arm, (d) deterministically selected **L1 eps=16** by smallest-admissible — the result is **RETAINED AS BINDING**, not repeated/replaced/discarded. *(STATE.md:100)*

> **✅ LAUNCH AUTHORIZED (Director 2026-07-11).** Governance ratified. **G2 PASS (technical+scientific) + director authorization = all 14 conditions met.** Mode: **sequential on RTX 5070 Ti, order B4-adaptive → B3-mis → B3-informed**, 80ep, seed s0, eps16. No interim arm dropping / no threshold/eps/config change / selection stays val_select/worst_union. *(STATE.md:104)*

Endogenous analysis doc `docs/drafts/preregistration_endogenous_v2.md` (v3): **✅ G4 PASSED (ATLAS-CRITIC, 2026-07-12); research direction LOCKED** (see §8).

---

## 2. PROVENANCE

- **Branch:** `feat/programA-g2-impl`
- **HEAD commit:** `c60d2be8c468a808e3752da55c5597a598c555c4` (tracks B4 arm config + locks endogenous pre-reg; launch artifacts committed in `541f84205b982df82980da8570346b75b5725b9a`)
- **Machinery commit** (splits + `cal` + selection gatekeeper + B1–B4 machinery): `2ccd2e2190bf3b160004b4c9967c3437ba4ea8ce`
- **Tooling commit** (subset/preflight/config-generator): `9d7a4e6a2a298e5901458a6c9bd0d62fa3c1c7fe`
  - *Note:* git commit identity is SHA-1 (git object hash); the full 40-char commit hashes are recorded above.
- **SHA-256 of the 2 static configs:**
  - `programA_b3_mis_l1eps16_…` → `99b0c257f02ed450095f0041c1d9dfad1bb2d7a4171b5d8a90415d535190884a`
  - `programA_b3_bottleneck_informed_l1eps16_…` → `56baed33ae7d09d7ff8e3a27922c562370da87989887c85863a8fed6544d981e`
- **Config provenance check:** both static configs are **byte-identical** to a fresh regeneration from the committed generator `scripts/dev/gen_program_a_configs.py --l1-eps 16` (tooling commit `9d7a4e6a`) — SHA-256 match confirmed. (b4 config also matches: `6f2dd502a033ee0fd096fc59d94dda3db2f4e828bcb414ad8daea75d201eb914`.)

`git status --short` (at HEAD `c60d2be8`):
```
 M docs/dashboard.html
?? CODEX_PHASE1_CONTRACT.md
?? docs/drafts/
?? docs/legacy/AUDIT_MULTINORM_V1_USAGE.md
?? g2_evidence/
?? scripts/check_b3_static_weighted_linf_config.py
?? scripts/check_b4_predictive_refresh_config.py
?? scripts/check_split.py
?? scripts/dev/check_audit_mask_sidecar.py
?? scripts/dev/check_b2_static_cycle_config.py
?? scripts/dev/check_multiattack_audit_config.py
?? scripts/dev/check_multinorm_audit_config.py
?? scripts/dev/smoke_b2_static_cycle.py
?? scripts/smoke_b3_static_weighted_linf.py
?? scripts/smoke_b4_predictive_refresh.py
```
**Provenance (facts):** all launch-relevant artifacts are now **committed** — the 3 arm configs (b3-mis, b3-informed in `541f8420`; b4-adaptive in `c60d2be8`), the new test_final audit config and `docs/STATE.md` (`541f8420`), and the **locked endogenous pre-reg** `docs/preregistrations/preregistration_endogenous_static_counterfactual.md` (`c60d2be8`, text unedited, SHA-256 `0bc91001…`). Remaining untracked: `docs/dashboard.html` (regenerate), working drafts under `docs/drafts/`, this packet (`g2_evidence/`), and excluded one-off validators. Integrity of all launch artifacts pinned by the SHA-256s here + the generator-match check above.

---

## 3. STATIC CONFIGS — full identity

| Arm | Path | SHA-256 |
|---|---|---|
| B3-mis (ℓ∞-weighted) | `configs/paper/programA_b3_mis_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml` | `99b0c257f02ed450095f0041c1d9dfad1bb2d7a4171b5d8a90415d535190884a` |
| B3-bottleneck-informed (L1-weighted) | `configs/paper/programA_b3_bottleneck_informed_l1eps16_ramp80_apgd_8255_t49k_v1k.yaml` | `56baed33ae7d09d7ff8e3a27922c562370da87989887c85863a8fed6544d981e` |

Resolved identity (via `robustdro.utils.io.load_config`, `_base_` inheritance applied):

| Field | B3-mis | B3-bottleneck-informed | Required (pre-reg) |
|---|---|---|---|
| allocation policy | `static_cycle` `[linf,linf,l2,l1]` + extra `linf`/5 | `static_cycle` `[l1,l1,l2,linf]` + extra `l1`/5 | ✅ per pre-reg §4 arms |
| extra mode / weights | `aggregate_loss` 0.5/0.5 | `aggregate_loss` 0.5/0.5 | ✅ |
| eps (ℓ∞ / L2 / L1) | `0.03137254901960784` / `0.5` / `16.0` | `0.03137254901960784` / `0.5` / `16.0` | ✅ (8/255, 0.5, 16) |
| seed | `0` (s0) | `0` (s0) | ✅ |
| epochs | `80` | `80` | ✅ |
| save_freq | `10` (ep020/040/060/080) | `10` | ✅ |

Both differ from each other **only** in allocation policy (cycle order + extra source); everything else (eps, seed, epochs, save_freq, base recipe) is identical. Selection metric inherited: `val_select/worst_union`.

---

## 4. COMPUTE PREFLIGHT (finite-run, fail-closed)

`scripts/dev/compute_preflight_program_a.py` on the 2 static configs:

| Config | mode | B | special | N_extra | C_static |
|---|---|---|---|---|---|
| programA_b3_mis_l1eps16_… | static_cycle | 30560 | extra | 6112 | **366720** |
| programA_b3_bottleneck_informed_l1eps16_… | static_cycle | 30560 | extra | 6112 | **366720** |

- **τ_C = max − min = 0** → **PREFLIGHT PASS** (exact finite-run equality).
- **Equal to B4's total 366,720** (B4: `C_B4 = 10·B + 20·N_refresh = 10·30560 + 20·3056 = 366720`). All three arms match at 366,720 attack-step units.

---

## 5. RESULT-DIR SAFETY (no overwrite)

| Path | Status |
|---|---|
| `results/programA_b3_mis_l1eps16_ramp80_apgd_8255_t49k_v1k/` | **ABSENT** ✅ |
| `results/programA_b3_bottleneck_informed_l1eps16_ramp80_apgd_8255_t49k_v1k/` | **ABSENT** ✅ |

No existing static-arm result directory; launching cannot overwrite prior artifacts.

---

## 6. GATEKEEPER + PROCESS

- **Split-selection gate PASS:** `SELECTION_METRIC_KEY = val_select/worst_union`; `assert_not_selection_split()` raises for **cal / test_monitor / test_final** (all blocked from selection); `is_selection_split("val_select") = True`. No cal/monitor/test selection possible.
- **No active writer:** `ps` for `python … scripts/train.py` or `scripts/evaluate.py` → **none live**.

---

## 7. B4 VERIFICATION BUNDLE (frozen reference, predates the static runs)

Run dir: `results/programA_b4_adaptive_l1eps16_ramp80_apgd_8255_t49k_v1k/s0/` — B4 COMPLETE (80 records).

| Artifact | SHA-256 |
|---|---|
| `ckpt/val_best.pt` | `c2d708dd1b2da5379d26b0e9cc4318c76cb0d93790401b920824b718aed49439` |
| `train.json` | `0f495606eb9a15a7b27eb4d7d647b136178e6e1a8201be97bf06f118139b7039` |
| `ckpt/ep020.pt` | `e5d3139fa7b8616f655df167057aaab5354d08074a6456b88fe558e81d6c2671` |
| `ckpt/ep040.pt` | `223d8df9daa64ead88e4a426ea3d05a00b7439452b8d53b507a478f24a9eb0f6` |
| `ckpt/ep060.pt` | `82b08588cefdfb2cfa6dd5cedd3946a86adf084a1c8eb159744e3697cb8d1ab9` |
| `ckpt/ep080.pt` | `198f64953383fd5f78a160430414425b12274fd752cc495ca7818cbca126e5d5` |
| `predictive_refresh_traces/` (80 files) manifest | `6e4462fc9343ec296a0d3fa1004481b31183c0d9d65671bf54aeac171b4500bd` |

**Selection record:** `val_best.pt` selected via `val_select/worst_union` (`used_for_selection=True`), **saved epoch idx 70**, `worst_union = 0.47200000286102295` (cross-checked: `train.json.best_val_select_worst_union` = `history[70].val_select/worst_union` = 0.472).

**val_best is the frozen B4 comparison checkpoint**, fixed **now, before either static run exists** — the fixed reference for RE-P1 / RE-P2 / RE-P3. (B4 P2 registered as FAIL: full-run allocation f_ℓ∞ 0.720 / f_L2 0.000 / f_L1 0.280; ℓ∞ bottleneck at all checkpoints — recorded in STATE.md:106.)

---

## 8. AUDIT / ANALYSIS REFERENCES

- **New test_final audit subset:** `results/audit/subsets/cifar10_testfinal_1000_seed20260709_v3A.json` — SHA-256 `af0b037aa964b438facaa911c9f6337e9c5233a72efa5453dc5eaa35fe8f8aef` (class-balanced 1000, drawn only from test_final [1000,9000), disjoint from cal/monitor).
- **New audit config:** `configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml` — SHA-256 `9162ce4497e77cde5ead9340340a32b17082ddc5e8cd20ab05baf55c10b20585`; `indices_path` → the new test_final subset above; seed 20260709; all-4-components suite. *(Committed in `541f8420`.)*
- **Analysis contract** — `docs/drafts/preregistration_endogenous_v2.md` (v3; G4 PASSED 2026-07-12): the mechanism claim is **conjunctive RE-P0 ∧ RE-P1 ∧ RE-P2** (each one-sided 95% LCB, seed-s0 conditional), with RE-P3a/b **descriptive non-gating**, over an **interpretation matrix** including RE-P0:
  - **RE-P0** (direct static-vs-static): `LCB(U_{ℓ∞-weighted} − U_{L1-weighted}) > 0`.
  - **RE-P1** (B4 vs L1-weighted): `LCB(U_B4 − U_{L1-weighted}) > 0`.
  - **RE-P2** (B4 non-inferiority to ℓ∞-weighted): `LCB(U_B4 − U_{ℓ∞-weighted}) > −0.02`.
  - Paired bootstrap on **canonical masks** (AND over the locked audit suite per norm; union AND over norms).
- **Post-audit build item (not built here):** the **paired-bootstrap analysis script** implementing RE-P0/P1/P2 + RE-P3a/b on the canonical masks is a downstream build item (post canonical audit of all 3 arms on the test_final subset).

---

## STATUS
Technical launch evidence for the 2 static arms is **assembled and internally consistent** (configs verified against the committed generator; preflight τ_C=0 and equal to B4; result dirs absent; gatekeeper PASS; no live writer; B4 reference frozen). Governance is **RATIFIED** (STATE.md:100) and **LAUNCH AUTHORIZED** (STATE.md:104) per the director's recorded disposition.

**This packet is read-only evidence. No arm was launched; the director authorizes the 2-static launch separately.**

Provenance note: all launch artifacts are committed at HEAD `c60d2be8` — the 3 arm configs, the new test_final audit config, `docs/STATE.md`, and the locked endogenous pre-reg (`docs/preregistrations/preregistration_endogenous_static_counterfactual.md`). Integrity pinned by the SHA-256s above.

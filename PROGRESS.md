
## 2026-07-02 — P1 Step 3b: SEED-0 APGD RESULT (provisional, 1 seed)
AttackDRO++ (GroupDRO over linf+l2+l1) seed 0, locked protocol (eps_inf=0.03, n=1000, apgd), best ckpt @epoch 33:

| model | clean | l_inf | l2 | l1 | worst-union |
|---|---|---|---|---|---|
| AttackDRO++ s0 | 80.7 | 46.2 | 63.1 | 41.2 | **38.7** |
| AVG (baseline) | 84.6 | 40.7 | 65.5 | 47.7 | **38.7** |
| MSD (baseline, target) | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |

- **NO-GO vs MSD (−3.8); exact TIE with AVG (38.7).** q_final=[l_inf 0.83, l2 0.00, l1 0.17].
- **Mechanism (why):** GroupDRO weights by per-group AVERAGE loss → piled onto l_inf (highest loss) → best l_inf (46.2, beats MSD) but **starved l1 (q=0.17) → weakest l1 (41.2, loses to MSD 46.7)**. l1 is the union-binding norm; the DRO signal is MISALIGNED with the worst-union objective. l2=63.1 despite q_l2=0.00 → l2 robustness is FREE (spillover), so anti-starvation/floor fixes waste capacity.
- **P2 direction (from seed-0): attack BINDING-AWARE first** (reweight toward per-sample worst norm / union-binding), NOT floor/CVaR. Confirms the P1 thesis-probe expectation: difficulty-aware DRO over attack-groups ≈ AVG on worst-case, does not reach MSD's per-sample worst-case handling.
- Confounds: 1 seed; our recipe vs locuslab recipe for baselines; apgd not `standard`. Seeds 1–2 running for variance; final human GO/NO-GO after they land.

### 2026-07-02 — AttackDRO++ SEED-0 APGD result (first GO/NO-GO signal)
Best ckpt @epoch33, n=1000, APGD-CE+T. q_final [linf,l2,l1]=[0.832, 0.000, 0.168].

| model | clean | l_inf | l2 | l1 | worst-union |
|---|---|---|---|---|---|
| AttackDRO++ s0 | 80.7 | 46.2 | 63.1 | 41.2 | **38.7** |
| AVG (baseline) | 84.6 | 40.7 | 65.5 | 47.7 | 38.7 |
| MSD (target)   | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |

**Verdict (provisional, 1 seed): NO-GO vs MSD** — union 38.7 = ties AVG, −3.8 vs MSD. Expected (thesis improved avg-case, not worst-case) → triggers P2.

**P2 direction from this seed → BINDING-AWARE first (not group floor/CVaR):**
- l2 group starved (q=0) yet l2 acc highest (63.1) → l2 is "free"; group-level floor/CVaR would spend capacity on an already-solved norm = mis-targeted.
- q collapsed to ~fixed (0.83/0/0.17) → GroupDRO degenerated into a weighted (linf+l1) AVG → lands exactly at AVG (38.7), can't reach MSD.
- worst-union 38.7 < min per-norm (l1 41.2) → per-sample binding norm varies; a single per-group scalar q can't capture it. This is precisely MSD's edge (per-sample max-loss over norms).
- Binding norms of worst-union: l1 then linf; leave l2 alone.
- Caveats: 1 seed; APGD (not full AA); recipe confound (ours vs locuslab). Seeds 1/2 running for variance; full auto-verdict appended when they finish.

## 2026-07-02 — P2 RE-ROUTE (after seed-0 verdict)
Seed-0 accepted: NO-GO vs MSD, exact tie AVG (38.7) — expected branch. Re-routed P2:
- **CARD-4 (qfloor) FORMALLY OFF** — gate ℓ2<45% not met (ℓ2 APGD=63.1, free/spillover). F3 rewritten in PROJECT_MEMORY §5.
- **CARD-2 (cvar) DEPRIORITIZED** — seed-0 indicts group-level scalar signals (q-by-loss piled on ℓ∞, under-weighted binding ℓ1); CVaR same class. Fallback only.
- **CARD-3 split** → **CARD-3a** group binding-aware `q∝softmax(−robust_acc_g/τ)` (**launched, 1-seed pilot**) + **CARD-3b** per-sample soft binding-aware (temperature-softmax over per-sample per-norm losses, interpolates AVG↔MAX, NOT hard-max since MAX=25.0 fails) — **drafted per E1, awaiting Kiet approval before code/launch**.
- **CARD-1 avg_frozen** (freeze q → AVG in-house, kills recipe confound) — queued after p1.
- New F4 in §5: degeneration-to-AVG tie + signal↔objective misalignment.

**Code (verified):** `dro.freeze_q` (CARD-1) + `dro.weight_signal=robust_acc`,`dro.tau` (CARD-3a) in GroupDRO/trainer; unit-checked q→weakest-norm; both configs smoke-passed. `configs/{avg_frozen,bindaware}.yaml`.
**Running:** `p1pipe` (P1 seeds 1–2, variance) + `p2pipe` (CARD-3a pilot now → then CARD-1 avg_frozen ×3). VRAM fine (15.7 GB free; each run ~1.4 GB). All eval = eval_union APGD n=1000.
**Next:** ping when CARD-3a pilot APGD lands (~4h, contended) with worst-∪ / per-norm(ℓ2) / q / vs 38.7 & 42.5 & CARD-3a-vs-P1.

## 2026-07-02 — CARD-3b approved + implemented (3 amendments)
- **Code:** `objective: per_sample_soft` in GroupDROTrainer — per-sample CE per norm → [G,B] matrix → temperature-softmax over norms (detached weights) → weighted mean. Unit-checked AVG↔MAX interpolation (T=100→AVG 1.23, T=0.01→MAX 2.03, monotonic). Amendment #1: per-norm loss distributions (std/p10/p50/p90) logged every epoch (`dist/*`); `normalize_losses: zscore` variant implemented, evidence-gated. Generic `--set KEY=VALUE` added to train.py for the sweep. Both variants smoke-passed.
- **Scheduling (per Kiet):** T-sweep {0.25,0.5,1,2} 1-seed pilots armed in tmux `p3bsweep` — waits for 3a pilot APGD, then retires p2pipe (avg_frozen local stage cancelled) and runs sweep sequentially (≤2 trainers on GPU). Best T → 3 seeds.
- **CARD-1 → Colab:** `notebooks/colab_avg_frozen.ipynb` ready (clone → smoke → 3×train+eval → table → Drive). **BLOCKER: repo not pushed since initial scaffold — needs Kiet OK to commit+push.**
- **Lit-check (amendment #2, preliminary):** closest prior = Adaptive Smoothness-weighted AT (arXiv 2210.00557, per-TYPE weights); Tramèr&Boneh AVG/hard-MAX = the two poles; MSD = per-STEP selection (different axis); E-AT = fine-tuning. Per-sample temperature-softmax not directly found → frame as mechanism probe, read 2210.00557 before any novelty claim. New decision branch recorded: 3b>3a but <42.5 ⇒ remaining MSD edge = per-step selection inside attack gen (publishable mechanism finding).

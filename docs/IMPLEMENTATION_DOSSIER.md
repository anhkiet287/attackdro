# IMPLEMENTATION & RESULTS DOSSIER — CLAMP

Evidence document for external methodology audit without repo access.
Every value carries `⟵ source`. Nothing is reconstructed from the paper or from memory.
`⚠ UNTRACED` = exists only in prose/ledger, not derivable from code or results files.
`PENDING` = not yet run. `⚠ DISCREPANCY` = two sources disagree; both shown.

Generated 2026-07-22T04:57:52Z.

---

# FILL-LIST — values the paper is blocked on

Flat, paste-ready. Every line: `value  ⟵ source`. `NOT RUN` = the experiment does not exist and the
row should be dropped or marked future work — there is nothing pending to wait for.

## App B — harness pin (`§app:harness`)

- **AutoAttack library**: version `0.1`, installed from `https://github.com/fra31/auto-attack.git`,
  git commit **`a39220048b3c9f2cca9a4d3a54604793c68eca7e`**
  ⟵ `.venv/lib/python3.12/site-packages/autoattack-0.1.dist-info/{METADATA,direct_url.json}`
  (the package exposes no `__version__`; the commit id is the authoritative pin)
- **APGD-CE**: **100 iterations, 1 restart** ⟵ `audit_v3A_test_10k.yaml` `apgd_ce_{linf,l2,l1}: steps:100, restarts:1`; applied at `scripts/eval_multinorm_audit.py:342-346`
- **APGD-DLR**: **100 iterations, 1 restart** — ⚠ **this is `apgd-dlr` (UNTARGETED DLR), not `apgd-t` (targeted)** ⟵ `scripts/eval_multinorm_audit.py:46` `"apgd_dlr_linf": "apgd-dlr"`. If the paper writes "APGD-T", that is **wrong** and must be corrected to APGD-DLR. AutoAttack in this env supports `apgd-t`; it was simply not selected.
- **FAB-T**: **100 iterations, 1 restart**, targeted ⟵ config `fab_t_*: type: fab, targeted: true, steps:100, restarts:1`; `"fab_t_linf": "fab-t"` ⟵ `:47`; applied at `:351-352`
- **Square**: **5000 queries, 1 restart** ⟵ config `square_*: queries:5000, restarts:1`; applied at `:353-355`
- **Fixed evaluation seed**: **20260709** — used for every attack *and* for subset construction ⟵ config `seed: 20260709` in all 12 attack blocks and in `subset:`
- **Frozen harness config hash (@10k)**: **`4465ab20738e475494f4d08c4606b13a8f47f0e5664e64f4be51b7a989c088c1`** (sha256 of `results/eval/union_bench/_config/audit_v3A_test_10k.yaml`)
  ⚠ The often-quoted `9162ce44…` is the **@1k** config (`configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml`), full sha `9162ce4497e77cde…`. **Headline @10k numbers were produced under `4465ab20`.**
- **ε**: ℓ∞ `0.03137254901960784` (= 8/255), ℓ₂ `0.5`, ℓ₁ `12.0` ⟵ config `eps:`
- **Subset @10k**: n = 10000, `results/audit/subsets/cifar10_test_10000_full_v3A.json`, indices = `[0…9999]` (the complete CIFAR-10 test set in natural order)
- **ℓ₁ components = Croce & Hein 2021 sparse variant: CONFIRMED.** `autoattack.autopgd_base` contains `def L1_projection(x2, y2, eps1)` and calls `delta = L1_projection(x, t, self.eps)`; `autoattack.square` carries its own `L1` branch with the `p_selection` schedule. The repo has **no custom ℓ₁ code**; `norm='L1'` is passed straight to the library ⟵ `scripts/eval_multinorm_audit.py:43, 366`.

## `tab:bases` — Δclean column (@10k, tier=full)

| row | base arm | clean (base) | +term arm | clean (+term) | **Δclean** |
|---|---|---|---|---|---|
| MSD-10 | M0 | 0.8379 | M1a | 0.8164 | **−0.0215** |
| MSD-50 | M0_full | 0.8055 | M1a_full | 0.8063 | **+0.0008** |
| MAX | M0_max | 0.8138 | M1a_max | 0.8173 | **+0.0035** |
| FT-∞ | ft_none | 0.8662 | ft_clamp | 0.8542 | **−0.0120** |
| AVG | M0_avg | 0.8417 | M1a_avg | 0.8331 | **−0.0086** |
| RAMP | Rprime | 0.8118 | **B1** | 0.8010 | **−0.0108** |
| clean-base | — | — | M1a_cleance | **NOT RUN** | **NOT RUN** |

⟵ `results/main/<arm>/10k/eval.json` key `clean_acc`.
The MSD-10 value **−0.0215** is the paper's "−2.1". AVG lands with the gating run.
**RAMP `+term` = `B1`, and it was mislabelled in this repo.** `B1` is **RAMP + pull-push**, not a plain
RAMP baseline: its checkpoint path is `Bet1_out/B1_pullpush_seed0/val_best.pth` and its training loss goes
**negative** (−0.4413 at epoch 80 ⟵ `log_train.txt`), which a pure CE objective cannot do — only the glue term
(`−cos/τ ≈ −9`) can. The matched base is `Rprime` = `Rprime_out/R_prime_ramponly_seed0/val_best.pth`.
`B2` in this repo is a **different experiment** — `b2_static_cycle_ramp80_apgd_8255_t49k_v1k`
(allocation-arc static-cycle; own results dir + 5 offline W&B runs) — and has nothing to do with CLAMP.
The stray empty `results/eval/union_bench/B2/` was **deleted 2026-07-22** (via `rmdir`, which refuses a
non-empty directory); the static-cycle results themselves are untouched. See §6.2a for the corrected pair.
(AVG @1k no_square, screening tier, do not mix: base 0.8360, +term 0.8300, Δ −0.0060.)
**All rows now landed except the two marked NOT RUN.**

## Minors flagged by R2

- **Single-view ablation uses the MSD view**, not ℓ∞/ℓ₂/ℓ₁. Arm `M1a_msdglue` runs `--glue-view msd`, described in its own record as `"1 MSD view (x_MSD, H3 ablation)"` ⟵ `scripts/dev/c5_fromscratch.py:735`. A single-**ℓ∞**-view arm exists as a code path (`--glue-view linf`, "redundancy ablation #14" ⟵ `:737`) but was **NOT RUN**.
- **Selection-parity 2×2** (full-budget pair, @1k, tier=full, 12 attacks):

  | | val_best | final (last epoch) |
  |---|---|---|
  | CLAMP-50 (`M1a_full`) | **0.4340** | **0.4320** |
  | MSD-50 (`M0_full`) | **0.4350** | **0.4320** |
  | Δ (CLAMP − control) | **−0.0010** | **0.0000** |

  ⟵ `results/main/M1a_full/{1k,last}/eval.json`, `results/main/M0_full/{1k,last}/eval.json`.
  clean: CLAMP 0.8080 / 0.8260, control 0.8170 / 0.8370. The reversal's **sign does not depend on the
  selection rule**; at n=1000 both rules give Δ≈0 (the effect is ~1 pp, the @1k CI half-width ~1.5 pp).
- **Per-pair hardware/software identity** — see §0.3. Summary for the pairs the paper reports:
  - CLAIM A pairs (M1a/M0 seeds 0,2,3), M1a_full s0, M1b, M1a_msdglue, Rprime, B1, FT pair: **machine ⚠ UNTRACED** (no local execution log; several ckpts were downloaded from Drive).
  - Reversal pair seed 1 + M0_full: **both local, same box** (5070 Ti, torch 2.11.0+cu128, CUDA 12.8).
  - **MAX pair is cross-machine**: `M1a_max` trained and audited on **Colab** (env unrecorded), `M0_max` trained and audited **locally**. Agreement check: @1k no_square Δ = −0.0230, @10k full Δ = −0.0224.
  - AVG pair: `M1a_avg` local, `M0_avg` trained local + audited on **Colab** (ckpt sha `78b02da344ed…` verified identical before the audit; eval.json cross-checked against masks, 12/12 attacks exact).
  - **No cudnn determinism flags are set anywhere** ⟵ grep over `scripts/`; training is not bit-reproducible on any machine.
- **FT base data recipe: CONFIRMED Sehwag proxy-data.** Base model `Sehwag2021Proxy_R18` (RobustBench, ResNet-18), published reference clean 84.59 / AA-ℓ∞ 55.54 ⟵ `notebooks/eval_colab.ipynb` cells 638, 649. Both FT arms fine-tune from this same checkpoint.

## Ablation rows never run — safe to drop or mark future work

- **clean-CE (`M1a_cleance`)**: **NOT RUN.** No directory under `results/fromscratch/`; training cells are wired in `notebooks/train_colab.ipynb` but never executed.
- **α/β sweep**: **NOT RUN.** Every `train.json` in the repo carries `(alpha, beta) = (0.5, 0.5)` — the only pair ever used ⟵ scan of all `results/fromscratch/**/train.json`.
- **F1 exposure control / F2 strength-matched views / M5 pull-only & push-only**: **NOT RUN**, greenlight rescinded 2026-07-22 before any job started (`results/fromscratch/C5_R2/` is empty). Code paths exist (`--ce-views-w`, `--apgd-view-steps`, `--alpha/--beta`) and `M1a_full_v50` (CLAMP-50 with 50-step views) is **already trained but never audited @10k**.

## ⚠ Two data-quality flags a reviewer can hit

1. **`M1a_full/10k/eval.json` has `per_attack = {}`** — it was entered from a Colab paste; only the union, clean and per-norm fields survived. Its masks *are* local (12 keys) and the union recomputes exactly, so every paired statistic is sound, but **the 12 per-attack accuracies for the headline reversal arm are not on file** ⟵ `results/main/M1a_full/meta.json` provenance `"from user paste (Colab). masks ARE local"`.
2. **`M1b/1k` and `Rprime/1k` have no masks** — eval.json only; excluded from every bootstrap.

---

## 0. Provenance

| item | value | source |
|---|---|---|
| repo commit | `7ac786a668243f6da9acc025f23fa2000a05e357` | `git rev-parse HEAD` |
| branch | `feat/programA-g2-impl` | `git rev-parse --abbrev-ref HEAD` |
| working tree | **DIRTY — 44 modified/untracked paths** | `git status --porcelain \| wc -l` |
| audit schema | `audit_cifar10_preactrn18_multinorm_v1` | `scripts/eval_multinorm_audit.py:33` |
| mask schema | `masks_multinorm_v1` | `scripts/eval_multinorm_audit.py:34` |

### 0.1 Harness SHA — correction to the brief

`9162ce44` is **not** a single harness SHA; it is the sha256 of the **@1k** config file only.

| config | sha256[:16] | source |
|---|---|---|
| `configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml` (**@1k**) | `9162ce4497e77cde` | computed over file bytes |
| `results/eval/union_bench/_config/audit_v3A_test_10k.yaml` (**@10k**) | `4465ab20738e4754` | computed over file bytes |

**⚠ The headline @10k numbers were produced under config `4465ab20`, not `9162ce44`.** Both files declare the
same `name:` (`audit_cifar10_preactrn18_multinorm_v1`) because `validate_config` pins that field to
`SCHEMA_VERSION` ⟵ `scripts/eval_multinorm_audit.py:152-153`; the schema name is therefore **not** a content hash
and does not distinguish the two tiers. The produced `eval.json` files record `checkpoint_sha256` but **no
config hash field** ⟵ inspected keys of `results/main/*/10k/eval.json`.

### 0.2 Environment (local box)

| item | value |
|---|---|
| python | 3.12.3 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA / cuDNN | 12.8 / 91900 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| autoattack | `0.1`, installed from `git+https://github.com/fra31/auto-attack.git` ⟵ `notebooks/eval_colab.ipynb` setup cell |
| platform | `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39` |

⚠ **Colab environment not captured.** Runs executed on Colab (§0.3) used whatever torch/CUDA that runtime
provided; no version record was written into any result file. This is an open reproducibility gap.

### 0.3 Machine attribution per run

| run | machine | evidence |
|---|---|---|
| `M0_max`, `M1a_avg`, `M0_avg` | local 5070 Ti | training + eval lines in `scratchpad/local_queue.log` |
| `M0_full`, `M0_full_seed1`, `M1a_full_seed1` | local 5070 Ti | `scratchpad/seed1_chain.log`; `train.json` written locally with `epochs_completed` |
| `M1a_max` (train) | Colab | no local `train.json`; ckpt supplied via Drive link |
| `M1a_max` @10k audit | Colab | run stdout pasted by director; `eval.json` + masks transferred via Drive |
| all other @10k audits | local 5070 Ti | `results/eval/union_bench/*/10k/` written in-place |
| `M1a` s0/s2/s3, `M0` s2/s3, `M1a_full` s0, `M1b`, `Rprime`, `B1`, C100 pair | **⚠ UNTRACED** | `train.json` present locally but no log proves where it executed; `M0` seed0, `Rprime`, `B1` ckpts were *downloaded* from Drive, implying non-local training |

### 0.4 Determinism

| control | status | source |
|---|---|---|
| training seed | `--seed` **required** (no default) | `scripts/dev/c5_fromscratch.py:356` |
| attack seed (audit) | `seed: 20260709` per attack entry | `configs/eval/…_testfinal.yaml`, each attack block |
| subset seed | `subset.seed: 20260709`, `class_balanced: true` | same config |
| bootstrap seed | `np.random.default_rng(0)` | `scripts/dev/c5_attribution.py:29`, `scripts/dev/t1b_report.py:19` |
| val-proxy attack | starts from the clean point, **no random init ⇒ consumes no RNG** | `scripts/dev/c5_fromscratch.py:262-266` (docstring) |
| logging-only attacks | RNG-isolated via `torch.random.fork_rng()` | `scripts/dev/c5_fromscratch.py` per-norm train logging |
| **cudnn determinism flags** | **⚠ NOT SET** — no `torch.backends.cudnn.deterministic` / `benchmark` anywhere | grep over `scripts/` returns nothing |
| **dataloader worker seeding** | `--num-workers` default 4 (env `C5_NUM_WORKERS`); no `worker_init_fn` seeding found | `scripts/dev/c5_fromscratch.py:382` |

**Known nondeterminism:** cuDNN algorithm selection is unpinned, so bit-exact re-runs are not guaranteed even at
a fixed seed. This affects *training* reproduction; it does not affect any *audit*, which is deterministic given
a fixed checkpoint and fixed attack seeds except for Square's internal sampling (seeded per attack).

---

## 1. Repo map

| path | role |
|---|---|
| `scripts/dev/c5_fromscratch.py` | primary trainer (all 80-epoch from-scratch arms; CLAMP + all base/glue ablations) |
| `scripts/dev/train_full_msd.py` | full-budget trainer (Maini 50-epoch, MSD-50 base) — separate script |
| `scripts/dev/finetune_msd_clamp.py` | FT-∞ fine-tuning from a RobustBench base |
| `scripts/eval_multinorm_audit.py` | **frozen** 12-attack audit harness; writes `eval.json` + mask sidecar |
| `scripts/dev/union_bench_eval.py` | alternate audit driver; the only route to `--skip-square` (9-attack tier) |
| `scripts/dev/eval_arm.py` | single-arm CLI; `--scale {1k,10k}` and `--tier {full,no_square}` both **required** |
| `scripts/dev/sync_results_main.py` | normalises `union_bench/` → `results/main/`; re-verifies union vs masks |
| `scripts/dev/t1b_report.py` | 2×2 base-generality paired table @10k full |
| `scripts/dev/c5_attribution.py` | paired sample-level bootstrap (canonical definition) |
| `scripts/dev/t2_geometry.py` | RQ3 encoder-space geometry (pre-registered) |
| `scripts/dev/t2b_sensitivity_control.py` | post-hoc random-perturbation control for T2 |
| `scripts/dev/t2c_class_separation.py` | post-hoc class-separation / Fisher ratio |
| `scripts/dev/t2d_head_space.py` | RQ3b head-space (h) vs encoder (f) |
| `scripts/dev/t2e_head_attackstrength.py` | post-hoc train↔eval decomposition of head alignment |
| `scripts/dev/t3_threat_dominance.py` | per-norm responsibility among union failures (0 GPU) |
| `scripts/dev/t5_proxy_decompose.py` | val-proxy over-read decomposition |
| `scripts/dev/margin_analysis.py` | RQ4 clean-margin vs blocker-count (**results are a stub — see §9**) |
| `scripts/dev/mask_analytics.py` | RQ4 blocker census / flip analysis |
| `scripts/dev/procctl.py` | strict process matcher (operational; not science) |
| `configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml` | @1k audit config |
| `results/eval/union_bench/_config/audit_v3A_test_10k.yaml` | @10k audit config |

**No plotting scripts exist.** All figures referenced by the paper are `⚠ UNTRACED`.

### 1.1 How a matched pair is produced — exact code-level toggle

`--variant M0` vs `--variant M1a` ⟵ `scripts/dev/c5_fromscratch.py:354`.

```
578  if a.base == "msd":     lce = F.cross_entropy(model.logits(x_base), y)
580  elif a.base == "clean": lce = F.cross_entropy(model.logits(x), y)
582  else:                   lce = ce_at_loss(model, xs_ce, y, agg=("avg" if a.base == "avg" else "max"))
```

The CLAMP term is `decoupled_pullpush(...)` ⟵ `:99`, returning `alpha*scaffold + beta*glue` ⟵ `:148`.
Under `M0` the term is not constructed at all (`pullpush` false), described in the written record as
`"L_CE(base) ONLY — pure MSD-AT matched control (no head/positives/scaffold/glue)"` ⟵ `:712`.

**Consequence for pairing:** the control does **not** merely zero the loss weight — it skips crafting the
3 per-norm views entirely (`xs_adv = None`). The pair therefore differs in **compute and in RNG consumption**,
not only in the loss. See §10.

---

## 2. Training configs

Common to every `c5_fromscratch` arm ⟵ `scripts/dev/c5_fromscratch.py`:

| item | value | line |
|---|---|---|
| optimiser | `SGD(lr=a.lr, momentum=0.9, weight_decay=5e-4)` | `:434` |
| schedule | `MultiStepLR(milestones=a.milestones, gamma=0.1)`, default `[70]` | `:435`, `:360` |
| lr | 0.05 | `:359` |
| batch | 128 | `:358` |
| epochs | 80 | `:357` |
| α, β | 0.5, 0.5 | `:361-362` |
| τ | 0.1 | `:364` |
| warmup | 10 epochs, linear `min(1, ep/warmup)` | `:363`, `:454` |
| projection head | `Linear(512,512) → ReLU → Linear(512,128)` | `:73` |
| train split | `train[0:49000]` | `:51` |
| val split | `train[49000:50000]` | `:52` |
| MSD steps | 10 (`--msd-steps`) | `:378` |
| per-norm APGD view steps | 10 (`--n-iter`) | `:379` |

Per-arm, read from each run's `train.json` (values below are from the file, not the CLI):

| arm | variant | base | glue_view | neg | ep | α/β | τ | wu | msd_steps | seed | val_best epoch | best valWU | machine | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1a s0 | M1a | msd | *(null)* | adv | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 0 | **71/80** | 0.4730 | ⚠ | `results/fromscratch/C5/M1a_advneg/seed0/train.json` |
| M1a s2 | M1a | msd | *(null)* | adv | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 2 | 71/80 | 0.4620 | ⚠ | `…/M1a_advneg/seed2/train.json` |
| M1a s3 | M1a | msd | *(null)* | adv | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 3 | 71/80 | 0.4610 | ⚠ | `…/M1a_advneg/seed3/train.json` |
| M0 s2 | M0 | msd | *(null)* | none | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 2 | 72/80 | 0.4470 | ⚠ | `…/M0_matched/seed2/train.json` |
| M0 s3 | M0 | msd | *(null)* | none | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 3 | 71/80 | 0.4330 | ⚠ | `…/M0_matched/seed3/train.json` |
| **M0 s0** | — | — | — | — | — | — | — | — | — | 0 | **⚠ no local train.json** | — | Colab (ckpt downloaded) | ckpt sha `e1dd1dac7985` matches its audit |
| M0_max | M0 | max | 3norm | none | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 0 | 71/80 | 0.4770 | local | `results/fromscratch/C5_ablations/M0_max/train.json` |
| M1a_avg | M1a | avg | 3norm | adv | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 0 | 75/80 | 0.4390 | local | `…/C5_ablations/M1a_avg/train.json` |
| M0_avg | M0 | avg | 3norm | none | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 0 | 75/80 | 0.4060 | local | `…/C5_ablations/M0_avg/train.json` |
| **M1a_max** | — | — | — | — | — | — | — | — | — | 0 | **⚠ no local train.json** | — | Colab | ckpt sha `df12f8989ec3` |
| M1a_c100 | M1a | msd | 3norm | adv | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 0 | 76/80 | 0.2310 | local | `…/C5_c100/M1a_c100/train.json` |
| M0_c100 | M0 | msd | 3norm | none | 80 | 0.5/0.5 | 0.1 | 10 | 10 | 0 | 71/80 | 0.2185 | local | `…/C5_c100/M0_c100/train.json` |
| M0_full | M0 | **msd_v0** | *(null)* | *(null)* | **50** | 0.5/0.5 | 0.1 | 10 | **50** | 0 | **40/50** | 0.4810 | local | `results/fromscratch/C5_full/M0_full/train.json` |
| M0_full s1 | M0 | msd_v0 | *(null)* | *(null)* | 50 | 0.5/0.5 | 0.1 | 10 | 50 | 1 | 40/50 | 0.4920 | local | `…/C5_full/M0_full_seed1/train.json` |
| M1a_full s1 | M1a | msd_v0 | *(null)* | *(null)* | 50 | 0.5/0.5 | 0.1 | 10 | 50 | 1 | 39/50 | 0.4890 | local | `…/C5_full/M1a_full_seed1/train.json` |
| **M1a_full s0** | — | — | — | — | 50 | — | — | — | 50 | 0 | **⚠ no local train.json** | — | ⚠ | ckpt sha `9e8ea901ab46` |
| M1b | M1b | msd | — | clean | 80 | — | — | — | — | 0 | ⚠ no local train.json | — | ⚠ | ckpt sha `10f2049d59ad` |
| M1a_msdglue | M1a | msd | **msd** | adv | 80 | — | — | — | — | 0 | ⚠ no local train.json | — | ⚠ | ckpt sha `f6d966fe3456` |
| Rprime, B1 (RAMP) | — | — | — | — | — | — | — | — | — | 0 | **⚠ UNTRACED** (external `external/RAMP`) | — | Colab | ckpts downloaded, sha verified |
| ft_none / ft_clamp | — | — | — | — | — | — | — | — | — | — | **⚠ UNTRACED** | — | ⚠ | `finetune_msd_clamp.py`; no local train.json |
| **M1a_cleance** (clean-CE) | M1a | clean | 3norm | adv | 80 | 0.5/0.5 | 0.1 | 10 | — | 0 | **PENDING** | — | Colab (queued) | wired in `notebooks/train_colab.ipynb` |

`lr`, `milestones` are **null** in the `C5_full` rows because `train_full_msd.py` writes a different key set —
the 50-epoch schedule is `⚠ UNTRACED` from `train.json` and must be read from `train_full_msd.py` directly.

### 2a. `M1a_full_v50` — recipe verified (⚠ flag closed 2026-07-22)

The earlier caveat "views = 50 is traceable only to the directory name" is **resolved**. The run's own
`train.json` was recovered and placed at `results/fromscratch/C5_full/M1a_full_v50/train.json`:

| key | value | meaning |
|---|---|---|
| `msd_steps` | **50** | base MSD attack at 50 steps |
| `apgd_view_steps` | **50** | CLAMP positives at 50 steps — **this is the decisive field** |
| `n_iter` | 10 | present but **overridden**: `view_steps = apgd_view_steps if not None else n_iter` ⟵ `scripts/dev/train_full_msd.py:184` |
| `epochs` / `epochs_completed` | 50 / **50** | finished |
| `alpha`,`beta`,`tau`,`warmup` | 0.5, 0.5, 0.1, 10 | identical to the headline pair |
| `lr_peak`,`bs`,`wd` | 0.1, 128, 5e-4 | full-budget recipe |
| `best_val_worst_union` | **0.4710** at epoch **41/50** | vs `M0_full` 0.4810 @ 40/50 |

Independent corroboration: **657 s/epoch** (mean over 50 epochs, min 656 / max 659) versus **365 s/epoch**
for `M1a_full` — a **1.80×** slowdown, which is what moving the three per-norm views from 10 to 50 APGD
steps costs. Checkpoint sha `ea76bf0aa5bd79e86398ff33…` is identical across the Drive copy, the local
copy, and the sha recorded in its own audit.

So `M1a_full_v50` is genuinely **base-50 + views-50**, i.e. the strength-matched arm, and the staleness
caveat (§3.4 / §10 item 2) has a ready-made control that only lacks a @10k audit.

---

## 3. Training-attack configs

### 3.1 ε-triple (single source of truth)

```
49  EPS = {"Linf": 8/255, "L2": 0.5, "L1": 12.0}
50  NORMS = ["Linf", "L2", "L1"]
```
⟵ `scripts/dev/c5_fromscratch.py:49-50`.

### 3.2 MSD base attack

`msd_v0(model, x, y, EPS["Linf"], EPS["L2"], EPS["L1"], steps=a.msd_steps)` ⟵ `scripts/dev/c5_fromscratch.py:543`.

```
146  def msd_v0(model, x, y, eps_linf, eps_l2, eps_l1,
147             alpha_linf=0.003, alpha_l2=0.05, alpha_l1=0.05, steps=50):
```
⟵ `src/robustdro/attacks/norms.py:146-153`. Faithful port of `locuslab/robust_union` `cifar_funcs.py::msd_v0`
at commit `ef34194`. One gradient per step, three candidate steps, per-sample argmax-loss; `k ~ U{5..20}` for ℓ₁.
Step sizes kept at the original defaults. **No random init, no restarts.**

- 80-epoch arms: `steps = --msd-steps = 10` ⟵ `:378`
- full-budget arms: `msd_steps = 50` ⟵ `results/fromscratch/C5_full/*/train.json` key `msd_steps`

### 3.3 The three per-norm APGD views

`apgd_train(model, x, y, nm, EPS[nm], n_iter=a.n_iter, is_train=True)` ⟵ `scripts/dev/c5_fromscratch.py:549, 560, 568-575`.

| property | value | source |
|---|---|---|
| loss | **cross-entropy only** (`crit = F.cross_entropy(..., reduction='none')`) | `src/robustdro/attacks/apgd_train.py:96-97` |
| restarts | **1** (no restart loop in the function) | `src/robustdro/attacks/apgd_train.py:81-140` |
| random init | `x_adv = x.clone().clamp(0,1)` — **starts from the clean point, no random start** | `…/apgd_train.py:90` |
| step size | `step_size = alpha * eps`, `alpha = 2.` for Linf/L2, `alpha = 1.` for L1 | `…/apgd_train.py:106, 114, 118` |
| APGD schedule | `n_iter_2 = 0.22·n_iter`, `n_iter_min = 0.06·n_iter`, `size_decr = 0.03·n_iter` | `…/apgd_train.py:101-103` |
| iterations | `n_iter = --n-iter`, **default 10** | `scripts/dev/c5_fromscratch.py:379` |

### 3.4 Staleness caveat — CONFIRMED at code level

In the 50-step full-budget pair the **base** attack is MSD-50 but the **CLAMP views remain 10-step APGD**:

- `msd_steps = 50` and `train_apgd_niter = 10` are separate keys in the same `train.json`
  ⟵ `results/fromscratch/C5_full/M0_full/train.json`
- `--msd-steps` (`:378`) and `--n-iter` (`:379`) are independent flags; nothing couples them.

So in the reversal pair the glue/scaffold positives are crafted **5× weaker than the base**. This is the
single most exploitable asymmetry in the paper (§10).

---

## 4. Evaluation harness

### 4.1 The 12 components

```
39  "linf": ["apgd_ce_linf", "apgd_dlr_linf", "fab_t_linf", "square_linf"],
40  "l2":   ["apgd_ce_l2",   "apgd_dlr_l2",   "fab_t_l2",   "square_l2"],
41  "l1":   ["apgd_ce_l1",   "apgd_dlr_l1",   "fab_t_l1",   "square_l1"],
```
⟵ `scripts/eval_multinorm_audit.py:39-41`.

### 4.2 ⚠ DISCREPANCY with the brief — APGD-DLR, not APGD-T

```
45  "apgd_ce_linf": "apgd-ce",
46  "apgd_dlr_linf": "apgd-dlr",
47  "fab_t_linf": "fab-t",
48  "square_linf": "square",
```
⟵ `scripts/eval_multinorm_audit.py:44-56`.

The brief specifies `{APGD-CE, APGD-T(DLR), FAB-T, Square}`. The code runs **`apgd-dlr` — the UNTARGETED DLR
attack — not `apgd-t` (targeted)**. AutoAttack exposes both (`'apgd-dlr' in source: True`, `'apgd-t' in source:
True`), so this is a deliberate-or-accidental configuration choice, not a library limitation. Untargeted DLR is
**weaker** than targeted APGD-T; the standard AutoAttack `rand`/`standard` versions use `apgd-t`. **All reported
unions are therefore upper bounds relative to a standard-AutoAttack evaluation.** A reviewer will notice this.
`fab-t` *is* the targeted FAB variant ⟵ `:47`.

### 4.3 Per-attack parameters

| parameter | value | source |
|---|---|---|
| steps | `attack.get("steps", 100)` → **100** in config for every attack | `scripts/eval_multinorm_audit.py:342`; config attack blocks |
| restarts | `attack.get("restarts", 1)` → **1** for every attack | `:343`; config |
| Square queries | `attack.get("queries", 5000)` → **5000** | `:354`; config |
| attack seed | `attack.get("seed", 0)` → **20260709** in config | `:373`; config |
| ε | `eps_key` per norm → ℓ∞ `0.03137254901960784`, ℓ₂ `0.5`, ℓ₁ `12.0` | config `eps:` block |
| AutoAttack mode | `version="custom"`, `attacks_to_run=[aa_name]` — one component per adversary | `:368-369` |

Each component is run as its **own** AutoAttack instance against the **full** subset (not cascaded on survivors),
so per-attack accuracies are independent and the union is a true AND.

### 4.4 ℓ₁ convention

`norm=NORM_MAP[attack["norm"]]` with `NORM_MAP = {"linf": "Linf", "l2": "L2", "l1": "L1"}`
⟵ `scripts/eval_multinorm_audit.py:43, 366`. The installed AutoAttack asserts `norm in ['Linf','L2','L1']` and
`autoattack.autopgd_base` contains L1 handling — i.e. the **Croce & Hein ℓ₁ variant**, including `square` run at
`norm='L1'`. No custom ℓ₁ code path exists in this repo.

### 4.5 Union logic

```
431  def and_numpy_masks(masks: list[Any]):
...
437      out = np.asarray(masks[0], dtype=bool).copy()
438      for mask in masks[1:]:
439          out &= np.asarray(mask, dtype=bool)
440      return out
```
⟵ `scripts/eval_multinorm_audit.py:431-440` (torch twin at `:404-410`).
Union = per-example AND over **all** component masks; `full_audit_union` is its mean ⟵ `:540`.

### 4.6 Tier definitions

| tier | components | driver | flag |
|---|---|---|---|
| `full` | **12** | `scripts/eval_multinorm_audit.py` | — |
| `no_square` | **9** (12 − 3 `square_*`) | `scripts/dev/union_bench_eval.py` | `--skip-square` ⟵ `:124`, filter at `:145` |

`union(no_square) ≥ union(full)` necessarily (dropping an attack can only add survivors).
Tier-aware output paths prevent cross-tier overwrites ⟵ `scripts/dev/eval_arm.py:51-54`, with a refuse-to-clobber
guard at `:61-69` and a preflight for missing config/indices at `:49-64`.

**Which tier each run used** is recorded per scale-dir as `n_attacks` in §6; `12` ⇒ full, `9` ⇒ no_square.
`results/main/INDEX.md` labels them and `sync_results_main.py` re-verifies `union(eval.json) == union(masks)`
on every sync (last sync: **44 pass / 0 mismatch**).

---

## 5. Checkpoint selection

### 5.1 Exact proxy spec

```
245  def worst_union_acc(model, X, Y, idx, device, bs=250, n_iter=20):
246      """val_select worst-union robust acc (== B3/B4 selection metric): AND over 3-norm APGD."""
...
254          xa = apgd_train(model, xb, yb, nm, EPS[nm], n_iter=n_iter, is_train=False)
257      robust &= pred_ok
```
⟵ `scripts/dev/c5_fromscratch.py:245-258`.

| property | value |
|---|---|
| split | `train[49000:50000]`, n = 1000 ⟵ `:52` |
| attack | `apgd_train`, CE loss, **20 iterations**, **1 restart**, no random start |
| norms | Linf, L2, L1 at the standard triple |
| aggregation | per-example AND ⟵ `:257` |
| selection rule | `if wu > best: best = wu; save("val_best")` ⟵ `:660` |
| RNG | APGD starts from the clean point ⇒ selection consumes no RNG ⟵ `:262-266` |

`val_metrics` (`:262`) is documented as byte-identical to `worst_union_acc`, and is what actually runs, so
M1a/M0/M1b share one selection rule ⟵ `:263-266`.

**All compared 80-epoch arms use this rule** (§2 val_best-epoch column populated from each `train.json` history).
`C5_full` arms use `train_full_msd.py`'s own selection — same *concept*, but the code path is a different file;
**⚠ not independently verified byte-identical in this dossier.**

### 5.2 Proxy over-read — measured, and it contradicts the brief

Measured by re-running the exact selection function on both splits ⟵ `scripts/dev/t5_proxy_decompose.py`,
results `results/analysis/T5_proxy_decompose/proxy_decompose.json`:

| arm | val_proxy (weak/VAL) | test_proxy (weak/TEST) | test_strong (9-atk/TEST@1k) | distribution | attack strength |
|---|---|---|---|---|---|
| M1a | 0.4730 | 0.4800 | 0.4370 | **−0.0070** | **+0.0430** |
| M1a_max | 0.4720 | 0.4640 | 0.4300 | +0.0080 | +0.0340 |
| M1a_avg | 0.4390 | 0.4420 | 0.4250 | −0.0030 | +0.0170 |

Self-check: recomputed `val_proxy` reproduces the logged `best_val_worst_union` **exactly** for M1a (0.4730) and
M1a_avg (0.4390) ⟵ same script's `[logged … MATCH]` output.

**⚠ DISCREPANCY.** The brief states over-read "CLAMP −5.6 pp, MSD −5.1 pp". Traced values:
M1a over-read = 0.4730 − 0.4370 = **3.60 pp** (not 5.6). For MSD, seed 0's `train.json` is not local, so the
matching number is `⚠ UNTRACED`; the nearest traceable is M0 seed 2 = 0.4470 − 0.3943(@10k) = **5.27 pp**, which
mixes VAL@1k against TEST@10k and is therefore not comparable to the 3.60 pp figure. **The 5.6/5.1 pair cannot be
reproduced from any file in the repo.**

---

## 6. MASTER RUN INVENTORY

Source for every row: `results/main/<arm>/<scale>/eval.json` and `…/masks_multinorm_v1.npz`, mirrored from
`results/eval/union_bench/`, with `union(eval.json) == union(masks)` re-verified on sync (44/44 pass, 0 mismatch).
`atk` 12 = tier full, 9 = tier no_square. `ver` = union-vs-masks verification passed.

| arm | scale | n | atk | clean | union | ℓ∞ | ℓ₂ | ℓ₁ | masks | ver | ckpt sha[:12] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1a | 1k | 1000 | 12 | 0.8180 | **0.4370** | 0.4520 | 0.6700 | 0.5190 | Y | Y | f02924cb230b |
| M1a | 10k | 10000 | 12 | 0.8164 | **0.4173** | 0.4269 | 0.6636 | 0.5148 | Y | Y | f02924cb230b |
| M1a | 1k_nosq | 1000 | 9 | 0.8180 | 0.4360 | 0.4510 | 0.6700 | 0.5190 | Y | Y | f02924cb230b |
| M0 | 1k | 1000 | 12 | 0.8450 | **0.3920** | 0.4120 | 0.6670 | 0.4590 | Y | Y | e1dd1dac7985 |
| M0 | 10k | 10000 | 12 | 0.8379 | **0.3886** | 0.4040 | 0.6621 | 0.4667 | Y | Y | e1dd1dac7985 |
| M0 | 1k_nosq | 1000 | 9 | 0.8450 | 0.3930 | 0.4130 | 0.6670 | 0.4580 | Y | Y | e1dd1dac7985 |
| M1a_seed2 | 1k | 1000 | 12 | 0.8200 | 0.4340 | 0.4490 | 0.6810 | 0.5210 | Y | Y | c6f24dd50b88 |
| M1a_seed2 | 10k | 10000 | 12 | 0.8113 | **0.4195** | 0.4293 | 0.6658 | 0.5153 | Y | Y | c6f24dd50b88 |
| M0_seed2 | 1k | 1000 | 12 | 0.8320 | 0.4110 | 0.4270 | 0.6650 | 0.4690 | Y | Y | a224ed1df317 |
| M0_seed2 | 10k | 10000 | 12 | 0.8392 | **0.3943** | 0.4076 | 0.6550 | 0.4721 | Y | Y | a224ed1df317 |
| M1a_seed3 | 10k | 10000 | 12 | 0.8171 | **0.4227** | 0.4299 | 0.6638 | 0.5209 | Y | Y | 5ca4e558b47a |
| M0_seed3 | 10k | 10000 | 12 | 0.8300 | **0.3970** | 0.4122 | 0.6554 | 0.4741 | Y | Y | 3885c6e1fdfc |
| M1b | 1k | 1000 | *(none)* | 0.8210 | 0.4240 | 0.4340 | 0.6660 | 0.5240 | **n** | – | 10f2049d59ad |
| M1b | 10k | 10000 | 12 | 0.8157 | 0.4139 | 0.4220 | 0.6631 | 0.5150 | Y | Y | 10f2049d59ad |
| M1a_msdglue | 1k | 1000 | 9 | 0.8320 | 0.4140 | 0.4320 | 0.6650 | 0.4750 | Y | Y | f6d966fe3456 |
| M1a_msdglue | 10k | 10000 | 12 | 0.8215 | 0.4012 | 0.4162 | 0.6528 | 0.4778 | Y | Y | f6d966fe3456 |
| M1a_msdglue | 1k_nosq | 1000 | 9 | 0.8320 | 0.4140 | 0.4320 | 0.6650 | 0.4750 | Y | Y | f6d966fe3456 |
| M1a_full | 1k | 1000 | 12 | 0.8080 | 0.4340 | 0.4500 | 0.6620 | 0.5050 | Y | Y | 9e8ea901ab46 |
| M1a_full | 10k | 10000 | 12 | 0.8063 | **0.4201** | 0.4298 | 0.6563 | 0.5057 | Y | Y | 9e8ea901ab46 |
| M1a_full | last | 1000 | 12 | 0.8260 | 0.4320 | 0.4420 | 0.6700 | 0.5120 | Y | Y | 0d2e7ef443d2 |
| M1a_full | 1k_nosq | 1000 | 9 | 0.8080 | 0.4320 | 0.4500 | 0.6620 | 0.5020 | Y | Y | 9e8ea901ab46 |
| M0_full | 1k | 1000 | 12 | 0.8170 | 0.4350 | 0.4450 | 0.6540 | 0.5140 | Y | Y | 469975cb1f8d |
| M0_full | 10k | 10000 | 12 | 0.8055 | **0.4297** | 0.4375 | 0.6488 | 0.5049 | Y | Y | 469975cb1f8d |
| M0_full | last | 1000 | 12 | 0.8370 | 0.4320 | 0.4380 | 0.6520 | 0.5040 | Y | Y | df4b467756b2 |
| M0_full | 1k_nosq | 1000 | 9 | 0.8170 | 0.4350 | 0.4450 | 0.6540 | 0.5120 | Y | Y | 469975cb1f8d |
| M1a_full_seed1 | 1k | 1000 | 9 | 0.7960 | 0.4330 | 0.4470 | 0.6430 | 0.5160 | Y | Y | 64b8f812ff01 |
| M1a_full_seed1 | 10k | 10000 | 12 | 0.7851 | **0.4230** | 0.4327 | 0.6420 | 0.5042 | Y | Y | 64b8f812ff01 |
| M0_full_seed1 | 1k | 1000 | 9 | 0.8290 | 0.4370 | 0.4490 | 0.6640 | 0.5070 | Y | Y | cde00aea9a7e |
| M0_full_seed1 | 10k | 10000 | 12 | 0.8046 | **0.4288** | 0.4353 | 0.6465 | 0.5098 | Y | Y | cde00aea9a7e |
| M1a_full_v50 | 1k | 1000 | 9 | 0.8120 | 0.4490 | 0.4570 | 0.6580 | 0.5470 | Y | Y | ea76bf0aa5bd |
| M1a_full_v50 | 10k | 10000 | 12 | 0.8042 | **0.4267** | 0.4315 | 0.6498 | 0.5289 | Y | Y | ea76bf0aa5bd |
| M1a_max | 1k | 1000 | 9 | 0.8240 | 0.4300 | 0.4490 | 0.6720 | 0.5060 | Y | Y | df12f8989ec3 |
| M1a_max | 10k | 10000 | 12 | 0.8173 | **0.4125** | 0.4281 | 0.6612 | 0.4918 | Y | Y | df12f8989ec3 |
| M1a_max | 1k_nosq | 1000 | 9 | 0.8240 | 0.4280 | 0.4480 | 0.6720 | 0.5050 | Y | Y | df12f8989ec3 |
| M0_max | 10k | 10000 | 12 | 0.8138 | **0.4349** | 0.4485 | 0.6500 | 0.4879 | Y | Y | a981e3a3a6b3 |
| M0_max | 1k_nosq | 1000 | 9 | 0.8250 | 0.4530 | 0.4680 | 0.6630 | 0.4980 | Y | Y | a981e3a3a6b3 |
| M1a_avg | 1k_nosq | 1000 | 9 | 0.8300 | 0.4250 | 0.4330 | 0.6830 | 0.5340 | Y | Y | 56c8b04b4e14 |
| M1a_avg | 10k | 10000 | 12 | 0.8331 | **0.4049** | 0.4108 | 0.6767 | 0.5347 | Y | Y | 56c8b04b4e14 |
| M0_avg | 10k | 10000 | 12 | 0.8417 | **0.3846** | 0.3897 | 0.6736 | 0.5209 | Y | Y | 78b02da344ed |
| B1 | 1k | 1000 | 12 | 0.8080 | **0.4540** | 0.4640 | 0.6690 | 0.5140 | Y | Y | fa820e3c0489 |
| M0_avg | 1k_nosq | 1000 | 9 | 0.8360 | 0.4060 | 0.4100 | 0.6860 | 0.5360 | Y | Y | 78b02da344ed |
| ft_clamp | 10k | 10000 | 12 | 0.8542 | 0.4026 | 0.4342 | 0.6876 | 0.4754 | Y | Y | 149b1bf33b51 |
| ft_clamp | 1k_nosq | 1000 | 9 | 0.8400 | 0.4150 | 0.4490 | 0.6970 | 0.4840 | Y | Y | 149b1bf33b51 |
| ft_none | 10k | 10000 | 12 | 0.8662 | 0.3991 | 0.4268 | 0.6782 | 0.4645 | Y | Y | 8eb94b91b4eb |
| ft_none | 1k_nosq | 1000 | 9 | 0.8610 | 0.4020 | 0.4360 | 0.6860 | 0.4660 | Y | Y | 8eb94b91b4eb |
| Rprime | 1k | 1000 | *(none)* | 0.8270 | 0.4580 | 0.4780 | 0.6660 | 0.4970 | **n** | – | 10b9f7431c2c |
| Rprime | 10k | 10000 | 12 | 0.8118 | **0.4461** | 0.4581 | 0.6550 | 0.4915 | Y | Y | 10b9f7431c2c |
| B1 | 10k | 10000 | 12 | 0.8010 | **0.4456** | 0.4544 | 0.6558 | 0.5116 | Y | Y | fa820e3c0489 |
| msd (public) | 1k | 1000 | 12 | 0.8240 | 0.4420 | 0.4580 | 0.6660 | 0.4990 | Y | Y | 482bf2876572 |
| msd (public) | 1k_nosq | 1000 | 9 | 0.8240 | 0.4420 | 0.4570 | 0.6660 | 0.5000 | Y | Y | 482bf2876572 |
| max (public) | 1k | 1000 | 12 | 0.8190 | 0.2800 | 0.3990 | 0.6400 | 0.2980 | Y | Y | 9f05f649e8ba |
| avg (public) | 1k | 1000 | 12 | 0.8530 | 0.3880 | 0.4060 | 0.6970 | 0.5140 | Y | Y | 987b38f5b3af |
| M1a_c100 | 1k | 1000 | 9 | 0.5650 | 0.1960 | 0.1990 | 0.3750 | 0.2580 | Y | Y | 5a8070f62c6c |
| M0_c100 | 1k | 1000 | 9 | 0.5780 | 0.1890 | 0.1980 | 0.3710 | 0.2500 | Y | Y | 148798d696bf |
| b3_static_linf | 1k / 10k | 1000/10000 | 12 | 0.8410 / 0.8334 | 0.4100 / 0.4028 | 0.4340 / 0.4221 | 0.6700 / 0.6704 | 0.4920 / 0.4881 | Y | Y | 4676e75fe06b |
| b4_adaptive | 1k / 10k | 1000/10000 | 12 | 0.8320 / 0.8253 | 0.4200 / 0.4114 | 0.4640 / 0.4463 | 0.6760 / 0.6572 | 0.4660 / 0.4591 | Y | Y | 7ceb02ae76ac |
| **M1a_cleance** | any | — | — | — | — | — | — | — | — | — | **PENDING (not trained)** |
| **M1a_c100 / M0_c100** | **10k** | — | — | — | — | — | — | — | — | — | **PENDING** |

**⚠ Two scale-dirs carry no masks** (`M1b/1k`, `Rprime/1k`) — they were entered from pasted `eval.json` only and
cannot enter any paired bootstrap ⟵ sync report "2 dirs without masks".

### 6.1 Paired deltas @10k tier=full (recomputed from masks at dossier time)

| pair | A | B | Δ | 95% CI | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ |
|---|---|---|---|---|---|---|---|---|
| CLAIM A seed 0 | 0.4173 | 0.3886 | **+0.0287** | [+0.0234, +0.0339] | + | +0.0229 | +0.0015 | +0.0481 |
| CLAIM A seed 2 | 0.4195 | 0.3943 | **+0.0252** | [+0.0202, +0.0301] | + | +0.0217 | +0.0108 | +0.0432 |
| CLAIM A seed 3 | 0.4227 | 0.3970 | **+0.0257** | [+0.0212, +0.0303] | + | +0.0177 | +0.0084 | +0.0468 |
| REVERSAL seed 0 | 0.4201 | 0.4297 | **−0.0096** | [−0.0142, −0.0048] | − | −0.0077 | +0.0075 | +0.0008 |
| REVERSAL seed 1 | 0.4230 | 0.4288 | **−0.0059** | [−0.0110, −0.0006] | − | −0.0026 | −0.0045 | −0.0056 |
| **F2 views-50 − MSD-50** | 0.4267 | 0.4297 | **−0.0030** | [−0.0081, +0.0021] | **ns** | −0.0060 | +0.0010 | +0.0240 |
| F2 views-50 − views-10 | 0.4267 | 0.4201 | **+0.0066** | [+0.0015, +0.0118] | + | +0.0017 | −0.0065 | +0.0232 |
| MAX base (#11) | 0.4125 | 0.4349 | **−0.0224** | [−0.0274, −0.0175] | − | −0.0204 | +0.0112 | +0.0039 |
| MAX vs MSD, term OFF | 0.4349 | 0.3886 | **+0.0463** | [+0.0413, +0.0513] | + | +0.0445 | −0.0121 | +0.0212 |
| MAX vs MSD, term ON | 0.4125 | 0.4173 | −0.0048 | [−0.0097, +0.0001] | ns | +0.0012 | −0.0024 | −0.0230 |
| FT-∞ | 0.4026 | 0.3991 | +0.0035 | [−0.0012, +0.0083] | ns | +0.0074 | +0.0094 | +0.0109 |
| vs RAMP (M1a − Rprime) | 0.4173 | 0.4461 | **−0.0288** | [−0.0342, −0.0234] | − | −0.0312 | +0.0086 | +0.0233 |
| **RAMP row: B1 − Rprime** | 0.4456 | 0.4461 | **−0.0005** | [−0.0050, +0.0038] | **ns** | −0.0037 | +0.0008 | +0.0201 |   *(Δclean −0.0108; route ℓ₁)*
| **AVG base (#12)** | 0.4049 | 0.3846 | **+0.0203** | [+0.0155, +0.0250] | + | +0.0211 | +0.0031 | +0.0138 |

⟵ `scripts/dev/t1b_report.py` and equivalent mask-level recomputation; masks paths in §6.

### 6.3 F2 — staleness vs redundancy, decomposed

`M1a_full_v50` = CLAMP-50 base **with 50-step views** (recipe verified, §2a), audited @10k tier=full.

| contrast | Δunion | 95% CI | LCB | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ | Δclean |
|---|---|---|---|---|---|---|---|---|
| views-**50** − MSD-50 | **−0.0030** | [−0.0081, +0.0021] | −0.0081 | **ns** | −0.0060 | +0.0010 | **+0.0240** | −0.0013 |
| views-**10** − MSD-50 (seed 0) | −0.0096 | [−0.0142, −0.0048] | −0.0142 | − | −0.0077 | +0.0075 | +0.0008 | +0.0008 |
| views-**10** − MSD-50 (seed 1) | −0.0059 | [−0.0110, −0.0006] | −0.0110 | − | −0.0026 | −0.0045 | −0.0056 | −0.0195 |
| views-50 − views-10 (term ON both) | **+0.0066** | [+0.0015, +0.0118] | +0.0015 | **+** | +0.0017 | −0.0065 | **+0.0232** | −0.0021 |

**Reading, stated precisely.** Strengthening the views from 10 to 50 steps recovers **+0.0066**
(significant), which is roughly **two thirds** of the seed-0 reversal (−0.0096). Under matched views the
reversal is **no longer detectable** (CI straddles zero) but the point estimate **stays negative**
(−0.0030). So neither extreme is supported: staleness is a **major contributor**, and it is **not the
whole story**.

**The recovery is entirely ℓ₁** (+0.0232), while **Δℓ∞ remains negative** (−0.0060, versus −0.0077 at
views-10 — essentially unchanged). Stale views were costing ℓ₁ robustness; the ℓ∞ interference is a
**separate channel that survives strength matching**. Since the union is ℓ∞-bound (§9.1: ℓ∞ implicated
in 97–98 % of union failures), that is why fixing ℓ₁ does not turn the union positive.

**Caveats.** n = 1 seed at views-50, against 2 seeds at views-10 — the views-10 reversal needed both
seeds before it was called. Also note `M1a_full_v50`'s val proxy was **lower** than `M0_full`'s
(0.4710 vs 0.4810 ⟵ `train.json`) yet its @10k union is **higher** than `M1a_full`'s — another instance
of the weak proxy mis-ranking arms (§5.2).

### 6.2a ⚠ CORRECTION — `B1` is the RAMP **+term** arm, not a RAMP baseline

| arm | checkpoint path (from its own `eval.json`) | what it is |
|---|---|---|
| `Rprime` | `Rprime_out/R_prime_ramponly_seed0/val_best.pth` | RAMP **only** (base) |
| `B1` | `Bet1_out/B1_pullpush_seed0/val_best.pth` | RAMP **+ pull-push** (term ON) |

Corroborating evidence for `B1`: 80 epochs at 98.1 s/ep, `val_best` at epoch 78 with
`val_worst_union = 0.5920` ⟵ `val_best_meta.pth`; train loss **negative** from ~epoch 40 onward
(−0.4413 at ep 80) ⟵ `log_train.txt`. Checkpoint sha `fa820e3c0489…` is byte-identical to the `B1`
already audited in this repo, and the re-downloaded @10k `eval.json` reproduces union 0.4456 / clean
0.8010 exactly.

**Consequence:** any comparison previously written as "ours vs RAMP" that used `B1` was in fact
"ours vs RAMP+CLAMP". The correct RAMP baseline is `Rprime`. The row
`M1a_full − B1 = −0.0255` in earlier ledgers is therefore **mislabelled** and must be restated as
"CLAMP-50 vs RAMP+CLAMP", or replaced by `M1a_full − Rprime`.

`B1` @1k (12 attacks, masks present) was recovered at the same time: clean 0.8080, union **0.4540**,
ℓ∞ 0.4640, ℓ₂ 0.6690, ℓ₁ 0.5140 — previously `B1` existed only at @10k.

⚠ `Rprime` has **no @1k masks**, so the RAMP pair can only be bootstrapped at @10k.

### 6.2 2×2 @1k tier=no_square (screening — do **not** mix with §6.1)

| base | term OFF | term ON | Δ | 95% CI | sig |
|---|---|---|---|---|---|
| MSD-10 | 0.3930 | 0.4360 | +0.0441 | [+0.0270, +0.0610] | + |
| MAX | 0.4530 | 0.4280 | −0.0230 | [−0.0380, −0.0080] | − |
| AVG | 0.4060 | 0.4250 | +0.0190 | [+0.0030, +0.0350] | + |

---

### 6.4 Pre-registration timestamp disclosure (G4)

> **mtime-ordered only; no ledger and no commit timestamps exist. AVG screening (@1k) preceded both
> preregistrations.**

Supporting detail, so the line above can be checked rather than trusted:
`docs/claim_evidence_ledger.md` does not exist. `preregistration_T2_representation_geometry.md`,
`preregistration_T2b_head_space.md` and `scripts/dev/f1_grad_scale.py` are **untracked** in git
(`git status` → `??`) in a working tree with 44 dirty paths, so **no commit hash dates any of them**;
the only ordering evidence is filesystem mtime, which is mutable.

| artefact | mtime |
|---|---|
| `M1a_avg` @1k no_square (AVG screening observed) | 07-21 12:00:18 |
| prereg T2 (geometry) | 07-21 13:09:50 |
| prereg T2b (head space) | 07-21 14:35:51 |
| `f1_grad_scale.py` (source of w = 1.6) | 07-22 05:33:51 |
| `M1a_avg` @10k `eval.json` (AVG union observed) | 07-22 07:17:32 |

Mitigating, and independently checkable: the T2/T2b preregistrations concern **encoder/head geometry**
and make no prediction about the AVG arm; and `w = 1.6` belongs to **F1, which was never run**, so its
timing bears on no reported number. Nothing is backdated.

---

### 6.5 Per-norm paired-bootstrap CIs (L13b)

`tab:bases` per-norm deltas are point estimates; the CIs below exist so that any smallness or
directional claim can be checked. Same estimator as the union deltas (per-example masks, B=10,000,
`rng(0)`, percentile, LCB = q05); per-norm block = AND over that norm's 4 attacks at tier=full.
Full table in `docs/PAPER_STATE.md` §D; raw in `results/analysis/pernorm_ci/pernorm_ci.json`;
code `scripts/dev/pernorm_ci.py`.

Load-bearing entries:

| contrast | Δℓ∞ | Δℓ₂ | Δℓ₁ |
|---|---|---|---|
| F2 v50 − MSD-50 | **−0.0060 [−0.0111, −0.0009]** − | +0.0010 [−0.0040, +0.0059] ns | **+0.0239 [+0.0186, +0.0293]** + |
| F2 v50 − views-10 | +0.0017 [−0.0033, +0.0067] ns | −0.0065 [−0.0115, −0.0016] − | **+0.0232 [+0.0178, +0.0287]** + |
| MSD-10 ℓ₂, 3 seeds | — | +0.0015 ns / +0.0109 + / +0.0084 + — **seed-min LCB −0.0037** | — |

**Consequences.** (a) The F2 reading "recovery is entirely ℓ₁, Δℓ∞ stays negative" is now supported on
both halves — ℓ₁ recovery significant, ℓ∞ recovery ns, ℓ∞ level significantly negative. (b) **ℓ₂ is not
a null channel**: it is significantly positive on MSD-10 seeds 2 and 3, on MAX (+0.0112) and on FT
(+0.0094); only seed 0 is ns, and the seed-min LCB is negative, so ℓ₂ neither holds across seeds nor is
absent. Prose treating ℓ₂ as "flat" is unsupported. (c) `MAX` Δℓ₁ (+0.0039) and `RAMP` Δℓ∞ (−0.0037) are
ns — magnitude-only language required (L13b).

---

## 7. Statistical method

Canonical implementation ⟵ `scripts/dev/c5_attribution.py:29-31`:

```
29      rng = np.random.default_rng(0); n = len(a); D = []
30      for _ in range(10000):
31          i = rng.integers(0, n, n); D.append(a[i].mean() - b[i].mean())
```

| property | value |
|---|---|
| resampling unit | **per-example union mask entry** (paired: the same resampled indices index both arms) |
| B | 10000 |
| seed | `default_rng(0)` |
| CI method | **percentile** of the bootstrap distribution of Δ |
| 1-sided LCB | `np.quantile(D, 0.05)` ⟵ `c5_attribution.py:52` docstring "1-sided 95% LCB" |
| 2-sided CI (used in §6.1 tables) | `quantile(D, .05)`, `quantile(D, .95)` ⟵ `scripts/dev/t1b_report.py:21` |
| T2/T2b CIs | `quantile(.025)`, `quantile(.975)` — **2-sided 95%** ⟵ `t2_geometry.py:187`, `t2d_head_space.py:133` |

**⚠ Two different quantile conventions are in use** — `[q05, q95]` for accuracy deltas and `[q025, q975]` for
geometry deltas. Each script is internally consistent and labels itself, but the paper must not describe them
with one sentence. The `[q05,q95]` interval is an 90% two-sided interval whose lower end is the 95% LCB;
calling it "95% CI" in §6.1 is the repo's convention and is **imprecise**.

**Multi-seed aggregation:** no code computes a 3-seed mean±std. The reported robustness statement is
"seed-minimum LCB > 0" (seed 0/2/3 LCBs +0.0234 / +0.0202 / +0.0212 → min +0.0202), derived by inspection of
the three per-seed rows in §6.1. **⚠ There is no script that performs this aggregation** — it is manual.

**No multiplicity correction** is applied anywhere, despite ~15 pairwise comparisons.

---

## 8. Mechanism experiment (RQ3)

Pre-registrations, both committed before the corresponding run and both recording "no deviations":
`docs/preregistrations/preregistration_T2_representation_geometry.md`,
`docs/preregistrations/preregistration_T2b_head_space.md`.

### 8.1 Measurement spec (T2, encoder space)

| property | value | source |
|---|---|---|
| layer | pooled 512-d encoder output, **before** `g` (`self.linear`) and **before** `h` | `scripts/dev/t2_geometry.py:57-66` (`Enc.features`) |
| metric | cosine `1 − cos` **and** Euclidean | `t2_geometry.py:131-137` (`dist`) |
| n samples | 1000 (frozen audit subset, identical indices across models) | `t2_geometry.py:96-101` |
| adv view | APGD-CE, `n_iter=50`, standard triple, crafted **against each model itself** | `t2_geometry.py:113-116` |
| CI | paired bootstrap B=10000, `rng(0)`, **2-sided 95%** | `t2_geometry.py:186-188` |
| models | M0, M1a (weak pair, `val_best`); M0_full, M1a_full (strong pair, `val_best`) | `t2_geometry.py:34-39` |

### 8.2 Result — 6/6 encoder cells, sign OPPOSITE to hypothesis

| base | Δd ℓ∞ | 95% CI | Δd ℓ₂ | 95% CI | Δd ℓ₁ | 95% CI |
|---|---|---|---|---|---|---|
| WEAK (MSD-10) | **+0.03290** | [+0.02937, +0.03656] | **+0.00439** | [+0.00355, +0.00525] | **+0.01556** | [+0.01252, +0.01858] |
| STRONG (MSD-50) | **+0.03343** | [+0.03098, +0.03595] | **+0.00506** | [+0.00450, +0.00565] | **+0.02392** | [+0.02169, +0.02626] |

⟵ `results/analysis/T2_geometry/geometry.json` key `relative_movement`; rendered in
`results/analysis/T2_geometry/REPORT.md`. Positive = CLAMP pushes adversarial views **further** from clean.
Euclidean agrees in sign. Pre-registered decision-rule row 3 fired: **H-geometry falsified**.

Supporting, same file: within every model, union-survivors are **closer** to clean than failures
(M0 0.0415 vs 0.0810; M1a 0.0763 vs 0.1166) — the metric is not broken; only the across-model ordering inverts.

### 8.3 Random-perturbation control (post-hoc)

⟵ `scripts/dev/t2b_sensitivity_control.py`, `results/analysis/T2_geometry/sensitivity_control.json`.
Same random directions for every model (`torch.Generator().manual_seed(0)`).

| pair | norm | sensitivity to random noise | adversarial displacement |
|---|---|---|---|
| WEAK | ℓ∞ | **0.97×** | **1.51×** |
| WEAK | ℓ₂ | 1.26× | 1.33× |
| WEAK | ℓ₁ | 1.20× | 1.31× |
| STRONG | ℓ∞ | 1.60× | 1.80× |

⚠ Absolute `d_random` for ℓ₂/ℓ₁ is ~1e-5, so only the **model-to-model ratios** in this table are meaningful;
the raw `ratio_adv_over_random` values (445×, 7208×) in the JSON are numerically unstable and must not be quoted.

### 8.4 Head-space (T2b) and the random-head baseline

Checkpoint constraint: `val_best.pt` stores `model.b.state_dict()` — **backbone only**
⟵ `scripts/dev/c5_fromscratch.py:656`. The head survives only in `resume.pt` / `ckpt_latest.pt`
⟵ `:662`, which are **last-epoch**. T2b therefore re-measures `f` at last-epoch weights too and regenerates the
views there ⟵ `scripts/dev/t2d_head_space.py` docstring + `:95-108`.

| model | norm | d_h | d_f | h − f | 95% CI |
|---|---|---|---|---|---|
| M1a | ℓ∞ | 0.2233 | 0.1086 | **+0.1147** | [+0.1072, +0.1226] |
| M1a | ℓ₂ | 0.0443 | 0.0200 | +0.0243 | [+0.0218, +0.0270] |
| M1a | ℓ₁ | 0.1485 | 0.0720 | +0.0765 | [+0.0701, +0.0831] |
| M1a_full | ℓ∞ | 0.2011 | 0.0860 | **+0.1151** | [+0.1087, +0.1217] |
| M1a_full | ℓ₂ | 0.0404 | 0.0162 | +0.0242 | [+0.0223, +0.0263] |
| M1a_full | ℓ₁ | 0.1505 | 0.0644 | +0.0861 | [+0.0806, +0.0917] |
| **M0 (untrained head)** | ℓ∞ | 0.0702 | 0.0806 | **−0.0104** | [−0.0121, −0.0087] |
| **M0_full (untrained head)** | ℓ∞ | 0.0340 | 0.0494 | **−0.0154** | [−0.0168, −0.0141] |

⟵ `results/analysis/T2_geometry/head_space.json` key `h_vs_f`.

**Random-head baseline validity:** under `variant=M0` the head is constructed but receives no gradient; its
`head.2.weight.std()` is **0.02549 in both controls, identical**, versus ~0.047 when trained ⟵ printed by
`t2d_head_space.py` at load, tabulated in `preregistration_T2b_head_space.md` §2. A random projection
**compresses** (d_h < d_f); the trained head **expands**. P1 and P3 both falsified.

### 8.5 In-sample cosine and train→eval transfer

Training log value at the last epoch: `glue = −9.18`; with `τ = 0.1` and `glue = mean(−cos/τ)` ⟵
`scripts/dev/c5_fromscratch.py:134`, this is `cos = 0.918`, `d_h = 0.082`.
Direct re-measurement reproduces it: per-view TRAIN/APGD-10 distances 0.1673 / 0.0278 / 0.0666, mean **0.0872**
⟵ `results/analysis/T2_geometry/head_attack_strength.json`.

| model | norm | TRAIN apgd-10 | TEST apgd-10 | TEST apgd-50 | Δ generalisation | Δ attack |
|---|---|---|---|---|---|---|
| M1a | ℓ∞ | 0.1673 | 0.2155 | 0.2233 | +0.0482 | +0.0078 |
| M1a | ℓ₁ | 0.0666 | 0.1000 | 0.1485 | +0.0333 | **+0.0486** |
| M1a_full | ℓ₁ | 0.0728 | 0.0956 | 0.1505 | +0.0228 | **+0.0549** |

⟵ `scripts/dev/t2e_head_attackstrength.py` (post-hoc, explicitly labelled). Conclusion: the term **does** hit its
own objective in-sample; the alignment does not survive the move to test data, and for ℓ₁ not a stronger attack.

### 8.6 Class separation (post-hoc)

Δ margin (term ON − OFF), L2-normalised embeddings: WEAK −0.0288/−0.0345/−0.0333/−0.0288 (clean/ℓ∞/ℓ₂/ℓ₁),
Δ Fisher **−0.3687** (0.820 → 0.452); STRONG Δ Fisher −0.2898 ⟵ `results/analysis/T2_geometry/class_separation.json`.

---

## 9. Anatomy experiment (RQ4)

### 9.1 Threat dominance — DONE

⟵ `scripts/dev/t3_threat_dominance.py`, `results/analysis/T3_threat_dominance/threat_dominance.json`.
Per-norm block = AND over that norm's attacks in the tier; union failure = ¬union.

| model | tier | union | ℓ∞% | ℓ₂% | ℓ₁% | ℓ∞only | ℓ₂only | ℓ₁only | all3 |
|---|---|---|---|---|---|---|---|---|---|
| msd (public) | full | 0.4420 | 97.1 | 59.9 | 89.8 | 10.2 | **0.0** | 2.9 | 59.9 |
| M0 (MSD-10) | full | 0.3886 | 97.5 | 55.3 | 87.2 | 12.8 | **0.0** | 2.5 | 55.3 |
| M1a (CLAMP) | full | 0.4173 | 98.4 | 57.7 | 83.3 | 16.7 | **0.0** | 1.6 | 57.7 |
| M0_full (MSD-50) | full | 0.4297 | 98.6 | 61.6 | 86.8 | 13.2 | **0.0** | 1.4 | 61.6 |
| M1a_full | full | 0.4201 | 98.3 | 59.3 | 85.2 | 14.8 | **0.0** | 1.7 | 59.3 |
| Rprime (RAMP) | full | 0.4461 | 97.8 | 62.3 | 91.8 | 8.2 | **0.0** | 2.2 | 62.3 |

**ℓ₂-only = 0.0% in all 10 audited models** — ℓ₂ is never solely responsible for a union failure.

### 9.2 Clean-margin / |B| binning / AUC — **PENDING, not merely unfinished**

`scripts/dev/margin_analysis.py` exists (197 lines) and defines
`margin_i = logit_y(x_i) − max_{k≠y} logit_k(x_i)` on **clean** audit inputs ⟵ `margin_analysis.py:3`,
with a stated decision threshold `AUC ≳ 0.85` ⟵ `:7-9`.

**But `results/analysis/margin_analysis_2026-07-18.json` contains only four keys** — `date`, `subset`,
`auc_threshold`, `margin_def` — and **no AUC value, no binning, no per-model result**. The file is a metadata
stub. **⚠ No RQ4 margin/AUC number in this repo is traceable.** Any such value in the paper is `⚠ UNTRACED`.

`results/analysis/mask_analytics_2026-07-18.json` does carry `per_norm_delta`, `blocker_census`, `flip_analysis`,
`gate` ⟵ top-level keys, and is the only substantive RQ4 artefact. Its contents are **not audited in this
dossier** and should be treated as unverified until read.

Models actually audited for RQ4: **⚠ UNTRACED** (the brief names MSD-50 and RAMP; no result file confirms).

---

## 10. Known implementation caveats

| # | issue | status | detail |
|---|---|---|---|
| 1 | **APGD-DLR instead of APGD-T** | **OPEN — most severe** | `apgd_dlr_*` → `"apgd-dlr"` (untargeted) ⟵ `eval_multinorm_audit.py:46`. Standard AutoAttack uses targeted `apgd-t`, which is strictly stronger. Every union in this dossier is an **upper bound** relative to a standard evaluation. Affects all arms equally, so *paired deltas* are largely protected; *absolute* numbers are not comparable to published RobustBench figures. |
| 2 | **50-step base / 10-step views staleness** | **OPEN** | `msd_steps=50` with `train_apgd_niter=10` in the same run ⟵ `C5_full/*/train.json`. The reversal pair's CLAMP positives are 5× weaker than its base. `M1a_full_v50` (views at 50) exists **only @1k no_square** (0.4490); the decisive @10k run is PENDING. |
| 3 | **Compute-budget confound** | **OPEN, quantified** | `M0` crafts 1 attack/step (MSD); `M1a` crafts 1 + 3 = 4 ⟵ `c5_fromscratch.py:543, 549`. Measured wall-clock: M1a_full 365 s/ep, 8.11 h; RAMP 268 s/ep, 5.96 h ⟵ `docs/RESULTS_MASTER.md` §6. **The efficiency claim vs RAMP has been dropped** for this reason. The CLAMP-vs-control gain is *not* compute-matched: a reviewer can argue the control deserves 4× the attack budget. |
| 4 | **Cross-machine MAX pair** | **PARTIALLY CONTROLLED** | `M1a_max` trained + audited on Colab, `M0_max` local. Agreement check: @1k no_square Δ = **−0.0230**, @10k full Δ = **−0.0224** — two tiers, two scales, two machines, agreeing to 0.0006. Both audits used the same frozen subset indices (now shipped in the code zip so Colab cannot regenerate a different subset). Residual risk: different torch/CUDA on Colab is unrecorded (§0.2). |
| 5 | **Single-seed rows** | **OPEN** | #11 (MAX), #12 (AVG), M1a_msdglue, C100 pair, FT pair, v50 are **all n=1 seed**. Only CLAIM A (3 seeds) and the reversal (2 seeds) are replicated. |
| 6 | **Tier mixing** | **CONTROLLED** | Tier-aware paths + clobber guard ⟵ `eval_arm.py:51-69`; `n_attacks` recorded per scale-dir; `sync_results_main.py` re-verifies union vs masks (44/44). Measured no_square−full drift 0…+0.0010. |
| 7 | **Control skips view crafting entirely** | **OPEN** | Under `M0` the 3 per-norm views are never crafted (`xs_adv = None` ⟵ `:564`), so the pair differs in **RNG consumption**, not only in loss. Two runs at the same seed do not see the same data-augmentation/dropout stream after the first CLAMP-active step. |
| 8 | **cuDNN nondeterminism** | **OPEN** | No `cudnn.deterministic`/`benchmark` set; no `worker_init_fn`. Training is not bit-reproducible. |
| 9 | **Config hash ambiguity** | **OPEN** | `9162ce44` is the @1k config only; @10k is `4465ab20`; `eval.json` records **no** config hash. Provenance of which config produced which result rests on directory convention, not on a recorded field. |
| 10 | **Two CI conventions** | **OPEN (presentational)** | `[q05,q95]` in accuracy tables vs `[q025,q975]` in geometry tables (§7). Calling the former "95% CI" is imprecise. |
| 11 | **No multiplicity correction** | **OPEN** | ~15 pairwise tests, no adjustment. |
| 12 | **Data augmentation / BN** | **CONTROLLED by construction** | Both arms run the same `c5_fromscratch` loop, same loaders, same BN; the only difference is the loss branch and the crafted views. No separate augmentation code path exists per variant. |
| 13 | **Two masks-less rows** | **CONTROLLED** | `M1b/1k`, `Rprime/1k` are `eval.json`-only and are excluded from every bootstrap. |
| 14 | **`M1a_full` seed 0, `M0` seed 0, `M1a_max`, `M1b`, `M1a_msdglue` have no local `train.json`** | **OPEN** | Their recipes cannot be verified from a written record on this machine; only ckpt sha256 continuity to the audit is verified. |

---

## 11. Open items / gaps

**Submission is wording-only. Exactly ONE run gates the paper.**

| item | lands in | status |
|---|---|---|
| **`M1a_avg` @10k tier=full** | `results/main/M1a_avg/10k/` | **DONE 2026-07-22.** union 0.4049, clean 0.8331, 12/12 attacks, union(eval.json)==union(masks) exact. AVG row closed: **Δ +0.0203 [+0.0155, +0.0250], LCB +0.0155, significant +**. **No run now gates the paper.** |
| F1 exposure control (`--ce-views-w`) | — | **FUTURE WORK** — greenlight rescinded before launch; code path implemented and smoke-tested, `w = 1.6` measured (grad-norm ratio 1.560 at init / 1.842 trained ⟵ `scripts/dev/f1_grad_scale.py`) |
| F2 strength-matched views | `results/main/M1a_full_v50/10k/` | **DONE 2026-07-22** (Colab, 12/12 attacks cross-verified). Reversal **loses significance** with matched views: −0.0030 [−0.0081, +0.0021] vs −0.0096 [−0.0142, −0.0048] at views-10. It does **not** flip positive. n=1 seed. See §6.3. |
| M5 pull-only / push-only @10k | — | **FUTURE WORK** — never trained |
| clean-CE ablation | — | **NOT RUN** — drop the row or mark future work |
| α/β sweep | — | **NOT RUN** — only (0.5, 0.5) exists |
| RAMP + CLAMP | `results/main/B1/` | **DONE** — it is `B1` (was mislabelled as a RAMP baseline). Δunion −0.0005 [−0.0050, +0.0038] ns · Δℓ∞ −0.0037 · Δℓ₂ +0.0008 · Δℓ₁ +0.0201 · Δclean −0.0108 · route ℓ₁. Stray `union_bench/B2/` deleted 2026-07-22. |
| CIFAR-100 12-AA @10k | `results/main/{M1a_c100,M0_c100}/10k/` | **NOT RUN** (only @1k no_square exists) |
| RQ4 margin / AUC numbers | `results/analysis/margin_analysis_*.json` | **NOT RUN** — current file is a metadata stub (§9.2); any AUC in the paper is `⚠ UNTRACED` |
| T4 failure analysis | — | **NOT RUN** (unblocked but not required for submission) |
| Colab environment record | — | **MISSING** — no torch/CUDA version captured for Colab-executed runs |
| plotting scripts / figures | — | **MISSING** — none exist in the repo |
| `claim_evidence_ledger.md` | `docs/` | **MISSING**; `docs/paperC_master.md` exists but is **0 bytes** |
| multi-seed aggregation script | — | **MISSING** — seed-min LCB computed by hand (§7) |
| cudnn determinism flags | — | **MISSING** — training not bit-reproducible (§0.4) |

---

## § ℓ₂ geometry & per-example confirmation (Option-B probe)

Script: `scripts/dev/l2_geometry_probe.py` (CPU only; no GPU job, no attack re-run — Task 2 re-reads
frozen masks). Run 2026-07-22. **Registration discipline (G4): this probe was written and run AFTER
the ℓ₂ 0.0 % sole-blocker observation (T3, §9.1). It is a post-hoc explanation of an existing
finding, not a prediction. mtime-ordered only; no commit timestamp.**

### Constants — centred, UNCLIPPED balls in R^d, d = 3·32·32 = 3072

ε₁ = 12, ε_∞ = 8/255 = 0.0313725490, ε₂ = 0.5.

| quantity | definition | value |
|---|---|---|
| `r_box1` | ε₁/√d — largest ℓ₂ ball inside `B_1(ε₁)` alone | **0.2165** |
| `r_union` | min_u max(ε₁/‖u‖₁, ε_∞/‖u‖_∞) — inside `B_1 ∪ B_∞` | **0.2182** |
| `r_hull` | min_u max(ε₁‖u‖_∞, ε_∞‖u‖₁) — inside `conv(B_1 ∪ B_∞)` | **0.6140** |

Both radial functions are minimised over unit-ℓ₂ directions. For the **union**, each ray meets each
(convex, origin-containing) ball in an interval `[0,·]`, so the union's radial function is the *max*
of the two exit distances. For the **hull**, the relevant object is the support function, whose
dual-norm form gives the `ε‖u‖` expressions.

**Method and what actually found each minimum.** The `s`-spike family (`s` coords carrying ℓ₂ mass β,
remainder spread equally), `s ∈ {1..10}`, coarse grid 2·10⁵ + 6 refinements:
- `r_union`: minimum at **s = 1, β = 0.143851** → 0.218242. Scanning `s` up to 10 confirms `s = 1` is
  optimal (the value rises monotonically with `s`).
- `r_hull`: **the spike family is the wrong family here** — with `s ≤ 10` it only reaches 1.6135,
  because the hull optimum needs ≈383 active coordinates. The authoritative computation is the
  **exact integer k-sparse scan** over `k ∈ [1, 3072]`: minimum **0.613973 at k = 383**, against the
  continuous closed form `√(ε₁ε_∞) = 0.613572` with `k* = ε₁/ε_∞ = 382.5`. The gap **4.01e-04** is the
  integer constraint (k* is not an integer) and is inside the 1e-3 tolerance → **no flag**.

**Monte-Carlo sanity**, 10⁶ random unit directions, `rng(0)`, never undercut a structured minimum:
union MC min 0.283005 ≥ 0.218242 ✓; hull MC min 1.355304 ≥ 0.613973 ✓. (Random directions in 3072-d
are near-dense, so they land far from both optima — expected, and the check is one-sided anyway.)

### Zones for ε₂ = 0.5

| zone | condition | holds? |
|---|---|---|
| 1 — ℓ₂ trivially covered by the raw union | ε₂ ≤ r_union = 0.2182 | no |
| **2 — covered only via the convex-hull / affine argument** | 0.2182 < ε₂ ≤ 0.6140 | **YES** |
| 3 — geometry no longer protects | ε₂ > 0.6140 | no |

ε₂/r_union = **2.2910** · ε₂/r_hull = **0.8144**.

For a deep net the hull argument is **not** a guarantee — it holds exactly for affine classifiers.
At ε₂ = 0.5 the ℓ₂ ball sticks out of the raw union by a factor 2.29, so whether ℓ₂ adds anything is
an **empirical** question. Task 2 answers it.

### Per-example anatomy, @10k, 12 components, over union-FAILING examples

**23 arms** — every arm under `results/main/` carrying a 12-key @10k mask set: B1, M0, M0_avg,
M0_full, M0_full_seed1, M0_max, M0_seed2, M0_seed3, M1a, M1a_avg, M1a_full, M1a_full_seed1,
M1a_full_v50, M1a_max, M1a_msdglue, M1a_seed2, M1a_seed3, M1b, Rprime, b3_static_linf, b4_adaptive,
ft_clamp, ft_none.

`n_fail` 5539–6154. **ℓ₂-SOLE = 0.00 % on all 23.** `{ℓ₂,ℓ∞}` and `{ℓ₂,ℓ₁}` are 0.00 % everywhere
except `M0_full_seed1` `{ℓ₂,ℓ∞} = 0.02 %` (a single example). Consequently
**"ℓ₂ blocks" ≈ "all three block"** for every arm (e.g. M0 55.27 / 55.27; Rprime 62.29 / 62.29):
ℓ₂ blocking is essentially *nested* inside all-three blocking, never a partial or exclusive blocker.
Full per-arm table: run the script.

### Consistency (Task 3): **PASS**

(i) ε₂ = 0.5 is in zone 2 — outside the raw union, inside the hull. (ii) ℓ₂-sole = 0.00 % across all
23 arms (max observed 0.0000 %). The hull-level protection, guaranteed only for affine classifiers,
**empirically holds for these deep nets at this ε-triple** (scope rider L7).

### Ledger line (to be pasted by CLAUDEUS — `claim_evidence_ledger.md` lives in paper1, not here)

> ℓ₂ zone-2 geometry + 0.00 % ℓ₂-sole across 23 @10k mask sets — `scripts/dev/l2_geometry_probe.py`,
> dossier § ℓ₂ geometry; r_union 0.2182, r_hull 0.6140, ε₂/r_union 2.29; post-hoc to T3, not preregistered.

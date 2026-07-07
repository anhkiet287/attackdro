# Progress log

## 2026-07-01
- **Done:**
  - Scaffolded the minimal PGD-AT training path under the protocol:
    - `configs/base.yaml` (THE protocol: threat model linf/l2/l1 eps, eval-attack knobs, wandb) + `configs/pgd_at.yaml` (inherits base via `_base_`).
    - `src/robustdro/`: `data/datasets.py` (CIFAR-10, raw [0,1]), `models/preact_resnet.py` (PreActResNet-18), `attacks/norms.py` (PGD-linf), `training/trainer.py` (PGD-AT loop + cheap robust probe + checkpoint + JSON results), `utils/` (`io.py` config `_base_` merge, `seed.py`, `wandb_log.py`).
    - `scripts/train.py` thin CLI (`--config`, `--smoke`, `--wandb-mode`, overrides).
  - Reusable W&B logging: `WandbLogger` (config-driven, 3 modes, never crashes a run, always echoes stdout + writes `results/<run>.json`). One-time `wandb login` documented in `SETUP.md` §7b.
  - Env re-verified: torch 2.11.0+cu128, RTX 5070 Ti, capability (12,0), real GPU matmul OK.
  - **Smoke test VERIFIED on real CIFAR-10.** torchvision integrity-checked the local batches and skipped download; loss=2.32 finite, GPU, artifacts written. (Also passed a synthetic-FakeData smoke first while the dataset was blocked.)
  - **CIFAR-10 resolved:** the canonical mirror was down, but a local copy from the thesis repo (`/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/data`) was found, md5-verified (`c58f30108f718f92721af3b95e74349a` ✓), and copied into `data/`.
  - **Full PGD-AT run LAUNCHED** in tmux `pgdat`, W&B online (run `pgd_at_linf`, entity `kietna-...`), log at `logs/pgdat.log`. 100 epochs, ~57s/epoch → ETA ~1h40m.
  - **Sanity gate PASSED** (first 3 epochs): robust_acc_pgd 0.2597 → 0.2755 → 0.2894 (monotonic climb, >0), clean 0.34 → 0.44, loss 2.10 → 1.88, no NaN.
  - **Attack sanity tool**: `scripts/visualize_attack.py` plots clean | perturbation | adversarial for N images (CPU by default so it doesn't touch the training GPU). Supports `--norm {linf,l2,l1}` (linf via in-repo PGD, l2/l1 via APGD), reports linf/l2/l1 of each perturbation + a budget assertion. Verified all three: max linf=8.000/255, l2=0.500, l1=12.000 — each exactly at budget. l1 perturbations are visibly sparse (few pixels, huge per-pixel linf ~200/255) — textbook. Figures: `results/figures/attack_viz_{linf,l2,l1}.png`.
  - **eval_union built (P0 linchpin).** `src/robustdro/eval/eval_union.py` + `attacks_aa.py` (AutoAttack wrapper, `version: apgd`=APGD-CE+APGD-T fast, `standard`=full AA) + `attacks/union.py` (union rule = AND of per-norm robust masks). `scripts/evaluate.py` thin CLI (`--checkpoint --n-examples --norms --version`). Independent of training; same harness for our checkpoints + downloaded baselines. autoattack pinned in requirements (git).
  - **First real union eval** (epoch-12 ckpt, 128 imgs, APGD): clean 70.3%, robust linf 35.2% / l2 51.6% / **l1 5.5%**, avg 30.7%, **worst-union 5.5%**. Sanity checks pass: APGD-linf 35.2% ≈ in-training PGD probe (~35%); worst-union collapses to l1 — the expected "l1 bottleneck" for an linf-only model.
  - **BASELINE VALIDATION PASSED (golden rule #3).** Downloaded all 6 official `locuslab/robust_union` CIFAR10 checkpoints (Google Drive → `external/robust_union/CIFAR10/Selected/`: MSD, LINF, L2, L1, AVG, MAX; loaded with THEIR PreActResNet18, raw [0,1], no normalization — matches ours). `scripts/validate_baseline.py` runs OUR eval_union under THEIR protocol (eps_inf=0.03, first 1000, APGD). MSD result vs their reported (PGD/10-restart):
    | metric | ours (APGD) | reported (PGD) | Δ |
    |---|---|---|---|
    | clean | 82.1 | 81.7 | +0.4 |
    | linf  | 44.6 | 47.6 | −3.0 |
    | l2    | 64.5 | 64.3 | +0.2 |
    | l1    | 46.7 | 53.4 | −6.7 |
    | union | 42.5 | 46.1 | −3.6 |
    Every delta is ≤0 within noise (clean/l2 match; linf/l1/union lower) — exactly right: our AutoAttack is stronger than their PGD. The l1 −6.7 gap is a real signal (their PGD-L1-topk over-stated l1 robustness; AA-L1 finds more). → **eval_union is trustworthy; P0 done.**
- **Results (worst-union / per-norm):** MSD baseline under our harness (eps_inf=0.03): worst-union **42.5%** (n=1000, APGD). Our PGD-AT model still training.
- **Open issues:**
  - In-training robust acc uses a cheap PGD probe; final numbers MUST come from `eval_union` (golden rule #4).
  - `--version standard` (full AutoAttack) untested for timing on 10k; only `apgd` (=APGD-CE+APGD-T) used so far. Consider it for final numbers.
  - Our protocol uses eps_inf=8/255 (0.0314); robust_union uses 0.03. When comparing OUR models to these baselines, decide one eps and re-eval all under it (validation above used 0.03 to match their table).
  - `configs/default.yaml` + `src/train.py` are the old toy MLP; keep or delete later.
- **Next step:** when `pgdat` finishes, `scripts/evaluate.py --checkpoint checkpoints/pgd_at_linf_best.pt -n 1000` for our model's worst-union; then start **P1** — probe AttackDRO++ in the union setting vs these validated baselines (MSD union 42.5% is the number to beat).

## 2026-07-01 — P1
- **STEP 0 DONE — protocol LOCKED.** `configs/base.yaml` threat model is now the single protocol for train AND eval: **eps_inf=0.03, eps_l2=0.5, eps_l1=12** (switched eps_inf 8/255→0.03 to match the validated locuslab baselines). Verified `pgd_at.yaml` inherits eps_inf=0.03. Attack-version policy: `apgd` (APGD-CE+APGD-T) for all iteration/comparison; `standard` (full AA) reserved for final numbers. NOTE: the running `pgdat` model was trained at the OLD 8/255 — it stays a pipeline proof, not a table baseline (the locuslab LINF ckpt @0.03 is the proper l_inf-AT baseline).
- **AttackDRO++ method understood** (from thesis `ardg`): GroupDRO with **attacks-as-groups** — L = Σ_g q_g·CE(f(A_g x), y); difficulty-aware weights `log_q[g] += eta_q·loss_g` then softmax (upweights hardest attack). Thesis used **only linf+l2** (pgd.py rejects l1); **P1 adds l1 as a source attack**.
- **STEP 3 code BUILT + smoke-passed** (ahead of Steps 1/2 completion, which are GPU-eval-bound):
  - Training l2/l1 attacks ported (fp32, [0,1]) from robust_union into `attacks/norms.py` (`pgd_l2`, `pgd_l1_topk` + l1 top-k / simplex projection). Budget smoke: linf 0.0300, l2 0.4999, l1 12.0000 — exact, no NaN.
  - `training/losses.py` (`GroupDRO`: log-space q update), `attacks/sources.py` (spec→attack factory), `training/groupdro.py` (`GroupDROTrainer`: 3 source attacks/step, q update, weighted loss, cheap union probe). `configs/attackdro.yaml` inherits locked base. `scripts/train.py` dispatches `method: attackdro`.
  - Smoke (2 steps, offline): groups=[linf,l2,l1] eps=(0.03,0.5,12), q init uniform 0.333, per-group losses ~2.31, union probe + artifacts written, no NaN. → ready to train once GPU frees.
- **STEP 1 DONE — FIXED BASELINE TABLE** (locked protocol eps_inf=0.03, n=1000, APGD-CE+APGD-T). `results/baseline_table.{md,json}`:

  | model | clean | l_inf | l2 | **l1** | **worst-union** | (their reported union) |
  |---|---|---|---|---|---|---|
  | **MSD** | 82.1 | 44.6 | 64.5 | **46.7** | **42.5** | 46.1 |
  | AVG | 84.6 | 40.7 | 65.5 | **47.7** | **38.7** | 40.6 |
  | MAX | 81.7 | 39.2 | 62.1 | **26.2** | **25.0** | 34.9 |
  | LINF | 83.9 | 47.7 | 57.7 | **7.0** | **7.0** | 15.6 |
  | L2 | 90.7 | 22.7 | 63.1 | **21.1** | **17.5** | 27.5 |
  | L1 | 73.7 | 0.0 | 0.1 | **0.0** | **0.0** | 0.0 |

  - **Target to beat: MSD worst-union = 42.5%.** Every model ≤ its reported union (our APGD is stronger than their PGD) — table is internally consistent.
  - **l1 is the bottleneck everywhere:** worst-union = the l1 column for MSD/LINF/L2/L1 (l1 is each model's weakest norm). Our APGD-l1 is notably stronger than their PGD-l1 (LINF l1 7.0 vs reported 16.0; MAX 26.2 vs 39.4) — published l1 robustness was overstated.
  - L1-only model collapses to 0 across the board (over-specialization). LINF-only collapses on l1 (7.0). This is why single-norm AT fails the union.
- **UNATTENDED PIPELINE running** (tmux `p1pipe`, `scripts/run_p1_pipeline.sh`, self-contained — survives disconnect). After pgdat finishes it auto-runs: Step 2 (eval pgdat best, n=1000) → Step 3b (train AttackDRO++ seeds 0/1/2, eval_union each) → **auto-writes a GO/NO-GO verdict block to PROGRESS.md + `results/p1_summary.md`** (compares best/mean AttackDRO++ union vs MSD 42.5%, with the training-recipe-confound caveat baked in). Incremental: seed-0 result ≈ +3.5h, all seeds ≈ +9–10h. A human-reviewed writeup follows when I'm next active.

## 2026-07-02 — P1 Steps 2–3 status (for planner)
**TL;DR:** Steps 0, 1, 3a done. Step 2 done. Step 3b (AttackDRO++ training) IN PROGRESS — first APGD result in ~2h, all 3 seeds in ~10h. No GO/NO-GO yet.

- **Process note (honest):** the unattended overnight pipeline STALLED and made no progress for ~16h — the wait-loop matched the tmux wrapper's command string (a ghost process), so it "waited for pgdat" forever. pgdat itself DID finish. Bug fixed (gate on output files + specific pgrep); pipeline relaunched and now progressing.

- **STEP 2 DONE — our PGD-AT reference** (`pgd_at_linf_best`, trained at OLD 8/255, eval at locked 0.03, n=1000, APGD):
  clean 82.0 | l_inf 48.9 | l2 58.2 | **l1 9.4** | avg 38.8 | **worst-union 9.4%**.
  As expected, an l_inf-only model collapses on l1 → worst-union = l1. (Its l_inf 48.9% even tops locuslab LINF's 47.7% because it trained at eps 8/255 > 0.03; still NOT a table baseline — recipe/eps differ. The locuslab LINF ckpt @0.03 is the proper l_inf-AT baseline: union 7.0%.)

- **STEP 3b IN PROGRESS — AttackDRO++ (GroupDRO over linf+l2+l1, l1 ADDED):** seed 0 at epoch ~22/50, ~4 min/epoch (~2h/seed; 3 seeds ~10h). Eval + auto GO/NO-GO written per seed to `results/p1_summary.md` and PROGRESS.md.
  - Early mechanism signal (in-training CHEAP PGD probe — NOT the final APGD number): the difficulty-aware weights collapse to **q ≈ (l_inf 0.82, l2 0.00, l1 0.18)** — GroupDRO starves the easy l2 group and concentrates on l_inf/l1. Probe worst-union ~40% at epoch ~22. TREAT AS PROVISIONAL; the authoritative number is APGD eval_union after training, which will be lower.

- **Target (from Step 1 table): MSD worst-union = 42.5%** (APGD, n=1000). AttackDRO++ must beat this to be a GO.

- **Next:** let 3 seeds finish → APGD eval each → auto verdict → I write the human-reviewed GO/NO-GO (with the training-recipe-confound caveat: baselines use locuslab's recipe, AttackDRO++ uses ours; a fully controlled table needs AVG/MAX retrained in-house). Expectation per plan: AttackDRO++ likely does NOT beat MSD on worst-case (thesis improved average-case) — an informative negative result that triggers P2.

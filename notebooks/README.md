# Colab notebooks

Four notebooks, one per purpose. All run on Colab, mount Drive, and load code by extracting
`attackdro_code.zip` from Drive (the GitHub repo is private, so Colab cannot clone it).

| notebook | purpose |
|---|---|
| **`train_colab.ipynb`** | **Train** the full-budget arms (Maini MSD recipe + CLAMP term). Resume-safe, Drive-direct, W&B. `--variant M1a\|M0`. |
| **`eval_colab.ipynb`** | **Eval** — the audit gate. Frozen 12-AA (`9162ce44`): merge → per-norm concordance/masking → gated audit queue, fast-review PROBE, PROBE_LAST diagnostic, MSD baseline, paired bootstrap. Writes to Drive `union_bench/`. |
| **`analysis_perclass_colab.ipynb`** | **Analysis** — per-class × norm breakdown (M1a vs M0, bootstrap CIs, sign test, bottom-3 lift). CPU-only. |
| **`explore_colab.ipynb`** | **Explore** — non-paper idea probes. Not paper-grade; kept separate from the arms above. |

## legacy/
Superseded and pre-CLAMP (program-A) notebooks, archived not deleted:
`audit_colab` (→ superseded by `eval_colab`), `M0_colab` / `M1b_colab` (per-arm trainers, now
trained locally / via `train_colab`), `Bet1_ramp_pullpush_colab` / `ClaimB_colab` (Claim-B RAMP
trainers, runs complete), `avg_bindaware_rerun_8_255` / `colab_anchors_8255` / `colab_avg_frozen`
/ `generality_ramp_predalloc_colab` (allocation-study / baseline runners).

## Conventions
- Re-upload `attackdro_code.zip` whenever the trainer/harness changes; the setup cell asserts the
  zip is current and fails with the exact missing files if it is stale.
- All audit outputs are Drive-direct under `MyDrive/attackdro/union_bench/`.
- Mask convention: `true` = robust.

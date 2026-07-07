# RAMP Baseline Integration

This note records the upstream RAMP checkout and the commands for evaluating a
RAMP checkpoint under our independent worst-union harness.

## Upstream Checkout

Repository:

```bash
git clone https://github.com/uiuc-focal-lab/RAMP external/RAMP
```

Current cloned commit:

```text
be4971f04cf8e70bd8255874a1ed2ab489cae682
```

Keep `external/RAMP` as an external source checkout. Do not vendor or edit their
code unless a future card explicitly requires it.

Local compatibility patch: upstream `external/RAMP/utils.py` calls
`copy.deepcopy(...)` in `gp()` but is missing `import copy`. Because the official
RAMP command uses `--gp`, our checkout adds that one import. No algorithmic code
is changed.

## Model Format

RAMP's CIFAR-10 ResNet-18 model is:

```python
from model_zoo.fast_models import PreActResNet18
model = PreActResNet18(10, activation="softplus1", normal="none")
```

`normal="none"` means no input normalization: the model expects raw `[0,1]`
CIFAR-10 tensors, matching our evaluator.

RAMP usually saves a plain PyTorch `state_dict`. If `--save_optim` is used, the
checkpoint is wrapped as `{"state_dict": ..., "optim": ...}`. Our loader supports
both forms and strips a leading `module.` prefix if present.

## Checkpoint Availability

The cloned upstream repo currently contains only:

```text
external/RAMP/models/pretr_L1.pth
external/RAMP/models/pretr_L2.pth
external/RAMP/models/pretr_Linf.pth
```

These are pretraining/fine-tuning starting points, not the final RAMP
train-from-scratch checkpoint behind the reported CIFAR-10 result. Do not claim
we evaluated RAMP's reported `44.6%` union / `81.2%` clean model unless a final
RAMP checkpoint is trained locally or obtained from the authors.

## Smoke Test

This only verifies import + checkpoint loading. It does not load data or run
attacks.

```bash
python scripts/smoke_ramp_loader.py \
  --checkpoint external/RAMP/models/pretr_Linf.pth \
  --device cpu
```

If `external/RAMP` or the checkpoint is missing, the smoke test skips clearly.

## Train RAMP From Scratch

Run RAMP's own code from inside the upstream checkout. Upstream's script names
seed 0 as `RAMP_beta_0.5_lbd_5_0`, so the seed-0 final checkpoint path is:

```text
external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth
```

Preferred wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/ramp/train_ramp_seed0.sh
```

The wrapper uses `${PYTHON}` if set, otherwise `.venv/bin/python` if present,
otherwise `python`. It prints the expected checkpoint path at the end and fails
if the file was not written.

W&B logging: upstream RAMP does not log to W&B natively. Our wrapper runs
`scripts/ramp/train_ramp_with_wandb.py`, which launches RAMP unchanged, echoes
its stdout, parses each `[epoch] ...` line, and logs live metrics to the
`union-robustness-dro` W&B project. You should see the W&B run initialize near
the start of training. It logs `ramp_live/train_loss` every epoch and
`ramp_live/test_*` / `ramp_live/train_*` every `--eval_freq` epochs. At the end,
it reads RAMP's `metrics.pth`, writes
`results/ramp_RAMP_beta_0.5_lbd_5_<seed>.json`, and logs final summary metrics.
Metrics are prefixed with `ramp_live/` or `ramp_internal/` because they come
from RAMP's own training/eval path, not from our independent `eval_union`
harness.

Control logging mode the same way as our native runs:

```bash
RAMP_WANDB_MODE=online  CUDA_VISIBLE_DEVICES=0 bash scripts/ramp/train_ramp_seed0.sh
RAMP_WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 bash scripts/ramp/train_ramp_seed0.sh
RAMP_WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 bash scripts/ramp/train_ramp_seed0.sh
```

Manual re-log after a run:

```bash
.venv/bin/python scripts/ramp/log_ramp_metrics.py \
  --run-dir external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0 \
  --run-name ramp_seed0 \
  --seed 0 \
  --wandb-mode offline
```

Dependency note: RAMP imports `robustbench` at module import time. The training
wrappers check this before launching and stop with a clear message if RAMP's
dependencies are not installed. Install from `external/RAMP/requirements.txt`
or use a RAMP-ready environment via `PYTHON=/path/to/python`.

Equivalent upstream command:

```bash
cd external/RAMP
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python RAMP.py \
  --lr-max 0.05 \
  --lr-schedule=static \
  --at_iter 10 \
  --epochs 80 \
  --save_freq 10 \
  --eval_freq 10 \
  --fname RAMP_beta_0.5_lbd_5_0 \
  --kl \
  --max \
  --final_eval \
  --gp \
  --lbd 5 \
  --seed 0
```

RAMP's script writes checkpoints under:

```text
external/RAMP/trained_models/<fname>/ep_<epoch>_<iteration>.pth
```

For seed 0:

```text
external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth
```

## Reproduction Plan

### Train Seed 0

Start with one seed only:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/ramp/train_ramp_seed0.sh
```

This runs 80 epochs and also triggers RAMP's internal `--final_eval`. Treat that
internal final eval as RAMP's own diagnostic, not as our independent result.

### Evaluate Seed 0 Under Our Harness

Set:

```bash
CKPT=external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth
```

RAMP-comparable APGD CE+T at `eps_inf=8/255`:

```bash
bash scripts/ramp/eval_ramp_seed0_apgd_eps8_255.sh
```

Full AutoAttack standard at `eps_inf=8/255`:

```bash
bash scripts/ramp/eval_ramp_seed0_aa_standard_eps8_255.sh
```

Our locked protocol at `eps_inf=0.03`:

```bash
bash scripts/ramp/eval_ramp_seed0_apgd_eps003.sh
bash scripts/ramp/eval_ramp_seed0_aa_standard_eps003.sh
```

Each eval wrapper defaults to `N_EXAMPLES=10000`, `BS=250`, and the seed-0
checkpoint above. Override as needed:

```bash
N_EXAMPLES=1000 CKPT=/path/to/ep_80_0.pth bash scripts/ramp/eval_ramp_seed0_apgd_eps8_255.sh
```

### When To Run Seeds 1-2

Decision rule:

- If seed-0 APGD CE+T at `eps_inf=8/255` is around `44-45%` worst-union, run
  seeds 1-2:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash scripts/ramp/train_ramp_seeds_0_2.sh
  ```

- If seed 0 is far below `44%`, debug before spending more GPU.
- If full AutoAttack standard is much lower than APGD CE+T, report both and use
  full AA standard for final claims.

### Debugging Protocol Mismatch

If seed 0 misses badly, check these before launching more seeds:

- RAMP model constructor is `PreActResNet18(10, activation="softplus1",
  normal="none")`.
- Inputs are raw `[0,1]` CIFAR-10 tensors, no normalization.
- Seed-0 path is `RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`, not
  `RAMP_beta_0.5_lbd_5_seed0/...`.
- RAMP-comparable eval uses `eps_inf=8/255`, `eps_l2=0.5`, `eps_l1=12`.
- Our `--version apgd` is APGD-CE + APGD-T, but the config default is 1 restart.
  For RAMP parity, set `eval_attack.{linf,l2,l1}.restarts=10`.
- The union metric is still the AND mask across all three norms.

## Evaluate With Our Harness

Use `scripts/evaluate.py --model_family ramp`. Our evaluator keeps the same
worst-union rule:

```text
union_mask = linf_mask & l2_mask & l1_mask
```

The default `configs/base.yaml` protocol uses `eps_inf=0.03`. For direct RAMP
comparison, override `eps_inf` to `8/255 = 0.03137254901960784`.

Set `CKPT` to a trained RAMP checkpoint, for example:

```bash
CKPT=external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth
```

### A. RAMP-Comparable APGD CE+T

RAMP's own final eval uses APGD-CE + APGD-T with 10 restarts and 100 steps. Our
`--version apgd` maps to APGD-CE + APGD-T. The step count comes from
`eval_attack.*.steps` and is 100 in `configs/base.yaml`; the default restart
count is 1, so RAMP parity requires the `--set eval_attack.*.restarts=10`
overrides below.

Direct RAMP protocol, `eps_inf=8/255`:

```bash
python scripts/evaluate.py \
  --model_family ramp \
  --config configs/base.yaml \
  --checkpoint "$CKPT" \
  --n-examples 10000 \
  --version apgd \
  --bs 250 \
  --out results/eval_ramp_eps8_255_apgd.json \
  --set threat_model.linf.eps=0.03137254901960784 \
  --set eval_attack.linf.restarts=10 \
  --set eval_attack.l2.restarts=10 \
  --set eval_attack.l1.restarts=10
```

Equivalent wrapper:

```bash
bash scripts/ramp/eval_ramp_seed0_apgd_eps8_255.sh
```

Our locked protocol, `eps_inf=0.03`:

```bash
python scripts/evaluate.py \
  --model_family ramp \
  --config configs/base.yaml \
  --checkpoint "$CKPT" \
  --n-examples 10000 \
  --version apgd \
  --bs 250 \
  --out results/eval_ramp_eps003_apgd.json \
  --set eval_attack.linf.restarts=10 \
  --set eval_attack.l2.restarts=10 \
  --set eval_attack.l1.restarts=10
```

Equivalent wrapper:

```bash
bash scripts/ramp/eval_ramp_seed0_apgd_eps003.sh
```

For a quick check before a full 10k run, replace `--n-examples 10000` with
`--n-examples 1000`.

### B. Full AutoAttack Standard

For final paper claims, prefer full AutoAttack `standard` when feasible. Our
harness runs each norm separately and computes the same AND union mask.

Direct RAMP protocol, `eps_inf=8/255`:

```bash
python scripts/evaluate.py \
  --model_family ramp \
  --config configs/base.yaml \
  --checkpoint "$CKPT" \
  --n-examples 10000 \
  --version standard \
  --bs 250 \
  --out results/eval_ramp_eps8_255_standard.json \
  --set threat_model.linf.eps=0.03137254901960784
```

Equivalent wrapper:

```bash
bash scripts/ramp/eval_ramp_seed0_aa_standard_eps8_255.sh
```

Our locked protocol, `eps_inf=0.03`:

```bash
python scripts/evaluate.py \
  --model_family ramp \
  --config configs/base.yaml \
  --checkpoint "$CKPT" \
  --n-examples 10000 \
  --version standard \
  --bs 250 \
  --out results/eval_ramp_eps003_standard.json
```

Equivalent wrapper:

```bash
bash scripts/ramp/eval_ramp_seed0_aa_standard_eps003.sh
```

## Interpretation Rules

- Do not mix per-norm average robustness with worst-union robustness.
- Do not use RAMP's training-time or our training-time probes as final numbers.
- Do not claim SOTA from infrastructure alone.
- Until a final RAMP checkpoint exists locally, the RAMP `44.6%` number remains a
  cited prior result, not a result re-evaluated by our harness.
- Do not compare against the reported `44.6%` as if it were our result unless we
  have trained or obtained a final RAMP checkpoint and evaluated it here.

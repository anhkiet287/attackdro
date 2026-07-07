#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
SEED=0
PY="$(python_bin_from_repo)"
CKPT="$(ramp_checkpoint_for_seed "$SEED")"

require_ramp_checkout
mkdir -p "$ROOT/external/RAMP/trained_models"
check_ramp_training_deps "$PY"

echo "[ramp-train] repo: $ROOT/external/RAMP"
echo "[ramp-train] seed: $SEED"
echo "[ramp-train] python: $PY"
echo "[ramp-train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT/external/RAMP"
"$PY" "$ROOT/scripts/ramp/train_ramp_with_wandb.py" \
  --ramp-root "$ROOT/external/RAMP" \
  --python "$PY" \
  --seed "$SEED" \
  --fname "RAMP_beta_0.5_lbd_5_${SEED}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}" \
  --wandb-mode "${RAMP_WANDB_MODE:-${WANDB_MODE:-online}}" \
  --out "$ROOT/results/ramp_RAMP_beta_0.5_lbd_5_${SEED}.json"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: expected checkpoint was not written: $CKPT" >&2
  exit 1
fi

echo "[ramp-train] checkpoint: $CKPT"

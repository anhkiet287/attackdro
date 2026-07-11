#!/usr/bin/env bash
# RAMP generality read on the 5070ti (EXPLORATION lane; Colab credits exhausted).
# Arm A = RAMP-full (FLOPs 1.0). Arm B = RAMP + predictive allocation (pred_alloc, pred-once fix
# -> FLOPs ~0.60). Same recipe/seed/data; only Arm B's per-sample attack budget shrinks. Writes
# ONLY to results/exploration/ (gitignored). Does NOT touch the paper frontier or results/ files.
#
# GPU is SERIAL — launch this ONLY after the paper frontier finishes (check: pgrep -af scripts/train.py).
# Usage:  bash scripts/ramp/run_generality_5070ti.sh          # ep25 clean read, both arms
#         EPOCHS=80 bash scripts/ramp/run_generality_5070ti.sh # proper-length run
#         RUN_ARM_A=0 bash scripts/ramp/run_generality_5070ti.sh # skip Arm A (reuse Colab 38.4)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
ROOT="$(repo_root)"
PY="$(python_bin_from_repo)"
require_ramp_checkout
check_ramp_training_deps "$PY"

EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-0}"
FLOOR="${PREDALLOC_FLOOR:-2}"
RUN_ARM_A="${RUN_ARM_A:-1}"
DATA="${DATA:-$ROOT/data}"                  # local CIFAR (torchvision layout: $DATA/cifar-10-batches-py)
OUT="$ROOT/results/exploration"; mkdir -p "$OUT"

compact_ramp_log() {
  # RAMP uses tqdm progress bars, which are useful interactively but explosive in
  # saved logs. Convert carriage returns to records, keep setup/eval/final-epoch
  # summaries and errors, and drop per-batch progress updates.
  tr '\r' '\n' | grep --line-buffered -E \
    '^\[generality|^\[train\]|^\[eval\]|^\[predalloc\]|^[[:space:]]*100%\||robust accuracy|clean accuracy|attack_flops_ratio|DONE|ERROR|Error|Traceback|Exception' || true
}

if [[ ! -d "$DATA/cifar-10-batches-py" ]]; then
  echo "ERROR: CIFAR-10 not at $DATA/cifar-10-batches-py — set DATA=..." >&2; exit 1
fi

# Regenerate RAMP_predalloc.py from pristine RAMP.py + drop in pred_alloc (idempotent, verifies anchors).
cp "$ROOT/scripts/ramp/pred_alloc.py" "$ROOT/external/RAMP/pred_alloc.py"
( cd "$ROOT" && "$PY" scripts/dev/make_ramp_predalloc.py )
"$PY" -m py_compile "$ROOT/external/RAMP/RAMP_predalloc.py"

BASE=(--lr-max 0.05 --lr-schedule=static --at_iter 10 --save_freq 10
      --kl --max --gp --lbd 5 --seed "$SEED" --final_eval
      --data_dir "$DATA" --epochs "$EPOCHS" --eval_freq "$EPOCHS")

echo "[generality-5070ti] epochs=$EPOCHS seed=$SEED floor=$FLOOR data=$DATA out=$OUT"
cd "$ROOT/external/RAMP"

if [[ "$RUN_ARM_A" == "1" ]]; then
  echo "[generality-5070ti] Arm A = RAMP-full (FLOPs 1.0) ..."
  "$PY" -u RAMP.py "${BASE[@]}" --fname armA_rampfull_5070ti \
    2>&1 | compact_ramp_log | tee "$OUT/ramp_armA_full_5070ti.log"
fi

echo "[generality-5070ti] Arm B = RAMP + predictive allocation (target ~0.60 FLOPs) ..."
PREDALLOC_FLOOR="$FLOOR" "$PY" -u RAMP_predalloc.py "${BASE[@]}" --fname armB_predalloc_5070ti \
  2>&1 | compact_ramp_log | tee "$OUT/ramp_armB_predalloc_5070ti.log"

echo
echo "[generality-5070ti] DONE. Read the comparison:"
echo "  Arm A (FLOPs 1.0):  grep 'robust accuracy' $OUT/ramp_armA_full_5070ti.log | tail"
echo "  Arm B (FLOPs ~0.60): grep -E 'attack_flops_ratio|robust accuracy' $OUT/ramp_armB_predalloc_5070ti.log | tail"
echo "  SIGNAL (1-seed, not verdict): union(B) ~ union(A) at ~0.60 FLOPs => allocation transfers to RAMP."

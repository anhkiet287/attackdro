#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
PY="$(python_bin_from_repo)"
CKPT="${CKPT:-$(ramp_checkpoint_for_seed 0)}"

require_ramp_checkout
require_checkpoint "$CKPT"
mkdir -p "$ROOT/results"

cd "$ROOT"
"$PY" scripts/evaluate.py \
  --model_family ramp \
  --config configs/base.yaml \
  --checkpoint "$CKPT" \
  --n-examples "${N_EXAMPLES:-10000}" \
  --version apgd \
  --bs "${BS:-250}" \
  --out results/eval_ramp_seed0_eps003_apgd.json \
  --set eval_attack.linf.restarts=10 \
  --set eval_attack.l2.restarts=10 \
  --set eval_attack.l1.restarts=10

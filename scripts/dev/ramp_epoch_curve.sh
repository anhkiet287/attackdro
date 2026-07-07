#!/bin/bash
# RAMP epoch curve — re-eval RAMP's SAVED checkpoints at 8/255 (n=1000 apgd r=1, matched to
# the 46.1 comparator) to quantify the RECIPE (epochs) contribution vs method. Evals only
# what exists (ep_10..80, save_freq=10); does NOT retrain RAMP. ep40/ep50 already done.
set -u
ROOT=/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO
cd "$ROOT" || exit 1
export WANDB_MODE=online
PY=.venv/bin/python
CKDIR=external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0
for EP in 10 20 30 60 70 80; do
  out=results/eval_ramp_ep${EP}_eps8255_apgd_n1000.json
  [ -f "$out" ] && { echo "SKIP ep$EP (done)"; continue; }
  echo "===== RAMP ep_${EP} eval $(date -u +%T) ====="
  $PY scripts/evaluate.py --model_family ramp --config configs/base.yaml \
    --checkpoint $CKDIR/ep_${EP}_0.pth -n 1000 --version apgd --bs 125 \
    --run-name ramp_repro_ep${EP} --tier repro --out "$out" || echo "FAIL ep$EP"
done
$PY scripts/dev/ramp_epoch_curve_table.py
echo "===== RAMP EPOCH CURVE DONE $(date -u +%T) ====="

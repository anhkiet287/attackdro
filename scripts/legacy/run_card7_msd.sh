#!/usr/bin/env bash
# CARD-7 msd_inhouse — gated: waits for finalpipe to finish (do not preempt).
set -uo pipefail
cd /mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO
PY=.venv/bin/python
log() { echo "[card7 $(date +%H:%M:%S)] $*"; }
log "waiting for finalpipe to finish..."
while tmux has-session -t finalpipe 2>/dev/null; do sleep 120; done
log "finalpipe done. TRAIN msd_inhouse_s0 (~6h: 50 MSD steps x 4 passes/step)"
$PY scripts/train.py --config configs/msd_inhouse.yaml --seed 0 --run-name msd_inhouse_s0 \
    > logs/msd_inhouse_s0.log 2>&1 || { log "TRAIN FAILED"; exit 1; }
$PY scripts/evaluate.py --config configs/msd_inhouse.yaml \
    --checkpoint checkpoints/msd_inhouse_s0_best.pt -n 1000 --version apgd \
    --out results/eval_msd_inhouse_s0.json > logs/eval_msd_inhouse_s0.log 2>&1 || { log "EVAL FAILED"; exit 1; }
$PY - <<'PYEOF'
import json
m = json.load(open("results/eval_msd_inhouse_s0.json"))["metrics"]
row = (f"| msd_inhouse_s0 | {m['clean_acc']*100:.1f} | {m['per_norm_robust_acc']['linf']*100:.1f} | "
       f"{m['per_norm_robust_acc']['l2']*100:.1f} | {m['per_norm_robust_acc']['l1']*100:.1f} | "
       f"**{m['worst_union_acc']*100:.1f}** |")
open("results/p2_summary.md","a").write(row+"\n"); print(row)
PYEOF
log "CARD-7 COMPLETE"

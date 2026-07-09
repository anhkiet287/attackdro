#!/usr/bin/env bash
# =============================================================================
# CARD-3b T-sweep — gated launcher (Kiet-approved scheduling 2026-07-02):
#   1. WAIT for the CARD-3a pilot APGD eval (results/eval_bindaware_s0.json).
#   2. Kill the p2pipe session BEFORE its local avg_frozen stage runs
#      (CARD-1 avg_frozen ×3 moved to Colab) — keeps ≤2 trainers on the GPU.
#   3. Run T-sweep {0.25, 0.5, 1, 2}, 1-seed pilots, sequentially; eval each.
# Rows append to results/p2_summary.md. tmux: p3bsweep.
# =============================================================================
set -uo pipefail
cd /mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO
PY=/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO/.venv/bin/python
SUMM=results/p2_summary.md
log() { echo "[3bsweep $(date +%H:%M:%S)] $*"; }

log "waiting for CARD-3a pilot eval (results/eval_bindaware_s0.json)..."
until [ -f results/eval_bindaware_s0.json ]; do sleep 60; done
log "3a pilot landed. Retiring p2pipe (avg_frozen moved to Colab)."
sleep 30   # let p2pipe finish appending its summary row
tmux kill-session -t p2pipe 2>/dev/null || true
pkill -f "configs/avg_frozen.yaml" 2>/dev/null || true

for T in 0.25 0.5 1 2; do
  TAG=$(echo "$T" | tr -d '.')
  RUN="bindaware_sample_T${TAG}"
  log "TRAIN $RUN (T=$T, seed 0)"
  $PY scripts/train.py --config configs/bindaware_sample.yaml --seed 0 \
      --run-name "$RUN" --set "train.groupdro.temperature=${T}" \
      > "logs/${RUN}.log" 2>&1 || { log "$RUN TRAIN FAILED"; continue; }
  log "EVAL $RUN (n=1000, apgd)"
  $PY scripts/evaluate.py --config configs/bindaware_sample.yaml \
      --checkpoint "checkpoints/${RUN}_best.pt" -n 1000 --version apgd \
      --out "results/eval_${RUN}.json" > "logs/eval_${RUN}.log" 2>&1 || { log "$RUN EVAL FAILED"; continue; }
  $PY - "$RUN" "results/eval_${RUN}.json" "$SUMM" <<'PYEOF'
import json, sys
run, path, summ = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(path))["metrics"]
row = (f"| {run} | {m['clean_acc']*100:.1f} | {m['per_norm_robust_acc']['linf']*100:.1f} | "
       f"{m['per_norm_robust_acc']['l2']*100:.1f} | {m['per_norm_robust_acc']['l1']*100:.1f} | "
       f"**{m['worst_union_acc']*100:.1f}** |")
open(summ, "a").write(row + "\n"); print("appended:", row)
PYEOF
  log "$RUN done"
done
log "3B SWEEP COMPLETE. See $SUMM (pick best T -> 3 seeds)."

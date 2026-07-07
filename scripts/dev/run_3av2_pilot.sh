#!/usr/bin/env bash
# =============================================================================
# CARD-3a-v2 pilot — gated launcher (Kiet scheduling 2026-07-02):
#   Gate: (a) CARD-3a pilot APGD eval landed, AND (b) p1's attackdro seed runs
#   finished — so with the 3b sweep (slot 1) this stays <= 2 trainers on GPU.
#   Then: train bindaware_v2 seed 0 -> eval_union (n=1000, apgd) -> append.
# Goal: completes the 2x2 (signal: probe vs val-APGD  x  aggregation: group vs
# per-sample). tmux: p3av2.
# =============================================================================
set -uo pipefail
cd /mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO
PY=/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO/.venv/bin/python
SUMM=results/p2_summary.md
log() { echo "[3av2 $(date +%H:%M:%S)] $*"; }

log "gate: waiting for 3a pilot eval + p1 attackdro trainers to finish..."
until [ -f results/eval_bindaware_s0.json ] \
      && ! pgrep -f "[.]venv/bin/python .*scripts/train.py --config configs/attackdro.yaml" >/dev/null; do
  sleep 60
done
log "gate open (3a landed, p1 done). Launching bindaware_v2 seed-0 pilot."

RUN=bindaware_v2_s0
$PY scripts/train.py --config configs/bindaware_v2.yaml --seed 0 --run-name "$RUN" \
    > "logs/${RUN}.log" 2>&1 || { log "TRAIN FAILED"; exit 1; }
log "training done; eval_union"
$PY scripts/evaluate.py --config configs/bindaware_v2.yaml \
    --checkpoint "checkpoints/${RUN}_best.pt" -n 1000 --version apgd \
    --out "results/eval_${RUN}.json" > "logs/eval_${RUN}.log" 2>&1 || { log "EVAL FAILED"; exit 1; }
$PY - "$RUN" "results/eval_${RUN}.json" "$SUMM" <<'PYEOF'
import json, sys
run, path, summ = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(path))["metrics"]
row = (f"| {run} | {m['clean_acc']*100:.1f} | {m['per_norm_robust_acc']['linf']*100:.1f} | "
       f"{m['per_norm_robust_acc']['l2']*100:.1f} | {m['per_norm_robust_acc']['l1']*100:.1f} | "
       f"**{m['worst_union_acc']*100:.1f}** |")
open(summ, "a").write(row + "\n"); print("appended:", row)
PYEOF
log "3A-V2 PILOT COMPLETE. See $SUMM"

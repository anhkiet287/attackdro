#!/usr/bin/env bash
# =============================================================================
# FINAL PIPELINE (Kiet GO 2026-07-03) — one tmux queue, ~16-20h unattended:
#   A. CARD-5 max_inhouse s0 (T->0 anchor; T-axis story gate)
#   B. Finalist seeds: T025 s1,s2 -> 3a s1,s2 -> T05 s1,s2   (skip-if-done)
#   C. Standard-AA pack (finalists s0 + MSD + avg_frozen if present) + delta
#   D. avg_frozen s0,s1,s2 (skip-if-done — Colab may deliver first, no dup work)
#   E. Standard-AA pack re-run (picks up avg_frozen if D trained it)
# Every run: train -> eval_union (apgd, n=1000) -> append results/p2_summary.md.
# Launch:  tmux new-session -d -s finalpipe 'bash scripts/run_final_pipeline.sh'
# =============================================================================
set -uo pipefail
cd /mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO
PY=/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO/.venv/bin/python
SUMM=results/p2_summary.md
log() { echo "[finalpipe $(date +%H:%M:%S)] $*"; }

# run_one <config> <run-name> <seed> [extra --set args...]
run_one() {
  local cfg=$1 run=$2 seed=$3; shift 3
  if [ -f "results/eval_${run}.json" ]; then log "SKIP $run (eval exists — Colab/local already did it)"; return 0; fi
  log "TRAIN $run (cfg=$cfg seed=$seed $*)"
  $PY scripts/train.py --config "configs/${cfg}.yaml" --seed "$seed" --run-name "$run" "$@" \
      > "logs/${run}.log" 2>&1 || { log "$run TRAIN FAILED"; return 1; }
  log "EVAL $run (apgd n=1000)"
  $PY scripts/evaluate.py --config "configs/${cfg}.yaml" \
      --checkpoint "checkpoints/${run}_best.pt" -n 1000 --version apgd \
      --out "results/eval_${run}.json" > "logs/eval_${run}.log" 2>&1 || { log "$run EVAL FAILED"; return 1; }
  $PY - "$run" "results/eval_${run}.json" "$SUMM" <<'PYEOF'
import json, sys
run, path, summ = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(path))["metrics"]
row = (f"| {run} | {m['clean_acc']*100:.1f} | {m['per_norm_robust_acc']['linf']*100:.1f} | "
       f"{m['per_norm_robust_acc']['l2']*100:.1f} | {m['per_norm_robust_acc']['l1']*100:.1f} | "
       f"**{m['worst_union_acc']*100:.1f}** |")
open(summ, "a").write(row + "\n"); print("appended:", row)
PYEOF
  log "$run done"
}

log "=== STAGE A: CARD-5 max_inhouse (T->0 anchor) ==="
run_one max_inhouse max_inhouse_s0 0

log "=== STAGE B: finalist seeds ==="
run_one bindaware_sample bindaware_sample_T025_s1 1 --set train.groupdro.temperature=0.25
run_one bindaware_sample bindaware_sample_T025_s2 2 --set train.groupdro.temperature=0.25
run_one bindaware bindaware_s1 1
run_one bindaware bindaware_s2 2
run_one bindaware_sample bindaware_sample_T05_s1 1 --set train.groupdro.temperature=0.5
run_one bindaware_sample bindaware_sample_T05_s2 2 --set train.groupdro.temperature=0.5

log "=== STAGE C: standard-AA pack (fires now that finalist seeds landed) ==="
$PY scripts/eval_standard_pack.py > logs/std_pack.log 2>&1 || log "std pack FAILED (see logs/std_pack.log)"

log "=== STAGE D: avg_frozen x3 (skip-if-done vs Colab) ==="
run_one avg_frozen avg_frozen_s0 0
run_one avg_frozen avg_frozen_s1 1
run_one avg_frozen avg_frozen_s2 2

log "=== STAGE E: standard-AA pack re-run (pick up avg_frozen) ==="
$PY scripts/eval_standard_pack.py > logs/std_pack2.log 2>&1 || log "std pack 2 FAILED"

log "FINAL PIPELINE COMPLETE. p2_summary.md + standard_vs_apgd.md ready."

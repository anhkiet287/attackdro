#!/bin/bash
# Ablations #11-#14 (base-generality + redundancy). GATED: refuses until seed-1 + v50 done, needs --go.
# All from-scratch 80ep seed0, matched to M1a except --base/--glue-view/--variant. Report bất kể.
set -e; cd "$(dirname "$0")/../.."; export ATTACKDRO_ROOT=$PWD C5_NUM_WORKERS=4
GO=0; [ "$1" = "--go" ] && GO=1
UB=results/eval/union_bench
if [ ! -f $UB/M0_full_seed1/eval.json ] || [ ! -f $UB/M1a_full_v50/eval.json ]; then
  echo "GATED: seed-1 (M0_full_seed1) / v50 (M1a_full_v50) not done. Not launching."; [ $GO -eq 0 ] && exit 0
fi
BASE=results/fromscratch/C5_ablations; PY=.venv/bin/python
train(){ echo "=== TRAIN $1 ($(date +%H:%M)) ==="; $PY scripts/dev/c5_fromscratch.py --seed 0 --epochs 80 \
  --outdir $BASE/$1 --wandb-mode offline ${@:2}; }
nosq(){ $PY scripts/dev/union_bench_eval.py --config configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml \
  --checkpoint $BASE/$1/ckpt/val_best.pt --arch robustdro --skip-square --run-id ${1}_1k_nosq \
  --checkpoint-role val_best --out $UB/$1/eval.json --export-masks --bs 128; }
if [ $GO -eq 0 ]; then echo "DRY (gate open). Pass --go to launch. Arms:"; fi
for spec in \
  "M0_max|--variant M0 --base max" \
  "M1a_max|--variant M1a --base max --glue-view 3norm --neg adv" \
  "M0_avg|--variant M0 --base avg" \
  "M1a_avg|--variant M1a --base avg --glue-view 3norm --neg adv" \
  "M1a_max_msdglue|--variant M1a --base max --glue-view msd --neg adv" \
  "M1a_linfglue|--variant M1a --base msd --glue-view linf --neg adv" \
  "M1a_cleance|--variant M1a --base clean --glue-view 3norm --neg adv"; do
  name=${spec%%|*}; flags=${spec#*|}
  echo "  $name : c5_fromscratch --seed 0 --epochs 80 $flags"
  if [ $GO -eq 1 ]; then train $name $flags; nosq $name; fi
done
[ $GO -eq 1 ] && echo "ABLATIONS_DONE $(date +%H:%M)"

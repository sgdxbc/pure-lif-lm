#!/bin/bash
# Extra phase-1 run: gamma initialised spread over [0.05, 0.999] so some neurons start almost without reset.
cd "$(dirname "$0")"
until grep -q "=== done" runs/phase1.log; do sleep 20; done
COMMON="--steps 1500 --batch 32 --seq 256 --warmup 150 --eval_every 500 --eval_batches 40 --log_every 100"
SNN="--hard_reset true --sg fast_sigmoid --detach_reset false --lr 3e-3"
echo "=== reset_spread ($(date +%H:%M))"
python3 train.py --out runs/p1_reset_spread $COMMON $SNN --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05 2>&1 \
  | grep -v Warning | grep --line-buffered -E "^params|step +(100|500|1000|1500) |val_loss|Error|Traceback"
echo "=== done ($(date +%H:%M))"

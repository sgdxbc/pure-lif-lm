#!/bin/bash
# Runs after the phase-2 training: evaluate it, then phase 3 (learning rate of per-neuron parameters).
cd "$(dirname "$0")"
while pgrep -f "train.py --out runs/p2_large" > /dev/null; do sleep 30; done
grep -q "== eval step 14000" runs/p2_large/stdout.log && echo "=== phase 2 finished ($(date +%H:%M))" \
  || echo "=== PHASE2_INCOMPLETE: training process exited before step 14000 ($(date +%H:%M))"

echo "=== phase 2 evaluation ($(date +%H:%M))"
python3 evaluate.py --ckpt runs/p2_large/ckpt.pt --out runs/p2_large/final_eval.json 2>&1 | grep -v Warning | head -12
python3 probe_memory.py first_12.6M=runs/main/model.pt large_25.2M=runs/p2_large/ckpt.pt --out runs/probe_phase2.json 2>&1 | grep -v Warning

COMMON="--steps 1500 --batch 32 --seq 256 --warmup 150 --eval_every 500 --eval_batches 40 --log_every 100"
SNN="--hard_reset true --sg fast_sigmoid --detach_reset false --lr 3e-3 --reset_learn true"
SPREAD="--gamma_init 0.999 --gamma_init_min 0.05"
run() { name=$1; shift; echo "=== $name: $@ ($(date +%H:%M))"; python3 train.py --out runs/p3_$name $COMMON "$@" 2>&1 | grep -v Warning | grep --line-buffered -E "^params|step +(500|1000|1500) |val_loss|Error|Traceback"; }
run spread_x10  $SNN $SPREAD --neuron_lr_mult 10
run spread_x30  $SNN $SPREAD --neuron_lr_mult 30
run g09_x30     $SNN --gamma_init 0.9 --neuron_lr_mult 30
echo "=== phase 3 probe ($(date +%H:%M))"
python3 probe_memory.py g09_x1=runs/p1_reset/ckpt.pt g09_x30=runs/p3_g09_x30/ckpt.pt spread_x1=runs/p1_reset_spread/ckpt.pt \
  spread_x10=runs/p3_spread_x10/ckpt.pt spread_x30=runs/p3_spread_x30/ckpt.pt --out runs/probe_phase3.json 2>&1 | grep -v Warning
echo "=== all done ($(date +%H:%M))"

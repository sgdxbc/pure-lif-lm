#!/bin/bash
# Phase 8: training methods, same network (12.6M, gamma spread + leak-modulating synapses rank 64), 1500 steps
# each as a first screen; compare with runs/p4_leak64 (2.234) and runs/p4_leak64_seed1 (2.230).
# DEVICE=cuda bash phase8.sh  to run on a GPU.
cd "$(dirname "$0")/.."   # the repository root
DEVICE=${DEVICE:-cpu}
while pgrep -x -f "/bin/bash ./phase7.sh" > /dev/null; do sleep 60; done
COMMON="--device $DEVICE --steps 1500 --batch 32 --seq 256 --warmup 150 --eval_every 500 --eval_batches 40 --log_every 100"
BASE="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05 --leak_rank 64"
run() { name=$1; shift; echo "=== $name: $@ ($(date +%H:%M))"; mkdir -p runs/p8_$name
  python3 train.py --out runs/p8_$name $COMMON $BASE "$@" --resume >> runs/p8_$name/stdout.log 2>&1
  grep -E "^params|val_loss|Error|Traceback" runs/p8_$name/stdout.log; }
run distill05      --lr 3e-3 --teacher runs/tf_full/ckpt.pt --distill_weight 0.5 --distill_temp 1
run carry          --lr 3e-3 --carry_state
run lr45           --lr 4.5e-3
run neuronlr10     --lr 3e-3 --neuron_lr_mult 10
run alpha2         --lr 3e-3 --alpha 2
run alpha8         --lr 3e-3 --alpha 8
run lr2            --lr 2e-3
run ema998         --lr 3e-3 --ema 0.998
run distill09      --lr 3e-3 --teacher runs/tf_full/ckpt.pt --distill_weight 0.9 --distill_temp 1
echo "=== phase 8 recall probe ($(date +%H:%M))"
M=""; for r in distill05 carry lr45 neuronlr10 alpha2 alpha8 lr2 ema998 distill09; do M="$M $r=runs/p8_$r/ckpt.pt"; done
python3 probe_recall.py --threads 4 leak64=runs/p4_leak64/ckpt.pt $M --out runs/recall_phase8.json 2>&1 | grep -v -i warn
python3 probe_memory.py leak64=runs/p4_leak64/ckpt.pt $M --out runs/probe_phase8.json 2>&1 | grep -v -i warn
echo "=== all done ($(date +%H:%M))"

#!/bin/bash
cd "$(dirname "$0")/.."   # the repository root
COMMON="--steps 400 --batch 32 --warmup 100 --eval_every 400 --eval_batches 20 --log_every 50"
run() { name=$1; shift; echo "=== $name: $@"; python3 train.py --out runs/abl_$name $COMMON "$@" 2>&1 | grep -v Warning | grep --line-buffered -E "step +(100|200|300|400) |val_loss|Error|Traceback"; }
run h_fastsg --hard_reset true --sg fast_sigmoid
run h_lr4e-3 --hard_reset true --lr 4e-3
run h_nodetach --hard_reset true --detach_reset false
run h_lr1e-3 --hard_reset true --lr 1e-3
run tf --arch tf --n_layers 5

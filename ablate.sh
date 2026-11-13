#!/bin/bash
# Short ablation runs (400 steps x 8192 tokens), run sequentially.
COMMON="--steps 400 --batch 32 --warmup 100 --eval_every 400 --eval_batches 20 --log_every 50"
run() { name=$1; shift; echo "=== $name: $@"; python3 train.py --out runs/abl_$name $COMMON "$@" 2>&1 | grep -v Warning | grep -E "step +(1|100|200|300|400) |val_loss"; }
run base
run w_init --in_norm weight
run hard --hard_reset true
run fastsg --sg fast_sigmoid
run lr1e-3 --lr 1e-3
run lr4e-3 --lr 4e-3

#!/bin/bash
# Phase 1: same size (8 x 1536), same budget (1500 steps x 8192 tokens = 12.3M tokens), same schedule.
cd "$(dirname "$0")"
COMMON="--steps 1500 --batch 32 --seq 256 --warmup 150 --eval_every 500 --eval_batches 40 --log_every 100"
SNN="--hard_reset true --sg fast_sigmoid --detach_reset false --lr 3e-3"
run() { name=$1; shift; echo "=== $name: $@ ($(date +%H:%M))"; python3 train.py --out runs/p1_$name $COMMON "$@" 2>&1 | grep -v Warning | grep --line-buffered -E "^params|step +(100|500|1000|1500) |val_loss|Error|Traceback"; }
run base     $SNN
run reset    $SNN --reset_learn true --gamma_init 0.9
run rec      $SNN --rec_rank 128
run both     $SNN --reset_learn true --gamma_init 0.9 --rec_rank 128
run tf       --arch tf --n_layers 5 --lr 2e-3
echo "=== done ($(date +%H:%M))"

#!/bin/bash
# Phase 5: is the gain from leak-modulating synapses just extra parameters? (width-matched control)
# and do delayed synapses (direction C) help?  Same 1500-step setting as phases 1 and 4.
cd "$(dirname "$0")"
COMMON="--steps 1500 --batch 32 --seq 256 --warmup 150 --eval_every 500 --eval_batches 40 --log_every 100"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --lr 3e-3 --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
run() { name=$1; shift; echo "=== $name: $@ ($(date +%H:%M))"; mkdir -p runs/p5_$name
  python3 train.py --out runs/p5_$name $COMMON "$@" --resume >> runs/p5_$name/stdout.log 2>&1
  grep -E "^params|step +(500|1000|1500) |val_loss|Error|Traceback" runs/p5_$name/stdout.log | grep -v "^ *>>"; }
run wide1696       $SPREAD --d_hidden 1696                 # 13.62M parameters, same as leak64
run delay2         $SPREAD --delay_k 2 --delay_rank 64
run delay2_seed1   $SPREAD --delay_k 2 --delay_rank 64 --seed 1
echo "=== phase 5 recall probe ($(date +%H:%M))"
python3 probe_recall.py --threads 4 wide1696=runs/p5_wide1696/ckpt.pt delay2=runs/p5_delay2/ckpt.pt \
  delay2_seed1=runs/p5_delay2_seed1/ckpt.pt --out runs/recall_phase5.json 2>&1 | grep -v -i warn
python3 probe_memory.py wide1696=runs/p5_wide1696/ckpt.pt delay2=runs/p5_delay2/ckpt.pt \
  delay2_seed1=runs/p5_delay2_seed1/ckpt.pt --out runs/probe_phase5.json 2>&1 | grep -v -i warn
echo "=== all done ($(date +%H:%M))"

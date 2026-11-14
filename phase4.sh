#!/bin/bash
# Phase 4 (after the fair Transformer reference): leak-modulating synapses (direction A), plus seed replicates
# to measure how much two runs with identical settings differ.
cd "$(dirname "$0")"
while pgrep -x -f "/bin/bash ./after_phase3.sh" > /dev/null; do sleep 30; done
COMMON="--steps 1500 --batch 32 --seq 256 --warmup 150 --eval_every 500 --eval_batches 40 --log_every 100"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --lr 3e-3 --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
run() { name=$1; shift; echo "=== $name: $@ ($(date +%H:%M))"; mkdir -p runs/p4_$name
  python3 train.py --out runs/p4_$name $COMMON "$@" --resume >> runs/p4_$name/stdout.log 2>&1
  grep -E "^params|step +(500|1000|1500) |val_loss|Error|Traceback" runs/p4_$name/stdout.log | grep -v "^ *>>"; }
run spread_seed1   $SPREAD --seed 1
run leak64         $SPREAD --leak_rank 64
run leak64_seed1   $SPREAD --leak_rank 64 --seed 1
run leak256        $SPREAD --leak_rank 256
echo "=== phase 4 recall probe ($(date +%H:%M))"
python3 probe_recall.py --threads 4 spread=runs/p1_reset_spread/ckpt.pt spread_seed1=runs/p4_spread_seed1/ckpt.pt \
  leak64=runs/p4_leak64/ckpt.pt leak64_seed1=runs/p4_leak64_seed1/ckpt.pt leak256=runs/p4_leak256/ckpt.pt \
  --out runs/recall_phase4.json 2>&1 | grep -v -i warn
python3 probe_memory.py spread=runs/p1_reset_spread/ckpt.pt spread_seed1=runs/p4_spread_seed1/ckpt.pt \
  leak64=runs/p4_leak64/ckpt.pt leak64_seed1=runs/p4_leak64_seed1/ckpt.pt leak256=runs/p4_leak256/ckpt.pt \
  --out runs/probe_phase4.json 2>&1 | grep -v -i warn
echo "=== all done ($(date +%H:%M))"

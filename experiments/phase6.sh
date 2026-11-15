#!/bin/bash
# Phase 6: does the gain from leak-modulating synapses hold up with more training?
# Same settings as phase 4, but 5000 steps (41M tokens) instead of 1500 (12.3M).
cd "$(dirname "$0")/.."   # the repository root
while pgrep -x -f "/bin/bash ./phase5.sh" > /dev/null; do sleep 30; done
COMMON="--steps 5000 --batch 32 --seq 256 --warmup 300 --eval_every 1000 --eval_batches 40 --log_every 100"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --lr 3e-3 --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
run() { name=$1; shift; echo "=== $name: $@ ($(date +%H:%M))"; mkdir -p runs/p6_$name
  python3 train.py --out runs/p6_$name $COMMON "$@" --resume >> runs/p6_$name/stdout.log 2>&1
  grep -E "^params|val_loss|Error|Traceback" runs/p6_$name/stdout.log | grep -v "^ *>>"; }
run spread_5k   $SPREAD
run leak64_5k   $SPREAD --leak_rank 64
echo "=== phase 6 recall probe ($(date +%H:%M))"
python3 probe_recall.py --threads 4 spread_5k=runs/p6_spread_5k/ckpt.pt leak64_5k=runs/p6_leak64_5k/ckpt.pt \
  --out runs/recall_phase6.json 2>&1 | grep -v -i warn
python3 probe_memory.py spread_5k=runs/p6_spread_5k/ckpt.pt leak64_5k=runs/p6_leak64_5k/ckpt.pt \
  --out runs/probe_phase6.json 2>&1 | grep -v -i warn
python3 leak_stats.py runs/p6_leak64_5k --out runs/leak_stats_phase6.json 2>&1 | grep -v -i warn
echo "=== all done ($(date +%H:%M))"

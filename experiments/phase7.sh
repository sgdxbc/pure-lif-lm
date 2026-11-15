#!/bin/bash
# Phase 7: the 25.25M model of phase 2 (runs/p2_large, val loss 1.793) with leak-modulating synapses added,
# everything else identical, on the full 115M tokens.
cd "$(dirname "$0")/.."   # the repository root
while pgrep -x -f "/bin/bash ./phase6.sh" > /dev/null; do sleep 30; done
mkdir -p runs/p7_large_leak64
echo "=== p7_large_leak64 ($(date +%H:%M))"
python3 train.py --out runs/p7_large_leak64 --d_model 512 --d_hidden 2048 --n_layers 10 \
  --hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05 \
  --leak_rank 64 --lr 2.5e-3 --steps 14000 --batch 32 --seq 256 --warmup 300 \
  --eval_every 1000 --eval_batches 40 --log_every 100 --resume >> runs/p7_large_leak64/stdout.log 2>&1
grep -E "^params|val_loss|Error|Traceback" runs/p7_large_leak64/stdout.log
echo "=== evaluation and probes ($(date +%H:%M))"
python3 evaluate.py --ckpt runs/p7_large_leak64/ckpt.pt --out runs/p7_large_leak64/final_eval.json > runs/p7_large_leak64/eval.log 2>&1
grep -v "^ *->" runs/p7_large_leak64/eval.log | head -12
python3 probe_recall.py --threads 4 large_25.2M=runs/p2_large/ckpt.pt large_leak64=runs/p7_large_leak64/ckpt.pt \
  tf_115M_tokens=runs/tf_full/ckpt.pt --out runs/recall_phase7.json 2>&1 | grep -v -i warn
python3 probe_memory.py large_25.2M=runs/p2_large/ckpt.pt large_leak64=runs/p7_large_leak64/ckpt.pt \
  tf_115M_tokens=runs/tf_full/ckpt.pt --out runs/probe_phase7.json 2>&1 | grep -v -i warn
python3 leak_stats.py runs/p7_large_leak64 --out runs/leak_stats_phase7.json 2>&1 | grep -v -i warn
echo "=== all done ($(date +%H:%M))"

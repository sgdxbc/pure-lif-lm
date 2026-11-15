#!/bin/bash
# Runs after after_phase2.sh: recall probe and parameter-drift diagnostics on the phase-2/3 models,
# then the fair Transformer reference trained on the same 115M tokens as the spiking models.
cd "$(dirname "$0")/.."   # the repository root
while pgrep -x -f "/bin/bash ./after_phase2.sh" > /dev/null; do sleep 30; done
echo "=== recall probe, phase 2/3 ($(date +%H:%M))"
python3 probe_recall.py --threads 4 first_12.6M=runs/main/model.pt large_25.2M=runs/p2_large/ckpt.pt \
  g09_x1=runs/p1_reset/ckpt.pt g09_x30=runs/p3_g09_x30/ckpt.pt spread_x1=runs/p1_reset_spread/ckpt.pt \
  spread_x10=runs/p3_spread_x10/ckpt.pt spread_x30=runs/p3_spread_x30/ckpt.pt --out runs/recall_phase23.json 2>&1 | grep -v -i warn
echo "=== neuron drift ($(date +%H:%M))"
python3 neuron_drift.py runs/main runs/p2_large runs/p1_reset runs/p3_g09_x30 runs/p1_reset_spread runs/p3_spread_x10 \
  runs/p3_spread_x30 --out runs/drift_phase23.json 2>&1 | grep -v -i warn
echo "=== phase 2 evaluation, generation speed and samples (re-run: the first run was cut short) ($(date +%H:%M))"
python3 evaluate.py --ckpt runs/p2_large/ckpt.pt --out runs/p2_large/final_eval.json > runs/p2_large/eval.log 2>&1
grep -v Warning runs/p2_large/eval.log | head -12
echo "=== tf_full: Transformer on the same 115M tokens ($(date +%H:%M))"
mkdir -p runs/tf_full
python3 train.py --out runs/tf_full --arch tf --n_layers 5 --lr 2e-3 --steps 14000 --batch 32 --seq 256 --warmup 300 \
  --eval_every 1000 --eval_batches 40 --log_every 100 --resume >> runs/tf_full/stdout.log 2>&1
grep "== eval" runs/tf_full/stdout.log | tail -3
echo "=== tf_full evaluation and probes ($(date +%H:%M))"
python3 evaluate.py --ckpt runs/tf_full/ckpt.pt --out runs/tf_full/final_eval.json > runs/tf_full/eval.log 2>&1
grep -v Warning runs/tf_full/eval.log | head -12
python3 probe_memory.py first_12.6M=runs/main/model.pt large_25.2M=runs/p2_large/ckpt.pt tf_12M_tokens=runs/p1_tf/ckpt.pt \
  tf_115M_tokens=runs/tf_full/ckpt.pt --out runs/probe_fair.json 2>&1 | grep -v -i warn
python3 probe_recall.py --threads 4 tf_12M_tokens=runs/p1_tf/ckpt.pt tf_115M_tokens=runs/tf_full/ckpt.pt \
  first_12.6M=runs/main/model.pt large_25.2M=runs/p2_large/ckpt.pt --out runs/recall_fair.json 2>&1 | grep -v -i warn
echo "=== all done ($(date +%H:%M))"

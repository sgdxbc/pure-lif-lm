#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 02:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Both memory probes with the moving-average (EMA) weights, for runs trained with --ema before slurm/train.sbatch
# did this itself. Writes recall_ema.json and probe_memory_ema.json into each run directory. From the repo root:
#   sbatch -J probes_ema --gres=gpu:a100-40:1 slurm/probes_ema.sbatch runs/A runs/B ...
source slurm/env.sh
for out in "$@"; do
  python3 probe_recall.py --device cuda --ema "m=$out/ckpt.pt" --out "$out/recall_ema.json" > "$out/probe_ema.log" 2>&1
  python3 probe_memory.py --device cuda --ema "m=$out/ckpt.pt" --out "$out/probe_memory_ema.json" >> "$out/probe_ema.log" 2>&1
  echo "$out: exit $? $(date)"
done

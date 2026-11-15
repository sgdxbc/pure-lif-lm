#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 01:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Recall probe over the first 512 tokens of the longer validation stories (probe_recall.py --length 512, which adds
# a d=257-512 bucket), written as recall512.json into each run directory, and also with the moving-average weights
# (recall512_ema.json) for runs trained with --ema. Spiking runs only: a Transformer trained on 256-token windows has
# never trained its position embeddings beyond 256, so its numbers past that point mean nothing. From the repo root:
#   sbatch -J probe_long --gres=gpu:a100-40:1 slurm/probe_long.sbatch runs/A runs/B ...
source slurm/env.sh
for out in "$@"; do
  python3 probe_recall.py --device cuda --length 512 "m=$out/ckpt.pt" --out "$out/recall512.json" > "$out/probe512.log" 2>&1
  code=$?
  if python3 -c "import json, sys; sys.exit(0 if json.load(open('$out/args.json')).get('ema', 0) else 1)"; then
    python3 probe_recall.py --device cuda --length 512 --ema "m=$out/ckpt.pt" --out "$out/recall512_ema.json" >> "$out/probe512.log" 2>&1
    code="$code, ema $?"
  fi
  echo "$out: exit $code $(date)"
done

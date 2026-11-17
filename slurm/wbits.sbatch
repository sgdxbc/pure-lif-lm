#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 01:30:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Validation loss (full validation set, evaluate.py) of each spiking run given, with its weight matrices rounded after
# training to 8 and to 4 bits (one scale per 128 inputs; model.py quantize_weight; docs/experiments.md 5.38), with the
# moving-average weights where the run has them. Unlike quant_sim.py it also covers the plastic synapses, the coded
# input and the two-level readout; it rounds only the weights (8-bit bus and 24-bit potentials cost nothing, 5.20).
# Writes runs/X/final_eval[_ema]_w8.json and _w4.json. From the repository root:
#   sbatch -J wbits --gres=gpu:a100-40:1 slurm/wbits.sbatch runs/A runs/B ...
source slurm/env.sh
for out in "$@"; do
  e=""; python3 -c "import json, sys; sys.exit(0 if json.load(open('$out/args.json')).get('ema') else 1)" && e="--ema"
  for b in 8 4; do
    python3 evaluate.py --device cuda $e --wbits $b --samples_per_prompt 1 --ckpt "$out/ckpt.pt" \
      --out "$out/final_eval${e:+_ema}_w$b.json" > "$out/eval_w$b.log" 2>&1
    echo "$out $b-bit weights: $(grep -h '"val_loss"' "$out/final_eval${e:+_ema}_w$b.json") ($(date +%H:%M))"
  done
done

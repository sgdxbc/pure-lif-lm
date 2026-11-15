#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 02:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Diagnostics of FineWeb-Edu runs (README 5.29): why LAMBADA is so hard for them (probe_lambada.py, with GPT-2 124M
# and 355M for reference, writes runs/lambada_probe.json), and the loss at each position of validation documents of
# at least 1025 tokens, each read from its start (eval_long.py --length 1024, writes runs/X/long_valid[_ema].json):
#   sbatch -J diag13 --gres=gpu:a100-40:1 slurm/diag.sbatch runs/g13_snn212 runs/g13_tf163 runs/g13_tf77
source slurm/env.sh
python3 probe_lambada.py --device cuda --ema --check 20 "$@" openai-community/gpt2 openai-community/gpt2-medium
echo "probe_lambada: exit $? $(date)"
for r in "$@"; do
  python3 eval_long.py --device cuda --ema --length 1024 --batch 16 "$r"
  echo "$r eval_long: exit $? $(date)"
done

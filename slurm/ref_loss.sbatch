#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 03:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Validation loss on FineWeb-Edu of Hugging Face models with GPT-2's tokenizer (ref_loss.py -> runs/ref_loss.json):
# targets for stage goal 1 and candidate teachers. Weights in float32 (TF32 multiplies; the first run, 0e77aaf, used
# bfloat16 and took 52 min on an A100-80). From the repository root, all eight models or the ones named:
#   sbatch -J ref_loss --gres=gpu:a100-80:1 slurm/ref_loss.sbatch [MODEL ...]
source slurm/env.sh
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(openai-community/gpt2 openai-community/gpt2-medium openai-community/gpt2-large
  openai-community/gpt2-xl EleutherAI/gpt-neo-125m EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-2.7B EleutherAI/gpt-j-6b)
python3 ref_loss.py --device cuda "${models[@]}" 2>&1 | grep -v -E "Loading weights|Warning|deprecated"

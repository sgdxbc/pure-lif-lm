#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 01:30:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Validation loss on FineWeb-Edu of Hugging Face models with GPT-2's tokenizer (ref_loss.py -> runs/ref_loss.json):
# targets for stage goal 1 and candidate teachers. From the repository root:
#   sbatch -J ref_loss --gres=gpu:a100-80:1 slurm/ref_loss.sbatch
source slurm/env.sh
python3 ref_loss.py --device cuda openai-community/gpt2 openai-community/gpt2-medium openai-community/gpt2-large \
  openai-community/gpt2-xl EleutherAI/gpt-neo-125m EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-2.7B EleutherAI/gpt-j-6b \
  2>&1 | grep -v -E "Loading weights|Warning|deprecated"

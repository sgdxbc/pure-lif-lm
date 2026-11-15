#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 03:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Zero-shot benchmarks (eval_lm.py, lm-evaluation-harness) of reference models from Hugging Face and/or our runs:
#   sbatch -J zeroshot --gres=gpu:a100-40:1 slurm/zeroshot.sbatch openai-community/gpt2 EleutherAI/pythia-160m runs/X
# A Hugging Face id writes runs/ref_<name>/zeroshot.json; a run directory writes zeroshot.json and, with moving-average
# weights, zeroshot_ema.json. Needs, once in the venv (where there is internet; check torch is unchanged afterwards):
#   pip install lm_eval transformers accelerate
source slurm/env.sh
python3 -c "import lm_eval, transformers" || { echo "pip install lm_eval transformers accelerate first"; exit 1; }
for x in "$@"; do
  if [ -d "$x" ]; then
    python3 eval_lm.py --device cuda "$x" && python3 eval_lm.py --device cuda --ema "$x"
  else
    python3 eval_lm.py --device cuda --hf "$x"
  fi
  echo "$x: exit $? $(date)"
done

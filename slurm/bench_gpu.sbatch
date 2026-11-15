#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 02:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Transformer inference measured on a GPU with vLLM (bench_gpu.py; README section 0, stage goal 3), for each model given:
#   sbatch -J bench_gpu --gres=gpu:h100-96:1 slurm/bench_gpu.sbatch openai-community/gpt2 EleutherAI/pythia-160m ...
# vLLM pins its own torch, so it lives in a separate venv, made once where there is internet:
#   python3 -m venv ~/venv-vllm && ~/venv-vllm/bin/pip install vllm nvidia-ml-py
source ~/venv-vllm/bin/activate
export XDG_CACHE_HOME=~/.cache HF_HOME=~/.cache/huggingface TMPDIR=${SLURM_TMPDIR:-/tmp}
cd ~/pure-lif-lm
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
for m in "$@"; do
  python3 bench_gpu.py --model "$m" 2>&1 | grep -v -E "^(INFO|DEBUG)|it/s\]" | tail -40
  echo "$m: exit ${PIPESTATUS[0]} $(date)"
done

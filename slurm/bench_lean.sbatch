#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 01:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Transformer decoding with a minimal program (bench_gpu_lean.py; docs/experiments.md 5.43), for each shape given:
#   sbatch -J bench_lean --gres=gpu:h100-96:1 slurm/bench_lean.sbatch 256,6,4 384,6,6 512,8,8
# Uses the vLLM venv of slurm/bench_gpu.sbatch, which has torch and nvidia-ml-py (no vLLM is used).
source ~/venv-vllm/bin/activate
export XDG_CACHE_HOME=~/.cache TMPDIR=${SLURM_TMPDIR:-/tmp}
cd ~/pure-lif-lm
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
python3 bench_gpu_lean.py --check
for s in "$@"; do
  python3 bench_gpu_lean.py --shape "$s" 2>&1 | tail -20
  echo "$s: exit ${PIPESTATUS[0]} $(date)"
done

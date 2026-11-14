#!/bin/bash
#SBATCH -J lif-gpucheck -p gpu --gres=gpu:h100-96:1 -c 8 --mem=64G -t 01:00:00 -o slurm/logs/%x-%j.out
# Checks part of gpu_setup.sh (steps 1, 4, 5); data was prepared and verified by prepare.sbatch.
source slurm/env.sh
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
curl -s -m 10 -o /dev/null -w "compute-node internet: HTTP %{http_code}\n" https://huggingface.co || echo "compute-node internet: none"
sed -n '/^echo "== 1/,/^EOF$/p;/^echo "== 4/,$p' gpu_setup.sh > $TMPDIR/checks.sh
bash $TMPDIR/checks.sh

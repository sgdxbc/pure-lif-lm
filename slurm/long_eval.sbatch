#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 01:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Long-text evaluation (eval_long.py, docs/experiments.md 5.22) of each run given: loss by position on the test books, read from
# the start of each book to LENGTH tokens (default 8192); EMA weights where the run has them. Transformers are also
# evaluated with a sliding window of their training length. From the repository root:
#   sbatch -J long_eval --gres=gpu:a100-80:1 slurm/long_eval.sbatch runs/A runs/B ...
source slurm/env.sh
L=${LENGTH:-8192}
for out in "$@"; do
  python3 eval_long.py --device cuda --ema --split test --length "$L" "$out"
  seq=$(python3 -c "import json; a = json.load(open('$out/args.json')); print(a['seq'] if a.get('arch') == 'tf' else 0)")
  [ "$seq" != "0" ] && python3 eval_long.py --device cuda --ema --split test --length "$L" --window "$seq" "$out"
  echo "$out done $(date)"
done

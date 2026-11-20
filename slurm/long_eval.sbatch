#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 02:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Long-text evaluation (eval_long.py, docs/experiments.md 5.22) of each run given: loss by position on the test books, read from
# the start of each book to LENGTH tokens (default 8192); EMA weights where the run has them. Transformers are also
# evaluated with a sliding window of their training length. From the repository root:
#   sbatch -J long_eval --gres=gpu:a100-80:1 slurm/long_eval.sbatch runs/A runs/B ...
source slurm/env.sh
L=${LENGTH:-8192}
# LONG_ARGS="--restart 1024 --recall" (docs 5.60): one paired run per network instead, scoring the same targets with the
# full context and with the state (or, for a softmax Transformer, the window) restarted, plus the planted-recall probe.
for out in "$@"; do
  seq=$(python3 -c "import json; a = json.load(open('$out/args.json')); print(a['seq'] if a.get('arch') == 'tf' and a.get('attn', 'softmax') != 'linear' else 0)")
  if [ -n "$LONG_ARGS" ]; then
    w=""; [ "$seq" != "0" ] && w="--window $seq"
    python3 eval_long.py --device cuda --ema --split test --length "$L" $w $LONG_ARGS "$out"
  else
    python3 eval_long.py --device cuda --ema --split test --length "$L" "$out"
    [ "$seq" != "0" ] && python3 eval_long.py --device cuda --ema --split test --length "$L" --window "$seq" "$out"
  fi
  echo "$out done $(date)"
done

#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 02:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Diagnostics of FineWeb-Edu runs (docs/experiments.md 5.28, point 3): why LAMBADA is so hard for them (probe_lambada.py, with GPT-2
# 124M and 355M for reference, adds to runs/lambada_probe.json), and the loss at each position of validation documents
# of at least 1025 tokens, each read from its start (eval_long.py --length 1024, writes runs/X/long_valid[_ema].json;
# skipped where that file exists already):
#   sbatch -J diag13 --gres=gpu:a100-40:1 slurm/diag.sbatch runs/g13_snn212 runs/g13_tf163 runs/g13_tf77
# PROBE=0 skips probe_lambada.py. WINDOWS="16 64 256": for Transformer runs, the same documents again with only the
# last W tokens as context (sliding by W/2, so each token sees between W/2 and W tokens; long_valid_wW.json): how far
# back a Transformer must see to match a spiking network's loss.
source slurm/env.sh
if [ "${PROBE:-1}" != "0" ]; then
  python3 probe_lambada.py --device cuda --ema --check 20 "$@" openai-community/gpt2 openai-community/gpt2-medium
  echo "probe_lambada: exit $? $(date)"
fi
for r in "$@"; do
  if [ ! -e "$r/long_valid.json" ] && [ ! -e "$r/long_valid_ema.json" ]; then
    python3 eval_long.py --device cuda --ema --length 1024 --batch 16 "$r"
    echo "$r eval_long: exit $? $(date)"
  fi
  if [ -n "$WINDOWS" ] && python3 -c "import json, sys; sys.exit(json.load(open('$r/args.json')).get('arch') != 'tf')"; then
    for w in $WINDOWS; do
      python3 eval_long.py --device cuda --length 1024 --batch 16 --window $w "$r"
      echo "$r eval_long window $w: exit $? $(date)"
    done
  fi
done

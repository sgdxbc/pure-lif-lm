#!/bin/bash
# GPU wave 41 (docs/experiments.md 5.52): relays that may send several spikes per channel and token. In wave 40 the
# 3-valued relays (one spike at most) cost 0.37 on top of the 0/1 MLP; these give the large inputs, which one spike
# clips, more spikes (model_tf.py --relay quant --relay_levels K), at the same shape, recipe and seed as wave 40.
# From the repository root:
#   bash experiments/gpu41.sh [NAME]     every run below, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
#   q2   g40_laspk with relays of up to 2 spikes per channel and token (5 values), no membrane
#   q4   the same with up to 4 spikes (9 values)
#   q4t  q4 held to 0.5 spikes per channel and token, weighted by the columns of the matrices the relays feed, that is
#        half the block matrices' dense multiply-adds (train.py --relay_target 0.5; q4 met the quality bar with 1.43)
# (K = 1 is g40_quant.)
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
[ -s data/fineweb/train.bin ] || { echo "data/fineweb/train.bin missing"; exit 1; }
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
S77="--arch tf --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"
only=$1
case "$only" in ""|q2|q4|q4t) ;; *) echo "NAME is q2, q4 or q4t"; exit 1 ;; esac
for s in "q2 a100-80 2" "q4 a100-40 4" "q4t a100-40 4 0.5"; do
  set -- $s; name=$1; gpu=${GPU:-$2}; K=$3; T=${4:-}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  X="--attn linear --mlp spike --relay quant --relay_levels $K${T:+ --relay_target $T}"
  j=$(sbatch --parsable -p gpu -J g41_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g41_$name $DATA $SHORT $S77 $X --accum $acc)
  echo "$j g41_$name ($gpu, accum $acc) $X"
done

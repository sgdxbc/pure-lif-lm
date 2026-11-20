#!/bin/bash
# GPU wave 45 (docs/experiments.md 5.56): where the spikes themselves could pay, and training methods for sparse
# spikes. All at wave 40's 77M shape and recipe (gated linear attention, 0/1 MLP units, relays of up to 4 spikes per
# channel and token held to a target of weighted spikes; g41_q4t: target 0.5, 3.815; g40_laspk without relays 3.704).
# The owner (2026-11-20): first a competitive spiking structure, with weight on the training method too.
#   How sparse can the spikes go (same width):
#     t25     target 0.25                     t10     target 0.1
#   Wider and sparser at about the same additions as g41_q4t (the MLP's hidden units per channel, model_tf.py
#   --mlp_mult; a dense network pays for every weight, a spiking one for the rows of the channels that fire):
#     w8      2x wider MLP (8 d), target 0.33   w16     4x wider MLP (16 d), target 0.2
#   Training methods, each against t25:
#     ft25    start from the trained g40_laspk (no relays) and continue 5,000 steps with relays at 0.25
#             (train.py --init_weights; lr 3e-4, warmup 200; seed 1, so it does not replay g40_laspk's first batches)
#     t25L    t25 for 25,000 steps: ft25's control at the same total steps (20,000 dense + 5,000 sparse)
#     twin25  t25 distilled from its non-spiking twin g40_la (same shape, GELU MLP, no relays; averaged weights)
#     kd25    t25 distilled from g13_tf163 (163M Transformer), the teacher of the 163M runs
#     ann25   t25 with a budget that tightens: target 1.5 at the start, lowered linearly to 0.25 by step 10,000
#             (train.py --relay_target_start / --relay_target_steps)
# From the repository root:
#   bash experiments/gpu45.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g40_laspk/ckpt.pt runs/g40_la/ckpt.pt runs/g13_tf163/ckpt.pt; do
  [ -s $f ] || { echo "$f missing"; exit 1; }
done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
S77="--arch tf --attn linear --mlp spike --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"   # g40_laspk's
Q4="--relay quant --relay_levels 4"
NAMES="t25 t10 w8 w16 ft25 t25L twin25 kd25 ann25"
only=$1
case " $NAMES " in *" ${only:-t25} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
for s in "t25 a100-40" "w8 a100-40" "ft25 h100-47" "t25L h100-47" "twin25 h100-47" "kd25 h100-96" "t10 a100-40" \
         "ann25 h100-47" "w16 a100-80"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h200-141) acc=1 ;; h100-96|a100-80) acc=2 ;; *) acc=4 ;; esac   # g41_q4t: 17.4 GB at accum 4
  [ $name = w16 ] && acc=$((acc * 2))                                             # 4x the MLP's activations
  acc=${ACCUM:-$acc}
  X="$SHORT $S77 $Q4"
  case $name in
    t25) X="$X --relay_target 0.25 --keep_ckpt 15000" ;;
    t25L) X="$X --relay_target 0.25 --steps 25000" ;;
    t10) X="$X --relay_target 0.1" ;;
    w8) X="$X --mlp_mult 8 --relay_target 0.33" ;;
    w16) X="$X --mlp_mult 16 --relay_target 0.2" ;;
    ft25) X="$X --relay_target 0.25 --init_weights runs/g40_laspk/ckpt.pt --steps 5000 --warmup 200 --lr 3e-4 --seed 1" ;;
    twin25) X="$X --relay_target 0.25 --teacher runs/g40_la/ckpt.pt --teacher_ema --distill_weight 0.9 --keep_ckpt 15000" ;;
    kd25) X="$X --relay_target 0.25 --teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --keep_ckpt 15000" ;;
    ann25) X="$X --relay_target 0.25 --relay_target_start 1.5 --relay_target_steps 10000" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g45_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g45_$name $DATA $X --accum $acc)
  echo "$j g45_$name ($gpu, accum $acc) $X"
done

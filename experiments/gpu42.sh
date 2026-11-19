#!/bin/bash
# GPU wave 42 (docs/experiments.md 5.53): which features of gated linear attention made the spiking network 0.28
# better when it replaced the plastic synapses (g39_snngla 3.583 against g38_norm 3.860). Each run is g39_snngla with
# one change, on the same recipe (wave 38's: 20,000 steps x 32 x 1,024, cosine, distilled from g13_tf163, same seed):
#   all       all four features below removed: gated linear attention cut down to the plastic synapses' features
#   spkkq     0/1 keys and queries (as the plastic synapses' key and query neurons), values real
#   fixdecay  a constant decay per head (as the plastic synapses' groups), not a function of the input
#   h16       16 heads of 48 (state 16 x 48 x 48, near the plastic synapses' 8 x 64 x 64; g39_snngla 12 x 64 x 64)
#   noconv    no short convolution in front of q, k, v
#   nogate    no output gate
# From the repository root:
#   bash experiments/gpu42.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
H8=runs/classes/tf163_head225_s8192.npy; E2=runs/classes/tf163_embed220_s2048.npy
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $H8 $E2; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
IO="--out_classes 225 --out_short 8192 --class_file $H8 --in_groups 2 --in_short 2048 --in_class_file $E2"
GLA="$SPREAD $N138 $SNN $IO --bus_norm true --mem_kind gla"
only=$1
case "$only" in ""|all|spkkq|fixdecay|h16|noconv|nogate) ;; *) echo "NAME is all, spkkq, fixdecay, h16, noconv or nogate"; exit 1 ;; esac
for s in "all h100-47" "spkkq h100-47" "fixdecay a100-80" "h16 a100-80" "noconv h100-96" "nogate h100-96"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    all) X="--mem_heads 12 --mem_gla_ablate fixdecay,noconv,nogate,spkkq" ;;
    h16) X="--mem_heads 16" ;;
    *) X="--mem_heads 12 --mem_gla_ablate $name" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g42_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g42_$name $DATA $SHORT $KD $GLA $X --accum $acc)
  echo "$j g42_$name ($gpu, accum $acc) $X"
done

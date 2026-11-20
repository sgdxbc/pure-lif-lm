#!/bin/bash
# GPU wave 48 (docs/experiments.md 5.60), agreed with the GPT advisor after the third self-audit:
#   gla31    g46_tf31s (g33_tf31's shape: 256 wide, 6 layers, 4 heads, GELU MLP, distilled from g13_tf163 at 0.9, the
#            77M runs' short recipe) with gated linear attention (--attn linear): the dense control the audit found
#            missing; judged against g45_kd25 (and reported against g46_tf31s)
#   gla50    the same at g33_tf50's shape (384 wide, 6 heads); judged against g44_all3
#   lr0      g47_ref (g44_all3 with seed 1) with the learning rate decayed to 0 instead of 10% of its peak
#            (--min_lr_frac 0): a training-method test on the LIF line; judged against g47_ref. Takes the cancelled
#            g47_half's slot. HW=1 like wave 47.
# From the repository root:
#   bash experiments/gpu48.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
H8=runs/classes/tf163_head225_s8192.npy; E2=runs/classes/tf163_embed220_s2048.npy
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $H8 $E2; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
TF31="--arch tf --d_model 256 --n_layers 6 --n_heads 4 --lr 1e-3 --ema 0.999"      # g33_tf31's shape (as gpu46.sh)
TF50="--arch tf --d_model 384 --n_layers 6 --n_heads 6 --lr 1e-3 --ema 0.999"      # g33_tf50's shape
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
IO="--out_classes 225 --out_short 8192 --class_file $H8 --in_groups 2 --in_short 2048 --in_class_file $E2"
NORM="$SPREAD $N138 $SNN --mem_heads 8 --mem_write_avg true $IO --bus_norm true"   # g38_norm (as gpu47.sh)
ALL3="$KD $NORM --mem_kq_levels 4 --mem_delays 4 --mem_gate channel --seed 1"       # g44_all3, seed 1 = g47_ref
only=$1
NAMES="gla31 gla50 lr0"
case " $NAMES " in *" ${only:-gla31} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
for s in "gla31 h100-47" "gla50 h100-47" "lr0 h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141|a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  hw=0
  case $name in
    gla31) X="$TF31 $KD --attn linear" ;;
    gla50) X="$TF50 $KD --attn linear" ;;
    lr0) X="$ALL3 --min_lr_frac 0"; hw=1 ;;
  esac
  j=$(sbatch --parsable -p gpu -J g48_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=$hw,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g48_$name $DATA $SHORT $X --accum $acc)
  echo "$j g48_$name ($gpu, accum $acc) $X"
done

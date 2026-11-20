#!/bin/bash
# GPU wave 47 (docs/experiments.md 5.59): which parts of the wave-44 spiking memory earn their hardware cost. Every
# run is g44_all3 (g38_norm's spiking network with 4-level keys and queries, delays 0-3 and the channel gate; 3.688)
# on its recipe (20,000 steps x 32 x 1,024, cosine, distilled from g13_tf163) with seed 1, judged against the seed-1
# reference here:
#   ref      g44_all3 with seed 1
#   bin      0/1 keys and queries (--mem_kq_levels 1): fewer memory additions
#   half     plastic synapses in front of every second population only (--mem_every 2)
#   sless    key and query neurons without a carried membrane (--mem_kq_stateless true; model.py docstring)
# HW=1: the hardware cost (hw_cost.py, with the key/query event counts) and the integer simulation run after training.
# From the repository root:
#   bash experiments/gpu47.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
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
NORM="$SPREAD $N138 $SNN --mem_heads 8 --mem_write_avg true $IO --bus_norm true"   # g38_norm
ALL3="$KD $NORM --mem_kq_levels 4 --mem_delays 4 --mem_gate channel --seed 1"       # g44_all3, seed 1
only=$1
NAMES="ref bin half sless"
case " $NAMES " in *" ${only:-ref} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
for s in "ref h100-96" "bin a100-80" "half h100-47" "sless h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141|a100-80) acc=2 ;; *) acc=4 ;; esac   # g44_all3: 45.0 GB at accum 2 on h100-96
  acc=${ACCUM:-$acc}
  case $name in
    ref) X="$ALL3" ;;
    bin) X="$ALL3 --mem_kq_levels 1" ;;
    half) X="$ALL3 --mem_every 2" ;;
    sless) X="$ALL3 --mem_kq_stateless true" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g47_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g47_$name $DATA $SHORT $X --accum $acc)
  echo "$j g47_$name ($gpu, accum $acc) $X"
done

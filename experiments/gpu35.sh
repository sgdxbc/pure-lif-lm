#!/bin/bash
# GPU wave 35: less dense arithmetic per token at the same quality (docs/experiments.md 5.46). From the repository root:
#   bash experiments/gpu35.sh [NAME]     all four short runs, or only NAME (base, r256, mspk, r256mspk);
#                                        GPU=... puts them on another card type, ACCUM=n overrides the batch split
# Against a Transformer of equal quality on a GPU (5.43), the hardware network's FPGA advantage holds only at a few
# streams: it needs 58.6M dense multiply-adds per token against the Transformer's 18-30M, and at many streams the GPU
# does dense arithmetic more efficiently. Of the 58.6M, 48% is the populations reading the real-valued bus (W_in) and
# 32% the plastic synapses' key, query and value synapses and W_o. Two options aim at these (model.py):
#  - in_rank 256: W_in through rank 256, 256 x (768 + 3,072) instead of 768 x 3,072 per population;
#  - mem_src spikes: the key, query and value synapses in front of population l > 0 read population l - 1's spikes
#    (3,072 neurons firing at a few percent): additions of the rows that fired instead of dense multiply-adds.
# Four runs on the hardware network's recipe (g29_hw: 138M-class, 64 keys, coded input, two-level readout, distilled
# from g13_tf163) at full precision, 20,000 steps x 32 x 1,024 = 655M tokens with the cosine schedule (as wave 27),
# the same seed and so the same batches:
#   base       the recipe as is (dense multiply-adds 58.6M per token)
#   r256       --in_rank 256 (42.1M)
#   mspk       --mem_src spikes (45.7M dense + the spike-driven additions)
#   r256mspk   both (29.1M)
# HW=1: hw_cost.py and the 8- and 4-bit rounding evaluations after training.
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
MEM="--mem_heads 8 --mem_write_avg true"
IO="--out_classes 225 --out_short 8192 --class_file $H8 --in_groups 2 --in_short 2048 --in_class_file $E2"
only=$1
case "$only" in ""|base|r256|mspk|r256mspk) ;; *) echo "NAME is base, r256, mspk or r256mspk"; exit 1 ;; esac
for s in "base h100-96" "r256 h100-47" "mspk a100-40" "r256mspk a100-40"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    base) X="" ;;
    r256) X="--in_rank 256" ;;
    mspk) X="--mem_src spikes" ;;
    r256mspk) X="--in_rank 256 --mem_src spikes" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g35_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch \
      runs/g35_$name $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IO $X --accum $acc)
  echo "$j g35_$name ($gpu, accum $acc) $X"
done

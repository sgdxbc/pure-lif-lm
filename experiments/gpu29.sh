#!/bin/bash
# GPU wave 29: the network for the FPGA comparison trained in full (stage goal 3; docs/experiments.md 5.38, 5.39).
# From the repository root:
#   bash experiments/gpu29.sh                     the main run (g29_hw)
#   bash experiments/gpu29.sh branch START NAME   a decay branch from the main run's kept checkpoint at START
# The network: 138M shape, plastic synapses before every layer (8 groups x 64 keys, write scaled by 1 - lambda), coded
# input with a short list of 2,048 words (g27_io2k) and two-level readout with a short list of 8,192 words (g27_cls8k:
# 0.028 better than 2,048 at step 10,000, at every evaluation from step 2,500 on; 5.36), distilled from g13_tf163, the
# recipe of waves 25-28. With 4-bit weights and the readout and input rows at 8 bits, the weights every step uses come
# to 48.1 MB, which with the state of 16 streams (1.33 MB each) makes 69.3 MB and fits the 75.6 MB of usable on-chip
# memory of the AMD V80; the word tables (input rows, word readout neurons) stay in HBM.
# Schedule: --schedule wsd, constant learning rate, then down to 0 over the last fifth, shaped 1 - sqrt (as wave 26).
# 4-bit training only during that decay (--wbits_from at its start): on the CPU this costs the same as starting earlier,
# +0.018 all at 4 bits and +0.002 with the readout, input rows and W_in at 8 bits (5.38 point 7).
#  - main run    75,000 steps (2.46B tokens, the length of waves 13-21; g18_kdmem138, the same shape with the full
#                vocabulary and 32-bit weights, reached 3.621), decay from 60,000, 4-bit with the readout and input rows
#                at 8 bits; keeps the checkpoints of 30,000 and 60,000
#  branches (each ends at 1.25 x START, decay over the last 0.25 x START):
#  - START=60000 NAME=fp    the same decay at full precision: what 4 bits cost at this length, and the full-precision
#                           point of the network
#  - START=60000 NAME=m8w   4-bit with W_in also at 8 bits (62.2 MB of weights: 10 streams on chip instead of 16)
#  - START=30000 NAME=q     the main run's own setting ending at 37,500 steps (1.23B tokens; as g26_snn212_b37500)
# Expected time on the H100-96: about 75-80k tok/s (the same shape with the full readout, g21_kdwa138, 105k; the
# two-level readout costs about a quarter in training, g25_cls 77k on an H200), so about 9 hours for the main run,
# 1.8 hours for each branch from 60,000 and 0.9 hours for the one from 30,000.
# GPU=... picks another card type; ACCUM=n splits each batch in n (1 on 96 GB and more, 2 on 80 GB, 4 on 40-47 GB).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
H8=runs/classes/tf163_head225_s8192.npy; E2=runs/classes/tf163_embed220_s2048.npy
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $H8 $E2; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
gpu=${GPU:-h100-96}
case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
acc=${ACCUM:-$acc}
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10 --accum $acc"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
IO="--out_classes 225 --out_short 8192 --class_file $H8 --in_groups 2 --in_short 2048 --in_class_file $E2"
M8='^head\.,^embed\.'                          # readout and input rows at 8 bits
M8W='^head\.,^embed\.,^layers\.\d+\.w_in\.'    # and W_in
sub() {  # sub NAME train.py-flags...   (prints the job id of the first part)
  local name=$1; shift
  sbatch --parsable -p gpu -J g29_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g29_$name "$@"
}
if [ "$1" != "branch" ]; then
  LEN="--batch 32 --seq 1024 --steps 75000 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
  WSD="--schedule wsd --decay_steps 15000 --min_lr_frac 0 --keep_ckpt 30000,60000"
  j=$(sub hw $DATA $SPREAD $LEN $WSD $N138 $SNN $KD $MEM $IO --wbits 4 --wbits_from 60000 --wbits8 "$M8")
  echo "$j g29_hw ($gpu, accum $acc)"
  exit 0
fi
start=$2; name=$3
end=$((start * 5 / 4)); decay=$((end - start))
ck=runs/g29_hw/ckpt_step$start.pt
[ -s $ck ] || { echo "$ck missing"; exit 1; }
LEN="--batch 32 --seq 1024 --steps $end --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps $decay --min_lr_frac 0 --init_ckpt $ck"
case $name in
  fp)  Q="" ;;
  q)   Q="--wbits 4 --wbits_from $start --wbits8 $M8" ;;
  m8w) Q="--wbits 4 --wbits_from $start --wbits8 $M8W" ;;
  *) echo "NAME is fp, q or m8w"; exit 1 ;;
esac
j=$(sub hw_${name}_b$end $DATA $SPREAD $LEN $WSD $N138 $SNN $KD $MEM $IO $Q)
echo "$j g29_hw_${name}_b$end ($gpu, accum $acc): steps $start -> $end, decay over $decay"

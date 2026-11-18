#!/bin/bash
# GPU wave 37: what the 0/1 spikes cost, and soft-to-hard training (docs/experiments.md 5.48). From the repository root:
#   bash experiments/gpu37.sh [NAME]     both runs, or only NAME (soft, anneal); GPU=... / ACCUM=n to override
# The linear-attention control (5.47) has the spiking network's fixed-size state but not its spikes, and so far it is
# far ahead of it (0.24 at step 10,000 without a teacher, 0.45 at step 5,000 with one): most of the gap looks like the
# cost of the spiking. model.py --spike_soft rho makes every LIF neuron output (1 - rho) [V >= theta] + rho f(z), f a
# smooth step around the threshold (lif.py soft_fn), with the reset using that output. Two runs on wave 35's control
# recipe (g35_base: the hardware network's shape and recipe at full precision, 20,000 steps, cosine, distilled from
# g13_tf163, 3.863 averaged), the same seed and so the same batches:
#   soft     rho = 1 throughout: a non-spiking twin (real-valued outputs, exact gradients). g35_base - soft = the cost
#            of the 0/1 signals and the surrogate gradient, everything else equal.
#   anneal   rho = 1 until step 5,000, linearly to 0 at step 15,000, then 5,000 steps as a plain spiking network:
#            soft-to-hard training. Its end is an ordinary spiking network, so it is compared with g35_base directly
#            (HW=1: hardware cost and the 8- and 4-bit rounding after training).
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
case "$only" in ""|soft|anneal) ;; *) echo "NAME is soft or anneal"; exit 1 ;; esac
for s in "soft a100-80" "anneal h100-96"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    soft) X="--spike_soft 1"; HW=0 ;;
    anneal) X="--spike_soft 1 --soft_hold 5000 --soft_end 15000"; HW=1 ;;
  esac
  j=$(sbatch --parsable -p gpu -J g37_$name --gres=gpu:$gpu:1 \
      --export=ALL,RELAY=$gpu,HW=$HW,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g37_$name $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IO $X --accum $acc)
  echo "$j g37_$name ($gpu, accum $acc) $X"
done

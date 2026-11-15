#!/bin/bash
# GPU wave 3: do the three training methods that helped in wave 1 (c) add up? Run on the cluster from the
# repository root:  bash experiments/gpu3.sh
# Same network and length as wave 1 (c): 12.6M spiking (gamma spread + leak synapses, rank 64), 14000 steps
# (115M tokens), lr 3e-3; reference g1_s_base 1.7523. Single methods there: distillation from g1_tf_full with
# weight 0.9 -> 1.6720, EMA of the weights (0.998) -> 1.7198, neuron parameters at 10x learning rate -> 1.7248.
#  - pairs and all three together
#  - two more distillation settings: weight 1.0 (teacher only, no hard labels) and temperature 2
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LONG="--batch 32 --seq 256 --steps 14000 --warmup 300 --eval_every 1000 --eval_batches 40 --log_every 100"
B12="$SPREAD --leak_rank 64 $LONG --lr 3e-3"
T="--teacher runs/g1_tf_full/ckpt.pt --distill_temp 1"
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g3_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g3_$name "$@")
  echo "$id g3_$name ($gpu)"
}
sub all          h100-47 $B12 $T --distill_weight 0.9 --ema 0.998 --neuron_lr_mult 10
sub d09_ema      h100-47 $B12 $T --distill_weight 0.9 --ema 0.998
sub d09_nlr10    a100-40 $B12 $T --distill_weight 0.9 --neuron_lr_mult 10
sub ema_nlr10    a100-40 $B12 --ema 0.998 --neuron_lr_mult 10
sub d10          a100-80 $B12 $T --distill_weight 1.0
sub d09_T2       h100-96 $SPREAD --leak_rank 64 $LONG --lr 3e-3 --teacher runs/g1_tf_full/ckpt.pt --distill_weight 0.9 \
                 --distill_temp 2
echo "submitted; squeue -u \$USER to follow"

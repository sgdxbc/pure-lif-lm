#!/bin/bash
# GPU wave 6: is the spiking network short of data? Run on the cluster from the repository root:  bash experiments/gpu6.sh
# In wave 4, going from 117M to 257M parameters gained only 0.030 (plain) / 0.036 (with the three methods), against
# about 0.065 per doubling from 27M to 117M in wave 2. The 257M network saw 549M tokens, about 2 tokens per
# parameter (117M: 4.7), so it may simply not have seen enough text. Here the two "best" networks of wave 4
# (distillation 0.9 from g2_tf105_lr6e-4 + EMA 0.999 + neuron lr x10) train for two passes over the training set
# (134000 steps, 1.10B tokens), the 117M also for three (201000 steps, 1.65B tokens); the training set has 552M
# tokens, so the extra steps reuse the same stories. The 105M Transformer also gets two passes, to see whether
# the same extra data helps it as much. Everything else is as in gpu4.sh (reference: g4_snn117_best 1.2776,
# g4_snn257_best 1.2420, g4_tf105_lr4e-4 1.1363, all EMA where used).
# Expected times: snn257 2 passes ~1.9 h on the H200; snn117 3 passes ~3 h on an h100-47, so it may stop at the 3 h
# limit once: submit the same command again and it continues from its last checkpoint (--resume).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 256 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
EP2="$RUN --steps 134000"
EP3="$RUN --steps 201000"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
N257="--d_model 1536 --d_hidden 6144 --n_layers 12 --leak_rank 192"
BEST="--teacher runs/g2_tf105_lr6e-4/ckpt.pt --distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g6_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g6_$name "$@")
  echo "$id g6_$name ($gpu)"
}
sub snn257_best_2ep  h200-141 $SPREAD $EP2 $N257 --lr 5e-4 $BEST
sub snn117_best_2ep  h100-96  $SPREAD $EP2 $N117 --lr 6e-4 $BEST
sub snn117_best_3ep  h100-47  $SPREAD $EP3 $N117 --lr 6e-4 $BEST
sub tf105_2ep        h100-47  --arch tf $EP2 --d_model 896 --n_layers 10 --n_heads 14 --lr 4e-4
echo "submitted; squeue -u \$USER to follow"

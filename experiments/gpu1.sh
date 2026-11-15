#!/bin/bash
# GPU wave 1. Run on the cluster login node from the repository root:  bash experiments/gpu1.sh
# Only submits jobs (slurm/train.sbatch, one GPU each) and prints their ids; SLURM queues whatever exceeds the
# per-user GPU limits. Results go to runs/g1_*.
#  (a) Does the GPU reproduce the CPU results? Same settings as runs/tf_full (Transformer, 1.480),
#      runs/p2_large (spiking 25M, 1.793) and runs/p7_large_leak64 (phase 7, still running on the CPU).
#  (b) Why does 117M diverge? The speed test warmed up for only 5 steps. Learning-rate sweep with the usual
#      300-step warm-up, 2000 steps each (no probes).
#  (c) Phase 8 on the GPU: training methods for the 12.6M network (gamma spread + leak synapses, rank 64), each at
#      full length (14000 steps = 115M tokens) instead of the 1500-step CPU screen; s_base is the reference.
#      The two distillation runs wait for the Transformer from (a), their teacher.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LONG="--batch 32 --seq 256 --steps 14000 --warmup 300 --eval_every 1000 --eval_batches 40 --log_every 100"
DEP=""
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g1_$name --gres=gpu:$gpu:1 $DEP slurm/train.sbatch runs/g1_$name "$@")
  echo "$id g1_$name ($gpu)" >&2
  echo $id
}

# (a)
TF=$(sub tf_full         a100-40 $(python3 slurm/args_of.py runs/tf_full))
sub p2_large             a100-40 $(python3 slurm/args_of.py runs/p2_large) > /dev/null
sub p7_large_leak64      h100-47 $(python3 slurm/args_of.py runs/p7_large_leak64) > /dev/null

# (b)
L117="$SPREAD --leak_rank 128 --d_model 1024 --d_hidden 4096 --n_layers 12 --batch 32 --seq 256 --steps 2000 \
      --warmup 300 --eval_every 500 --eval_batches 20 --log_every 50"
for lr in 2.5e-3 1.2e-3 6e-4; do
  PROBES=0 sub 117m_lr$lr h100-96 $L117 --lr $lr > /dev/null
done

# (c)
B12="$SPREAD --leak_rank 64 $LONG"
sub s_base               h100-47 $B12 --lr 3e-3 > /dev/null
sub s_carry              a100-80 $B12 --lr 3e-3 --carry_state > /dev/null
sub s_lr45               a100-40 $B12 --lr 4.5e-3 > /dev/null
sub s_lr2                a100-40 $B12 --lr 2e-3 > /dev/null
sub s_neuronlr10         h100-47 $B12 --lr 3e-3 --neuron_lr_mult 10 > /dev/null
sub s_alpha2             a100-80 $B12 --lr 3e-3 --alpha 2 > /dev/null
sub s_alpha8             a100-40 $B12 --lr 3e-3 --alpha 8 > /dev/null
sub s_ema998             h100-47 $B12 --lr 3e-3 --ema 0.998 > /dev/null
DEP="--dependency=afterok:$TF"
for w in 0.5 0.9; do
  sub s_distill${w/./}   a100-80 $B12 --lr 3e-3 --teacher runs/g1_tf_full/ckpt.pt --distill_weight $w --distill_temp 1 \
    > /dev/null
done
echo "submitted; squeue -u \$USER to follow" >&2

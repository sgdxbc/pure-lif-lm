#!/bin/bash
# GPU wave 15: hidden-state distillation, screened on TinyStories (docs/experiments.md 5.24). From the repository root:  bash experiments/gpu15.sh
# So far the only training change that clearly improved the spiking network's far recall is distillation from the
# Transformer's output distribution (docs/experiments.md 5.16-5.18); a better teacher passes on less and less (5.19, 5.23). This
# wave gives the student a denser signal from the same teacher: at 10 pairs of layers at matching relative depth, a
# linear map (training only) must predict the teacher's residual stream from the student's bus (train.py
# --hint_weight). The teacher's residual stream holds what its attention gathered from the context at every position.
#  - the 117M with the three methods (teacher g2_tf105_lr6e-4, 10 layers), hint weight 0.1 and 1.0, against
#    g4_snn117_best (EMA 1.2776, same settings without hints)
# Expected: ~2-2.5 h each (H100 47GB / A100 40GB); each run chains 3-hour parts by itself (RELAY).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
METHODS="--distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
T1="--teacher runs/g2_tf105_lr6e-4/ckpt.pt"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g15_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu slurm/train.sbatch runs/g15_$name "$@"
}
a=$(sub hint117_w01 h100-47 $SPREAD $RUN $N117 $T1 $METHODS --lr 6e-4 --hint_weight 0.1); echo "$a g15_hint117_w01 (h100-47)"
b=$(sub hint117_w1  a100-40 $SPREAD $RUN $N117 $T1 $METHODS --lr 6e-4 --hint_weight 1.0); echo "$b g15_hint117_w1 (a100-40)"
echo "submitted; squeue -u \$USER to follow"

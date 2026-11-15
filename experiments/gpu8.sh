#!/bin/bash
# GPU wave 8: why did doubling the spiking network from 117M to 257M gain so little? Run from the repository root:
#   bash experiments/gpu8.sh
# With the three methods, 117M -> 257M improved validation loss by only 0.036 with one pass over the data and 0.039
# with two (wave 6), against about 0.065 per doubling from 27M to 117M in wave 2. Wave 6 ruled out the data: the
# second pass helped 117M and 257M equally (0.040 / 0.043). Two causes remain, tested here on one pass, each against
# g4_snn257_best (EMA 1.2420; recall 1.92 / 1.49 at 65-128 / 129-256):
#  - shape: 117M -> 257M only widened the layers (d_model 1024 -> 1536, still 12 layers), whereas 27M -> 48M also went
#    from 10 to 12 layers. deep27 keeps the 117M width and stacks 27 layers instead: 253.0M parameters.
#  - learning rate: 257M was only trained at 5e-4; here also 3e-4 and 8e-4.
# A fourth run asks whether a better teacher helps: the 105M Transformer trained for two passes (g6_tf105_2ep, 1.091)
# instead of one (g2_tf105_lr6e-4, 1.139).
# Everything else as in gpu4.sh (three methods: distillation 0.9 / temperature 1, EMA 0.999, neuron lr x10).
# Expected times: 257M ~55 min on the H200 and ~1.5 h on an H100; deep27 is slower per token (more layers and neurons).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
N257="--d_model 1536 --d_hidden 6144 --n_layers 12 --leak_rank 192"
D27="--d_model 1024 --d_hidden 4096 --n_layers 27 --leak_rank 128"
METHODS="--distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
T1="--teacher runs/g2_tf105_lr6e-4/ckpt.pt"
T2="--teacher runs/g6_tf105_2ep/ckpt.pt"
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g8_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g8_$name "$@")
  echo "$id g8_$name ($gpu)"
}
sub deep27_best        h200-141 $SPREAD $RUN $D27  --lr 5e-4 $T1 $METHODS
sub snn257_best_lr3e-4 h100-47  $SPREAD $RUN $N257 --lr 3e-4 $T1 $METHODS
sub snn257_best_lr8e-4 h100-47  $SPREAD $RUN $N257 --lr 8e-4 $T1 $METHODS
sub snn257_best_t2ep   h100-96  $SPREAD $RUN $N257 --lr 5e-4 $T2 $METHODS
echo "submitted; squeue -u \$USER to follow"

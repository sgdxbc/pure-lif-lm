#!/bin/bash
# GPU wave 5: is long-range recall limited by how many neurons (how much state) the spiking network has?
# Run on the cluster from the repository root:  bash experiments/gpu5.sh
# A recurrent spiking network keeps everything it knows about the past in its membrane potentials, so its state
# is (neurons per layer) x (layers), whatever the parameter count. Wave 2 scaled the bus width (d_model) and
# the neurons per layer together, so parameters grew faster than state (27M -> 117M: 4.4x parameters, 2.4x
# neurons) and recall at 129-256 tokens grew only from 1.17 to 1.29.
# Here the parameter count stays about the same as g2_snn117 (117.1M, 49,152 neurons; 1.411 at lr 1.2e-3,
# 1.3905 at 8e-4) while the neurons double or triple, by narrowing the bus:
#  - wide2x: d_model 512, 8192 neurons per layer, 12 layers: 118.6M parameters, 98,304 neurons (both learning rates)
#  - wide3x: d_model 384, 12288 neurons per layer, 12 layers: 136.4M parameters, 147,456 neurons
# Same data and schedule as wave 2 (549M tokens); compare validation loss and, above all, recall by distance.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
FULL="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g5_$name --gres=gpu:$gpu:1 --exclude=xgpj0 slurm/train.sbatch runs/g5_$name "$@")
  echo "$id g5_$name ($gpu)"
}
W2="$SPREAD $FULL --d_model 512 --d_hidden 8192 --n_layers 12 --leak_rank 128"
W3="$SPREAD $FULL --d_model 384 --d_hidden 12288 --n_layers 12 --leak_rank 128"
sub wide2x_lr8e-4    a100-40 $W2 --lr 8e-4
sub wide2x_lr1.2e-3  a100-40 $W2 --lr 1.2e-3
sub wide3x_lr8e-4    a100-80 $W3 --lr 8e-4
echo "submitted; squeue -u \$USER to follow"

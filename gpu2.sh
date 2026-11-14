#!/bin/bash
# GPU wave 2: scaling on the whole training set. Run on the cluster login node from the repository root:
#   bash gpu2.sh
# Every run sees 67000 steps x 8192 tokens = 549M tokens (about one pass over the 552M training tokens; all
# earlier runs saw 115M). Batch, sequence length, warm-up and schedule are the same as before, so the only
# changes are model size and data.
#  - spiking (gamma spread + leak synapses): 27M, 48M, 117M
#  - Transformer baseline at four sizes: 12M, 30M, 55M, 105M (MLP width 4 x d_model, as in model_tf.py)
#  - one lower learning rate for each of the two largest models, to check that the default is not far off
# Learning rates: 27M spiking 2.5e-3 and 12M Transformer 2e-3 as before; 117M spiking 1.2e-3 (best of the
# 2000-step sweep in wave 1); the rest interpolated. Expected time per run 20-70 min depending on the card.
set -e
cd "$(dirname "$0")"
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
FULL="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g2_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g2_$name "$@")
  echo "$id g2_$name ($gpu)"
}
SNN="$SPREAD $FULL"
sub snn117          h200-141$SNN --d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128 --lr 1.2e-3
sub tf105           a100-80 --arch tf $FULL --d_model 896 --n_layers 10 --n_heads 14 --lr 1e-3
sub snn117_lr8e-4   h100-96 $SNN --d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128 --lr 8e-4
sub tf105_lr6e-4    h100-47 --arch tf $FULL --d_model 896 --n_layers 10 --n_heads 14 --lr 6e-4
sub snn48           h100-47 $SNN --d_model 640 --d_hidden 2560 --n_layers 12 --leak_rank 96 --lr 1.8e-3
sub tf55            a100-40 --arch tf $FULL --d_model 640 --n_layers 10 --n_heads 10 --lr 1.2e-3
sub snn27           a100-40 $SNN --d_model 512 --d_hidden 2048 --n_layers 10 --leak_rank 64 --lr 2.5e-3
sub tf30            a100-40 --arch tf $FULL --d_model 512 --n_layers 8 --n_heads 8 --lr 1.5e-3
sub tf12            a100-40 --arch tf $FULL --d_model 384 --n_layers 5 --n_heads 6 --lr 2e-3
echo "submitted; squeue -u \$USER to follow"

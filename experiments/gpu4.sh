#!/bin/bash
# GPU wave 4: push the spiking network towards the Transformers of wave 2. Run on the cluster from the
# repository root:  bash experiments/gpu4.sh      (after wave 3 has been submitted; jobs queue behind it)
# All runs: whole training set, 67000 steps x 8192 tokens = 549M tokens, as in wave 2.
# Wave 2 found the lower of two learning rates better for both large models (117M spiking 8e-4: 1.3905,
# 105M Transformer 6e-4: 1.1385), so
#  - snn117_lr5e-4, tf105_lr4e-4: one step lower again
#  - snn257_lr5e-4: a larger spiking network (d_model 1536, 6144 neurons per layer, 12 layers, 257M parameters)
#  - snn117_best / snn257_best: the same networks with the training methods that helped in phase 8: distillation
#    (weight 0.9) from the best Transformer of wave 2 (g2_tf105_lr6e-4, 1.1385), weight EMA (0.999 for this
#    longer run) and 10x learning rate for the neuron parameters
# Heavy runs go to the fastest cards; if one reaches the 3 h limit, submit the same command again (--resume).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
FULL="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
S117="$SPREAD $FULL --d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
S257="$SPREAD $FULL --d_model 1536 --d_hidden 6144 --n_layers 12 --leak_rank 192"
BEST="--teacher runs/g2_tf105_lr6e-4/ckpt.pt --distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
sub() {  # sub NAME GPU_TYPE train.py-flags...
  local name=$1 gpu=$2; shift 2
  local id=$(sbatch --parsable -J g4_$name --gres=gpu:$gpu:1 --exclude=xgpj0 slurm/train.sbatch runs/g4_$name "$@")
  echo "$id g4_$name ($gpu)"
}
sub snn257_best     h200-141 $S257 --lr 5e-4 $BEST
sub snn117_best     h100-96  $S117 --lr 6e-4 $BEST
sub snn257_lr5e-4   h100-47  $S257 --lr 5e-4
sub snn117_lr5e-4   h100-47  $S117 --lr 5e-4
sub tf105_lr4e-4    a100-80  --arch tf $FULL --d_model 896 --n_layers 10 --n_heads 14 --lr 4e-4
echo "submitted; squeue -u \$USER to follow"

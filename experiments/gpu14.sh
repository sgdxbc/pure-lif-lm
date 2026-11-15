#!/bin/bash
# GPU wave 14: a better optimizer, screened on TinyStories (docs/experiments.md 5.24). From the repository root:  bash experiments/gpu14.sh
# The owner suggested trying improvements to the training method first. Muon (train.py --optim muon) replaces AdamW
# for the 2-D weights inside the layers (W_in, W_out, the leak synapses; attention and MLP matrices in the
# Transformer): momentum, then the update matrix is orthogonalised (Newton-Schulz). In Transformer training
# (modded-nanogpt, Moonlight) it reaches the same loss with clearly fewer tokens than AdamW; whether it does the same
# for the spiking network is the question. The update is scaled so that AdamW's learning rate carries over.
#  - the 117M with the three methods (teacher g2_tf105_lr6e-4), Muon at lr 6e-4 and 1.2e-3, against g4_snn117_best
#    (EMA 1.2776, AdamW at 6e-4)
#  - the 105M Transformer with Muon at lr 6e-4, against g2_tf105_lr6e-4 (1.139, AdamW at 6e-4): the Transformer
#    baselines should use the better optimizer too if it helps them
# Expected: ~1.5 h each (A100 40GB / H100 47GB); each run chains 3-hour parts by itself (RELAY).
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
  sbatch --parsable -p gpu -J g14_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu slurm/train.sbatch runs/g14_$name "$@"
}
a=$(sub muon117      a100-40 $SPREAD $RUN $N117 $T1 $METHODS --optim muon --lr 6e-4); echo "$a g14_muon117 (a100-40)"
b=$(sub muon117_lr12 a100-40 $SPREAD $RUN $N117 $T1 $METHODS --optim muon --lr 1.2e-3); echo "$b g14_muon117_lr12 (a100-40)"
c=$(sub tf105_muon   h100-47 $RUN --arch tf --d_model 896 --n_layers 10 --n_heads 14 --optim muon --lr 6e-4); echo "$c g14_tf105_muon (h100-47)"
echo "submitted; squeue -u \$USER to follow"

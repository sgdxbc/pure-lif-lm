#!/bin/bash
# GPU wave 16: why Muon hurts the spiking network (README 5.27), on TinyStories. From the repository root:  bash gpu16.sh
# Wave 14: Muon improved the 105M Transformer by 0.023 but made the spiking 117M worse (+0.013 at lr 6e-4, +0.023 at
# 1.2e-3), and the loss is concentrated in far recall (net d=129-256 1.10 -> 0.87), while short context barely moved.
# Two cheap checks before setting Muon aside for the spiking network:
#  - Muon only for W_in and W_out; the leak-modulating synapses, which decide when neurons hold or forget, stay on
#    AdamW (--muon_skip leak_), lr 6e-4
#  - Muon for all layer matrices at a lower learning rate, 3e-4 (1.2e-3 was worse than 6e-4)
# Both against g4_snn117_best (AdamW, EMA 1.2776) and g14_muon117 (Muon, lr 6e-4, EMA 1.2904).
# Expected: ~2.5 h each (A100 40GB); each run chains 3-hour parts by itself (RELAY).
set -e
cd "$(dirname "$0")"
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
METHODS="--distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
T1="--teacher runs/g2_tf105_lr6e-4/ckpt.pt"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g16_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu slurm/train.sbatch runs/g16_$name "$@"
}
a=$(sub muon117_noleak a100-40 $SPREAD $RUN $N117 $T1 $METHODS --optim muon --muon_skip leak_ --lr 6e-4); echo "$a g16_muon117_noleak (a100-40)"
b=$(sub muon117_lr3    a100-40 $SPREAD $RUN $N117 $T1 $METHODS --optim muon --lr 3e-4); echo "$b g16_muon117_lr3 (a100-40)"
echo "submitted; squeue -u \$USER to follow"

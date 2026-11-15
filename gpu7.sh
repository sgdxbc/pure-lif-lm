#!/bin/bash
# GPU wave 7: is long-range recall limited by how rarely the training windows contain long-range context?
# Run on the cluster from the repository root:  bash gpu7.sh
# Wave 5 found that more neurons (state) do not improve recall at 129-256 tokens, and wave 4 that more parameters do
# not either, while distillation did (117M: 1.29 -> 1.51). So the limit looks like the training signal rather than
# capacity. One reason the signal is weak: TinyStories stories are short (mean 202 tokens, median 179; 13% reach 256
# tokens), and training reads random 256-token windows that usually start in the middle of a story. Measured on the
# training data, a position has more than 128 earlier tokens of its own story inside the window only 17.9% of the
# time, and more than 192 only 3.6%. With 512-token windows these become 27.1% and 9.2%.
# Here the 117M network of wave 2 (g2_snn117_lr8e-4: 1.391; recall 1.73 / 1.29 at 65-128 / 129-256) trains on
# 512- and 1024-token windows with the same tokens per step (batch 16 / 8, so 8192 tokens per step as before) and the
# same 67000 steps and schedule; nothing else changes. Validation loss is still measured on 256-token windows
# (evaluate.py), so it stays comparable with every earlier run.
# Afterwards slurm/probe_long.sbatch measures recall on the longer stories read to 512 tokens (adds d=257-512) for
# these two runs and for 117M/257M networks trained on 256-token windows. Expected times: the LIF kernel walks through
# time, so longer windows with smaller batches are slower; the 1024 run may stop at the 3 h limit once: then submit the
# same command again (it resumes), and submit the probe job again by hand after both runs are done.
set -e
cd "$(dirname "$0")"
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -J g7_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g7_$name "$@"
}
a=$(sub seq512  a100-40 $SPREAD $RUN --batch 16 --seq 512  $N117 --lr 8e-4); echo "$a g7_seq512 (a100-40)"
b=$(sub seq1024 a100-80 $SPREAD $RUN --batch 8  --seq 1024 $N117 --lr 8e-4); echo "$b g7_seq1024 (a100-80)"
c=$(sbatch --parsable -J probe_long --gres=gpu:a100-40:1 --dependency=afterok:$a:$b slurm/probe_long.sbatch \
    runs/g7_seq512 runs/g7_seq1024 runs/g2_snn117_lr8e-4 runs/g4_snn117_best runs/g4_snn257_best)
echo "$c probe_long (after both runs)"
echo "submitted; squeue -u \$USER to follow"

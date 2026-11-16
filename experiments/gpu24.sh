#!/bin/bash
# GPU wave 24: the learning-rate schedule for the long run (docs/running.md, --schedule wsd). From the repository root:
#   bash experiments/gpu24.sh
# With --schedule wsd the learning rate stays at --lr and only falls over the last --decay_steps, so one long run can
# branch off short decays at several points and give a finished result at each training length (the points of the
# compute-scaling plot) for about 10-20% more compute, instead of one run per length. Hägele et al. (2024, arXiv
# 2405.18392) report that a constant learning rate with such a decay ("cooldown", best shaped 1 - sqrt(progress))
# matches or beats a cosine schedule. Check that here, with a decay over the last 20%, before the long run uses it:
#  - kdk256t1wsd  g23_kdk256t1 (138M, KD, 8 x 256 keys, time constants from 1, cosine to 0.1 lr) with wsd: decay over
#                 the last 4,000 of 20,000 steps, to 0. Keeps the step-10,000 checkpoint for a decay branch later.
# 20,000 steps (655M tokens), seed 0, the same batches as g23_kdk256t1.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
WSD="--schedule wsd --decay_steps 4000 --min_lr_frac 0 --keep_ckpt 10000"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g24_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g24_$name "$@"
}
a=$(sub kdk256t1wsd h100-47 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $WSD --accum 4); echo "$a g24_kdk256t1wsd (h100-47)"
echo "submitted; squeue -u \$USER to follow"

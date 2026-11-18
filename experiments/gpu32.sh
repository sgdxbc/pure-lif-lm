#!/bin/bash
# GPU wave 32: a much stronger teacher late in the long run (docs/experiments.md 5.42). From the repository root:
#   bash experiments/gpu32.sh
# At a constant learning rate the spiking trunk g26_snn212 stalled from step 50,000 to 120,000 with its first teacher
# g13_tf163 (3.310): averaged weights 3.629 -> 3.610, and the distance to the teacher (distill_kl ~0.50) stopped
# shrinking too. Switched to g26_tf163 (3.162) at step 120,000, it fell again (3.610 -> 3.582 by 135,000), and the two
# decay branches from step 120,000 that differ only in the teacher (same checkpoint, same batches) are 0.025 apart
# after 2,500 and 5,000 steps (averaged weights, 5.41). GPT-2 XL (hf:openai-community/gpt2-xl, the same tokenizer) is
# 2.769 on this validation set (runs/ref_loss.json), 0.39 better than g26_tf163. Here:
#  - xl120  the trunk's exact recipe continued from runs/g26_snn212/ckpt_step120000.pt at the constant learning rate
#           for 10,000 steps, distilled from GPT-2 XL instead of g26_tf163. Its first relay part reads the same batches
#           as the trunk's part that started at step 120,000 (each part draws them from seed + its first step), so it
#           is paired with the trunk's evaluations at steps 125,000 and 130,000 (3.597 / 3.587, averaged weights).
# If it is clearly better, the trunk switches to GPT-2 XL at its next checkpoint after this result. Cost: GPT-2 XL's
# forward pass is about 1e14 FLOPs per batch of 32 x 1,024 tokens, about as much as the student's own step on an H100.
# PROBES=0: no full-validation or zero-shot evaluation afterwards (the evaluations in the log are paired).
# GPU=... picks the card (default h100-47), ACCUM=n the split of each batch (default 4; 2 on an 80 GB card or larger).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
ck=runs/g26_snn212/ckpt_step120000.pt
[ -s $ck ] || { echo "$ck missing"; exit 1; }
gpu=${GPU:-h100-47}; acc=${ACCUM:-4}
# exactly g26_snn212's flags (experiments/gpu26.sh, gpu26_branch.sh), the schedule constant to the last step
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LEN="--batch 32 --seq 1024 --steps 130000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps 1 --min_lr_frac 0 --init_ckpt $ck"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher hf:openai-community/gpt2-xl --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
name=g32_xl120
j=$(sbatch --parsable -p gpu -J $name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,PROBES=0 slurm/train.sbatch runs/$name \
    $DATA $SPREAD $LEN $WSD $N212 $SNN $KD $MEM --accum $acc)
echo "$j $name ($gpu, accum $acc): steps 120000 -> 130000 at the constant learning rate, teacher GPT-2 XL"

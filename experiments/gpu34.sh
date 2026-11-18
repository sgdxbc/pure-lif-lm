#!/bin/bash
# GPU wave 34: a moderately stronger teacher late in the long run (docs/experiments.md 5.45). From the repository root:
#   bash experiments/gpu34.sh
# Two teacher switches from the trunk's step-120,000 checkpoint so far (5.41, 5.42), at the constant learning rate:
#  - g13_tf163 (3.310) -> g26_tf163 (3.162), 0.15 better: the student improved on every measure (averaged weights
#    -0.025 after 2,500 and 5,000 steps in the paired decay branches; the trunk kept falling).
#  - g26_tf163 -> GPT-2 XL (2.769), 0.39 better: the student got worse (raw weights +0.042, paired training
#    cross-entropy +0.020 at step 125,000; KL to the teacher 1.1 against 0.6): too large a gap to the teacher.
# GPT-2 Medium (hf:openai-community/gpt2-medium, the same tokenizer) is 3.000 on this validation set (runs/ref_loss.json),
# 0.16 better than g26_tf163: the same step as the switch that worked. Here:
#  - med120  the trunk's exact recipe continued from runs/g26_snn212/ckpt_step120000.pt at the constant learning rate
#            for 10,000 steps, distilled from GPT-2 Medium. Its first relay part reads the same batches as the trunk's
#            part that started at step 120,000 and as g32_xl120's first part, so it is paired with both (trunk:
#            3.597 / 3.587 at steps 125,000 / 130,000, averaged weights; g32_xl120: 3.571 / 3.580 / 3.583 at
#            122,500 / 125,000 / 127,500).
# If it beats the trunk on the raw weights, the averaged weights and the paired training cross-entropy, the trunk
# switches to GPT-2 Medium at its next checkpoint. Cost: GPT-2 Medium's forward pass is about twice g26_tf163's
# (355M against 163M parameters), a fifth of GPT-2 XL's.
# PROBES=0: no full-validation or zero-shot evaluation afterwards (the evaluations in the log are paired).
# GPU=... picks the card (default a100-40), ACCUM=n the split of each batch (default 4; 2 on an 80 GB card or larger).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
ck=runs/g26_snn212/ckpt_step120000.pt
[ -s $ck ] || { echo "$ck missing"; exit 1; }
gpu=${GPU:-a100-40}; acc=${ACCUM:-4}
# exactly g26_snn212's flags (experiments/gpu26.sh, gpu26_branch.sh, gpu32.sh), the schedule constant to the last step
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LEN="--batch 32 --seq 1024 --steps 130000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps 1 --min_lr_frac 0 --init_ckpt $ck"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher hf:openai-community/gpt2-medium --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
name=g34_med120
j=$(sbatch --parsable -p gpu -J $name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,PROBES=0 slurm/train.sbatch runs/$name \
    $DATA $SPREAD $LEN $WSD $N212 $SNN $KD $MEM --accum $acc)
echo "$j $name ($gpu, accum $acc): steps 120000 -> 130000 at the constant learning rate, teacher GPT-2 Medium"

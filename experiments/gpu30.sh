#!/bin/bash
# GPU wave 30: no weight decay on the input rows (train.py --embed_wd 0; docs/experiments.md 5.40). From the repository
# root:  bash experiments/gpu30.sh
# Every 2-D weight is decayed (--wd 0.1), and the spiking network has no normalisation anywhere, so the input rows
# (the word vectors, initialised at size 1.0) shrink at almost exactly the rate weight decay alone would give:
# 0.52 after 20,000 cosine steps, 0.16 after the 37,500 wsd steps of g26_snn212_b37500, 0.045 after 75,000
# (experiments/embed_norms.py), while the other weight matrices stay at 0.03-0.05. With them the first population's
# output current grows from 0.7 to 19 times its input current (experiments/layer_stats.py): after a long run which word
# was read reaches the rest of the network almost only through the first population's ~200 spikes. A Transformer has
# LayerNorm after its input and does not see this.
#  - ewd0  exactly g26_snn212's recipe (212M, 8 x 256 keys from tau 1, distilled from g13_tf163, wsd), 37,500 steps
#          with the decay over the last 7,500, i.e. the same learning rate at every step as the trunk g26_snn212 up to
#          step 30,000 and as its branch g26_snn212_b37500 after it, only --embed_wd 0. Paired with both: the trunk's
#          evaluations every 5,000 steps (same validation batches) and the branch's final 3.625 (1.23B tokens).
#          The batches differ (each run draws them from its own part boundaries).
# GPU=... picks the card (default h100-47); ACCUM=n splits each batch in n (default 4: the 212M shape at 32 x 1024
# needs about 47 GB with --accum 2).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
gpu=${GPU:-h100-47}; acc=${ACCUM:-4}
# exactly g26_snn212's flags (experiments/gpu26.sh), with its branch's length and evaluations every 2,500 steps
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LEN="--batch 32 --seq 1024 --steps 37500 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps 7500 --min_lr_frac 0"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
j=$(sbatch --parsable -p gpu -J g30_ewd0 --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g30_ewd0 \
    $DATA $SPREAD $LEN $WSD $N212 $SNN $KD $MEM --accum $acc --embed_wd 0)
echo "$j g30_ewd0 ($gpu, accum $acc)"

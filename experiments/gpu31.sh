#!/bin/bash
# GPU wave 31: how much the Transformer's moving-average weights gain at a constant learning rate (docs/experiments.md
# 5.37). From the repository root:
#   bash experiments/gpu31.sh
# The spiking network's averaged weights (--ema 0.999) are 0.14 better than its current weights all through the stable
# phase of g26_snn212 (3.678 / 3.818 at step 30,000, 3.624 / 3.763 at 55,000), and its decay branches end only
# 0.06-0.07 below that average (g26_snn212_b37500 3.615, b75000 3.551, EMA): most of what the decay gives it, the
# average already had. The Transformer g26_tf163 has no average, so 5.37 compared current weights, and there the gap
# grows in the decay (0.165 -> 0.233 by step 37,500, ~0.17 -> 0.269 by 75,000). Two readings: (a) the Transformer gains
# more from the decay beyond averaging, i.e. a real deficit of the spiking network in the decay phase; (b) its current
# weights are just noisier at the high learning rate, its average is ~0.2-0.3 better, and the gap of the averaged
# weights is already ~0.23 in the stable phase (then the decay is not where it opens). These two runs tell them apart:
# g26_tf163's exact recipe continued from its kept checkpoints at the constant learning rate with --ema 0.999, i.e. the
# stable phase again (the batches are drawn afresh from the start step), evaluated every 2,500 steps on the same 40
# validation batches as all of wave 26, with and without the average:
#  - ema30  steps 30,000 -> 37,500, to compare with g26_tf163_b37500 (3.384) at its endpoint
#  - ema60  steps 60,000 -> 65,000 (the average is settled after ~3,000 steps), against g26_tf163_b75000 (3.284)
# --decay_steps 1 makes the wsd schedule constant through the last step (the 1 - sqrt decay starts at 0 at step
# steps - 1). PROBES=0: no full-validation or zero-shot evaluation afterwards, the evaluations in the log are paired.
# GPU=... picks the card (default h100-47; the 163M Transformer needs ~34 GB at --accum 1, ~116k tok/s on an H100,
# so about 35 and 25 minutes).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
gpu=${GPU:-h100-47}
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
TF="--arch tf --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4 --ema 0.999"
for p in "30000 37500" "60000 65000"; do
  set -- $p; start=$1; end=$2
  ck=runs/g26_tf163/ckpt_step$start.pt
  [ -s $ck ] || { echo "$ck missing"; exit 1; }
  name=g31_tf_ema$((start / 1000))
  LEN="--batch 32 --seq 1024 --steps $end --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
  j=$(sbatch --parsable -p gpu -J $name --gres=gpu:$gpu:1 --export=ALL,PROBES=0 slurm/train.sbatch runs/$name \
      $DATA $LEN --schedule wsd --decay_steps 1 --min_lr_frac 0 --init_ckpt $ck $TF)
  echo "$j $name ($gpu): steps $start -> $end at the constant learning rate, with the average"
done

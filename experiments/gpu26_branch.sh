#!/bin/bash
# A decay branch of a wave-26 long run (experiments/gpu26.sh): continue from one of its kept checkpoints with the
# learning rate falling to 0 over a fifth of the branch's final length, shaped 1 - sqrt, as at the end of the run.
# From the repository root:
#   bash experiments/gpu26_branch.sh RUN START [TEACHER]
#     RUN      snn212 or tf163
#     START    a kept step: 30000, 60000, 120000 or 180000 -> the branch ends at 1.25 x START (37,500, 75,000,
#              150,000 or 225,000 steps) in runs/g26_RUN_bEND
#     TEACHER  snn212 only: the teacher the run had at START (default runs/g13_tf163/ckpt.pt)
# The flags are those of gpu26.sh without --keep_ckpt, with evaluations every 2,500 steps. The weights, Adam moments
# and average come from the checkpoint; the batches are drawn afresh from its step on.
set -e
cd "$(dirname "$0")/.."   # the repository root
run=$1; start=$2; teacher=${3:-runs/g13_tf163/ckpt.pt}
end=$((start * 5 / 4)); decay=$((end - start))
ck=runs/g26_$run/ckpt_step$start.pt
[ -s $ck ] || { echo "$ck missing"; exit 1; }
mkdir -p slurm/logs
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LEN="--batch 32 --seq 1024 --steps $end --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps $decay --min_lr_frac 0 --init_ckpt $ck"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher $teacher --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
name=g26_${run}_b$end
case $run in
  snn212) flags="$DATA $SPREAD $LEN $WSD $N212 $SNN $KD $MEM --accum 2"; gpu=${GPU:-h200-141} ;;
  tf163)  flags="$DATA $LEN $WSD --arch tf --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4"; gpu=${GPU:-h100-96} ;;
  *) echo "RUN is snn212 or tf163"; exit 1 ;;
esac
j=$(sbatch --parsable -p gpu -J $name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/$name $flags)
echo "$j $name ($gpu): steps $start -> $end, decay over $decay"

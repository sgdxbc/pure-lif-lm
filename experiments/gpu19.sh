#!/bin/bash
# GPU wave 19: the plastic synapses' capacity (docs/experiments.md 5.33). From the repository root:
#   bash experiments/gpu19.sh
# Wave 18's 20,000-step runs (138M shape, 655M tokens, EMA), against g17_mem138 (8 groups x 64 keys, 4.029):
# 256 key and query neurons per group 3.947 (-0.082), 16 groups 4.171 (+0.142), lr 1.2e-3 4.162 (+0.133; the firing
# rate fell to ~2%). More keys per group help, more groups do not. All runs below are 20,000 steps and compare with
# g18_k256 (3.947), or with g17_kdmem138 (3.892) for distillation:
#  - kdk256   distillation + 256 keys: does the gain from more keys add to the gain from distillation? (the recipe for
#             the long run)
#  - h4k512   4 groups x 512 keys: the same number of key and query neurons, synapses and state as k256, but half as
#             many groups. Keys per group or the number of groups?
#  - k512     8 groups x 512 keys: twice k256's key and query neurons and state. Does capacity keep helping?
#  - tau1     k256 with the groups' decay time constants initialised from 1 token (not 8) to 1024: in wave 17 the
#             shortest one or two groups of almost every layer shrank to 0.5-2 tokens during training.
# h100-47 is a MIG 3g.47gb slice of an H100 NVL (about half a card); a100-40 on xgph[10-15] is half of a shared 80 GB
# card. --accum n splits each batch into n micro-batches (same result, less memory).
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
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g19_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g19_$name "$@"
}
a=$(sub kdk256 a100-80 $DATA $SPREAD $SHORT $N138 $SNN $KD --mem_heads 8 --mem_keys 256 --accum 2);   echo "$a g19_kdk256 (a100-80)"
b=$(sub h4k512 h100-47 $DATA $SPREAD $SHORT $N138 $SNN --mem_heads 4 --mem_keys 512 --accum 2);       echo "$b g19_h4k512 (h100-47)"
c=$(sub k512   h100-47 $DATA $SPREAD $SHORT $N138 $SNN --mem_heads 8 --mem_keys 512 --accum 4);       echo "$c g19_k512 (h100-47)"
d=$(sub tau1   a100-40 $DATA $SPREAD $SHORT $N138 $SNN --mem_heads 8 --mem_keys 256 --mem_tau_min 1 --accum 8); echo "$d g19_tau1 (a100-40)"
echo "submitted; squeue -u \$USER to follow"

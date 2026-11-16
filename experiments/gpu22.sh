#!/bin/bash
# GPU wave 22: plastic synapses written with (1 - lambda) v (--mem_write_avg, docs/experiments.md 5.34) at 256 key
# neurons per group, the capacity that worked best in waves 18-20 (5.35). With wave 21's kdwa138 (64 keys + KD) this
# makes a 2 x 2 at 256 keys (distillation or not, x the write) to decide the write for the long run. From the
# repository root:  bash experiments/gpu22.sh
#  - kdk256wa  138M, KD, 8 groups x 256 keys, --mem_write_avg   against g19_kdk256 (3.795)
#  - k256wa    138M,     8 groups x 256 keys, --mem_write_avg   against g18_k256 (3.947)
# 20,000 steps (655M tokens) each, the same seed as the runs they are compared with (the same batches).
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
MEM="--mem_heads 8 --mem_keys 256 --mem_write_avg true"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g22_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g22_$name "$@"
}
a=$(sub kdk256wa a100-80 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM --accum 2); echo "$a g22_kdk256wa (a100-80)"
b=$(sub k256wa   h100-47 $DATA $SPREAD $SHORT $N138 $SNN $MEM --accum 2);     echo "$b g22_k256wa (h100-47)"
echo "submitted; squeue -u \$USER to follow"

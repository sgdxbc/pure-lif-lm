#!/bin/bash
# GPU wave 20: fewer layers with plastic synapses, each with more key neurons (docs/experiments.md 5.33). From the
# repository root:  bash experiments/gpu20.sh
# Wave 18 (138M shape, 20,000 steps, EMA), against g17_mem138 (8 groups x 64 keys before every population, 4.029):
# 256 keys per group 3.947 (-0.082); plastic synapses before every second population only (--mem_every 2) 3.975
# (-0.054, at 81.0M instead of 90.7M dense MACs per token and 719 KB instead of 1.33 MB state). Both together?
#  - e2k256   --mem_every 2 --mem_keys 256: the plastic synapses before populations 2, 4, ..., 12
#  - e4k256   --mem_every 4 --mem_keys 256: before populations 4, 8, 12 only (deeper layers used them most, 5.32)
# Both on h100-47 (a MIG slice, about half an H100), so they start when wave 19's h4k512 and k512 end.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
[ -s data/fineweb/train.bin ] || { echo "data/fineweb/train.bin missing"; exit 1; }
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g20_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g20_$name "$@"
}
a=$(sub e2k256 h100-47 $DATA $SPREAD $SHORT $N138 $SNN --mem_heads 8 --mem_keys 256 --mem_every 2 --accum 2); echo "$a g20_e2k256 (h100-47)"
b=$(sub e4k256 h100-47 $DATA $SPREAD $SHORT $N138 $SNN --mem_heads 8 --mem_keys 256 --mem_every 4 --accum 2); echo "$b g20_e4k256 (h100-47)"
echo "submitted; squeue -u \$USER to follow"

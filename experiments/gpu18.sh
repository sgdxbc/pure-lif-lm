#!/bin/bash
# GPU wave 18: plastic synapses at full length, and how to get more out of them (docs/experiments.md 5.32). From the
# repository root:  bash experiments/gpu18.sh
# Wave 17 (138M shape, 655M tokens, EMA): base 4.195, plastic synapses (--mem_heads 8) 4.029 (-0.166), distillation
# 4.073 (-0.122), both 3.892 (-0.303): already below g13_snn138 after 2.5B tokens (3.906). The memory's gain grows
# during training (-0.09 at step 10000, -0.17 at 20000). The 163M Transformer (Muon) reached 3.446 on the same tokens.
#  - kdmem212, kdmem138   the full 2.5B tokens (76,300 steps) with distillation + plastic synapses, at the 212M and the
#                         138M shapes: the new recipe against g13_snn212 / g13_snn138 (3.814 / 3.906), the Transformers
#                         (g13_tf163 3.310, g13_tf77 3.535) and GPT-2 124M (3.277); and a size step for the scaling plot
#  - k256                 mem138 with 256 key and query neurons per group instead of 64 (a bigger, sparser memory: on
#                         the CPU recall toy the memory's capacity, not learning, was the limit)
#  - h16                  mem138 with 16 groups instead of 8
#  - lr12                 mem138 at lr 1.2e-3 (wave 17: 3e-4 was 0.15 worse than 6e-4; maybe higher is better)
#  - e2                   mem138 with plastic synapses in front of every second population only (half the cost)
#  - kd355mem             kdmem138 but distilled from GPT-2 355M (hf:openai-community/gpt2-medium, 3.000 on this
#                         validation set) instead of g13_tf163 (3.310): does a stronger teacher help?
# The 20,000-step runs compare with g17_mem138 (4.029) or g17_kdmem138 (3.892). --accum n splits the batch into n
# micro-batches (same result, less memory): runs with plastic synapses need ~44 GB at batch 32 (one did not fit the
# 46 GB h100-47), and the a100-40 on xgph[10-15] can be half of a shared 80 GB card. Every run chains 3-hour parts
# (RELAY) and ends with the zero-shot benchmarks and the hardware cost (HW=1).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
FULL="--batch 32 --seq 1024 --steps 76300 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
KD355="--teacher hf:openai-community/gpt2-medium --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g18_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g18_$name "$@"
}
a=$(sub kdmem212 h200-141 $DATA $SPREAD $FULL $N212 $SNN $KD $MEM --lr 6e-4);                    echo "$a g18_kdmem212 (h200-141)"
b=$(sub kdmem138 h100-96  $DATA $SPREAD $FULL $N138 $SNN $KD $MEM --lr 6e-4);                    echo "$b g18_kdmem138 (h100-96)"
c=$(sub k256     a100-80  $DATA $SPREAD $SHORT $N138 $SNN $MEM --mem_keys 256 --lr 6e-4);         echo "$c g18_k256 (a100-80)"
d=$(sub h16      h100-47  $DATA $SPREAD $SHORT $N138 $SNN --mem_heads 16 --accum 2 --lr 6e-4);   echo "$d g18_h16 (h100-47)"
e=$(sub lr12     h100-47  $DATA $SPREAD $SHORT $N138 $SNN $MEM --accum 2 --lr 1.2e-3);           echo "$e g18_lr12 (h100-47)"
f=$(sub e2       a100-40  $DATA $SPREAD $SHORT $N138 $SNN $MEM --mem_every 2 --accum 4 --lr 6e-4); echo "$f g18_e2 (a100-40)"
g=$(sub kd355mem a100-40  $DATA $SPREAD $SHORT $N138 $SNN $KD355 $MEM --accum 8 --lr 6e-4);       echo "$g g18_kd355mem (a100-40)"
echo "submitted; squeue -u \$USER to follow"

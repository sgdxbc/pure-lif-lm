#!/bin/bash
# GPU wave 23: the last checks before the long run (docs/experiments.md 5.34, 5.35). From the repository root:
#   bash experiments/gpu23.sh
# Waves 19-22 (138M, 20,000 steps): 8 groups x 256 key neurons is the best capacity (512 no better), the shortest
# time constant from 1 token helps a little (-0.018), and writing (1 - lambda) v (--mem_write_avg) helps only where
# the original write trains unstably (64 keys: 3.892 -> 3.816) and not at 256 keys (3.947 -> 3.945, with KD 3.795 ->
# 3.799, LAMBADA lower in both). The long run is meant to be 212M with 256 keys, so:
#  - kd212k256    212M, KD, 8 x 256 keys, --mem_tau_min 1, the original write: does 212M train stably at 256 keys?
#                 (spike lines from train.py --spike_log; 212M with 64 keys had gradient spikes of 88-117)
#  - kd212k256wa  the same with --mem_write_avg: does the averaged write help at 212M, where training is less stable?
#  - kdk256t1     138M, KD, 8 x 256 keys, --mem_tau_min 1: does the shorter time constant add to KD? (vs g19_kdk256, 3.795)
#  - kde4k256t1   the same with the plastic synapses only before layers 4, 8 and 12 (--mem_every 4): what the cheaper
#                 hardware variant (a quarter of the state) costs with KD (vs kdk256t1)
# All 20,000 steps (655M tokens), seed 0, so the pairs see the same batches. kd212k256 queues for the H200 after
# g21_wa212.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g23_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g23_$name "$@"
}
a=$(sub kd212k256   h200-141 $DATA $SPREAD $SHORT $N212 $SNN $KD $MEM --accum 2);                       echo "$a g23_kd212k256 (h200-141)"
b=$(sub kd212k256wa h100-96  $DATA $SPREAD $SHORT $N212 $SNN $KD $MEM --mem_write_avg true --accum 2); echo "$b g23_kd212k256wa (h100-96)"
c=$(sub kdk256t1    a100-80  $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM --accum 2);                       echo "$c g23_kdk256t1 (a100-80)"
d=$(sub kde4k256t1  h100-47  $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM --mem_every 4 --accum 4);         echo "$d g23_kde4k256t1 (h100-47)"
echo "submitted; squeue -u \$USER to follow"

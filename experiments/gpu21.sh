#!/bin/bash
# GPU wave 21: plastic synapses written with (1 - lambda) v (--mem_write_avg), against the gradient spikes of
# docs/experiments.md 5.34. From the repository root:  bash experiments/gpu21.sh
# grad_spikes.py on g18_kdmem212 found the rows of S in the long-time-constant groups at 35,000-56,000 (the raw read-out
# at up to 100,000; the short groups' around 30-50) and single sequences whose gradient is 10-25x the median, almost
# all of it in the key/query neurons' input synapses (mems.*.kq_in) and the read-out gain. With --mem_write_avg, S is a
# running average and its rows stay below the size of a value; the trained network is the same kind (the factor folds
# into W_v), so the hardware cost is unchanged.
#  - wa212    g18_kdmem212 (3.600) with --mem_write_avg: the same seed, so the same batches; the full 2.5B tokens
#  - kdwa138  g17_kdmem138 (138M, 20,000 steps, 3.892) with --mem_write_avg
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
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g21_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g21_$name "$@"
}
a=$(sub wa212   h200-141 $DATA $SPREAD $FULL $N212 $SNN $KD $MEM);  echo "$a g21_wa212 (h200-141)"
b=$(sub kdwa138 h100-96  $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM); echo "$b g21_kdwa138 (h100-96)"
echo "submitted; squeue -u \$USER to follow"

#!/bin/bash
# GPU wave 28: training with 4-bit weights (train.py --wbits; docs/experiments.md 5.38). From the repository root:
#   bash experiments/gpu28.sh
# The network meant for the FPGA comparison (138M, 64 keys, coded input and two-level readout with short lists of 2,048
# words; g27_io2k) fits the on-chip memory only with 4-bit weights. Rounding a trained spiking network's weights to 4
# bits after training cost +0.78 on FineWeb (g13_snn212, 5.28), against +0.02 on TinyStories: on general text the
# network has to be trained for it. On the CPU (TinyStories, 5.2M, 1,500 steps; 5.38) rounding after training cost
# +0.034, training with 4-bit weights from the start +0.033, from 1/3, 2/3 or 5/6 of the way +0.018 / +0.017 / +0.018,
# and from 2/3 with one scale per 64 inputs +0.014. Here, 20,000 steps, seed 0, the same batches as g27_io2k:
#  - io2k_q4l     at full precision, then with 4-bit weights (one scale per 128 inputs) for the last 5,000 steps (25%)
#  - io2k_q4lg64  the same with one scale per 64 inputs (4.25 instead of 4.125 bits per weight with 16-bit scales)
# and the validation loss of the full-precision runs of waves 21-27 with their weights rounded after training to 8 and
# 4 bits (slurm/wbits.sbatch), as the reference: those finished when this script runs (wave 27's, once finished, with
#   sbatch -J g28_wbits27 --gres=gpu:h100-47:1 slurm/wbits.sbatch runs/g27_cls2k runs/g27_cls8k runs/g27_io2k ).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
H2=runs/classes/tf163_head225_s2048.npy; E2=runs/classes/tf163_embed220_s2048.npy
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $H2 $E2; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
IO="--out_classes 225 --out_short 2048 --class_file $H2 --in_groups 2 --in_short 2048 --in_class_file $E2"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g28_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g28_$name "$@"
}
a=$(sub io2k_q4l    a100-40 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IO --wbits 4 --wbits_from 15000 --accum 4)
echo "$a g28_io2k_q4l (a100-40)"
b=$(sub io2k_q4lg64 a100-40 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IO --wbits 4 --wgroup 64 --wbits_from 15000 --accum 4)
echo "$b g28_io2k_q4lg64 (a100-40)"
runs=""
for r in g21_kdwa138 g21_wa212 g25_cls g25_inc g25_io g27_cls2k g27_cls8k g27_io2k; do
  [ -e runs/$r/done ] && runs="$runs runs/$r"
done
c=$(sbatch --parsable -J g28_wbits --gres=gpu:h100-47:1 slurm/wbits.sbatch $runs)
echo "$c rounded-weight evaluations of:$runs (h100-47)"
echo "submitted; squeue -u \$USER to follow"

#!/bin/bash
# GPU wave 28: training with 4-bit weights (train.py --wbits; docs/experiments.md 5.38). From the repository root:
#   bash experiments/gpu28.sh          the rounded-weight evaluations of the finished runs of waves 21-27
#   bash experiments/gpu28.sh branch   the two 4-bit runs, branches of runs/g27_io2k/ckpt_step15000.pt
# The network meant for the FPGA comparison (138M, 64 keys, coded input and two-level readout with short lists of 2,048
# words; g27_io2k) fits the on-chip memory only with 4-bit weights. Rounding a trained spiking network's weights to 4
# bits after training cost +0.78 on FineWeb (g13_snn212, 5.28), against +0.02 on TinyStories: on general text the
# network has to be trained for it. On the CPU (TinyStories, 5.2M, 1,500 steps; 5.38) rounding after training cost
# +0.034, training with 4-bit weights from the start +0.033, from 1/3, 2/3 or 5/6 of the way +0.018 / +0.017 / +0.018,
# and from 2/3 with one scale per 64 inputs +0.014. Here, g27_io2k's last 5,000 of 20,000 steps (25%) again, from its
# step-15,000 checkpoint (the same weights, optimizer state and averaged weights; the batches after it differ, since
# the data order restarts from the step at the start of every part), with 4-bit weights:
#  - io2k_q4l     all weight matrices at 4 bits (one scale per 128 inputs)
#  - io2k_q4lm8   the readout and the input rows kept at 8 bits (--wbits8): rounding only the readout of a trained
#                 network costs +0.07 to +0.14 of the +0.13 to +0.17 of rounding everything (wbits_sens.py, 5.38),
#                 and the word readout neurons sit in HBM, so 8 bits there costs no on-chip memory
# so the difference to g27_io2k itself is what 4 bits cost, paired. History: these first ran from scratch (20,000 steps
# each, the first 15,000 at full precision, i.e. g27_io2k again), with io2k_q4lg64 (one scale per 64 inputs) planned in
# place of io2k_q4lm8; with five runs on the two a100-40 cards of the per-user limit, wave 27 waited behind them, so on
# Nov 17 at ~20:10 io2k_q4l (at step 3,500) and io2k_q4lg64 (6,500) were cancelled, io2k_q4lm8 before it started, and
# g27_io2k was resubmitted with --keep_ckpt 15000 for these branches (5,000 steps each instead of 20,000).
# The rounded-weight evaluations (slurm/wbits.sbatch): the full-precision runs of waves 21-27 with their weights rounded
# after training to 8 and 4 bits, as the reference; wave 27's, once finished, with
#   sbatch -J g28_wbits27 --gres=gpu:h100-47:1 slurm/wbits.sbatch runs/g27_cls2k runs/g27_cls8k runs/g27_io2k
# GPU=... picks the card type of the branches (default h100-47); ACCUM=n splits each batch in n (default 4).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
H2=runs/classes/tf163_head225_s2048.npy; E2=runs/classes/tf163_embed220_s2048.npy
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $H2 $E2; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
if [ "$1" != "branch" ]; then
  runs=""
  for r in g21_kdwa138 g21_wa212 g25_cls g25_inc g25_io g27_cls2k g27_cls8k g27_io2k; do
    [ -e runs/$r/done ] && runs="$runs runs/$r"
  done
  c=$(sbatch --parsable -J g28_wbits --gres=gpu:h100-47:1 slurm/wbits.sbatch $runs)
  echo "$c rounded-weight evaluations of:$runs (h100-47)"
  exit 0
fi
gpu=${GPU:-h100-47}; acc=${ACCUM:-4}
ck=runs/g27_io2k/ckpt_step15000.pt
[ -s $ck ] || { echo "$ck missing (g27_io2k needs --keep_ckpt 15000 and to have passed step 15,000)"; exit 1; }
# exactly g27_io2k's flags (experiments/gpu27.sh), then the branch's
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
IO="--out_classes 225 --out_short 2048 --class_file $H2 --in_groups 2 --in_short 2048 --in_class_file $E2"
BR="--wbits 4 --wbits_from 15000 --init_ckpt $ck --accum $acc"
sub() {  # sub NAME train.py-flags...   (prints the job id of the first part)
  local name=$1; shift
  sbatch --parsable -p gpu -J g28_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g28_$name "$@"
}
for r in g28_io2k_q4l g28_io2k_q4lm8; do
  [ -e runs/$r/ckpt.pt ] && { echo "runs/$r/ckpt.pt exists (the cancelled run from scratch): move runs/$r away first"; exit 1; }
done
a=$(sub io2k_q4l   $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IO $BR)
echo "$a g28_io2k_q4l ($gpu, accum $acc)"
b=$(sub io2k_q4lm8 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IO $BR --wbits8 '^head\.,^embed\.')
echo "$b g28_io2k_q4lm8 ($gpu, accum $acc)"

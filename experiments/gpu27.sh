#!/bin/bash
# GPU wave 27: the two-level readout and the coded input with a short list of frequent words (an adaptive softmax;
# model.py out_short, in_short; docs/experiments.md 5.36). From the repository root:  bash experiments/gpu27.sh
# Wave 25 (138M, KD, 64 keys, --mem_write_avg; g21_kdwa138 3.816): the two-level readout cost +0.071 (225 classes of
# 224 words), the coded input +0.024. In the two-level readout every word, also "the", "," and ".", goes through
# "class, then word in class", and a class's probability is a linear readout of the bus instead of the sum of its
# words' probabilities. With out_short S, the S most frequent words have their own readout neurons beside the class
# neurons and only the other words are in classes; in_short does the same for the coded input (the S words have an
# input neuron of their own). On the CPU (TinyStories, 4,096 words, 64 classes from a trained network, a short list
# of 256 words = 77% of the tokens; baseline 2.714 / 2.699 with two seeds):
#   two-level readout 2.750 -> 2.717 with the short list;  coded input 2.745 -> 2.717;  both 2.763 -> 2.720.
# Here, 20,000 steps, seed 0, the same batches as g21_kdwa138 and wave 25:
#  - cls2k  two-level readout, 225 classes, short list of 2,048 words: (2,048 + 225 + 215) readout neurons per token
#  - cls8k  the same with 8,192 words: (8,192 + 225 + 188)
#  - io2k   coded input and two-level readout, both with a short list of 2,048 words (the network that fits the FPGA's
#           on-chip memory at 4 bits: about 121M weights)
# The class files are made from the teacher (runs/g13_tf163) by experiments/word_classes.py; the short lists are the
# most frequent words in the first 200M training tokens.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs runs/classes
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
WC="python3 experiments/word_classes.py --vocab 50304 --method kmeans --ckpt runs/g13_tf163/ckpt.pt --data data/fineweb/train.bin"
H2=runs/classes/tf163_head225_s2048.npy; H8=runs/classes/tf163_head225_s8192.npy; E2=runs/classes/tf163_embed220_s2048.npy
[ -s $H2 ] || $WC --rows head --classes 225 --short 2048 --out $H2
[ -s $H8 ] || $WC --rows head --classes 225 --short 8192 --out $H8
[ -s $E2 ] || $WC --rows embed --classes 220 --short 2048 --out $E2   # 220 x 220 >= 50,304 - 2,048
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g27_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g27_$name "$@"
}
a=$(sub cls2k h100-47 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM --out_classes 225 --out_short 2048 --class_file $H2 --accum 4)
echo "$a g27_cls2k (h100-47)"
b=$(sub cls8k a100-40 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM --out_classes 225 --out_short 8192 --class_file $H8 --accum 4)
echo "$b g27_cls8k (a100-40)"
c=$(sub io2k  a100-40 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM --out_classes 225 --out_short 2048 --class_file $H2 \
    --in_groups 2 --in_short 2048 --in_class_file $E2 --accum 4)
echo "$c g27_io2k (a100-40)"
echo "submitted; squeue -u \$USER to follow"

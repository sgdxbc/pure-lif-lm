#!/bin/bash
# GPU wave 25: smaller word input and readout through the spike coding (docs/experiments.md 5.36). From the
# repository root:  bash experiments/gpu25.sh
# The two vocabulary-sized parts are half of the parameters (2 x 50,304 x d_model) and the readout is 30-54% of the
# dense multiply-adds per token (the part where spikes save nothing). On the CPU (TinyStories, 1,500 steps, baseline
# 2.714 / 2.699 with two seeds):
#   two-level readout (model.py out_classes; the class neurons, then the word neurons of the class drawn), classes
#   from k-means on a trained readout   2.743;   coded input (model.py in_groups 2: a word fires its class's neuron
#   and the neuron of its position in the class), classes from k-means on trained input synapses   2.745;   both 2.763.
# Here on FineWeb-Edu at 138M with KD, 64 keys and --mem_write_avg (g21_kdwa138, 3.816; the same seed and batches):
#  - cls  the two-level readout, 225 classes of up to 224 words, from the teacher's readout rows
#  - inc  the coded input, 2 groups of 225 input neurons, classes from the teacher's input rows
#  - io   both
# The class files are made here from the teacher (runs/g13_tf163) by experiments/word_classes.py (about a minute).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs runs/classes
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
HC=runs/classes/tf163_head225.npy; EC=runs/classes/tf163_embed225.npy
[ -s $HC ] || python3 experiments/word_classes.py --vocab 50304 --classes 225 --method kmeans --rows head --ckpt runs/g13_tf163/ckpt.pt --out $HC
[ -s $EC ] || python3 experiments/word_classes.py --vocab 50304 --classes 225 --method kmeans --rows embed --ckpt runs/g13_tf163/ckpt.pt --out $EC
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
OUT="--out_classes 225 --class_file $HC"
IN="--in_groups 2 --in_class_file $EC"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g25_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g25_$name "$@"
}
a=$(sub cls h200-141 $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $OUT --accum 2);     echo "$a g25_cls (h200-141)"
b=$(sub inc h100-96  $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $IN --accum 2);      echo "$b g25_inc (h100-96)"
c=$(sub io  h100-47  $DATA $SPREAD $SHORT $N138 $SNN $KD $MEM $OUT $IN --accum 4); echo "$c g25_io (h100-47)"
echo "submitted; squeue -u \$USER to follow"

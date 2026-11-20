#!/bin/bash
# GPU wave 46, first part (docs/experiments.md 5.58): confirm distillation on sparse relays with a second seed, and a
# Transformer trained on the same short recipe as the 77M spiking-relay runs (the fair side of the FPGA comparison).
# Agreed with the GPT advisor (docs/discussions.md, 2026-11-20). The two-level-readout runs follow once model_tf.py has
# that readout.
#   t25s1    g45_t25 (relays at 0.25, no teacher) with seed 1
#   kd25s1   g45_kd25 (relays at 0.25, distilled from g13_tf163) with seed 1; keeps its step-15000 checkpoint for the
#            constant-versus-tapered distillation continuations
#   tf31s    g33_tf31's shape (256 wide, 6 layers, softmax attention, GELU MLP, distilled from g13_tf163 at weight 0.9)
#            on the 77M runs' short recipe (20,000 steps, cosine) instead of its 75,000-step WSD recipe
# From the repository root:
#   bash experiments/gpu46.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
S77="--arch tf --attn linear --mlp spike --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"   # g40_laspk's
Q25="--relay quant --relay_levels 4 --relay_target 0.25"
TF31="--arch tf --d_model 256 --n_layers 6 --n_heads 4 --lr 1e-3 --ema 0.999"                         # g33_tf31's shape
NAMES="t25s1 kd25s1 tf31s"
only=$1
case " $NAMES " in *" ${only:-t25s1} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
for s in "kd25s1 h100-96" "t25s1 a100-40" "tf31s h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h200-141) acc=1 ;; h100-96|a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    t25s1) X="$S77 $Q25 --seed 1" ;;
    kd25s1) X="$S77 $Q25 $KD --seed 1 --keep_ckpt 15000" ;;
    tf31s) X="$TF31 $KD" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g46_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g46_$name $DATA $SHORT $X --accum $acc)
  echo "$j g46_$name ($gpu, accum $acc) $X"
done

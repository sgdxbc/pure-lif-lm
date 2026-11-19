#!/bin/bash
# GPU wave 40: small-scale ranking for the plan of docs/experiments.md 5.51 (step 3): let the LIF neuron's ultra-light
# recurrence + nonlinearity do the work, and judge every variant by quality and by its cost on the same FPGA
# (experiments/fpga_compare.py). From the repository root:
#   bash experiments/gpu40.sh [NAME]     every run below, or only NAME; GPU=... / ACCUM=n to override
# One small shape for all (width 512, 8 layers, 8 heads: g33_tf77's shape, 77M parameters of which 51.5M are the
# word tables), wave 35's short schedule (20,000 steps x 32 x 1,024 = 655M tokens, cosine), learning rate 1e-3 (as
# the small Transformers of 5.43), no teacher (cheaper, and the main arm of the later scaling grid), the same seed and
# so the same batches:
#   la       gated linear attention, GELU MLP: the reference
#   laspk    the same with 0/1 MLP units (model_tf.py --mlp spike; at GPT-2 small's size this cost 0.067, 5.49)
#   tf       softmax attention at the same shape (the same-size Transformer)
# More variants (LIF relays in front of every matrix, ternary weights) are added when their code is ready.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
[ -s data/fineweb/train.bin ] || { echo "data/fineweb/train.bin missing"; exit 1; }
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
S77="--arch tf --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"
only=$1
case "$only" in ""|la|laspk|tf) ;; *) echo "NAME is la, laspk or tf"; exit 1 ;; esac
for s in "la a100-40" "laspk a100-40" "tf h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    la) X="--attn linear" ;;
    laspk) X="--attn linear --mlp spike" ;;
    tf) X="" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g40_$name --gres=gpu:$gpu:1 \
      --export=ALL,RELAY=$gpu,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g40_$name $DATA $SHORT $S77 $X --accum $acc)
  echo "$j g40_$name ($gpu, accum $acc) $X"
done

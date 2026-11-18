#!/bin/bash
# GPU wave 33: the Transformer of the same quality as the hardware network, for the FPGA comparison (stage goal 3;
# docs/experiments.md 5.43). From the repository root:
#   bash experiments/gpu33.sh [NAME]   the three small Transformers (training), or only NAME (tf31, tf50 or tf77),
#                                      e.g. GPU=a100-80 bash experiments/gpu33.sh tf50 to put one on another card type
#   bash experiments/gpu33.sh bench    their shapes measured on the GPU with vLLM (bench_gpu.py --model shape:D,L,H)
#   bash experiments/gpu33.sh bench_big   the small shapes again at 256, 1,024 and 4,096 streams
#   bash experiments/gpu33.sh bench_lean  the small shapes decoded by a minimal program (bench_gpu_lean.py)
# The hardware network g29_hw (121M, 4-bit weights, fits on chip) reaches 3.753 after 2.46B tokens, and fpga_est.py
# estimates 131k tokens/s and 1.45 mJ per token for it on an AMD V80. The only Transformers measured on the GPU so far
# are GPT-2 124M and larger (3.277 on this validation set): far better, so not a comparison at the same quality.
# A Transformer as good as 3.753 is much smaller; with a few streams the GPU's ~1 ms floor per token barely depends on
# the model's size, but with hundreds of streams a small one is cheaper than GPT-2 124M. Here three small Transformers
# trained exactly like the hardware network (same data, 75,000 steps of 32 x 1,024 tokens = 2.46B, wsd with the last
# 20% decayed, distilled from g13_tf163 with weight 0.9 as g29_hw, so the Transformer gets the same help), to find by
# interpolation the size that reaches 3.753:
#   tf31   d 256,  6 blocks, 4 heads   30.7M parameters (25.8M of them the word embedding and readout)
#   tf50   d 384,  6 blocks, 6 heads   49.6M
#   tf77   d 512,  8 blocks, 8 heads   77.2M (the shape of g13_tf77: 3.535 without distillation, cosine, 2.50B tokens)
# Learning rate 1e-3 (Pythia-70M's; small Transformers take a higher one than GPT-2 124M's 6e-4), moving average of the
# weights as for the spiking networks. GPU=... picks the card (default h100-47), ACCUM=n the split of each batch
# (default 1; the student's and the teacher's logits over the vocabulary are ~6.6 GB each at 32 x 1,024 tokens, so tf77
# ran out of memory on a 47 GB card at 1 and needs ACCUM=2 there).
# bench: the three shapes (and GPT-2 124M's, as a check against the real GPT-2 measured in 5.25) with random weights on
# an H100 NVL, as the earlier measurements; only the shape matters for time and energy.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
if [ "$1" = "bench" ]; then
  j=$(sbatch --parsable -J bench_gpu33 --gres=gpu:h100-96:1 slurm/bench_gpu.sbatch \
      shape:256,6,4 shape:384,6,6 shape:512,8,8 shape:768,12,12)
  echo "$j bench_gpu33 (h100-96): shapes 256/6, 384/6, 512/8 and 768/12 (GPT-2 124M's)"
  exit 0
fi
# bench_big: the three small shapes at 256, 1,024 and 4,096 streams (5.43: at 256 streams these small models leave the
# H100 nearly idle, 114-125 W, and every shape takes the same 3.7 ms per step; the comparison takes the Transformer's
# best numbers, so more streams). Writes runs/bench_gpu/<gpu>_gpt2shape_..._big.json.
if [ "$1" = "bench_big" ]; then
  j=$(BENCH_ARGS="--batches 256,1024,4096 --suffix _big" sbatch --parsable -J bench_gpu33b --gres=gpu:h100-96:1 \
      slurm/bench_gpu.sbatch shape:256,6,4 shape:384,6,6 shape:512,8,8)
  echo "$j bench_gpu33b (h100-96): shapes 256/6, 384/6, 512/8 at 256, 1024 and 4096 streams"
  exit 0
fi
# bench_lean: the same shapes decoded by a minimal program (bench_gpu_lean.py: static cache, CUDA graphs, nothing else),
# 1 to 4,096 streams. vLLM's time per step is the same for every shape (a fixed cost per step and per stream outside
# the GPU's arithmetic), so this bounds what the GPU itself can do for these small models.
if [ "$1" = "bench_lean" ]; then
  j=$(sbatch --parsable -J bench_lean33 --gres=gpu:h100-96:1 slurm/bench_lean.sbatch 256,6,4 384,6,6 512,8,8)
  echo "$j bench_lean33 (h100-96): shapes 256/6, 384/6, 512/8, minimal decoding program"
  exit 0
fi
[ -s runs/g13_tf163/ckpt.pt ] || { echo "runs/g13_tf163/ckpt.pt missing"; exit 1; }
gpu=${GPU:-h100-47}; acc=${ACCUM:-1}; only=$1
case "$only" in ""|tf31|tf50|tf77) ;; *) echo "NAME is tf31, tf50 or tf77 (or bench)"; exit 1 ;; esac
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
LEN="--batch 32 --seq 1024 --steps 75000 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps 15000 --min_lr_frac 0"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
TF="--arch tf --lr 1e-3 --ema 0.999"
for s in "tf31 256 6 4" "tf50 384 6 6" "tf77 512 8 8"; do
  set -- $s; name=g33_$1
  [ -n "$only" ] && [ "$only" != "$1" ] && continue
  j=$(sbatch --parsable -p gpu -J $name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu slurm/train.sbatch runs/$name \
      $DATA $LEN $WSD $KD $TF --d_model $2 --n_layers $3 --n_heads $4 --accum $acc)
  echo "$j $name ($gpu, accum $acc): d $2, $3 blocks, $4 heads"
done

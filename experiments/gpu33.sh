#!/bin/bash
# GPU wave 33: the Transformer of the same quality as the hardware network, for the FPGA comparison (stage goal 3;
# docs/experiments.md 5.43). From the repository root:
#   bash experiments/gpu33.sh          the three small Transformers (training)
#   bash experiments/gpu33.sh bench    their shapes measured on the GPU with vLLM (bench_gpu.py --model shape:D,L,H)
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
# weights as for the spiking networks. GPU=... picks the card (default h100-47; each needs well under 40 GB).
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
[ -s runs/g13_tf163/ckpt.pt ] || { echo "runs/g13_tf163/ckpt.pt missing"; exit 1; }
gpu=${GPU:-h100-47}
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
LEN="--batch 32 --seq 1024 --steps 75000 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps 15000 --min_lr_frac 0"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
TF="--arch tf --lr 1e-3 --ema 0.999"
for s in "tf31 256 6 4" "tf50 384 6 6" "tf77 512 8 8"; do
  set -- $s; name=g33_$1
  j=$(sbatch --parsable -p gpu -J $name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu slurm/train.sbatch runs/$name \
      $DATA $LEN $WSD $KD $TF --d_model $2 --n_layers $3 --n_heads $4)
  echo "$j $name ($gpu): d $2, $3 blocks, $4 heads"
done

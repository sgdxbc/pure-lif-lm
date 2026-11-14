#!/bin/bash
# One-time setup and self-check on a machine with an NVIDIA GPU. Run from the repository root:
#   bash gpu_setup.sh
# Assumes PyTorch with CUDA is already installed (pick the build for your driver at https://pytorch.org).
# Steps: check the GPU -> install the small extra packages -> download and tokenize TinyStories (skipped if the
# data is already there) -> verify the token files are identical to the ones used for all results so far ->
# check the Triton LIF kernels against the reference -> time a few training steps at several model sizes.
set -e
cd "$(dirname "$0")"

echo "== 1. GPU and PyTorch"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch does not see a GPU (CPU-only build, or driver problem)"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| bf16:", torch.cuda.is_bf16_supported())
import triton; print("triton", triton.__version__)
EOF

echo "== 2. packages"
python3 -m pip install -q numpy tokenizers ninja matplotlib

echo "== 3. data"
mkdir -p data
for f in train valid; do
  if [ ! -f data/$f.bin ] && [ ! -f data/TinyStoriesV2-GPT4-$f.txt ]; then
    curl -L -o data/TinyStoriesV2-GPT4-$f.txt \
      https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-$f.txt
  fi
done
if [ ! -f data/train.bin ] || [ ! -f data/valid.bin ]; then python3 prepare_data.py; fi
(cd data && md5sum -c checksums.md5)

echo "== 4. Triton LIF kernels vs reference"
python3 test_lif_gpu.py

echo "== 5. training speed (20 steps each, batch 32 x 256)"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
bench() { name=$1; shift; rm -rf runs/gpu_bench_$name
  python3 train.py --device cuda --out runs/gpu_bench_$name $SPREAD --steps 20 --batch 32 --seq 256 --warmup 5 \
    --eval_every 1000 --log_every 10 "$@" 2>&1 | grep -E "^params|step +20 " | sed "s/^/  $name: /"
  rm -rf runs/gpu_bench_$name; }
bench 12M        --leak_rank 0
bench 12M_leak   --leak_rank 64
bench 25M_leak   --leak_rank 64 --d_model 512 --d_hidden 2048 --n_layers 10
bench 117M_leak  --leak_rank 128 --d_model 1024 --d_hidden 4096 --n_layers 12
echo "== done"

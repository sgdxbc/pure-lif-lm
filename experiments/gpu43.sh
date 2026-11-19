#!/bin/bash
# GPU wave 43 (docs/experiments.md 5.54): relays held to 0.5 weighted spikes, ternary weights, and both, at GPT-2
# small's shape. At 77M (wave 41) the relays (g41_q4t) were the cheapest point on the same FPGA but cost 0.111 on
# g40_laspk; the ternary-weight network (g40_tern, ordinary MLP) was 0.050 behind g40_laspk with three times the
# additions. Which pays depends on the blocks' share of the work, which the full readout hides at 77M; and whether
# either cost grows or shrinks with the network is the first pair of points for the spiking network's own scaling.
# From the repository root:
#   bash experiments/gpu43.sh [NAME]     every run below, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
#   tern163   g38_laspk (gated linear attention + 0/1 MLP units, GPT-2 small's shape, distilled) with ternary
#             weights in the blocks' matrices (model_tf.py --wq ternary)
#   q4t163    g38_laspk with relays of up to 4 spikes held to 0.5 weighted spikes (as g41_q4t)
#   q4tt163   both
#   tern77    ternary weights at 77M: g40_laspk with --wq ternary (wave 40's shape, recipe and seed; g40_tern had
#             the ordinary MLP, so the ternary weights' own cost on the 0/1-unit network was not measured)
#   q4tt77    both at 77M: g41_q4t with ternary weights
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
[ -s data/fineweb/train.bin ] && [ -s runs/g13_tf163/ckpt.pt ] || { echo "data or teacher missing"; exit 1; }
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
LA163="--arch tf --attn linear --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4 --ema 0.999 --mlp spike $KD"
S77="--arch tf --attn linear --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999 --mlp spike"
RELAY="--relay quant --relay_levels 4 --relay_target 0.5"
only=$1
case "$only" in ""|tern163|q4t163|q4tt163|tern77|q4tt77) ;; *) echo "NAME is tern163, q4t163, q4tt163, tern77 or q4tt77"; exit 1 ;; esac
for s in "tern77 a100-40" "q4tt77 a100-40" "tern163 h100-47" "q4t163 h100-47" "q4tt163 h100-96"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $name in
    tern163) X="$LA163 --wq ternary" ;;
    q4t163) X="$LA163 $RELAY" ;;
    q4tt163) X="$LA163 $RELAY --wq ternary" ;;
    tern77) X="$S77 --wq ternary" ;;
    q4tt77) X="$S77 $RELAY --wq ternary" ;;
  esac
  case $name in   # 163M: g38_laspk used 24.7 GB at accum 4
    *163) case $gpu in h100-96|h200-141|a100-80) acc=2 ;; *) acc=4 ;; esac ;;
    *) case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac ;;
  esac
  acc=${ACCUM:-$acc}
  j=$(sbatch --parsable -p gpu -J g43_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g43_$name $DATA $SHORT $X --accum $acc)
  echo "$j g43_$name ($gpu, accum $acc) $X"
done

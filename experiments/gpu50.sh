#!/bin/bash
# GPU wave 50 (docs/experiments.md 5.62), agreed with the GPT advisor after wave 48 showed the dense gated-linear-
# attention control g48_gla50 ahead of every 77M / 126M spiking candidate at the same estimated FPGA cost. All
# distilled from g13_tf163 at 0.9 with the 77M runs' short recipe, seed 0:
#   la77kd    g40_la (dense GLA, 512 wide, 8 layers, GELU MLP) with distillation: is g43_q4t163 (relays at 0.5,
#             GPT-2 small's shape, 3.545 at 3.38 mJ) dominated by a cheaper distilled dense network (2.49 mJ)?
#   laspkkd   g40_laspk (the same with 0/1 MLP units, no relays) with distillation: the matched distilled baseline
#             for the relay penalty of g45_kd25 (same card type and accum as g45_kd25)
#   n320g     a narrower GLA, 320 wide, 8 layers, 5 heads (head size 64), MLP 6x wide, GELU, full readout (1.41 mJ)
#   n320s     the same with 0/1 MLP units (1.19 mJ at the 3-8% rates seen so far): spikes only in the MLP of a dense
#             GLA, judged against g48_gla50 (1.45 mJ) and against n320g
# From the repository root:
#   bash experiments/gpu50.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
LA77="--arch tf --attn linear --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"   # g40_la's shape
N320="--arch tf --attn linear --d_model 320 --n_layers 8 --n_heads 5 --mlp_mult 6 --lr 1e-3 --ema 0.999"
only=$1
NAMES="la77kd laspkkd n320g n320s"
case " $NAMES " in *" ${only:-la77kd} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
for s in "la77kd h100-96" "laspkkd h100-96" "n320g h100-47" "n320s h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h200-141) acc=1 ;; h100-96|a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    la77kd) X="$LA77 --mlp gelu" ;;
    laspkkd) X="$LA77 --mlp spike" ;;
    n320g) X="$N320 --mlp gelu" ;;
    n320s) X="$N320 --mlp spike" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g50_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g50_$name $DATA $SHORT $X $KD --seed 0 --accum $acc)
  echo "$j g50_$name ($gpu, accum $acc) $X"
done

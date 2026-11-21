#!/bin/bash
# GPU wave 49 (docs/experiments.md 5.61): does giving the true next word more weight late in training help a distilled
# spiking-relay network? Agreed with the GPT advisor (docs/discussions.md). Three continuations of g46_kd25s1 (relays at
# 0.25, distilled from g13_tf163 at weight 0.9, seed 1) from its step-15000 checkpoint to step 20000, with exactly its
# flags; all read the same batches (the data order restarts at seed + 15000), so they are paired with each other, not
# with g46_kd25s1's own last 5,000 steps:
#   kdc   distillation weight kept at 0.9 (the control)
#   kdt   distillation weight lowered linearly from 0.9 to 0 over the last 5,000 updates (--distill_taper_steps 5000)
#   kds   distillation kept at 0.9, the relays' spike-count gradient switched from straight-through to a smooth one
#         (--relay_count_grad smooth, width 0.5; added in wave 50, docs 5.62): a late-training screen, paired with kdc
# From the repository root:
#   bash experiments/gpu49.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override (all runs on the same
#                                        card type and accum; default a100-80 with accum 2, where all three and g51_kdf
#                                        ran; g46_kd25s1 itself ran on h100-96, also accum 2); NICE=n
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
ck=runs/g46_kd25s1/ckpt_step15000.pt
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $ck; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
# exactly g46_kd25s1's flags (experiments/gpu46.sh)
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
S77="--arch tf --attn linear --mlp spike --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"
Q25="--relay quant --relay_levels 4 --relay_target 0.25"
NAMES="kdc kdt kds"
only=$1
case " $NAMES " in *" ${only:-kdc} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
gpu=${GPU:-a100-80}
case $gpu in h200-141) acc=1 ;; h100-96|a100-80) acc=2 ;; *) acc=4 ;; esac
acc=${ACCUM:-$acc}
for name in $NAMES; do
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $name in
    kdc) X="" ;;
    kdt) X="--distill_taper_steps 5000 --distill_weight_final 0" ;;
    kds) X="--relay_count_grad smooth --relay_count_width 0.5" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g49_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g49_$name $DATA $SHORT $S77 $Q25 $KD --seed 1 --init_ckpt $ck $X --accum $acc)
  echo "$j g49_$name ($gpu, accum $acc) from $ck $X"
done

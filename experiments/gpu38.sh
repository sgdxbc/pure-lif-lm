#!/bin/bash
# GPU wave 38: where the spiking network's gap comes from, from both ends (docs/experiments.md 5.49). From the
# repository root:
#   bash experiments/gpu38.sh [NAME]     all three runs, or only NAME (norm, la, laspk); GPU=... / ACCUM=n to override
# So far (5.47, 5.48): gated linear attention, with a fixed-size state like ours, is as good as softmax attention, and
# far ahead of the spiking network; the spiking network's non-spiking twin (real-valued outputs, wave 37) is no better
# than the spiking one. So the gap is neither the fixed-size state nor the 0/1 outputs themselves. Two remaining
# suspects, one run each from either end, all on wave 35's control recipe (20,000 steps x 32 x 1,024, cosine,
# distilled from g13_tf163, the same seed and so the same batches as g35_base, 3.863 averaged):
#   norm    the spiking network (g35_base exactly) with model.py --bus_norm: every reader of the bus reads it at unit
#           RMS per token, as a Transformer's pre-norm does. Against g35_base: what the missing normalisation costs.
#           HW=1 (hardware cost and the 8- and 4-bit rounding after training).
#   la      gated linear attention at GPT-2 small's shape (as g36_la163) on this recipe: the reference for laspk, and
#           the gap to g35_base at this recipe.
#   laspk   la with 0/1 hidden units in every MLP (model_tf.py --mlp spike: no membrane, the spiking network's
#           surrogate gradient). Against la: what 0/1 units cost in an otherwise ordinary, normalised network.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
H8=runs/classes/tf163_head225_s8192.npy; E2=runs/classes/tf163_embed220_s2048.npy
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt $H8 $E2; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_write_avg true"
IO="--out_classes 225 --out_short 8192 --class_file $H8 --in_groups 2 --in_short 2048 --in_class_file $E2"
LA="--arch tf --attn linear --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4 --ema 0.999"
only=$1
case "$only" in ""|norm|la|laspk) ;; *) echo "NAME is norm, la or laspk"; exit 1 ;; esac
for s in "norm h100-47" "la h200-141" "laspk h200-141"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    norm) X="$SPREAD $N138 $SNN $MEM $IO --bus_norm true"; HW=1 ;;
    la) X="$LA"; HW=0 ;;
    laspk) X="$LA --mlp spike"; HW=0 ;;
  esac
  j=$(sbatch --parsable -p gpu -J g38_$name --gres=gpu:$gpu:1 \
      --export=ALL,RELAY=$gpu,HW=$HW,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g38_$name $DATA $SHORT $KD $X --accum $acc)
  echo "$j g38_$name ($gpu, accum $acc)"
done

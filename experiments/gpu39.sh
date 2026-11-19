#!/bin/bash
# GPU wave 39: the next rung of the ladder between the linear-attention network and the spiking network
# (docs/experiments.md 5.50). From the repository root:
#   bash experiments/gpu39.sh [NAME]     both runs, or only NAME (lalif, snngla); GPU=... / ACCUM=n to override
# Wave 38 so far: on wave 35's short recipe the linear-attention network (g38_la) is about 0.5 ahead of the spiking
# network (g35_base); 0/1 hidden units in it (g38_laspk) cost little, and normalising the spiking network's bus
# (g38_norm) changes little. What is left between them is mainly (ii) the hidden units being LIF neurons with a
# membrane and a reset, and (iii) the form of the memory. One run each, on the same recipe as wave 38 (20,000 steps x
# 32 x 1,024, cosine, distilled from g13_tf163, the same seed and so the same batches):
#   lalif    g38_laspk with LIF neurons as hidden units (model_tf.py --mlp lif: leak, threshold and reset learned per
#            unit, initialised as the spiking network's populations). Against g38_laspk: what the membrane costs.
#   snngla   g38_norm with gated linear attention (12 heads, model.py --mem_kind gla, as in g38_la) in place of the
#            plastic synapses. Against g38_norm: what the form of the memory costs. Not a spiking network (HW=0).
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
IO="--out_classes 225 --out_short 8192 --class_file $H8 --in_groups 2 --in_short 2048 --in_class_file $E2"
LA="--arch tf --attn linear --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4 --ema 0.999"
only=$1
case "$only" in ""|lalif|snngla) ;; *) echo "NAME is lalif or snngla"; exit 1 ;; esac
for s in "lalif a100-80" "snngla h200-141"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; a100-80) acc=2 ;; *) acc=4 ;; esac
  acc=${ACCUM:-$acc}
  case $name in
    lalif) X="$LA --mlp lif" ;;
    snngla) X="$SPREAD $N138 $SNN $IO --bus_norm true --mem_heads 12 --mem_kind gla" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g39_$name --gres=gpu:$gpu:1 \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g39_$name $DATA $SHORT $KD $X --accum $acc)
  echo "$j g39_$name ($gpu, accum $acc)"
done

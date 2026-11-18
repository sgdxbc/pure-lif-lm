#!/bin/bash
# GPU wave 36: a non-spiking network with a fixed-size state as a control (docs/experiments.md 5.47). From the
# repository root:
#   bash experiments/gpu36.sh [NAME]    both runs, or only NAME (la163, la163kd); GPU=... / ACCUM=n to override
# The spiking network's gap to the Transformer (about 0.26) mixes two costs: keeping the past in a fixed-size state
# (membrane potentials and plastic synapses: no exact look-up of every earlier token) and the spiking itself (0/1
# signals between neurons, surrogate gradients, no normalisation). Gated linear attention (model_tf.py --attn linear)
# has the first and not the second. Trained like our Transformers, it splits the gap:
#   la163     the exact recipe of g13_tf163 (GPT-2 small's shape, 2.5B tokens, cosine, no teacher: 3.310) with every
#             softmax attention replaced by gated linear attention (169.5M parameters against 163.0M: the output gates)
#   la163kd   the same, distilled from g13_tf163 (weight 0.9) like the spiking networks of 5.34 (212M + plastic
#             synapses + distillation, 2.5B tokens: 3.541)
# la163 - g13_tf163 = the cost of the fixed-size state; the spiking network - la163kd = the cost of the spiking.
# --ema 0.999 only adds the averaged weights' evaluation (it does not change training).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
RUN="--batch 32 --seq 1024 --steps 76300 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
LA="--arch tf --attn linear --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4 --ema 0.999"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
only=$1
case "$only" in ""|la163|la163kd) ;; *) echo "NAME is la163 or la163kd"; exit 1 ;; esac
for s in "la163 h100-96" "la163kd h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  case $gpu in h100-96|h200-141) acc=1 ;; *) acc=2 ;; esac   # the teacher's logits need room on a 47 GB card
  acc=${ACCUM:-$acc}
  X=""; [ $name = la163kd ] && X="$KD"
  j=$(sbatch --parsable -p gpu -J g36_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu slurm/train.sbatch \
      runs/g36_$name $DATA $RUN $LA $X --accum $acc)
  echo "$j g36_$name ($gpu, accum $acc)"
done

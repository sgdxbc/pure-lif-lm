#!/bin/bash
# GPU wave 44 (docs/experiments.md 5.55): the features of gated linear attention that wave 42 (5.53) found to matter,
# added back to the plastic synapses in a spiking form. Each run is g38_norm (the spiking network with bus_norm and
# plastic synapses, 3.860) plus one change, on its recipe (20,000 steps x 32 x 1,024, cosine, distilled from
# g13_tf163, same seed):
#   kq4    each key and query is 4 LIF neurons with graded thresholds (up to 4 spikes per token; model.py
#          --mem_kq_levels 4); 0/1 keys and queries cost 0.076 in 5.53
#   dl4    key, query and value currents through synapses with delays of 0-3 tokens (--mem_delays 4; the short
#          convolution, 0.048 in 5.53)
#   gate   the read gated per value channel by silu(W_g bus) (--mem_gate channel; the output gate, 0.072 in 5.53)
#   gateh  the same gate per group, the cheap form (--mem_gate head)
#   all3   kq4, dl4 and gate together
# and, for the self-audit of 2026-11-20 (docs/audits.md): in g39_lalif and g40_lalif the LIF units' parameters
# learned at the base rate, while the spiking network's neurons use 10 times it, so "the membrane did not help" is
# confounded. At 77M (wave 40's shape, recipe and seed; about 2.3 hours each on an a100-40):
#   laspk10  g40_laspk (0/1 MLP units) with --neuron_lr_mult 10 (their thresholds)
#   lalif10  g40_lalif (LIF MLP units: membrane, leak, reset) with --neuron_lr_mult 10
# Also from that audit: the seed noise (about 0.003) rests on one pair of runs of another network, and the 163M runs
# of wave 43 are distilled while their 77M counterparts were not, so "cost against size" mixes the two:
#   laspk_s1, laspk_s2   g40_laspk with seeds 1 and 2
#   kd_laspk77, kd_q4t77, kd_tern77   g40_laspk, g41_q4t and g43_tern77 distilled as the 163M runs are (from
#                                     g13_tf163, weight 0.9); the learning rate stays the 77M one (1e-3). Not in the
#                                     default list: dropped when the owner put scaling trends aside (2026-11-20);
#                                     still launchable by name
# From the repository root:
#   bash experiments/gpu44.sh [NAME]     every run, or only NAME; GPU=... / ACCUM=n to override;
#                                        NICE=n to let the next parts of runs already in flight go first
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
NORM="$SPREAD $N138 $SNN --mem_heads 8 --mem_write_avg true $IO --bus_norm true"   # g38_norm
S77="--arch tf --attn linear --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"   # g40_laspk's
RELAY="--relay quant --relay_levels 4 --relay_target 0.5"                                    # g41_q4t's
only=$1
NAMES="laspk_s1 laspk_s2 kd_laspk77 kd_q4t77 kd_tern77 laspk10 lalif10 gate gateh all3 kq4 dl4"
case " $NAMES " in *" ${only:-laspk_s1} "*) ;; *) echo "NAME is one of: $NAMES"; exit 1 ;; esac
for s in "laspk_s1 a100-40" "laspk_s2 a100-40" "kd_laspk77 a100-40" "kd_q4t77 a100-40" "kd_tern77 a100-40" \
         "laspk10 a100-40" "lalif10 a100-40" "gate h100-47" "gateh h100-47" "all3 h100-96" "kq4 a100-80" "dl4 h100-47"; do
  set -- $s; name=$1; gpu=${GPU:-$2}
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  [ -z "$only" ] && case $name in kd_*) continue ;; esac   # dropped from the default list (header)
  case $gpu in h100-96|h200-141|a100-80) acc=2 ;; *) acc=4 ;; esac   # g38_norm 24.8 GB at accum 4; accum 1 ran
                                                                      # out of memory on h100-96 (g42_noconv)
  acc=${ACCUM:-$acc}
  case $name in
    laspk_s1) X="$S77 --mlp spike --seed 1" ;;
    laspk_s2) X="$S77 --mlp spike --seed 2" ;;
    kd_laspk77) X="$S77 --mlp spike $KD" ;;
    kd_q4t77) X="$S77 --mlp spike $RELAY $KD" ;;
    kd_tern77) X="$S77 --mlp spike --wq ternary $KD" ;;
    laspk10) X="$S77 --mlp spike --neuron_lr_mult 10" ;;
    lalif10) X="$S77 --mlp lif --neuron_lr_mult 10" ;;
    kq4) X="$KD $NORM --mem_kq_levels 4" ;;
    dl4) X="$KD $NORM --mem_delays 4" ;;
    gate) X="$KD $NORM --mem_gate channel" ;;
    gateh) X="$KD $NORM --mem_gate head" ;;
    all3) X="$KD $NORM --mem_kq_levels 4 --mem_delays 4 --mem_gate channel" ;;
  esac
  j=$(sbatch --parsable -p gpu -J g44_$name --gres=gpu:$gpu:1 ${NICE:+--nice=$NICE} \
      --export=ALL,RELAY=$gpu,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
      runs/g44_$name $DATA $SHORT $X --accum $acc)
  echo "$j g44_$name ($gpu, accum $acc) $X"
done

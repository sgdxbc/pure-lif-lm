#!/bin/bash
# GPU wave 26: the long runs, 300,000 steps of 32 x 1024 = 9.8B tokens, about one pass over the FineWeb-Edu sample
# (stage goals 1 and 2; docs/experiments.md 5.37). From the repository root:  bash experiments/gpu26.sh
#  - snn212  the spiking network with the best recipe of waves 17-24: 212M shape, plastic synapses before every layer
#            (8 groups x 256 key neurons, time constants from 1 token, the original write; 212M at 256 keys trains
#            without large gradient spikes, g23_kd212k256), distilled from the Transformer g13_tf163.
#            --spike_skip 10 drops the update of the rare batch whose gradient norm is over 10x the recent median
#            (g23_kd212k256 had 4 such in 20,000 steps, up to 45x).
#  - tf163   the Transformer g13_tf163 (GPT-2 small's shape) trained the same way on the same tokens: the reference for
#            the spiking network at equal data, and a stronger teacher (g13_tf163 read 2.5B tokens). Distillation helps a
#            student only up to a training length that grows with its size, and a teacher that the student nearly
#            reaches holds it back (Busbridge et al. 2025, arXiv 2502.08606); 212M is projected at about 3.4 after 9.8B
#            tokens, close to g13_tf163's 3.31. So once tf163 is finished (about 11 hours), snn212 continues with it as
#            the teacher from its next kept checkpoint (stop the relay, resubmit the same command with
#            --teacher runs/g26_tf163/ckpt.pt; the checkpoint carries no teacher).
# Schedule: --schedule wsd, constant learning rate, then down to 0 over the last 20% (60,000 steps), shaped 1 - sqrt.
# The kept checkpoints (30k, 60k, 120k, 180k, 240k) are the starting points of decay branches of the same 20% shape,
# finished at 37.5k, 75k (2.46B tokens, the length of waves 13-21), 150k and 225k steps: with the end of the run, five
# points of the compute curve from one run (experiments/gpu26_branch.sh). 240k (where the decay starts) lets the run be
# extended later. On the 138M check (g24_kdk256t1wsd) this schedule ended 0.016 behind cosine with averaging (3.788 vs
# 3.772); the branches are worth that.
# Expected time: snn212 at about 79k tok/s on the H200 (g23_kd212k256) is about 35 hours of training; tf163 at about
# 250k tok/s on the H100-96 (g13_tf163) about 11 hours. Each checkpoint of snn212 (weights, Adam moments, average) is
# about 3-4 GB; five kept ones are about 20 GB.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
LONG="--batch 32 --seq 1024 --steps 300000 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
WSD="--schedule wsd --decay_steps 60000 --min_lr_frac 0 --keep_ckpt 30000,60000,120000,180000,240000"
N212="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN="--ema 0.999 --neuron_lr_mult 10 --lr 6e-4 --spike_skip 10"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8 --mem_keys 256 --mem_tau_min 1"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g26_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g26_$name "$@"
}
a=$(sub snn212 h200-141 $DATA $SPREAD $LONG $WSD $N212 $SNN $KD $MEM --accum 2); echo "$a g26_snn212 (h200-141)"
b=$(sub tf163  h100-96  $DATA $LONG $WSD --arch tf --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4); echo "$b g26_tf163 (h100-96)"
echo "submitted; squeue -u \$USER to follow"

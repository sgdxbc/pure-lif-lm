#!/bin/bash
# GPU wave 9: several simulation steps per token; a 250M Transformer; the best spiking setup so far for two passes.
# Run from the repository root:  bash experiments/gpu9.sh
# Wave 8 (docs/experiments.md 5.19): neither the shape (27 layers at the 117M width = 12 wider layers, EMA 1.2429 vs 1.2420) nor
# the learning rate (5e-4 stays best) explains why 117M -> 257M gained so little, while a better teacher helped
# (g6_tf105_2ep, 1.091, instead of g2_tf105_lr6e-4, 1.139: 1.2420 -> 1.2292). Three questions here:
#  - steps per token. So far every token is one simulation step, so a neuron sends at most one spike (one bit) per
#    token. With --steps_per_token T each token is shown for T steps (model.py docstring): neurons can fire up to T
#    times per token and the readout averages the T steps. No new parameters; T times the neuron steps per token.
#    On the 117M with the three methods, against g4_snn117_best (EMA 1.2776; lr 6e-4, teacher g2_tf105_lr6e-4,
#    loss at k=0 / k=128 of probe_memory: 1.798 / 1.319): T = 2 and 3, nothing else changed.
#  - does the Transformer also gain less from 105M to 250M? tf250 (d_model 1280, 12 layers, 20 heads, 247.8M) for two
#    passes, against g6_tf105_2ep (1.091). The spiking network gained 0.039 from 117M to 257M with two passes. If the
#    Transformer also gains only about that much, the slowdown is (partly) the data set's; if clearly more, it is the
#    spiking network's. It is also a stronger teacher for later waves.
#  - the best spiking setup so far for two passes: 257M, teacher g6_tf105_2ep, lr 5e-4 (against g6_snn257_best_2ep,
#    teacher g2_tf105_lr6e-4, EMA 1.1993).
# Afterwards slurm/probe_long.sbatch measures recall to 512 tokens for the three spiking runs.
# Expected times: snn257 two passes ~1.8 h on the H200. tf250 two passes ~3-4 h on the H100 NVL, and the T = 3 run
# ~2.5-3.5 h on an H100 47GB: either may stop at the 3 h limit once; then submit the same sub line again (it resumes),
# and submit the probe job again by hand once all three spiking runs are done.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
EP2="--batch 32 --seq 256 --steps 134000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
N257="--d_model 1536 --d_hidden 6144 --n_layers 12 --leak_rank 192"
METHODS="--distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
T1="--teacher runs/g2_tf105_lr6e-4/ckpt.pt"
T2="--teacher runs/g6_tf105_2ep/ckpt.pt"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -J g9_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g9_$name "$@"
}
a=$(sub snn257_t2ep_2ep h200-141 $SPREAD $EP2 $N257 --lr 5e-4 $T2 $METHODS); echo "$a g9_snn257_t2ep_2ep (h200-141)"
b=$(sub snn117_T2       h100-47  $SPREAD $RUN $N117 --lr 6e-4 $T1 $METHODS --steps_per_token 2); echo "$b g9_snn117_T2 (h100-47)"
c=$(sub snn117_T3       h100-47  $SPREAD $RUN $N117 --lr 6e-4 $T1 $METHODS --steps_per_token 3); echo "$c g9_snn117_T3 (h100-47)"
d=$(sub tf250_2ep       h100-96  --arch tf $EP2 --d_model 1280 --n_layers 12 --n_heads 20 --lr 3e-4); echo "$d g9_tf250_2ep (h100-96)"
e=$(sbatch --parsable -J probe_long --gres=gpu:a100-40:1 --dependency=afterok:$a:$b:$c slurm/probe_long.sbatch \
    runs/g9_snn257_t2ep_2ep runs/g9_snn117_T2 runs/g9_snn117_T3)
echo "$e probe_long (after the three spiking runs)"
echo "submitted; squeue -u \$USER to follow"

#!/bin/bash
# GPU wave 12: the input current scaled by (1 - beta) (--leak_gate, model.py). From the repository root:
#   bash experiments/gpu12.sh
# Wave 10 (docs/experiments.md 5.23): the spiking bus costs 0.09 and saves little on the target hardware; 2 steps per token gain
# the 257M 0.010 and the 250M teacher 0.005. The gap to the Transformer grows with context (5.19), so this wave tries
# a change aimed at memory:
#  - leak_gate: V[t] = beta[t] V[t-1] + (1 - beta[t]) I[t], the exact one-step discretisation of the LIF equation
#    when the leak-modulating synapses change the time constant. A neuron made to hold its potential (beta near 1)
#    then takes in little new current, so what it holds is not overwritten; one made to leak takes the new current
#    in fully. The threshold starts (1 - beta) times lower, so at initialisation it is exactly the same network
#    (checked on CPU, also with 2 steps per token, streaming, and quant_sim.py's step-by-step pass).
#  - on the 117M with the three methods (lr 6e-4, teacher g2_tf105_lr6e-4) against g4_snn117_best (EMA 1.2776),
#    and on the 257M (lr 5e-4, teacher g6_tf105_2ep, one pass) against g8_snn257_best_t2ep (EMA 1.2292).
# Then slurm/hw.sbatch on both (hardware cost and the low-precision simulation).
# Expected times: 117M ~1.5 h (A100 40GB), 257M ~1.5 h (H200). A run stopped at the 3 h limit resumes when its sub
# line is submitted again (then also submit the hw job again by hand).
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 256 --steps 67000 --warmup 300 --eval_every 5000 --eval_batches 40 --log_every 500"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
N257="--d_model 1536 --d_hidden 6144 --n_layers 12 --leak_rank 192"
METHODS="--distill_weight 0.9 --distill_temp 1 --ema 0.999 --neuron_lr_mult 10"
T1="--teacher runs/g2_tf105_lr6e-4/ckpt.pt"
T2="--teacher runs/g6_tf105_2ep/ckpt.pt"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g12_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g12_$name "$@"
}
a=$(sub gate117 a100-40  $SPREAD $RUN $N117 --lr 6e-4 $T1 $METHODS --leak_gate true); echo "$a g12_gate117 (a100-40)"
b=$(sub gate257 h200-141 $SPREAD $RUN $N257 --lr 5e-4 $T2 $METHODS --leak_gate true); echo "$b g12_gate257 (h200-141)"
c=$(sbatch --parsable -p gpu -J hw --gres=gpu:a100-40:1 --dependency=afterok:$a:$b slurm/hw.sbatch \
    runs/g12_gate117 runs/g12_gate257)
echo "$c hw (after the two runs)"
echo "submitted; squeue -u \$USER to follow"

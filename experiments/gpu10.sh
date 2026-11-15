#!/bin/bash
# GPU wave 10: the spiking bus, and whether wave 9's two improvements carry over to the 257M. From the repo root:
#   bash experiments/gpu10.sh
# Wave 9 (docs/experiments.md 5.21): 2 steps per token improved the 117M by 0.013 (EMA 1.2651 vs 1.2776), 3 steps did not; the
# two-pass teacher helped the two-pass 257M by 0.015; the Transformer gains as little per doubling beyond 100M as the
# spiking network does.
#  - spiking bus (model.py, --spike_bus): ON/OFF relay neurons turn the bus into spikes before each population, so the
#    input half of every layer also runs on spikes (docs/experiments.md 5.20: on every candidate hardware that half is the dense
#    part that sparsity cannot reduce). On the 117M with the three methods (lr 6e-4, teacher g2_tf105_lr6e-4), at 1
#    and 2 steps per token, against g4_snn117_best (EMA 1.2776) and g9_snn117_T2 (EMA 1.2651).
#  - 2 steps per token on the 257M: teacher g6_tf105_2ep, one pass, against g8_snn257_best_t2ep (EMA 1.2292).
#  - the 250M Transformer (g9_tf250_2ep, 1.049) as the teacher of the 257M, one pass, against the same run (whose
#    teacher had 1.091).
# Then slurm/hw.sbatch on the four runs (hardware cost; quant_sim.py skips the spiking-bus runs).
# Expected times: 117M bus ~1.5 h and 117M bus T=2 ~2.5 h (H100 47GB); 257M T=2 ~2.2 h (H200); 257M with the 250M
# teacher ~1.5 h (H100 NVL). A run stopped at the 3 h limit resumes when its sub line is submitted again (then also
# submit the hw job again by hand).
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
T3="--teacher runs/g9_tf250_2ep/ckpt.pt"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -J g10_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g10_$name "$@"
}
a=$(sub bus117      h100-47  $SPREAD $RUN $N117 --lr 6e-4 $T1 $METHODS --spike_bus true); echo "$a g10_bus117 (h100-47)"
b=$(sub bus117_T2   h100-47  $SPREAD $RUN $N117 --lr 6e-4 $T1 $METHODS --spike_bus true --steps_per_token 2); echo "$b g10_bus117_T2 (h100-47)"
c=$(sub snn257_T2   h200-141 $SPREAD $RUN $N257 --lr 5e-4 $T2 $METHODS --steps_per_token 2); echo "$c g10_snn257_T2 (h200-141)"
d=$(sub snn257_t250 h100-96  $SPREAD $RUN $N257 --lr 5e-4 $T3 $METHODS); echo "$d g10_snn257_t250 (h100-96)"
e=$(sbatch --parsable -J hw --gres=gpu:a100-40:1 --dependency=afterok:$a:$b:$c:$d slurm/hw.sbatch \
    runs/g10_bus117 runs/g10_bus117_T2 runs/g10_snn257_T2 runs/g10_snn257_t250)
echo "$e hw (after the four runs)"
echo "submitted; squeue -u \$USER to follow"

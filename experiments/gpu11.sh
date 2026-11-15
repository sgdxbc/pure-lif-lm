#!/bin/bash
# GPU wave 11: first runs of the long-text track on PG-19 (docs/experiments.md 5.22). From the repository root, once
# slurm/prepare_pg19.sbatch has written data/pg19/{train,valid,test}.bin:   bash experiments/gpu11.sh
# Questions: does the spiking network keep improving deeper into a book, as a Transformer does, and how do the two
# compare in quality and in hardware cost on long text? No teacher yet (a PG-19 Transformer from this wave can be the
# teacher of the next one). All runs see ~1.0B tokens (about a third of PG-19's training books), 8192 tokens per step.
#  - snn125_carry  the 117M spiking network (125M with the 8192-token vocabulary), lr 8e-4, EMA 0.999, neuron lr x10,
#                  reading 16 streams of 512-token segments with --carry_state: each row continues through the books and
#                  its membrane potentials pass on to the next segment (gradients stop at segment boundaries)
#  - snn125_rest   the same, but every 512-token segment starts from rest (as on TinyStories): does carrying matter?
#  - tf98          Transformer d768 x 12 (97.5M), rotary positions, 2048-token windows, lr 6e-4
#  - tf34          Transformer d512 x 8 (33.6M), rotary positions, 2048-token windows, lr 1.5e-3
# Validation loss during training is on windows of each run's own length, so it is not comparable across runs; the
# comparison is eval_long.py on the test books (loss by position to 8192 tokens; Transformers also with a 2048-token
# sliding window) and hw_cost.py, both submitted to run after all four (slurm/long_eval.sbatch, slurm/hw.sbatch).
# PROBES=0: the TinyStories evaluation and probes of slurm/train.sbatch do not apply here.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for s in train valid test; do [ -s data/pg19/$s.bin ] || { echo "data/pg19/$s.bin missing: run slurm/prepare_pg19.sbatch first"; exit 1; }; done
DATA="--data data/pg19 --tokenizer tok8192.json --vocab 8192"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
N117="--d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128"
SNN_RUN="--batch 16 --seq 512 --steps 122000 --warmup 500 --eval_every 5000 --eval_batches 40 --log_every 500"
TF_RUN="--batch 4 --seq 2048 --steps 122000 --warmup 500 --eval_every 5000 --eval_batches 40 --log_every 500"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable --export=ALL,PROBES=0 -J g11_$name --gres=gpu:$gpu:1 slurm/train.sbatch runs/g11_$name "$@"
}
a=$(sub snn125_carry h100-96 $DATA $SPREAD $N117 $SNN_RUN --lr 8e-4 --ema 0.999 --neuron_lr_mult 10 --carry_state); echo "$a g11_snn125_carry (h100-96)"
b=$(sub snn125_rest  h100-47 $DATA $SPREAD $N117 $SNN_RUN --lr 8e-4 --ema 0.999 --neuron_lr_mult 10); echo "$b g11_snn125_rest (h100-47)"
c=$(sub tf98         h100-47 $DATA --arch tf --rope --d_model 768 --n_layers 12 --n_heads 12 $TF_RUN --lr 6e-4); echo "$c g11_tf98 (h100-47)"
d=$(sub tf34         a100-80 $DATA --arch tf --rope --d_model 512 --n_layers 8 --n_heads 8 $TF_RUN --lr 1.5e-3); echo "$d g11_tf34 (a100-80)"
R="runs/g11_snn125_carry runs/g11_snn125_rest runs/g11_tf98 runs/g11_tf34"
e=$(sbatch --parsable -J long_eval --gres=gpu:a100-80:1 --dependency=afterok:$a:$b:$c:$d slurm/long_eval.sbatch $R); echo "$e long_eval (after the four)"
f=$(sbatch --parsable -J hw --gres=gpu:a100-40:1 --dependency=afterok:$a:$b:$c:$d slurm/hw.sbatch $R); echo "$f hw (after the four)"
echo "Expected: ~2-3 h per run; a run stopped at the 3 h limit resumes when its sub line is submitted again (then submit"
echo "the two evaluation jobs again by hand)."

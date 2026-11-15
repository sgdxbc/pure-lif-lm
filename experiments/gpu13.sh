#!/bin/bash
# GPU wave 13: the first runs on general text (README section 0, stage goals of 2026-11-15). From the repository root,
# once slurm/prepare_fineweb.sbatch has written data/fineweb/{train,valid}.bin and lm_eval + transformers are in the
# venv (slurm/zeroshot.sbatch):   bash experiments/gpu13.sh
# Data: FineWeb-Edu (10B-token sample), GPT-2's tokenizer (vocabulary padded to 50304), 1024-token windows. Every run
# sees 2.5B tokens (76,300 steps of 32 x 1024), about 20 tokens per parameter for a GPT-2-small-sized model; no teacher
# yet. Questions: where does the spiking network stand on general text against a Transformer trained the same way,
# and against the classic small models (GPT-2 124M / 355M, Pythia 160M / 410M, scored by slurm/zeroshot.sbatch with
# the same benchmarks)? These are also the first points of the compute curves of stage goal 2.
#  - snn212   spiking, bus 1024, 4096 neurons x 12 populations, leak synapses rank 128 (the TinyStories 117M; 212M
#             with the larger vocabulary, 109M without input and readout synapses), lr 6e-4, EMA, neuron lr x10
#  - snn138   spiking, bus 768, 3072 x 12, leak rank 96 (138M; 61M without input and readout synapses), same recipe
#  - tf163    Transformer in GPT-2 small's shape (d768 x 12, 12 heads, learned positions; 163M as the input and output
#             embeddings are not shared, 85M without them), lr 6e-4
#  - tf77     Transformer d512 x 8 (77M; 25M without the embeddings), lr 1e-3
# Each run chains 3-hour jobs (RELAY in slurm/train.sbatch) until it is finished, then writes the full-validation
# evaluation, the zero-shot benchmarks and the hardware cost (HW=1) into its directory.
set -e
cd "$(dirname "$0")/.."   # the repository root
mkdir -p slurm/logs
for s in train valid; do [ -s data/fineweb/$s.bin ] || { echo "data/fineweb/$s.bin missing: run slurm/prepare_fineweb.sbatch first"; exit 1; }; done
source slurm/env.sh
python3 -c "import lm_eval, transformers" || { echo "pip install lm_eval transformers accelerate into the venv first"; exit 1; }
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 1024 --steps 76300 --warmup 1000 --eval_every 5000 --eval_batches 40 --log_every 500"
SNN="--ema 0.999 --neuron_lr_mult 10"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g13_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g13_$name "$@"
}
a=$(sub snn212 h200-141 $DATA $SPREAD $RUN $SNN --d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128 --lr 6e-4); echo "$a g13_snn212 (h200-141)"
b=$(sub snn138 h100-47  $DATA $SPREAD $RUN $SNN --d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96 --lr 6e-4); echo "$b g13_snn138 (h100-47)"
c=$(sub tf163  h100-96  $DATA $RUN --arch tf --d_model 768 --n_layers 12 --n_heads 12 --lr 6e-4); echo "$c g13_tf163 (h100-96)"
d=$(sub tf77   a100-80  $DATA $RUN --arch tf --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3); echo "$d g13_tf77 (a100-80)"
e=$(sbatch --parsable -p gpu -J zeroshot --gres=gpu:a100-40:1 slurm/zeroshot.sbatch \
    openai-community/gpt2 openai-community/gpt2-medium EleutherAI/pythia-160m EleutherAI/pythia-410m)
echo "$e zeroshot references (GPT-2 124M / 355M, Pythia 160M / 410M)"
echo "Each run continues in 3-hour parts by itself; squeue -u \$USER to follow."

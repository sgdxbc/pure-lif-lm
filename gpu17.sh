#!/bin/bash
# GPU wave 17: short screening runs on general text (README 5.24, 5.28). From the repository root:  bash gpu17.sh
# Wave 13: on FineWeb-Edu the spiking 212M is 0.50 behind the 163M Transformer (3.814 against 3.310), and the gap
# grows during training (0.42 at step 5000, 0.40 at 20000, 0.50 at the end). Its zero-shot results are close to GPT-2
# 124M except LAMBADA (1%), which mostly asks for a word seen earlier in the passage. This wave screens the candidate
# fixes side by side on the 138M shape (bus 768, 3072 x 12, leak rank 96), each reading 655M tokens (20,000 steps of
# 32 x 1024, its own learning-rate schedule), with the zero-shot benchmarks and the hardware cost afterwards (HW=1):
#  - base138     no teacher, lr 6e-4 (the reference for the others)
#  - lr3_138     no teacher, lr 3e-4: the firing rate collapses while the learning rate is high (212M: 9% at step 500,
#                ~2.4% from step 2000 to 20000, layers 2-5 near 1%; back to 6.6% at the end); is the peak lr too high?
#  - mem138      plastic synapses (--mem_heads 8: 8 groups of 64 key and 64 query neurons, 64 values each, in front of
#                every population; model.py), lr 6e-4: a memory that can bring back a word read long before
#  - kd138       distillation from g13_tf163 (output distribution, weight 0.9, temperature 1), lr 6e-4: on TinyStories
#                the biggest single gain, and the only training change that improved far recall (README 5.16-5.18)
#  - kdmem138    both
#  - tf163m      the 163M Transformer with Muon at lr 6e-4 (on TinyStories Muon helped the Transformer, README 5.27),
#                the same 20,000 steps: the Transformer side of the comparison at this budget
# All spiking runs: the methods without the teacher (EMA 0.999, neuron lr x10). Distillation over the full 50k-word
# vocabulary needs an 80 GB card or more (the 212M alone peaked at 40 GB). Also: the reference losses of GPT-2 124M /
# 355M and GPT-Neo 125M again, in float32 (ref_loss.py now defaults to it; the first run used bfloat16).
# Expected: ~1-2 h each; each run chains 3-hour parts by itself (RELAY).
set -e
cd "$(dirname "$0")"
mkdir -p slurm/logs
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt; do [ -s $f ] || { echo "$f missing"; exit 1; }; done
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SPREAD="--hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05"
RUN="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
N138="--d_model 768 --d_hidden 3072 --n_layers 12 --leak_rank 96"
SNN="--ema 0.999 --neuron_lr_mult 10"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
MEM="--mem_heads 8"
sub() {  # sub NAME GPU_TYPE train.py-flags...   (prints the job id of the first part)
  local name=$1 gpu=$2; shift 2
  sbatch --parsable -p gpu -J g17_$name --gres=gpu:$gpu:1 --export=ALL,RELAY=$gpu,HW=1 slurm/train.sbatch runs/g17_$name "$@"
}
a=$(sub base138  h100-47  $DATA $SPREAD $RUN $N138 $SNN --lr 6e-4);             echo "$a g17_base138 (h100-47)"
b=$(sub lr3_138  a100-80  $DATA $SPREAD $RUN $N138 $SNN --lr 3e-4);             echo "$b g17_lr3_138 (a100-80)"
c=$(sub mem138   h100-47  $DATA $SPREAD $RUN $N138 $SNN $MEM --lr 6e-4);        echo "$c g17_mem138 (h100-47)"
d=$(sub kd138    h100-96  $DATA $SPREAD $RUN $N138 $SNN $KD --lr 6e-4);         echo "$d g17_kd138 (h100-96)"
e=$(sub kdmem138 h200-141 $DATA $SPREAD $RUN $N138 $SNN $KD $MEM --lr 6e-4);    echo "$e g17_kdmem138 (h200-141)"
f=$(sub tf163m   a100-40  $DATA $RUN --arch tf --d_model 768 --n_layers 12 --n_heads 12 --optim muon --lr 6e-4)
echo "$f g17_tf163m (a100-40)"
g=$(sbatch --parsable -J ref_loss32 --gres=gpu:a100-40:1 slurm/ref_loss.sbatch \
    openai-community/gpt2 openai-community/gpt2-medium EleutherAI/gpt-neo-125m)
echo "$g ref_loss32 (GPT-2 124M / 355M, GPT-Neo 125M in float32)"
echo "submitted; squeue -u \$USER to follow"

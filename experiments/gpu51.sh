#!/bin/bash
# Feature distillation pilot (docs/experiments.md 5.62), paired with g49_kdc.
# From the repository root: bash experiments/gpu51.sh   (NICE=n to let the next parts of runs in flight go first)
set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs
ck=runs/g46_kd25s1/ckpt_step15000.pt
feat=runs/g44_laspk_s1/ckpt.pt
for f in data/fineweb/train.bin runs/g13_tf163/ckpt.pt "$ck" "$feat"; do
  [ -s "$f" ] || { echo "$f missing"; exit 1; }
done
# g49_kdc's flags, from experiments/gpu49.sh, with the feature options added below.
DATA="--data data/fineweb --tokenizer gpt2.json --vocab 50304"
SHORT="--batch 32 --seq 1024 --steps 20000 --warmup 1000 --eval_every 2500 --eval_batches 40 --log_every 500"
KD="--teacher runs/g13_tf163/ckpt.pt --distill_weight 0.9 --distill_temp 1"
S77="--arch tf --attn linear --mlp spike --d_model 512 --n_layers 8 --n_heads 8 --lr 1e-3 --ema 0.999"
Q25="--relay quant --relay_levels 4 --relay_target 0.25"
j=$(sbatch --parsable -p gpu -J g51_kdf --gres=gpu:a100-80:1 ${NICE:+--nice=$NICE} \
    --export=ALL,RELAY=a100-80,HW=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train.sbatch \
    runs/g51_kdf $DATA $SHORT $S77 $Q25 $KD --seed 1 --init_ckpt "$ck" --accum 2 \
    --feat_teacher "$feat" --feat_positions 64 --feat_weight_frac 0.1)
echo "$j g51_kdf (a100-80, accum 2) from $ck with feature teacher $feat"

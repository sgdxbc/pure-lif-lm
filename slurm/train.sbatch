#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 03:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# One training run on one GPU, then (unless PROBES=0) full-validation evaluation and the two memory probes, all
# written into the run directory. Submit from the repository root:
#   sbatch -J NAME --gres=gpu:a100-40:1 slurm/train.sbatch runs/NAME [train.py flags...]
# A run stopped by the time limit continues from its last checkpoint when the same command is submitted again.
source slurm/env.sh
out=$1; shift
mkdir -p "$out"
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) | $out | $*"
python3 train.py --device cuda --out "$out" "$@" --resume >> "$out/stdout.log" 2>&1
code=$?
grep -E "^params|val_loss|Error|Traceback" "$out/stdout.log" | tail -20
[ $code -ne 0 ] && { echo "train.py failed with exit code $code"; exit $code; }
[ "${PROBES:-1}" = "0" ] && exit 0
python3 evaluate.py --device cuda --ckpt "$out/ckpt.pt" --out "$out/final_eval.json" > "$out/eval.log" 2>&1
if [[ " $* " == *" --ema "* ]]; then
  python3 evaluate.py --device cuda --ema --ckpt "$out/ckpt.pt" --out "$out/final_eval_ema.json" >> "$out/eval.log" 2>&1
fi
python3 probe_recall.py --device cuda "m=$out/ckpt.pt" --out "$out/recall.json" > "$out/probe.log" 2>&1
python3 probe_memory.py --device cuda "m=$out/ckpt.pt" --out "$out/probe_memory.json" >> "$out/probe.log" 2>&1
if [[ " $* " == *" --ema "* ]]; then  # the averaged weights are the ones reported, so probe them too
  python3 probe_recall.py --device cuda --ema "m=$out/ckpt.pt" --out "$out/recall_ema.json" >> "$out/probe.log" 2>&1
  python3 probe_memory.py --device cuda --ema "m=$out/ckpt.pt" --out "$out/probe_memory_ema.json" >> "$out/probe.log" 2>&1
fi
grep -h -E "val_loss|loss" "$out/final_eval.json" | head -3
echo "done $(date)"

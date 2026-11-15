#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 03:00:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# One training run on one GPU, then the evaluation, all written into the run directory. Submit from the repo root:
#   sbatch -J NAME --gres=gpu:a100-40:1 slurm/train.sbatch runs/NAME [train.py flags...]
# A run stopped by the time limit continues from its last checkpoint when the same command is submitted again.
# Long runs: with --export=ALL,RELAY=<gpu type> (e.g. RELAY=h100-96), each job first queues its own successor on the
# same partition and GPU type, which starts when this job ends (for whatever reason) and continues from the last
# checkpoint. The chain stops once the run is finished (runs/NAME/done) or has failed with an error (runs/NAME/FAILED,
# written when train.py exits with an error; a job stopped by the time limit writes nothing). Remove FAILED to retry.
# After training:
#   TinyStories runs: full-validation evaluation (evaluate.py) and the two memory probes;
#   runs with --data (general text): full-validation evaluation and the zero-shot benchmarks (eval_lm.py,
#   ZEROSHOT=0 skips them; needs lm_eval in the venv, see slurm/zeroshot.sbatch);
#   PROBES=0 skips all of it. Each is done with the moving-average weights too when the run has --ema.
#   HW=1 adds the hardware cost and, where it applies, the low-precision simulation (as slurm/hw.sbatch does).
source slurm/env.sh
out=$1; shift
mkdir -p "$out"
[ -e "$out/done" ] && { echo "$out is finished"; exit 0; }
[ -e "$out/FAILED" ] && { echo "$out failed earlier ($(cat "$out/FAILED")); not continuing"; exit 1; }
if [ -n "$RELAY" ]; then
  next=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" -J "$SLURM_JOB_NAME" --gres=gpu:$RELAY:1 \
         --dependency=afterany:$SLURM_JOB_ID --export=ALL,RELAY=$RELAY slurm/train.sbatch "$out" "$@")
  echo "queued the next part of $out: job $next"
fi
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) | $out | $*"
python3 train.py --device cuda --out "$out" "$@" --resume >> "$out/stdout.log" 2>&1
code=$?
grep -E "^params|val_loss|Error|Traceback" "$out/stdout.log" | tail -20
if [ $code -ne 0 ]; then
  echo "train.py failed with exit code $code (job $SLURM_JOB_ID)" | tee "$out/FAILED"
  exit $code
fi
ema=0; [[ " $* " == *" --ema "* ]] && ema=1
general=0; [[ " $* " == *" --data "* ]] && general=1
if [ "${PROBES:-1}" != "0" ]; then
  python3 evaluate.py --device cuda --ckpt "$out/ckpt.pt" --out "$out/final_eval.json" > "$out/eval.log" 2>&1
  if [ $ema = 1 ]; then
    python3 evaluate.py --device cuda --ema --ckpt "$out/ckpt.pt" --out "$out/final_eval_ema.json" >> "$out/eval.log" 2>&1
  fi
  if [ $general = 0 ]; then
    python3 probe_recall.py --device cuda "m=$out/ckpt.pt" --out "$out/recall.json" > "$out/probe.log" 2>&1
    python3 probe_memory.py --device cuda "m=$out/ckpt.pt" --out "$out/probe_memory.json" >> "$out/probe.log" 2>&1
    if [ $ema = 1 ]; then  # the averaged weights are the ones reported, so probe them too
      python3 probe_recall.py --device cuda --ema "m=$out/ckpt.pt" --out "$out/recall_ema.json" >> "$out/probe.log" 2>&1
      python3 probe_memory.py --device cuda --ema "m=$out/ckpt.pt" --out "$out/probe_memory_ema.json" >> "$out/probe.log" 2>&1
    fi
  elif [ "${ZEROSHOT:-1}" != "0" ]; then
    python3 eval_lm.py --device cuda "$out" > "$out/zeroshot.log" 2>&1
    [ $ema = 1 ] && python3 eval_lm.py --device cuda --ema "$out" >> "$out/zeroshot.log" 2>&1
    grep -E "^(hellaswag|lambada|piqa|arc|winogrande)" "$out/zeroshot.log"
  fi
  grep -h -E "val_loss|loss" "$out/final_eval.json" | head -3
fi
if [ "${HW:-0}" = "1" ]; then
  python3 hw_cost.py --device cuda --ema "$out" > "$out/hw_cost.log" 2>&1; tail -3 "$out/hw_cost.log"
  if python3 -c "import json, sys; a = json.load(open('$out/args.json')); sys.exit(0 if a.get('arch', 'snn') == 'snn' and not (a.get('spike_bus') or a.get('rec_rank') or a.get('delay_k') or a.get('mem_heads')) else 1)"; then
    python3 quant_sim.py --device cuda --ema "$out" > "$out/quant.log" 2>&1; cat "$out/quant.log"
  fi
fi
touch "$out/done"
echo "done $(date)"

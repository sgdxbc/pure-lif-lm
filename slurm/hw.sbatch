#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G -t 01:30:00 -o slurm/logs/%x-%j.out
#SBATCH --exclude=xgpj0  # numpy C extensions fail to import on this node (g2_tf105, g3_d10)
# Hardware cost (hw_cost.py -> hw_cost.json) for every run given, and the low-precision simulation (quant_sim.py ->
# quant.json, log in quant.log) for the spiking ones; both with the moving-average weights where the run has them
# (README 5.20). From the repository root:
#   sbatch -J hw --gres=gpu:a100-40:1 slurm/hw.sbatch runs/A runs/B ...
source slurm/env.sh
python3 hw_cost.py --device cuda --ema "$@"
echo "hw_cost exit $? $(date)"
for out in "$@"; do
  # quant_sim.py covers spiking runs without the spiking bus, same-layer synapses or delays
  if python3 -c "import json, sys; a = json.load(open('$out/args.json')); sys.exit(0 if a.get('arch', 'snn') == 'snn' and not (a.get('spike_bus') or a.get('rec_rank') or a.get('delay_k')) else 1)"; then
    python3 quant_sim.py --device cuda --ema "$out" > "$out/quant.log" 2>&1
    echo "$out: quant_sim exit $? $(date)"
    cat "$out/quant.log"
  fi
done

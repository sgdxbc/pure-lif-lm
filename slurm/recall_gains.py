"""Print the recall gains (repeat d=65-128 and d=129-256) of runs: python3 slurm/recall_gains.py runs/X ..."""
import json, os, sys
for r in sys.argv[1:]:
    d = json.load(open(os.path.join(r, "recall.json")))["m"]
    print(f"{os.path.basename(r.rstrip('/')):18s} gain d=65-128 {d['repeat d=65-128']['gain']:.3f}  "
          f"d=129-256 {d['repeat d=129-256']['gain']:.3f}")

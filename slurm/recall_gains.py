"""Print the recall gains (repeat d=65-128 and d=129-256) and the context gain (measurement A, loss at k=8 minus
k=128) of runs, with the EMA weights too where recall_ema.json / probe_memory_ema.json exist:
python3 slurm/recall_gains.py runs/X ..."""
import json, os, sys


def gains(r, suffix):
    out = ""
    p = os.path.join(r, f"recall{suffix}.json")
    if os.path.exists(p):
        d = json.load(open(p))["m"]
        out += f"recall d=65-128 {d['repeat d=65-128']['gain']:.3f}  d=129-256 {d['repeat d=129-256']['gain']:.3f}"
    p = os.path.join(r, f"probe_memory{suffix}.json")
    if os.path.exists(p):
        c = json.load(open(p))["m"]["context_loss"]
        out += f"  context {c['8'] - c['128']:.3f}"
    return out


for r in sys.argv[1:]:
    line = f"{os.path.basename(r.rstrip('/')):20s} {gains(r, '')}"
    ema = gains(r, "_ema")
    if ema:
        line += f"   | EMA: {ema}"
    print(line)

"""Print the recall gains (repeat d=65-128 and d=129-256; "net" subtracts the first-occurrence gain, README 5.1)
and the context gain (measurement A, loss at k=8 minus
k=128) of runs, with the EMA weights too where recall_ema.json / probe_memory_ema.json exist, and the gains on
512-token stories where recall512.json exists (slurm/probe_long.sbatch):
python3 slurm/recall_gains.py runs/X ..."""
import json, os, sys


def gains(r, suffix):
    out = ""
    p = os.path.join(r, f"recall{suffix}.json")
    if os.path.exists(p):
        d = json.load(open(p))["m"]
        g0 = d["first occurrence"]["gain"]   # net = gain minus the first-occurrence (control) gain, README 5.1
        out += (f"recall d=65-128 {d['repeat d=65-128']['gain']:.3f}  d=129-256 {d['repeat d=129-256']['gain']:.3f}"
                f"  net d=65-128 {d['repeat d=65-128']['gain'] - g0:.3f} / d=129-256 {d['repeat d=129-256']['gain'] - g0:.3f}")
    p = os.path.join(r, f"probe_memory{suffix}.json")
    if os.path.exists(p):
        c = json.load(open(p))["m"]["context_loss"]
        out += f"  context {c['8'] - c['128']:.3f}"
    p = os.path.join(r, f"recall512{suffix}.json")   # slurm/probe_long.sbatch: the longer stories, read to 512 tokens
    if os.path.exists(p):
        d = json.load(open(p))["m"]
        g0 = d["first occurrence"]["gain"]
        out += (f"  [512-token stories: d=65-128 {d['repeat d=65-128']['gain']:.3f}"
                f"  d=129-256 {d['repeat d=129-256']['gain']:.3f}  d=257-512 {d['repeat d=257-512']['gain']:.3f}"
                f"  net d=129-256 {d['repeat d=129-256']['gain'] - g0:.3f} / d=257-512 {d['repeat d=257-512']['gain'] - g0:.3f}]")
    return out


for r in sys.argv[1:]:
    line = f"{os.path.basename(r.rstrip('/')):20s} {gains(r, '')}"
    ema = gains(r, "_ema")
    if ema:
        line += f"   | EMA: {ema}"
    print(line)

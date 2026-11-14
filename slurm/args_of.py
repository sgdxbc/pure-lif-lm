"""Print the train.py flags that reproduce an earlier run, read from its args.json, so a CPU run can be repeated
on the GPU with exactly the same settings:

  python3 train.py --device cuda --out runs/g1_tf_full $(python3 slurm/args_of.py runs/tf_full) --resume

Leaves out where and how the run executed (out, device, threads, resume); flags given later on the command line
win over these, since argparse keeps the last value.
"""
import json, os, sys

SKIP = {"out", "device", "threads", "resume"}
STORE_TRUE = {"carry_state"}

a = json.load(open(os.path.join(sys.argv[1], "args.json")))
flags = []
for k, v in a.items():
    if k in SKIP or v is None or v == "":
        continue
    if k in STORE_TRUE:
        if v:
            flags.append(f"--{k}")
    elif isinstance(v, bool):
        flags += [f"--{k}", str(v).lower()]
    else:
        flags += [f"--{k}", str(v)]
print(" ".join(flags))

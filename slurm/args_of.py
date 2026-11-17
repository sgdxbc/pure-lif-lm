"""Print the train.py flags that reproduce an earlier run, read from its args.json, so a CPU run can be repeated
on the GPU with exactly the same settings:

  python3 train.py --device cuda --out runs/g1_tf_full $(python3 slurm/args_of.py runs/tf_full) --resume

Leaves out where and how the run executed (out, device, threads, resume); flags given later on the command line
win over these, since argparse keeps the last value.
"""
import json, os, re, sys

SKIP = {"out", "device", "threads", "resume"}
# options removed from model.py (its REMOVED; this script must not import torch): left out when off
REMOVED_OFF = {"rec_rank": 0, "rec_init": None, "delay_k": 0, "delay_rank": None, "leak_gate": False,
               "spike_bus": False, "bus_tau_min": None, "bus_tau_max": None, "bus_theta": None}
# train.py's on/off switches (action="store_true") take no value: given when on, left out when off. Read from
# train.py's source, so a new switch (e.g. --rope) is not missed
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")).read()
STORE_TRUE = set(re.findall(r'add_argument\("--(\w+)", action="store_true"', _src))

a = json.load(open(os.path.join(sys.argv[1], "args.json")))
flags = []
for k, v in a.items():
    if k in SKIP or v is None or v == "":
        continue
    if k in REMOVED_OFF:
        assert REMOVED_OFF[k] is None or v == REMOVED_OFF[k], f"{k}={v}: removed from the code (commit 490e9e6 has it)"
        continue
    if k in STORE_TRUE:
        if v:
            flags.append(f"--{k}")
    elif isinstance(v, bool):
        flags += [f"--{k}", str(v).lower()]
    else:
        flags += [f"--{k}", str(v)]
print(" ".join(flags))

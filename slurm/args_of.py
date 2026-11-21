"""Print the train.py flags that reproduce an earlier run, read from its args.json, so a CPU run can be repeated
on the GPU with exactly the same settings:

  python3 train.py --device cuda --out runs/g1_tf_full $(python3 slurm/args_of.py runs/tf_full) --resume

Leaves out where and how the run executed (out, device, threads, resume); flags given later on the command line
win over these, since argparse keeps the last value.
"""
import ast, json, os, re, sys

SKIP = {"out", "device", "threads", "resume"}
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# options removed from model.py (its REMOVED, read from the source: this script must not import torch) and from
# train.py (spike_soft's schedule, removed with it in 5fcacfd): left out when off
_m = re.search(r"^REMOVED = (\{.*?\})\n", open(os.path.join(_root, "model.py")).read(), re.S | re.M)
REMOVED_OFF = dict(ast.literal_eval(_m.group(1)), soft_hold=0, soft_end=0)
# train.py's on/off switches (action="store_true") take no value: given when on, left out when off. Read from
# train.py's source, so a new switch (e.g. --rope) is not missed
_src = open(os.path.join(_root, "train.py")).read()
STORE_TRUE = set(re.findall(r'add_argument\("--(\w+)", action="store_true"', _src))

a = json.load(open(os.path.join(sys.argv[1], "args.json")))
flags = []
for k, v in a.items():
    if k in SKIP or v is None or v == "":
        continue
    if k == "feat_weight_frac" and a.get("feat_weight") is not None:
        continue   # an explicit --feat_weight was given; train.py does not accept both
    if k in REMOVED_OFF:
        assert REMOVED_OFF[k] is None or v == REMOVED_OFF[k], f"{k}={v}: removed from the code (see model.Config.from_dict for the commit that has it)"
        continue
    if k in STORE_TRUE:
        if v:
            flags.append(f"--{k}")
    elif isinstance(v, bool):
        flags += [f"--{k}", str(v).lower()]
    else:
        flags += [f"--{k}", str(v)]
print(" ".join(flags))

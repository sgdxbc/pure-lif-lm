"""What the plastic synapses learned (docs/experiments.md 5.32): per layer, the decay time constant of each group
(in tokens; initialised log-uniform from mem_tau_min to mem_tau_max), the key and query neurons' mean threshold, and
how strongly the layer's memory feeds the bus (the size of W_o times the read-out gain, relative to the population's
own output synapses W_out). Runs on the CPU from the repository root:
  python3 experiments/mem_stats.py runs/g17_mem138 runs/g17_kdmem138 ...     (moving-average weights when present)
"""
import json, math, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sample import load

for run in sys.argv[1:]:
    ck = os.path.join(run, "ckpt.pt")
    try:
        model, step = load(ck, ema=True)
    except AssertionError:
        model, step = load(ck)
    c = model.cfg
    if not c.mem_heads:
        print(run, "has no plastic synapses"); continue
    print(f"== {run} (step {step}; {c.mem_heads} groups x {c.mem_keys} keys, every {c.mem_every} layer(s); "
          f"initial tau {c.mem_tau_min:g}-{c.mem_tau_max:g} tokens)")
    out = {}
    for i in model.mem_layers:
        mem = model.mems[str(i)]
        lam = torch.sigmoid(mem.decay_beta_logit.float())
        tau = (-1.0 / torch.log(lam)).tolist()
        th = mem.kq_log_theta.float().exp()
        hm = c.mem_heads * c.mem_keys
        wo = (mem.w_out.weight.float() * mem.gain.float()).norm().item()
        wpop = model.layers[i].w_out.weight.float().norm().item()
        per_group = (mem.w_out.weight.float() * mem.gain.float()).view(-1, c.mem_heads, c.mem_dv).norm(dim=(0, 2))
        out[i] = {"tau": tau, "key_theta": th[:hm].mean().item(), "query_theta": th[hm:].mean().item(),
                  "w_o_rel": wo / wpop, "w_o_group": per_group.tolist()}
        print(f"layer {i:2d}  tau " + " ".join(f"{t:7.1f}" for t in tau) +
              f" | theta key {out[i]['key_theta']:.2f} query {out[i]['query_theta']:.2f}"
              f" | |W_o g| / |W_out| {wo / wpop:.3f} | per group " + " ".join(f"{x:.2f}" for x in per_group.tolist()))
    with open(os.path.join(run, "mem_stats.json"), "w") as f:
        json.dump({"step": step, "layers": out}, f, indent=1)

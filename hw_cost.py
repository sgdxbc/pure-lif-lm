"""Hardware cost of a trained network (README 5.20): what one generated token costs on an FPGA or a dataflow
accelerator, with the spiking network's own firing measured on validation text, next to the same figures for a
Transformer. From the repository root:
  python3 hw_cost.py --device cuda [--ema] runs/A runs/B ...     (writes runs/X/hw_cost.json and prints a summary)

Spiking network, per generated token and per stream:
  dense MACs   W_in, the leak-modulating synapses and the readout. Their input, the current bus, is real-valued and
               dense, so every weight is used at every step (T times per token for W_in and the leak synapses).
  sparse adds  W_out: only the outgoing weights of neurons that fired, one addition per weight, no multiplication.
  bytes read   weights read from off-chip memory (an FPGA with HBM, when the weights do not fit on chip): all dense
               weights, plus the W_out row (the outgoing weights) of every neuron that fired at any of the token's T
               steps. Within a token, layer l's T steps need only layers < l, so each weight is read once per token.
               With 16 streams generated together, dense weights are read once for all 16 and the W_out row of every
               neuron that fired in any stream once; divided by 16. Plus one embedding row per stream.
  state        membrane potentials, 24-bit, per stream (they stay on chip).
  With the spiking bus (model.py, spike_bus), W_in and the first factor of the leak synapses read spikes too: they
  move from dense MACs to additions over the active bus channels, and from "all weights" to "the columns of the
  channels whose relay neuron fired"; the relay potentials add to the state.
Transformer (analytic): MACs = weights + attention over c earlier tokens; bytes read = weights (shared by the streams)
  + each stream's key/value history of c tokens (8-bit); state = that history.
Weights are counted at 8 and 4 bits. Firing comes from --windows validation windows of 256 tokens, each read from rest,
taken in groups of 16 as the 16 streams.
"""
import argparse, json, os
import numpy as np
import torch
from model import SpikingLM
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = (256, 2048, 8192)
STREAMS = 16


@torch.no_grad()
def firing(model, windows, seq=256, dev="cpu"):
    """Per layer: firing rate per step ("rate"), share of neurons that fire at least once in a token in one stream
    ("token") and in any of STREAMS streams ("token_any"). With the spiking bus also the same for the bus channels
    (a channel is active when its ON or OFF relay neuron fires): "bus_rate", "bus_token", "bus_token_any"."""
    arr = np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r")
    starts = np.linspace(0, len(arr) - seq - 2, windows).astype(np.int64)
    T, L = model.cfg.steps_per_token, len(model.layers)
    keys = ("rate", "token", "token_any") + (("bus_rate", "bus_token", "bus_token_any") if model.cfg.spike_bus else ())
    f = {k: np.zeros(L) for k in keys}
    n = 0
    for g in range(0, windows - STREAMS + 1, STREAMS):
        x = torch.from_numpy(np.stack([arr[i:i + seq] for i in starts[g:g + STREAMS]]).astype(np.int64)).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            _, _, _, spikes = model(x, return_spikes=True)
        for j, s in enumerate(spikes):
            l, pre = (j, "") if j < L else (j - L, "bus_")
            s = s > 0.5
            if pre:                                          # relay spikes: channel active if ON or OFF fired
                d = s.size(-1) // 2
                s = s[..., :d] | s[..., d:]
            s = s.view(STREAMS, seq, T, -1)
            tok = s.any(2)                                   # fired at any of the token's T steps
            f[pre + "rate"][l] += s.float().mean().item()
            f[pre + "token"][l] += tok.float().mean().item()
            f[pre + "token_any"][l] += tok.any(0).float().mean().item()   # ... in any of the streams
        n += 1
    return {k: v / n for k, v in f.items()}


def snn_cost(model, f):
    c = model.cfg
    V, d, H, L, r, T = c.vocab, c.d_model, c.d_hidden, c.n_layers, c.leak_rank, c.steps_per_token
    assert c.rec_rank == 0 and c.delay_k == 0, "cost model covers neither same-layer nor delayed synapses"
    into = d * H + (d * r if r else 0)                       # W_in and the leak synapses' first factor: read the bus
    leak2 = r * H if r else 0                                # second factor of the leak synapses: dense (rank-r input)
    if c.spike_bus:   # the bus reaches W_in and the leak synapses as spikes: only active channels' columns are used
        dense_step, dense_w = leak2, L * (leak2 + H) + d * V
        bus_adds = float(sum(f["bus_rate"][l] * (H + r) * d * T for l in range(L)))
        bus_rows = lambda k: sum(f[k][l] * (H + r) * d for l in range(L))
    else:
        dense_step, dense_w = into + leak2, L * (into + leak2 + H) + d * V
        bus_adds, bus_rows = 0.0, lambda k: 0.0
    out = {"arch": "snn", "params": model.num_params(), "steps_per_token": T, "spike_bus": c.spike_bus,
           "dense_macs": T * L * dense_step + d * V,
           "sparse_adds": float(sum(f["rate"][l] * H * T * d for l in range(L))) + bus_adds,
           "state_bytes": L * H * 3 + (L * 2 * d * 3 if c.spike_bus else 0),
           "firing": {k: v.tolist() for k, v in f.items()}}
    for bits in (8, 4):
        b = bits / 8
        one = (dense_w + sum(f["token"][l] * H * d for l in range(L)) + bus_rows("bus_token")) * b + d * b
        many = (dense_w + sum(f["token_any"][l] * H * d for l in range(L)) + bus_rows("bus_token_any")) * b / STREAMS + d * b
        out[f"weights_bytes_{bits}bit"] = out["params"] * b
        out[f"bytes_per_token_{bits}bit"] = {"1 stream": one, "%d streams" % STREAMS: many}
    return out


def tf_cost(model):
    c = model.cfg
    V, d, L = c.vocab, c.d_model, c.n_layers
    w = 12 * L * d * d + d * V                               # blocks + head, read every step; + embedding and position rows
    out = {"arch": "tf", "params": sum(p.numel() for p in model.parameters()),
           "macs": {str(k): w + 2 * L * d * k for k in CTX},
           "state_bytes": {str(k): 2 * L * d * k for k in CTX}}
    for bits in (8, 4):
        b = bits / 8
        out[f"weights_bytes_{bits}bit"] = out["params"] * b
        out[f"bytes_per_token_{bits}bit"] = {("%d stream%s, context %d" % (s, "s" if s > 1 else "", k)):
                                             w * b / s + 2 * L * d * k + 2 * d * b for s in (1, STREAMS) for k in CTX}
    return out


def summary(name, o):
    M = 1e6
    j = lambda vals: "/".join("%.1f" % (v / M) for v in vals)
    bt = o["bytes_per_token_8bit"]
    if o["arch"] == "snn":
        many = bt["%d streams" % STREAMS]
        return ("%-24s snn %6.1fM  8-bit %6.1f MB  dense MACs %6.1fM + sparse adds %5.2fM  state %6.1f KB  "
                "MB/token 8-bit: 1 stream %6.1f, %d streams %5.1f  (rate %.3f)"
                % (name, o["params"] / M, o["weights_bytes_8bit"] / M, o["dense_macs"] / M, o["sparse_adds"] / M,
                   o["state_bytes"] / 1e3, bt["1 stream"] / M, STREAMS, many / M, np.mean(o["firing"]["rate"])))
    one = [bt["1 stream, context %d" % k] for k in CTX]
    many = [bt["%d streams, context %d" % (STREAMS, k)] for k in CTX]
    return ("%-24s tf  %6.1fM  8-bit %6.1f MB  at context %s: MACs %sM  state %s MB  MB/token 8-bit: 1 stream %s, "
            "%d streams %s" % (name, o["params"] / M, o["weights_bytes_8bit"] / M, "/".join(map(str, CTX)),
                               j(o["macs"][str(k)] for k in CTX), j(o["state_bytes"][str(k)] for k in CTX),
                               j(one), STREAMS, j(many)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories with ckpt.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true", help="moving-average weights where the checkpoint has them")
    p.add_argument("--windows", type=int, default=64)
    a = p.parse_args()
    dev = torch.device(a.device)
    for run in a.runs:
        args = json.load(open(os.path.join(run, "args.json")))
        has_ema = bool(a.ema and args.get("ema", 0))
        model, _ = load(os.path.join(run, "ckpt.pt"), ema=has_ema, device=dev)
        if isinstance(model, SpikingLM):
            o = snn_cost(model, firing(model, a.windows, dev=dev))
        else:
            o = tf_cost(model)
        o["ema"] = has_ema
        json.dump(o, open(os.path.join(run, "hw_cost.json"), "w"), indent=1)
        print(summary(os.path.basename(run.rstrip("/")), o), flush=True)


if __name__ == "__main__":
    main()

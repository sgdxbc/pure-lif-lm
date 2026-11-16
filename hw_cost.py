"""Hardware cost of a trained network (docs/experiments.md 5.20): what one generated token costs on an FPGA or a dataflow
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
  Plastic synapses (model.py, mem_heads): the key, query and value synapses and W_o are dense MACs, like W_in; each
  key neuron that fires adds the value (mem_dv additions) to its row of S, each query neuron that fires adds its row
  to the readout (sparse adds); the decay of S is one multiplication per synapse and step; S (24-bit) and the key and
  query potentials add to the state. With mem_rule = "delta" each firing key neuron's row is also read before the
  write (another mem_dv additions), the write strength is d MACs per group, and u = b (v - read) is mem_dv
  multiplications per group.
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
def firing(model, windows, seq=256, dev="cpu", data="data"):
    """Per layer: firing rate per step ("rate"), share of neurons that fire at least once in a token in one stream
    ("token") and in any of STREAMS streams ("token_any"); with plastic synapses also the key and query neurons'
    firing rates."""
    arr = np.memmap(os.path.join(HERE, data, "valid.bin"), dtype=np.uint16, mode="r")   # the run's own data
    starts = np.linspace(0, len(arr) - seq - 2, windows).astype(np.int64)
    T, L, M = model.cfg.steps_per_token, len(model.layers), len(getattr(model, "mem_layers", []))
    f = {k: np.zeros(L) for k in ("rate", "token", "token_any")}
    f.update({k: np.zeros(M) for k in (("mem_key_rate", "mem_query_rate") if M else ())})
    n = 0
    for g in range(0, windows - STREAMS + 1, STREAMS):
        x = torch.from_numpy(np.stack([arr[i:i + seq] for i in starts[g:g + STREAMS]]).astype(np.int64)).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            _, _, _, spikes = model(x, return_spikes=True)
        if M:   # key and query spikes of the plastic synapses come last
            for l, s in enumerate(spikes[-M:]):
                hm = s.size(-1) // 2
                f["mem_key_rate"][l] += (s[..., :hm] > 0.5).float().mean().item()
                f["mem_query_rate"][l] += (s[..., hm:] > 0.5).float().mean().item()
            spikes = spikes[:-M]
        for l, s in enumerate(spikes):
            s = (s > 0.5).view(STREAMS, seq, T, -1)
            tok = s.any(2)                                   # fired at any of the token's T steps
            f["rate"][l] += s.float().mean().item()
            f["token"][l] += tok.float().mean().item()
            f["token_any"][l] += tok.any(0).float().mean().item()   # ... in any of the streams
        n += 1
    return {k: v / n for k, v in f.items()}


def snn_cost(model, f):
    c = model.cfg
    V, d, H, L, r, T = c.vocab, c.d_model, c.d_hidden, c.n_layers, c.leak_rank, c.steps_per_token
    into = d * H + (d * r if r else 0)                       # W_in and the leak synapses' first factor: read the bus
    leak2 = r * H if r else 0                                # second factor of the leak synapses: dense (rank-r input)
    dense_step, dense_w = into + leak2, L * (into + leak2 + H) + d * V
    M = len(getattr(model, "mem_layers", []))
    mem_w = mem_macs = mem_adds = mem_state = 0
    if M:
        h, m, dv = c.mem_heads, c.mem_keys, c.mem_dv
        mem_w = M * d * (2 * h * m + 2 * h * dv)                       # key/query, value synapses and W_o: dense
        mem_macs = mem_w + M * (h * m * dv + 3 * h * dv)                # + decay of S, RMS scaling and gain
        kw = 2 if c.mem_rule == "delta" else 1                         # delta: read the key's row, then write it
        mem_adds = float(sum((kw * f["mem_key_rate"][l] + f["mem_query_rate"][l]) * h * m * dv for l in range(M)))
        if c.mem_rule == "delta":
            mem_w += M * d * h                                          # write-strength synapses
            mem_macs += M * (d * h + 2 * h * dv)                        # them, and b (v - read), 1/sqrt(n) scaling
        mem_state = M * (h * m * dv + 2 * h * m) * 3
        dense_w += mem_w
    out = {"arch": "snn", "params": model.num_params(), "steps_per_token": T,
           "dense_macs": T * L * dense_step + d * V + mem_macs,
           "sparse_adds": float(sum(f["rate"][l] * H * T * d for l in range(L))) + mem_adds,
           "state_bytes": L * H * 3 + mem_state,
           "dense_weights": dense_w, "plastic_layers": M,
           "firing": {k: v.tolist() for k, v in f.items()}}
    for bits in (8, 4):
        b = bits / 8
        one = (dense_w + sum(f["token"][l] * H * d for l in range(L))) * b + d * b
        many = (dense_w + sum(f["token_any"][l] * H * d for l in range(L))) * b / STREAMS + d * b
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
            o = snn_cost(model, firing(model, a.windows, dev=dev, data=args.get("data", "data")))
        else:
            o = tf_cost(model)
        o["ema"] = has_ema
        json.dump(o, open(os.path.join(run, "hw_cost.json"), "w"), indent=1)
        print(summary(os.path.basename(run.rstrip("/")), o), flush=True)


if __name__ == "__main__":
    main()

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
  key event adds the value (mem_dv additions) to its row of S, each query event adds its row to the readout.
  A multi-spike count repeats the addition. The decay of S and of the key/query potentials each cost one
  multiplication per state value. Causal delay filters and the output gate add dense arithmetic. S, membrane
  potentials and stored delay currents are each budgeted at 24 bits. Gate SiLU and reciprocal square root use
  lookup tables in this model; their latency and storage are outside the operation and byte estimates.
Transformer (analytic): MACs = weights + attention over c earlier tokens; bytes read = weights (shared by the streams)
  + each stream's key/value history of c tokens (8-bit); state = that history.
Weights are counted at 8 and 4 bits. Firing comes from --windows validation windows of 256 tokens, each read from rest,
taken in groups of 16 as the 16 streams.
"""
import argparse, json, math, os, re
import numpy as np
import torch
from model import SpikingLM
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = (256, 2048, 8192)
STREAMS = 16


def memory_firing(spikes):
    """Mean event count and fraction of rows with at least one event, over tokens and streams."""
    k, q = spikes.chunk(2, dim=-1)
    return {"mem_key_count": k.float().mean().item(), "mem_query_count": q.float().mean().item(),
            "mem_key_rate": (k > 0).float().mean().item(), "mem_query_rate": (q > 0).float().mean().item()}


@torch.no_grad()
def firing(model, windows, seq=256, dev=torch.device("cpu"), data="data"):
    """Per layer: firing rate per step ("rate"), share of neurons that fire at least once in a token in one stream
    ("token") and in any of STREAMS streams ("token_any"); with plastic synapses also the key and query neurons'
    active-row fractions (mem_*_rate) and mean spike counts (mem_*_count)."""
    if windows < STREAMS:
        raise ValueError(f"windows must be at least {STREAMS}")
    arr = np.memmap(os.path.join(HERE, data, "valid.bin"), dtype=np.uint16, mode="r")   # the run's own data
    starts = np.linspace(0, len(arr) - seq - 2, windows).astype(np.int64)
    T, L, M = model.cfg.steps_per_token, len(model.layers), len(getattr(model, "mem_layers", []))
    f = {k: np.zeros(L) for k in ("rate", "token", "token_any")}
    if M:
        f.update({k: np.zeros(M) for k in ("mem_key_rate", "mem_query_rate", "mem_key_count", "mem_query_count")})
    n = 0
    for g in range(0, windows - STREAMS + 1, STREAMS):
        x = torch.from_numpy(np.stack([arr[i:i + seq] for i in starts[g:g + STREAMS]]).astype(np.int64)).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            _, _, _, spikes = model(x, return_spikes=True)
        if M:   # key and query spikes of the plastic synapses come last
            for l, s in enumerate(spikes[-M:]):
                for k, v in memory_firing(s).items():
                    f[k][l] += v
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
    into = d * H + (d * r if r else 0)   # W_in and the leak synapses' first factor: read the bus
    leak2 = r * H if r else 0                                # second factor of the leak synapses: dense (rank-r input)
    # readout: all vocab neurons, or with out_classes the S short-list and C class neurons (shared by all streams) and
    # the m word neurons of the class each stream draws (its own); input: one synapse row per word, or in_groups rows
    C, S = getattr(c, "out_classes", 0), getattr(c, "out_short", 0)
    m = math.ceil((V - S) / C) if C else 0
    head_shared, head_own = (d * (S + C), d * m) if C else (d * V, 0)
    own = head_own + d * max(1, getattr(c, "in_groups", 0))   # weights each stream reads for itself per token
    dense_step, dense_w = into + leak2, L * (into + leak2 + H) + head_shared
    M = len(getattr(model, "mem_layers", []))
    mem_w = mem_macs = mem_adds = mem_state = 0
    if M:
        h, m, dv = c.mem_heads, c.mem_keys, c.mem_dv
        K = max(1, c.mem_kq_levels)
        taps = c.mem_delays if c.mem_delays > 1 else 0
        currents = h * (2 * m + dv)
        gate_channels = {"": 0, "head": h, "channel": h * dv}[c.mem_gate]
        potentials = 0 if c.mem_kq_stateless else 2 * h * m * K
        mem_w = M * (d * (currents + h * dv + gate_channels) + taps * currents)
        # State decay, membrane integration, RMS scaling, gain, and applying the gate.
        mem_macs = mem_w + M * (h * m * dv + potentials + 3 * h * dv + bool(gate_channels) * h * dv)
        if K > 1 and ("mem_key_count" not in f or "mem_query_count" not in f):
            raise ValueError("multi-spike memory needs mean key and query counts")
        keys = f["mem_key_count"] if "mem_key_count" in f else f["mem_key_rate"]
        queries = f["mem_query_count"] if "mem_query_count" in f else f["mem_query_rate"]
        mem_adds = float(sum((k + q) * h * m * dv for k, q in zip(keys, queries)))
        mem_state = M * (h * m * dv + potentials + max(0, taps - 1) * currents) * 3
        dense_w += mem_w
    norm_macs = (T * L + M + 1) * 2 * d if getattr(c, "bus_norm", False) else 0   # bus_norm: squares and scaling
    out = {"arch": "snn", "params": model.num_params(), "steps_per_token": T,
           "dense_macs": T * L * dense_step + head_shared + head_own + mem_macs + norm_macs,
           "sparse_adds": float(sum(f["rate"][l] * H * T * d for l in range(L))) + mem_adds,
           "state_bytes": L * H * 3 + mem_state,
           "dense_weights": dense_w, "per_stream_weights": own, "plastic_layers": M,
           "memory_dense_macs": mem_macs, "memory_sparse_adds": mem_adds, "memory_state_bytes": mem_state,
           "firing": {k: v.tolist() for k, v in f.items()}}
    # word tables: rows used by one word at a time (the input rows; with out_classes the word readout neurons, of which
    # each stream uses one class's block per token), so they can stay in off-chip memory (fpga_est.py)
    tables = ("embed.weight", "head.w_word") if C else ("embed.weight",)
    params = dict(model.named_parameters())
    out["table_weights"] = sum(params[n].numel() for n in tables if n in params)
    if getattr(c, "wbits", 0):   # trained for low-precision weights (train.py --wbits; docs/experiments.md 5.38)
        pats = [re.compile(x) for x in c.wbits8.split(",") if x]
        wb = tb = 0.0
        for name, p in params.items():
            if p.dim() == 2:
                bits = 8 if any(x.search(name) for x in pats) else c.wbits
                g = c.wgroup if c.wgroup and p.shape[1] % c.wgroup == 0 else p.shape[1]
                nb = p.numel() * bits / 8 + p.numel() / g * 2   # integers + one 16-bit scale per group
                wb += nb
                tb += nb if name in tables else 0.0
            else:
                wb += p.numel() * 2                              # per-neuron parameters at 16 bits
        out.update(wbits=c.wbits, wbits8=c.wbits8, weights_bytes_trained=wb, table_bytes_trained=tb)
    for bits in (8, 4):
        b = bits / 8
        one = (dense_w + sum(f["token"][l] * H * d for l in range(L))) * b + own * b
        many = (dense_w + sum(f["token_any"][l] * H * d for l in range(L))) * b / STREAMS + own * b
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
            c = model.cfg   # options that make it not a spiking network: skipped, so the other runs still go through
            if getattr(c, "mem_kind", "plastic") != "plastic":
                print(f"{run}: mem_kind gla, not a spiking network: no hardware cost", flush=True)
                continue
            o = snn_cost(model, firing(model, a.windows, dev=dev, data=args.get("data", "data")))
        else:
            c = model.cfg   # tf_cost covers the plain Transformer only; the others: fpga_est.tf_hw (experiments/fpga_compare.py)
            if (getattr(c, "attn", "softmax") != "softmax" or getattr(c, "mlp", "gelu") != "gelu"
                    or getattr(c, "relay", "none") != "none" or getattr(c, "wq", "none") != "none"
                    or getattr(c, "out_classes", 0) or getattr(c, "mlp_mult", 4) != 4):
                print(f"{run}: linear attention, 0/1 units, relays, ternary weights, a two-level readout or a wider MLP: "
                      "use experiments/fpga_compare.py", flush=True)
                continue
            o = tf_cost(model)
        o["ema"] = has_ema
        json.dump(o, open(os.path.join(run, "hw_cost.json"), "w"), indent=1)
        print(summary(os.path.basename(run.rstrip("/")), o), flush=True)


if __name__ == "__main__":
    main()

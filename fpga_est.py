"""Estimated inference cost of a trained spiking network on an FPGA (README section 0, stage goal 3), to set against
the Transformer measured on a GPU (bench_gpu.py). tf_hw / estimate_tf give the same estimate for a Transformer or a
linear-attention network on the same FPGA (experiments/fpga_compare.py; docs/experiments.md 5.51). An estimate from the network's measured firing (hw_cost.py ->
hw_cost.json) and the device's published figures, with the assumptions below, until a prototype exists.
From the repository root:
  python3 fpga_est.py runs/X [--bits 8] [--device u55c --cards 2] [--compare runs/bench_gpu/NVIDIA_H100_NVL_gpt2.json]
  (writes runs/X/fpga_est.json for one V80, runs/X/fpga_est_u55cx2.json etc. for the others)

Devices (--device; the figures below can each be changed by an option):
  v80   AMD Alveo V80 (product page, data sheet DS1013): HBM2e 820 GB/s, 673 Mb (84 MB) of on-chip memory, 10,848 DSP58
        slices, each one 3-element INT8 dot product per cycle (its INT8 vector mode), at 400 MHz: 13.0 TMAC/s peak;
        thermal design power up to 190 W. Not in the AMD HACC clusters (the default, as before).
  u55c  AMD Alveo U55C (data sheet DS978; HACC@NUS has 6, two per server): HBM2 460 GB/s, 70.9 Mb of block RAM + 270 Mb
        of UltraRAM = 341 Mb (42.6 MB) of on-chip memory (the 36.7 Mb of LUT memory not counted), 9,024 DSP48E2 slices,
        each two INT8 multiply-accumulates per cycle (two weights packed against one shared input), at 300 MHz (the
        Vitis platforms' default kernel clock): 5.4 TMAC/s peak; 150 W total card load (thermal design power 115 W).
  --cards N: N boards as one pipeline (layers split between them, only d_model values per token and stream pass from
  one to the next): on-chip memory, compute, bandwidth and power all N times one board's. With fewer streams than
  boards the pipeline is not full: B < N streams keep only B boards busy at a time, so compute and bandwidth are
  min(B, N)/N of the N boards' (one stream on two boards is as fast as on one, at twice the power; docs 5.43 made this
  correction by hand, and the estimates of 5.51-5.54 lacked it until the second self-audit, docs/audits.md).
Assumptions (all can be changed by options):
  - the additions driven by spikes (W_out) are counted on the same units as the multiply-accumulates, although they
    need no multiplier;
  - 60% of the peak bandwidth and of the peak compute are reached;
  - the board draws its full 190 W whenever it runs (no credit for lighter load; the GPU is credited with its
    measured power instead, so this side is the conservative one);
  - weights stay on chip when they and the streams' membrane potentials fit in 90% of the on-chip memory; otherwise
    weights are read from HBM every step, and the potentials stay on chip if they fit, else are read and written
    back each step. The word tables (hw_cost.py table_weights: the input rows and, with the two-level readout, the
    word readout neurons) do not count against the on-chip memory: a stream uses only a few of their rows per token
    (the word's input rows, its class's block of word neurons), read from HBM each step;
  - B streams are generated together, one token each per step: dense weights (W_in, leak synapses, readout) are read
    once per step for all streams, the W_out row of a neuron once if it fired in any stream. The share of neurons
    that fire in any of B streams is 1 - (1 - q)^B, with q set per layer so that it equals the measured share for 16
    streams (for 1 stream the measured one-stream share is used);
  - time per step = the larger of reading time and compute time (the two overlap). Per stream, a token takes one
    step: that is the latency. The membrane potentials make the cost independent of the context length.
"""
import argparse, json, math, os

STREAMS = (1, 4, 16, 64, 256)
DEVICES = {"v80": dict(bw_gbs=820.0, onchip_mb=84.0, dsp=10848, macs_per_dsp=3, clock_mhz=400.0, watts=190.0),
           "u55c": dict(bw_gbs=460.0, onchip_mb=42.6, dsp=9024, macs_per_dsp=2, clock_mhz=300.0, watts=150.0)}


def estimate(hw, args, bits, dev):
    V, d, H, L, r = args["vocab"], args["d_model"], args["d_hidden"], args["n_layers"], args.get("leak_rank", 0)
    b = bits / 8
    f = hw["firing"]
    dense_w = hw.get("dense_weights", L * (d * H + (d * r + r * H if r else 0) + H) + d * V)   # as in hw_cost.py
    weights, table, own = hw["params"] * b, hw.get("table_weights", 0) * b, hw.get("per_stream_weights", d) * b
    if hw.get("weights_bytes_trained") and bits == hw.get("wbits"):   # the bits it was trained for, scales included
        weights, table = hw["weights_bytes_trained"], hw["table_bytes_trained"]
        own *= table / (hw["table_weights"] * b) if hw.get("table_weights") else 1.0
    ops = hw["dense_macs"] + dev.get("add_cost", 1.0) * hw["sparse_adds"]    # per token and stream (add_cost: below)
    n = dev["cards"]
    rate = n * dev["dsp"] * dev["macs_per_dsp"] * dev["clock_mhz"] * 1e6 * dev["eff"]   # MAC/s reached
    bw = n * dev["bw_gbs"] * 1e9 * dev["eff"]
    budget = n * 0.9 * dev["onchip_mb"] * 1e6
    rows = []
    for B in STREAMS:
        busy = min(B, n) / n                    # share of the pipeline's boards at work (module docstring)
        state = B * hw["state_bytes"]
        if weights - table + state <= budget:
            read, where = (B * own, "weights on chip, word tables in HBM") if table else (0.0, "weights on chip")
        else:
            def share(l):
                if B == 1:
                    return f["token"][l]
                q = 1 - (1 - f["token_any"][l]) ** (1 / 16)
                return 1 - (1 - q) ** B
            read = (dense_w + sum(share(l) * H * d for l in range(L))) * b + B * own
            where = "weights in HBM"
            if state > budget:
                read += 2 * state
                where += ", potentials in HBM"
        t = max(read / (bw * busy), B * ops / (rate * busy))
        rows.append({"streams": B, "where": where, "bytes_per_step": read, "ops_per_step": B * ops,
                     "bound": "memory" if read / bw >= B * ops / rate else "compute",
                     "tok_s": B / t, "ms_per_token": t * 1000, "mJ_per_token": n * dev["watts"] * t / B * 1000})
    return rows


def tf_hw(args, rates=None, readout=None, ctx=1024, bits=4):
    """The figures hw_cost.py gives for a spiking network, for a Transformer or a linear-attention network
    (model_tf.py), from its shape (args.json) and, where it has 0/1 units or relays, their measured rates (the
    val_rates its training log records: model_tf.GPT.rates). Per token and stream:
      attn softmax  every weight once, plus attention over the key/value cache of ctx earlier tokens (2 L d ctx
                    multiply-adds); the cache (8-bit, 2 L d ctx bytes) is the per-stream state and is read every step
      attn linear   every weight once, plus the decay, write and read of each head's hd x hd state (3 L d hd); the
                    state (24-bit, like the spiking network's potentials, with the convolution's last 3 inputs) is
                    fixed in size
      mlp spike/lif the MLP's second matrix reads 0/1 units: additions of the rows of the units that are 1; lif adds a
                    24-bit potential per unit to the state
      relay lif/quant every block matrix reads 3-valued spikes: additions or subtractions of the rows of the input
                    channels that fired (the q/k/v/gate/decay matrices together, the output projection, the MLP's
                    first matrix); lif adds two 24-bit potentials per channel and relay to the state. With
                    relay_levels > 1 the rate is spikes per channel, and each spike adds its row once (and is
                    counted as reading it once: conservative, a channel with several spikes reads its row once)
      wq ternary    the block matrices' weights take 2 bits (values -1, 0, +1) instead of `bits`, and each of their
                    weights costs an addition (dense_adds) instead of a multiply-add
    The input rows (and learned positions) are word tables: one row per token, read from HBM. readout (S, C) models
    unfiltered, unit-temperature sampling from a two-level head: S shortlist rows and C class rows shared across
    streams, plus up to one word block per stream (m padded rows, none for a shortlist draw). The estimate budgets
    a block for every draw. With readout=None, count the run's configured head, flat or two-level. Byte counts at
    `bits` per weight cover the matrices."""
    V, d, L, h = args["vocab"], args["d_model"], args["n_layers"], args.get("n_heads") or 12
    hd, lin, mlp = d // h, args.get("attn") == "linear", args.get("mlp") or "gelu"
    H = (args.get("mlp_mult") or 4) * d                              # the MLP's hidden units
    relay, tern = args.get("relay") or "none", (args.get("wq") or "none") == "ternary"
    b, bm = bits / 8, (2 / 8 if tern else bits / 8)                 # bytes per weight: others, block matrices
    r = list(rates) if rates is not None else []
    mlp_r = r[:L] if mlp != "gelu" else []
    rel = r[len(mlp_r):len(mlp_r) + 3 * L] if relay != "none" else []
    if mlp != "gelu" and len(mlp_r) < L:
        mlp_r = [0.03] * L                                           # no measurement: 3% (g38_laspk ended at 3.5%)
    if relay != "none" and len(rel) < 3 * L:
        rel = [0.3] * (3 * L)                                        # no measurement: 30%, about their rate at start
    in_cols = 3 * d + d + h if lin else 3 * d                        # what the stream feeds: q, k, v (gate, decay)
    out_cols, small = d, (12 * d if lin else 0)                      # output projection; the convolution
    if readout is None and args.get("out_classes", 0):
        readout = (args.get("out_short", 0), args["out_classes"])
    if readout:
        S, C = readout
        m = math.ceil((V - S) / C)
        head_shared, head_own, head_table = d * (S + C), d * m, d * C * m
    else:
        head_shared, head_own, head_table = d * V, 0, 0
    block_dense = L * ((0 if relay != "none" else d * (in_cols + out_cols + H))
                       + (0 if mlp != "gelu" else H * d))            # block matrices read densely
    sparse = []                                                     # (rate, input rows, bytes per row)
    for l in range(L):
        if relay != "none":
            ri, ro, rm = rel[3 * l:3 * l + 3]
            sparse += [(ri, d, in_cols * bm), (ro, d, out_cols * bm), (rm, d, H * bm)]
        if mlp != "gelu":
            sparse.append((mlp_r[l], H, d * bm))
    block_all = L * d * (in_cols + out_cols + 2 * H)
    table = V * d + (0 if lin else args.get("seq", 1024) * d) + head_table
    params = block_all + L * small + table + head_shared + L * 2 * d + d
    state_vals = L * (h * hd * hd + 9 * d) if lin else 0
    state_vals += L * H if mlp == "lif" else 0
    state_vals += L * 3 * 2 * d if relay == "lif" else 0
    kv = 0 if lin else 2 * L * d * ctx
    return {"arch": "la" if lin else "tf", "mlp": mlp, "relay": relay, "wq": "ternary" if tern else "none",
            "ctx": None if lin else ctx, "readout": readout, "bits": bits, "params": params,
            "dense_macs": (0 if tern else block_dense) + L * small + head_shared + head_own
                          + (3 * L * d * hd if lin else 2 * L * d * ctx),
            "dense_adds": block_dense if tern else 0,
            "sparse_adds": sum(rt * rows * (rb / bm) for rt, rows, rb in sparse),
            "weight_bytes_onchip": block_all * bm + (L * small + head_shared + L * 2 * d + d) * b,
            "dense_bytes": block_dense * bm + (L * small + head_shared) * b,
            "sparse_layers": sparse, "per_stream_bytes": (head_own + d) * b,
            "table_weights": table, "state_bytes": 3 * state_vals, "kv_bytes": kv}


def estimate_tf(t, dev):
    """estimate() for tf_hw's figures (bytes at the bits given to tf_hw), with the same assumptions; the key/value
    cache counts against the on-chip memory like the state, and is read (not written back) from HBM when it does not
    fit. The share of a sparse layer's inputs that fire in any of B streams is 1 - (1 - rate)^B (independent
    streams). dev["add_cost"] (default 1): the cost of an addition (a spike's row, a ternary weight) in multiply-adds'
    time; 1 counts both on the DSPs, as every figure in docs/experiments.md before 5.52 does. Additions can instead
    go to the LUTs, which would make them cheaper, but this is not designed or modelled: values below 1 are a
    sensitivity check, not an estimate."""
    ops = t["dense_macs"] + dev.get("add_cost", 1.0) * (t["sparse_adds"] + t.get("dense_adds", 0))
    n = dev["cards"]
    rate = n * dev["dsp"] * dev["macs_per_dsp"] * dev["clock_mhz"] * 1e6 * dev["eff"]
    bw = n * dev["bw_gbs"] * 1e9 * dev["eff"]
    budget = n * 0.9 * dev["onchip_mb"] * 1e6
    rows = []
    for B in STREAMS:
        busy = min(B, n) / n                    # share of the pipeline's boards at work (module docstring)
        state, kv = B * t["state_bytes"], B * t["kv_bytes"]
        own = B * t["per_stream_bytes"]
        if t["weight_bytes_onchip"] + state + kv <= budget:
            read, where = own, "weights on chip, word tables in HBM"
        else:
            # share of rows read in any of B streams; with relay_levels > 1 the rate is spikes per channel and may
            # exceed 1, while a row is read at most once per step: the share of channels with spikes is at most min(r, 1)
            read = (t["dense_bytes"] + sum((1 - (1 - min(r, 1.0)) ** B) * n_in * rb for r, n_in, rb in t["sparse_layers"])
                    + own)
            where = "weights in HBM"
            if state + kv > budget:
                read += 2 * state + kv
                where += ", state in HBM"
        tm = max(read / (bw * busy), B * ops / (rate * busy))
        rows.append({"streams": B, "where": where, "bytes_per_step": read, "ops_per_step": B * ops,
                     "bound": "memory" if read / bw >= B * ops / rate else "compute",
                     "tok_s": B / tm, "ms_per_token": tm * 1000, "mJ_per_token": n * dev["watts"] * tm / B * 1000})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--bits", type=int, default=0, help="weight bits; default: the network's own (train.py --wbits), else 8 "
                   "(8 is lossless, docs/experiments.md 5.20; 4 needs training for it, 5.38)")
    p.add_argument("--device", default="v80", choices=sorted(DEVICES))
    p.add_argument("--cards", type=int, default=1, help="boards in one pipeline")
    for k, t in (("bw_gbs", float), ("onchip_mb", float), ("dsp", int), ("macs_per_dsp", int), ("clock_mhz", float),
                 ("watts", float)):
        p.add_argument("--" + k, type=t, default=None, help="default: the device's")
    p.add_argument("--eff", type=float, default=0.6, help="share of peak bandwidth and compute reached")
    p.add_argument("--out", default="", help="default: runs/X/fpga_est.json for the V80, runs/X/fpga_est_DEVICExN.json "
                   "otherwise")
    p.add_argument("--compare", nargs="*", default=[], help="bench_gpu.py results to print alongside")
    a = p.parse_args()
    hw = json.load(open(os.path.join(a.run, "hw_cost.json")))
    args = json.load(open(os.path.join(a.run, "args.json")))
    a.bits = a.bits or args.get("wbits", 0) or 8
    assert hw["arch"] == "snn" and not hw.get("spike_bus"), "for spiking networks without the spiking bus"
    dev = dict(DEVICES[a.device], name=a.device, cards=a.cards, eff=a.eff)
    dev.update({k: getattr(a, k) for k in DEVICES[a.device] if getattr(a, k) is not None})
    rows = estimate(hw, args, a.bits, dev)
    out = a.out or os.path.join(a.run, "fpga_est.json" if (a.device, a.cards) == ("v80", 1) else
                                f"fpga_est_{a.device}x{a.cards}.json")
    json.dump({"device": dev, "bits": a.bits, "params": hw["params"], "rows": rows}, open(out, "w"), indent=1)
    print(f"{os.path.basename(a.run.rstrip('/'))}: {hw['params'] / 1e6:.0f}M parameters, {a.bits}-bit weights, "
          f"{(hw['dense_macs'] + hw['sparse_adds']) / 1e6:.0f}M operations per token; estimate on {a.cards} x "
          f"{a.device} (any context length):")
    for r in rows:
        print(f"  streams {r['streams']:4d}: {r['tok_s']:10,.0f} tok/s, {r['ms_per_token']:6.3f} ms/token, "
              f"{r['mJ_per_token']:7.2f} mJ/token  ({r['bound']}-bound, {r['where']})")
    for c in a.compare:
        g = json.load(open(c))
        print(f"{g['model']} on {g['gpu']} (measured, vLLM):")
        for r in g["rows"]:
            print(f"  streams {r['streams']:4d} context {r['context']:5d}: {r['decode_tok_s']:10,.0f} tok/s, "
                  f"{r['ms_per_token']:6.3f} ms/token, {r['decode_J_per_token'] * 1000:7.2f} mJ/token")


if __name__ == "__main__":
    main()

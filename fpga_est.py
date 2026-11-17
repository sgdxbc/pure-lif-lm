"""Estimated inference cost of a trained spiking network on an FPGA (README section 0, stage goal 3), to set against
the Transformer measured on a GPU (bench_gpu.py). An estimate from the network's measured firing (hw_cost.py ->
hw_cost.json) and the device's published figures, with the assumptions below, until a prototype exists.
From the repository root:
  python3 fpga_est.py runs/X [--bits 8] [--compare runs/bench_gpu/NVIDIA_H100_NVL_gpt2.json]   (writes runs/X/fpga_est.json)

Device (defaults: AMD Alveo V80, product page and datasheet DS1013): HBM2e 820 GB/s, 673 Mb (84 MB) of on-chip
memory, 10,848 DSP58 slices, thermal design power up to 190 W.
Assumptions (all can be changed by options):
  - every DSP58 does one 3-element INT8 dot product per cycle (its INT8 vector mode), at 400 MHz: 13.0 TMAC/s peak;
    the additions driven by spikes (W_out) are counted on the same units, although they need no multiplier;
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


def estimate(hw, args, bits, dev):
    V, d, H, L, r = args["vocab"], args["d_model"], args["d_hidden"], args["n_layers"], args.get("leak_rank", 0)
    b = bits / 8
    f = hw["firing"]
    dense_w = hw.get("dense_weights", L * (d * H + (d * r + r * H if r else 0) + H) + d * V)   # as in hw_cost.py
    weights, table, own = hw["params"] * b, hw.get("table_weights", 0) * b, hw.get("per_stream_weights", d) * b
    if hw.get("weights_bytes_trained") and bits == hw.get("wbits"):   # the bits it was trained for, scales included
        weights, table = hw["weights_bytes_trained"], hw["table_bytes_trained"]
        own *= table / (hw["table_weights"] * b) if hw.get("table_weights") else 1.0
    ops = hw["dense_macs"] + hw["sparse_adds"]                               # per token and stream
    rate = dev["dsp"] * 3 * dev["clock_mhz"] * 1e6 * dev["eff"]              # MAC/s reached
    bw = dev["bw_gbs"] * 1e9 * dev["eff"]
    budget = 0.9 * dev["onchip_mb"] * 1e6
    rows = []
    for B in STREAMS:
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
        t = max(read / bw, B * ops / rate)
        rows.append({"streams": B, "where": where, "bytes_per_step": read, "ops_per_step": B * ops,
                     "bound": "memory" if read / bw >= B * ops / rate else "compute",
                     "tok_s": B / t, "ms_per_token": t * 1000, "mJ_per_token": dev["watts"] * t / B * 1000})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--bits", type=int, default=0, help="weight bits; default: the network's own (train.py --wbits), else 8 "
                   "(8 is lossless, docs/experiments.md 5.20; 4 needs training for it, 5.38)")
    p.add_argument("--bw_gbs", type=float, default=820.0)
    p.add_argument("--onchip_mb", type=float, default=84.0)
    p.add_argument("--dsp", type=int, default=10848)
    p.add_argument("--clock_mhz", type=float, default=400.0)
    p.add_argument("--eff", type=float, default=0.6, help="share of peak bandwidth and compute reached")
    p.add_argument("--watts", type=float, default=190.0)
    p.add_argument("--compare", nargs="*", default=[], help="bench_gpu.py results to print alongside")
    a = p.parse_args()
    hw = json.load(open(os.path.join(a.run, "hw_cost.json")))
    args = json.load(open(os.path.join(a.run, "args.json")))
    a.bits = a.bits or args.get("wbits", 0) or 8
    assert hw["arch"] == "snn" and not hw.get("spike_bus"), "for spiking networks without the spiking bus"
    dev = {k: getattr(a, k) for k in ("bw_gbs", "onchip_mb", "dsp", "clock_mhz", "eff", "watts")}
    rows = estimate(hw, args, a.bits, dev)
    json.dump({"device": dev, "bits": a.bits, "params": hw["params"], "rows": rows},
              open(os.path.join(a.run, "fpga_est.json"), "w"), indent=1)
    print(f"{os.path.basename(a.run.rstrip('/'))}: {hw['params'] / 1e6:.0f}M parameters, {a.bits}-bit weights, "
          f"{(hw['dense_macs'] + hw['sparse_adds']) / 1e6:.0f}M operations per token; FPGA estimate (any context length):")
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

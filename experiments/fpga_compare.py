"""The spiking hardware network, Transformers and linear-attention networks estimated on the same FPGAs, with the
same assumptions (fpga_est.py; docs/experiments.md 5.51). From the repository root:
  python3 experiments/fpga_compare.py [--bits 4] [--ctx 1024]     (prints a table, writes runs/fpga_compare.json)
  python3 experiments/fpga_compare.py --runs g40_la g40_laspk g40_relay ...   only these runs (model_tf.py networks),
      each from its shape and the rates of its last evaluation, with its final loss (EMA where there is one) or, while
      it trains, the last evaluation's (marked with the step)

Rows (validation loss on FineWeb-Edu where the network was trained; "-" for a shape that was not trained):
  snn  g29_hw                the hardware network (4-bit weights, coded input, two-level readout), from its
                             measured firing (runs/g29_hw/hw_cost.json)
  tf   g33_tf31, g33_tf50    the Transformers on either side of g29_hw's quality (5.43), as trained (full readout),
                             and with a hypothetical two-level readout like the spiking network's (8,192 own words +
                             225 classes; its quality cost, about 0.07-0.11 for the spiking network, not included)
  la   the same two shapes   gated linear attention instead of softmax attention (not trained at these sizes; at
                             GPT-2 small's size it was 0.07 better than the Transformer, 5.47)
  la   g36_la163, + spike    GPT-2 small's shape, and the same with 0/1 MLP units (the rate per block measured in
                             g38_laspk's last evaluation when its log is here, else --spike_rate)
  tf   g26_tf163             GPT-2 small's shape, softmax attention, for scale
Weights at --bits for every network (the Transformers' and linear-attention networks' loss at 4 bits was not
measured; the spiking network was trained for it). Contexts: the Transformers' cache holds --ctx tokens.
"""
import argparse, copy, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import fpga_est  # noqa: E402

DEVS = (("u55c", 1), ("u55c", 2), ("v80", 1))
SHOW = (1, 16, 256)


def loss(run):
    for f in ("final_eval_ema.json", "final_eval.json"):
        p = os.path.join(HERE, "runs", run, f)
        if os.path.exists(p):
            return json.load(open(p))["val_loss"]
    return None


def rates_of(run):
    p = os.path.join(HERE, "runs", run, "log.jsonl")
    if not os.path.exists(p):
        return None
    ev = [json.loads(x) for x in open(p) if '"val_rates"' in x]
    return ev[-1]["val_rates"] if ev else None


def last_eval(run):
    """(loss, step) of the last evaluation in a run's log (EMA where there is one), or (None, None)."""
    p = os.path.join(HERE, "runs", run, "log.jsonl")
    ev = [json.loads(x) for x in open(p) if '"val_loss"' in x] if os.path.exists(p) else []
    if not ev:
        return None, None
    return ev[-1].get("val_loss_ema", ev[-1]["val_loss"]), ev[-1]["step"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--spike_rate", type=float, default=0.03, help="0/1 MLP units' rate if g38_laspk's log is absent")
    p.add_argument("--runs", nargs="*", default=None, help="only these model_tf.py runs, with their measured rates")
    a = p.parse_args()
    args = lambda r: json.load(open(os.path.join(HERE, "runs", r, "args.json")))
    nets = []
    if a.runs:
        for r in a.runs:
            ar, lo, note = args(r), loss(r), ""
            if lo is None:
                lo, st = last_eval(r)
                note = f" (step {st})" if st is not None else ""
            nets.append((f"{r}{note}", lo, fpga_est.tf_hw(ar, rates=rates_of(r), ctx=a.ctx, bits=a.bits), ar))
        return table(nets, a, None)
    hw = json.load(open(os.path.join(HERE, "runs/g29_hw/hw_cost.json")))
    nets.append(("snn g29_hw", loss("g29_hw"), hw, args("g29_hw")))
    for r in ("g33_tf31", "g33_tf50"):
        ar = args(r)
        nets.append((f"tf {r}", loss(r), fpga_est.tf_hw(ar, ctx=a.ctx, bits=a.bits), ar))
        nets.append((f"tf {r} + 2-level readout", None, fpga_est.tf_hw(ar, readout=(8192, 225), ctx=a.ctx, bits=a.bits), ar))
        la = dict(copy.deepcopy(ar), attn="linear")
        nets.append((f"la {r} shape", None, fpga_est.tf_hw(la, bits=a.bits), la))
        nets.append((f"la {r} shape + 2-level readout", None, fpga_est.tf_hw(la, readout=(8192, 225), bits=a.bits), la))
    la = args("g36_la163")
    nets.append(("la g36_la163", loss("g36_la163"), fpga_est.tf_hw(la, bits=a.bits), la))
    rs = rates_of("g38_laspk")
    spk = dict(la, mlp="spike")
    nets.append((f"la163 + 0/1 MLP units (rate {'measured' if rs else a.spike_rate})", None,
                 fpga_est.tf_hw(spk, rates=rs or [a.spike_rate] * la["n_layers"], bits=a.bits), spk))
    nets.append(("la163 + 0/1 MLP units + 2-level readout", None,
                 fpga_est.tf_hw(spk, rates=rs or [a.spike_rate] * la["n_layers"], readout=(8192, 225), bits=a.bits), spk))
    tf = args("g26_tf163")
    nets.append(("tf g26_tf163", loss("g26_tf163"), fpga_est.tf_hw(tf, ctx=a.ctx, bits=a.bits), tf))
    table(nets, a, os.path.join(HERE, "runs/fpga_compare.json"))


def table(nets, a, path):
    """Print the table of nets [(name, loss, figures, args)]; write it to path (JSON) unless path is None."""
    out = []
    head = "| network | loss | params | dense MACs | adds | " + " | ".join(
        f"{d} x{n} ms/mJ at {', '.join(map(str, SHOW))} streams" for d, n in DEVS) + " |"
    print(head)
    print("|" + "---|" * (5 + len(DEVS)))
    for name, lo, t, ar in nets:
        cells = []
        rec = {"network": name, "loss": lo, "params": t["params"], "dense_macs": t["dense_macs"],
               "sparse_adds": t["sparse_adds"], "devices": {}}
        for d, n in DEVS:
            dev = dict(fpga_est.DEVICES[d], name=d, cards=n, eff=0.6)
            rows = fpga_est.estimate(t, ar, a.bits, dev) if name.startswith("snn") else fpga_est.estimate_tf(t, dev)
            rec["devices"][f"{d}x{n}"] = rows
            sel = [r for r in rows if r["streams"] in SHOW]
            cells.append(" / ".join(f"{r['ms_per_token']:.3f}, {r['mJ_per_token']:.2f}" for r in sel))
        out.append(rec)
        print(f"| {name} | {'-' if lo is None else f'{lo:.3f}'} | {t['params'] / 1e6:.1f}M | {t['dense_macs'] / 1e6:.1f}M | "
              f"{t['sparse_adds'] / 1e6:.2f}M | " + " | ".join(cells) + " |")
    if path:
        json.dump({"bits": a.bits, "ctx": a.ctx, "rows": out}, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()

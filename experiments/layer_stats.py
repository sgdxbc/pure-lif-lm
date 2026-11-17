"""How much each layer of a trained spiking network fires and how much it contributes (docs/experiments.md 5.37).
In the long runs of the 212M shape the second and third populations end up nearly silent (under 0.5% of their 4,096
neurons fire per token, against 2-3% in the 20,000-step runs), so: per population, the firing rate, how many neurons
never or almost never fire, how concentrated the spikes are on a few neurons, and the size of its output current
relative to the bus; then the validation loss with the output of one population (or one layer's plastic synapses) set
to zero, i.e. how much the network loses without it. The same validation batches for every setting (as
wbits_sens.py), so differences are paired. Moving-average weights where the run has them. From the repository root:
  python3 experiments/layer_stats.py --device cuda runs/A [runs/B ...]     (writes runs/X/layer_stats.json)
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import load
from wbits_sens import batches, val_loss


@torch.no_grad()
def firing(model, X, n, dev):
    """Per population: per-neuron firing rates over the first n batches, and the RMS of its output current W_out s
    and of the bus it adds to (per token, averaged)."""
    L, H = model.cfg.n_layers, model.cfg.d_hidden
    cnt = torch.zeros(L, H, dtype=torch.float64, device=dev)
    out_ms = torch.zeros(L, dtype=torch.float64)
    bus_ms = torch.zeros(L, dtype=torch.float64)
    hooks = []
    for i, layer in enumerate(model.layers):
        def hook(mod, inp, res, i=i):
            out, s, _ = res
            cnt[i] += s.float().sum((0, 1)).double()
            out_ms[i] += out.float().pow(2).mean().item()
            bus_ms[i] += inp[0].float().pow(2).mean().item()
        hooks.append(layer.register_forward_hook(hook))
    tokens = 0
    for x in X[:n]:
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            model(x.to(dev))
        tokens += x.numel()
    for h in hooks:
        h.remove()
    return (cnt / tokens).cpu(), (out_ms / n).sqrt(), (bus_ms / n).sqrt(), tokens


def zeroed(module):
    """Make module's forward return a zero current (its state and spikes unchanged); returns the undo function."""
    orig = module.forward

    def fwd(*a, **k):
        out = orig(*a, **k)
        return (torch.zeros_like(out[0]),) + tuple(out[1:])
    module.forward = fwd
    return lambda: setattr(module, "forward", orig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rate_batches", type=int, default=8, help="batches for the firing statistics")
    p.add_argument("--no_ablate", action="store_true")
    a = p.parse_args()
    dev = torch.device(a.device)
    for run in a.runs:
        args = json.load(open(os.path.join(run, "args.json")))
        model, step = load(os.path.join(run, "ckpt.pt"), ema=bool(args.get("ema")), device=dev, wbits=0)
        model.eval()
        X, Y = batches(args.get("data", "data"), a.batches, a.batch, args.get("seq", 256), dev)
        rate, out_rms, bus_rms, ntok = firing(model, X, min(a.rate_batches, len(X)), dev)
        res = {"step": step, "rate_tokens": ntok, "layers": []}
        print(f"== {run} (step {step}; firing over {ntok:,} tokens)", flush=True)
        print("layer  rate    never  <0.1%  median   top-1% share   |W_out s| / |bus|")
        for i in range(model.cfg.n_layers):
            r = rate[i]
            srt = r.sort(descending=True).values
            k = max(1, len(r) // 100)
            st = {"rate": r.mean().item(), "never": (r == 0).float().mean().item(),
                  "below_0.1%": (r < 1e-3).float().mean().item(), "median": r.median().item(),
                  "top1_share": (srt[:k].sum() / srt.sum().clamp_min(1e-12)).item(),
                  "out_rel": (out_rms[i] / bus_rms[i]).item()}
            res["layers"].append(st)
            print(f"{i:5d}  {st['rate']:.4f}  {st['never']:5.1%}  {st['below_0.1%']:5.1%}  {st['median']:.4f}"
                  f"   {st['top1_share']:5.1%}         {st['out_rel']:.3f}", flush=True)
        if not a.no_ablate:
            base = val_loss(model, X, Y, dev)
            res["full"] = base
            res["tokens"] = int(Y.numel())
            print(f"full network {base:.4f} ({Y.numel():,} tokens); loss increase without:", flush=True)
            for i, layer in enumerate(model.layers):
                undo = zeroed(layer)
                res["layers"][i]["without"] = val_loss(model, X, Y, dev) - base
                undo()
                line = f"  population {i:2d}: {res['layers'][i]['without']:+.4f}"
                if i in model.mem_layers:
                    undo = zeroed(model.mems[str(i)])
                    res["layers"][i]["without_plastic"] = val_loss(model, X, Y, dev) - base
                    undo()
                    line += f"   its plastic synapses: {res['layers'][i]['without_plastic']:+.4f}"
                print(line, flush=True)
        json.dump(res, open(os.path.join(run, "layer_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

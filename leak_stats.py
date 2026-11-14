"""How much do the leak-modulating synapses actually change each neuron's leak, and when?

For every layer with leak_rank > 0, runs validation stories through the model and records the
per-step time constant tau[t] = -1 / ln(beta[t]) of every neuron, compared with its own resting
value tau0 (beta from beta_logit alone, i.e. no input on the leak synapses).

Reports per layer:
  - median over neurons of the spread over time of log10(tau[t])  (0.30 = a factor of 2)
  - share of (neuron, step) where tau[t] is more than 2x longer / 2x shorter than tau0
  - share of steps where a slow neuron (tau0 > 20 steps) drops to tau < 2 steps (a "forget" event)
Layer 0 sees only the current token's embedding, so its leak change is a function of the token;
the tokens that shorten / lengthen layer-0 memory most (averaged over slow neurons) are listed.

  python3 leak_stats.py runs/p4_leak64 runs/p4_leak256 --out runs/leak_stats.json
"""
import argparse, json, os
import numpy as np
import torch
from tokenizers import Tokenizer
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
LOG2 = float(np.log10(2.0))


def tau_of(logit):
    return -1.0 / torch.log(torch.sigmoid(logit).clamp(1e-7, 1 - 1e-7))


@torch.no_grad()
def analyse(run, X, tok):
    model, step = load(os.path.join(run, "ckpt.pt"))
    buses = {}
    hooks = [layer.register_forward_pre_hook(lambda m, a, i=i: buses.__setitem__(i, a[0].detach().float()))
             for i, layer in enumerate(model.layers)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(torch.from_numpy(X))
    for h in hooks:
        h.remove()
    res = {"step": step, "layers": []}
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "leak_in"):
            continue
        base = layer.beta_logit.float()
        mod = layer.leak_out(layer.leak_in(buses[i])).float()             # (B, L, H)
        lt = torch.log10(tau_of(base + mod))                               # log10 tau[t]
        lt0 = torch.log10(tau_of(base))                                    # (H,)
        r = lt - lt0
        slow = tau_of(base) > 20
        forget = (lt[..., slow] < np.log10(2.0)).float().mean().item() if slow.any() else float("nan")
        res["layers"].append({
            "layer": i,
            "median_spread_log10_tau": float(lt.flatten(0, 1).std(0).median()),
            "p90_spread_log10_tau": float(lt.flatten(0, 1).std(0).quantile(0.9)),
            "mean_shift_log10_tau": float(r.mean()),
            "share_2x_longer": float((r > LOG2).float().mean()),
            "share_2x_shorter": float((r < -LOG2).float().mean()),
            "slow_neurons": int(slow.sum()),
            "slow_forget_share": forget,
            "leak_out_norm": float(layer.leak_out.weight.norm()),
        })
    # layer 0: leak change as a function of the current token
    l0 = model.layers[0]
    if hasattr(l0, "leak_in"):
        base = l0.beta_logit.float()
        slow = tau_of(base) > 20
        mod = l0.leak_out(l0.leak_in(model.embed.weight.float())).float()   # (V, H)
        r = (torch.log10(tau_of(base + mod)) - torch.log10(tau_of(base)))[:, slow].mean(1)
        counts = np.bincount(X.ravel(), minlength=r.numel())
        seen = torch.from_numpy(counts >= 20)
        rr = r.masked_fill(~seen, float("nan"))
        order = torch.argsort(rr.nan_to_num(0.0))
        dec = lambda j: tok.id_to_token(int(j))
        res["layer0_shortening_tokens"] = [(dec(j), round(float(r[j]), 3)) for j in order[:15] if seen[j]]
        res["layer0_lengthening_tokens"] = [(dec(j), round(float(r[j]), 3)) for j in order.flip(0)[:15] if seen[j]]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--stories", type=int, default=64)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(2)
    arr = np.asarray(np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r"))
    eos = np.flatnonzero(arr == 0)
    starts, ends = eos[:-1] + 1, eos[1:]
    S = starts[(ends - starts) >= 256][:a.stories]
    X = np.stack([arr[s:s + 256] for s in S]).astype(np.int64)
    tok = Tokenizer.from_file(os.path.join(HERE, "data", "tok4096.json"))
    out = {}
    for run in a.runs:
        r = analyse(run, X, tok)
        out[os.path.basename(run.rstrip("/"))] = r
        print(f"== {run} (step {r['step']})")
        print(f"  {'layer':>5s} {'spread':>7s} {'p90':>6s} {'shift':>7s} {'2x long':>8s} {'2x short':>9s} "
              f"{'slow':>5s} {'forget':>7s} {'|W_out|':>8s}")
        for d in r["layers"]:
            print(f"  {d['layer']:5d} {d['median_spread_log10_tau']:7.3f} {d['p90_spread_log10_tau']:6.3f} "
                  f"{d['mean_shift_log10_tau']:7.3f} {d['share_2x_longer']:8.1%} {d['share_2x_shorter']:9.1%} "
                  f"{d['slow_neurons']:5d} {d['slow_forget_share']:7.1%} {d['leak_out_norm']:8.2f}")
        if "layer0_shortening_tokens" in r:
            print("  layer 0, tokens that shorten slow-neuron memory most:", r["layer0_shortening_tokens"])
            print("  layer 0, tokens that lengthen it most:", r["layer0_lengthening_tokens"])
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

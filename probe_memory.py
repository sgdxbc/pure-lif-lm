"""How far back does a model actually use its context?

Takes validation stories that are at least 193 tokens long and measures the mean loss on story
positions 128..191 when the model is only allowed to start reading k tokens before position 128
(spiking models start from rest, V = 0). If more preceding text lowers the loss, the model is using it.

For spiking models it also reports how often the slow neurons (tau > 50 tokens) fire: with a
reset, every spike wipes (part of) what the membrane had accumulated.

  python3 probe_memory.py base=runs/p1_base/ckpt.pt tf=runs/p1_tf/ckpt.pt --out runs/probe.json
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
P0, P1 = 128, 192
KS = (0, 8, 32, 64, 128)


def stories(n=600):
    arr = np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r")
    eos = np.flatnonzero(np.asarray(arr) == 0)
    starts, ends = eos[:-1] + 1, eos[1:]
    keep = (ends - starts) >= P1 + 1
    return arr, starts[keep][:n]

DEV = torch.device("cpu")


@torch.no_grad()
def context_ablation(model, arr, S):
    out = {}
    for k in KS:
        tot = 0.0
        for b in range(0, len(S), 50):
            s = S[b:b + 50]
            x = torch.from_numpy(np.stack([arr[i + P0 - k:i + P1] for i in s]).astype(np.int64)).to(DEV)
            y = torch.from_numpy(np.stack([arr[i + P0 - k + 1:i + P1 + 1] for i in s]).astype(np.int64)).to(DEV)
            with torch.autocast(DEV.type, dtype=torch.bfloat16):
                logits, _, _, _ = model(x)
            nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
            tot += nll[:, k:].sum().item()
        out[k] = tot / (len(S) * (P1 - P0))
    return out


@torch.no_grad()
def slow_neuron_stats(model, arr, S):
    x = torch.from_numpy(np.stack([arr[i:i + P1] for i in S[:200]]).astype(np.int64)).to(DEV)
    with torch.autocast(DEV.type, dtype=torch.bfloat16):
        _, _, _, spikes = model(x, return_spikes=True)
    gaps, never, rates, frac_slow = [], [], [], []
    T = model.cfg.steps_per_token   # spikes come per simulation step; time constants and gaps are reported in tokens
    for layer, s in zip(model.layers, spikes):
        tau = -1.0 / torch.log(layer.beta().detach().clamp(1e-6, 1 - 1e-6)) / T
        slow = tau > 50
        frac_slow.append(slow.float().mean().item())
        ss = s[:, :, slow].float()
        rates.append(ss.mean().item())
        never.append((ss.sum(1) == 0).float().mean().item())
        for row in ss.permute(0, 2, 1).reshape(-1, ss.size(1)).cpu().numpy():
            t = np.flatnonzero(row)
            if len(t) > 1:
                gaps.append(np.diff(t) / T)
    g = np.concatenate(gaps) if gaps else np.array([np.nan])
    res = {"share_slow": float(np.mean(frac_slow)), "slow_rate": float(np.mean(rates)),
           "slow_never_fire": float(np.mean(never)), "slow_gap_median": float(np.median(g)),
           "slow_gap_p90": float(np.percentile(g, 90)), "slow_gaps_le_10": float(np.mean(g <= 10))}
    if model.cfg.reset_learn:
        gam = torch.cat([l.gamma().detach() for l in model.layers]).cpu().numpy()
        res["gamma_quantiles_10_50_90"] = [float(q) for q in np.percentile(gam, [10, 50, 90])]
        res["gamma_share_below_0.5"] = float(np.mean(gam < 0.5))
    if model.cfg.rec_rank > 0:
        res["rec_weight_rms"] = float(np.mean([(l.rec_A @ l.rec_B).detach().pow(2).mean().sqrt().item()
                                               for l in model.layers]))
        res["w_in_weight_rms"] = float(np.mean([l.w_in.weight.detach().pow(2).mean().sqrt().item()
                                                for l in model.layers]))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="name=path/to/checkpoint")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true", help="use the moving-average weights (train.py --ema)")
    a = p.parse_args()
    global DEV
    DEV = torch.device(a.device)
    torch.set_num_threads(4)
    arr, S = stories()
    results = {}
    print(f"mean loss on story positions {P0}-{P1 - 1} when reading starts k tokens before {P0} "
          f"({len(S)} stories)")
    print(f"{'model':16s}" + "".join(f"   k={k:<4d}" for k in KS) + "   gain 8->128")
    for spec in a.models:
        name, path = spec.split("=", 1)
        model, _ = load(path, ema=a.ema, device=DEV)
        r = {"context_loss": context_ablation(model, arr, S)}
        c = r["context_loss"]
        print(f"{name:16s}" + "".join(f"   {c[k]:.3f} " for k in KS) + f"   {c[8] - c[128]:.3f}", flush=True)
        if hasattr(model, "layers"):
            r["slow_neurons"] = slow_neuron_stats(model, arr, S)
        results[name] = r
    for name, r in results.items():
        if "slow_neurons" in r:
            print(f"{name}: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in r["slow_neurons"].items()))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

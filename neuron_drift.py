"""Did the per-neuron dynamics parameters (tau, theta, gamma) actually learn, or stay near their init?

Rebuilds each run's initial weights (train.py seeds with --seed right before building the model),
compares them with the checkpoint, and reads Adam's moment estimates: |m| / sqrt(v) is near 1 when a
parameter's gradient keeps the same sign step after step, and about 0.15 when it is pure noise
(beta1 = 0.9, beta2 = 0.95).

  python3 neuron_drift.py runs/p1_reset runs/p3_g09_x30 --out runs/drift.json
"""
import argparse, json, os
import numpy as np
import torch
from model import Config, SpikingLM

KEYS = [("beta_logit", "log10_tau"), ("log_theta", "theta"), ("gamma_logit", "gamma")]


def transform(key, x):
    if key == "beta_logit":
        return torch.log10(-1.0 / torch.log(torch.sigmoid(x).clamp(1e-6, 1 - 1e-6)))
    if key == "log_theta":
        return torch.exp(x)
    return torch.sigmoid(x)


def param_order(model, args):
    """Order of parameters in the optimizer state, matching the groups built in train.py."""
    mult = args.get("neuron_lr_mult", 1.0)
    is_neuron = lambda n: mult != 1.0 and n.endswith(tuple(k for k, _ in KEYS))
    named = list(model.named_parameters())
    decay = [n for n, p in named if p.dim() >= 2 and p.requires_grad]
    no_decay = [n for n, p in named if p.dim() < 2 and p.requires_grad and not is_neuron(n)]
    neuron = [n for n, p in named if p.requires_grad and is_neuron(n)] if mult != 1.0 else []
    return decay + no_decay + neuron


def analyse(run):
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cpu")
    args = json.load(open(os.path.join(run, "args.json")))
    cfg = Config.from_dict(ck["cfg"])
    torch.manual_seed(args.get("seed", 0))
    model = SpikingLM(cfg)
    init, final = model.state_dict(), ck["model"]
    order = param_order(model, args)
    state = ck["opt"]["state"]
    res = {"step": ck["step"], "neuron_lr_mult": args.get("neuron_lr_mult", 1.0)}
    for key, name in KEYS:
        names = [n for n in order if n.endswith(key)]
        if not names:
            continue
        a = torch.cat([transform(key, init[n].float()) for n in names]).numpy()
        b = torch.cat([transform(key, final[n].float()) for n in names]).numpy()
        ratio = np.concatenate([(state[order.index(n)]["exp_avg"].abs() /
                                 (state[order.index(n)]["exp_avg_sq"].sqrt() + 1e-12)).numpy() for n in names])
        with np.errstate(invalid="ignore", divide="ignore"):   # constant init (e.g. gamma 0.9) has no correlation
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 else float("nan")
        d = {"corr_init_final": corr,
             "median_abs_change": float(np.median(np.abs(b - a))),
             "init_quantiles_10_50_90": [float(q) for q in np.percentile(a, [10, 50, 90])],
             "final_quantiles_10_50_90": [float(q) for q in np.percentile(b, [10, 50, 90])],
             "adam_consistency_median": float(np.median(ratio)),
             "adam_consistency_p90": float(np.percentile(ratio, 90))}
        if key == "gamma_logit":
            d["share_below_0.5_init"] = float(np.mean(a < 0.5))
            d["share_below_0.5_final"] = float(np.mean(b < 0.5))
        res[name] = d
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(2)
    out = {}
    for run in a.runs:
        r = analyse(run)
        out[os.path.basename(run.rstrip("/"))] = r
        print(f"== {run}  (step {r['step']}, neuron lr x{r['neuron_lr_mult']:g})")
        for _, name in KEYS:
            if name in r:
                d = r[name]
                q0, q1 = d["init_quantiles_10_50_90"], d["final_quantiles_10_50_90"]
                extra = (f"  below 0.5: {d['share_below_0.5_init']:.1%} -> {d['share_below_0.5_final']:.1%}"
                         if name == "gamma" else "")
                print(f"  {name:10s} corr {d['corr_init_final']:.3f}  median|change| {d['median_abs_change']:.3f}"
                      f"  p10/50/90 [{q0[0]:.2f} {q0[1]:.2f} {q0[2]:.2f}] -> [{q1[0]:.2f} {q1[1]:.2f} {q1[2]:.2f}]"
                      f"  adam |m|/sqrt(v) {d['adam_consistency_median']:.2f}{extra}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Is it the per-step change of the leak that helps, or only a different average leak?

Evaluates a trained model with leak-modulating synapses three ways on the same validation windows:
  as trained        - beta[t] = sigmoid(beta_logit + M[t])
  average leak      - M[t] replaced by each neuron's average of M over all windows and steps
                      (so each neuron keeps its learned average time constant, but it no longer changes)
  resting leak      - M = 0 (the leak the neuron has without any input on the leak synapses)

  python3 leak_ablate.py runs/p4_leak64 runs/p4_leak64_seed1 --out runs/leak_ablate.json
"""
import argparse, json, os
import numpy as np
import torch
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def loss_of(model, X):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model(X[:, :-1].contiguous(), targets=X[:, 1:].contiguous())[1].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(2)
    arr = np.asarray(np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r"))
    ix = np.random.default_rng(0).integers(0, len(arr) - 257, a.windows)
    X = torch.from_numpy(np.stack([arr[i:i + 257] for i in ix]).astype(np.int64))
    out = {}
    for run in a.runs:
        model, step = load(os.path.join(run, "ckpt.pt"))
        layers = [l for l in model.layers if hasattr(l, "leak_out")]
        res = {"as trained": loss_of(model, X)}
        sums = {}
        hooks = [l.leak_out.register_forward_hook(
            lambda m, i, o, k=k: sums.__setitem__(k, o.float().mean((0, 1)))) for k, l in enumerate(layers)]
        loss_of(model, X)
        for h in hooks:
            h.remove()
        for name, fill in (("average leak", lambda k, o: sums[k].to(o.dtype).expand_as(o)),
                           ("resting leak", lambda k, o: torch.zeros_like(o))):
            hooks = [l.leak_out.register_forward_hook(lambda m, i, o, k=k, f=fill: f(k, o))
                     for k, l in enumerate(layers)]
            res[name] = loss_of(model, X)
            for h in hooks:
                h.remove()
        out[os.path.basename(run.rstrip("/"))] = res
        print(f"{run} (step {step}): " + "  ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

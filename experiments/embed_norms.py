"""How large the input rows (the synapses of the input neurons, i.e. the word vectors) and a few other weight matrices
are after training (docs/experiments.md 5.37). layer_stats.py found that in the long runs the first population's output
current is 7-19 times the size of the input current it is added to (0.7 times in the 20,000-step runs). One
explanation: the input rows are weight-decayed (train.py --wd, 0.1, on every 2-D weight) and there is no normalisation
anywhere, so over a long training at a high learning rate the rows of all but the most frequent words shrink, and the
first population has to carry the word itself. This prints the root-mean-square size of the input rows by how often the
word occurs in the validation set, against their initial size (1.0), and of the first population's W_in and W_out and
the readout. Reads only the checkpoint (moving-average weights where present). From the repository root, on a CPU:
  python3 experiments/embed_norms.py runs/A [runs/B ...]     (writes runs/X/embed_norms.json)
"""
import json, os, sys
import numpy as np
import torch

BUCKETS = [(0, 1000), (1000, 8000), (8000, 50304)]   # words ranked by validation-set frequency


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


def main():
    counts = {}
    for run in sys.argv[1:]:
        args = json.load(open(os.path.join(run, "args.json")))
        data = args.get("data", "data")
        if data not in counts:
            arr = np.memmap(os.path.join(data, "valid.bin"), dtype=np.uint16, mode="r")
            counts[data] = np.bincount(arr, minlength=args.get("vocab", 50304))
        cnt = counts[data]
        ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cpu", weights_only=False)
        sd = dict(ck["model"])
        if "ema" in ck:
            sd.update({k: v for k, v in ck["ema"].items() if k in sd})
        res = {"step": ck.get("step"), "ema": "ema" in ck}
        line = f"{run} (step {res['step']}):"
        if "embed.weight" in sd:
            E = sd["embed.weight"].float()
            V = E.shape[0]
            order = np.argsort(-cnt[:V], kind="stable")
            row = E.pow(2).mean(1).sqrt()   # RMS of each row
            res["input_rows"] = rms(E)
            line += f" input rows {res['input_rows']:.3f} (by frequency rank"
            for lo, hi in BUCKETS:
                ids = torch.from_numpy(order[lo:min(hi, V)].copy())
                if len(ids):
                    v = row[ids].mean().item()
                    res[f"input_rows_{lo}_{hi}"] = v
                    line += f" {lo}-{hi}: {v:.3f}"
            unseen = torch.from_numpy(np.nonzero(cnt[:V] == 0)[0])
            res["input_rows_unseen"] = row[unseen].mean().item() if len(unseen) else None
            line += f"; not in valid: {res['input_rows_unseen']:.3f})" if len(unseen) else ")"
        else:
            line += " coded input (no per-word rows)"
        for name in ["layers.0.w_in.weight", "layers.0.w_out.weight", "layers.1.w_out.weight", "head.weight"]:
            if name in sd:
                res[name] = rms(sd[name])
                line += f" | {name} {res[name]:.4f}"
        print(line, flush=True)
        json.dump(res, open(os.path.join(run, "embed_norms.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

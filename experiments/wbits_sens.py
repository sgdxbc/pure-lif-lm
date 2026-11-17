"""Which weight matrices lose the most when rounded to 4 bits (docs/experiments.md 5.38). For each run given: the
validation loss of the full-precision network (moving-average weights where the run has them), then with only one
kind of weight matrix rounded (model.py quantize_weight), all of them, and all of them but the most sensitive kinds
kept at 8 bits. The same validation batches for every setting (as train.py's evaluation), so differences are paired.
From the repository root:
  python3 experiments/wbits_sens.py --device cuda runs/A [runs/B ...]     (writes runs/X/wbits_sens.json)
"""
import argparse, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import quantize_weight
from sample import load

KINDS = {   # name -> pattern on the parameter name (2-D parameters only)
    "input rows": r"^embed\.",
    "readout": r"^head\.",
    "W_in": r"^layers\.\d+\.w_in\.",
    "W_out": r"^layers\.\d+\.w_out\.",
    "leak synapses": r"\.leak_(in|out)\.",
    "plastic: key/query in": r"^mems\.\d+\.kq_in\.",
    "plastic: value in": r"^mems\.\d+\.v_in\.",
    "plastic: out": r"^mems\.\d+\.w_out\.",
}


def batches(data, n, B, seq, dev):
    arr = np.memmap(os.path.join(data, "valid.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1234)
    ix = rng.integers(0, len(arr) - seq - 1, n * B)
    x = np.stack([arr[i:i + seq] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + seq + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).view(n, B, seq), torch.from_numpy(y).view(n, B, seq)


@torch.no_grad()
def val_loss(model, X, Y, dev):
    tot = 0.0
    for x, y in zip(X, Y):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            logits = model(x.to(dev))[0]
        tot += F.cross_entropy(logits.float().flatten(0, 1), y.to(dev).flatten()).item()
    return tot / len(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--group", type=int, default=128)
    a = p.parse_args()
    dev = torch.device(a.device)
    for run in a.runs:
        args = json.load(open(os.path.join(run, "args.json")))
        model, _ = load(os.path.join(run, "ckpt.pt"), ema=bool(args.get("ema")), device=dev, wbits=0)
        X, Y = batches(args.get("data", "data"), a.batches, a.batch, args.get("seq", 256), dev)
        full = {n: q.detach().clone() for n, q in model.named_parameters() if q.dim() == 2}
        kind = {n: next((k for k, pat in KINDS.items() if re.search(pat, n)), "other") for n in full}

        def setting(bits_of):   # bits_of(name) -> bits (0 = unrounded); returns the loss
            with torch.no_grad():
                for n, q in model.named_parameters():
                    if n in full:
                        q.copy_(quantize_weight(full[n].float(), bits_of(n), a.group).to(q.dtype))
            return val_loss(model, X, Y, dev)

        base = setting(lambda n: 0)
        res = {"full": base, "weights": {}, "only": {}, "group": a.group, "tokens": int(Y.numel())}
        print(f"{run}: full precision {base:.4f} ({Y.numel():,} tokens)", flush=True)
        for k in list(KINDS) + ["other"]:
            names = [n for n in full if kind[n] == k]
            if not names:
                continue
            res["weights"][k] = sum(full[n].numel() for n in names)
            res["only"][k] = setting(lambda n: 4 if kind[n] == k else 0) - base
            print(f"  only {k:24s} ({res['weights'][k] / 1e6:6.2f}M weights) to 4 bits: {res['only'][k]:+.4f}", flush=True)
        res["all4"] = setting(lambda n: 4) - base
        print(f"  all to 4 bits: {res['all4']:+.4f}", flush=True)
        # keep the most sensitive kinds (by loss per weight) at 8 bits, one more at a time
        order = sorted(res["only"], key=lambda k: -res["only"][k] / res["weights"][k])
        res["keep8"] = []
        for i in range(1, min(4, len(order)) + 1):
            keep = set(order[:i])
            d = setting(lambda n: 8 if kind[n] in keep else 4) - base
            extra_mb = sum(res["weights"][k] for k in keep) * 0.5 / 1e6   # 4 more bits per weight
            res["keep8"].append({"kinds": order[:i], "loss": d, "extra_MB": extra_mb})
            print(f"  all to 4 bits but {', '.join(order[:i])} at 8 bits (+{extra_mb:.1f} MB): {d:+.4f}", flush=True)
        json.dump(res, open(os.path.join(run, "wbits_sens.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

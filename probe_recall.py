"""Can a model recall a specific earlier token, and from how far back?

For a next token that already appeared earlier in the same story, d tokens before the current input,
compare the model's loss on it in two conditions:
  full   - the model has read the story from its start (the earlier occurrence is in what it has read)
  short  - the model only reads the last W tokens (the earlier occurrence is cut off; spiking models
           start from rest, V = 0)
The difference (short - full) is what that earlier occurrence, together with the rest of the cut-off text,
is worth to the model. Bucketed by d it shows how well recall holds up with distance. Very frequent tokens
("the", ".", "he", ...) are excluded so that the targets are mostly names, objects and other content words.
First occurrences of content words are reported as a control: there the long context can only help
through the general topic of the story, not by recalling the same word.

  python3 probe_recall.py tf=runs/p1_tf/ckpt.pt snn=runs/p1_reset_spread/ckpt.pt --out runs/recall.json
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
L, W = 256, 16
BUCKETS = [(1, 5), (5, 17), (17, 33), (33, 65), (65, 129), (129, 257)]
N_FREQUENT = 150


def build(n_stories):
    arr = np.asarray(np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r"))
    eos = np.flatnonzero(arr == 0)
    starts, ends = eos[:-1] + 1, eos[1:]
    S = starts[(ends - starts) >= L + 1][:n_stories]
    X = np.stack([arr[s:s + L + 1] for s in S]).astype(np.int64)
    frequent = set(np.argsort(-np.bincount(arr, minlength=4096))[:N_FREQUENT].tolist())
    rows = []   # (story, t, d) ; d = 0 means no earlier occurrence (first occurrence)
    for i, row in enumerate(X):
        last = {}
        for t in range(L):
            last[int(row[t])] = t                  # the model has read row[0..t]
            tgt = int(row[t + 1])
            if tgt in frequent:
                continue
            d = t - last[tgt] + 1 if tgt in last else 0   # d = 1: the current input token itself
            rows.append((i, t, d))
    return X, np.array(rows)

DEV = torch.device("cpu")


@torch.no_grad()
def nll_full(model, X):
    out = []
    for b in range(0, len(X), 25):
        x = torch.from_numpy(X[b:b + 25, :L]).to(DEV); y = torch.from_numpy(X[b:b + 25, 1:]).to(DEV)
        with torch.autocast(DEV.type, dtype=torch.bfloat16):
            logits = model(x)[0]
        out.append(F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def nll_short(model, X, sel):
    out = []
    for b in range(0, len(sel), 1000):
        s = sel[b:b + 1000]
        x = torch.from_numpy(np.stack([X[i, t - W + 1:t + 1] for i, t, _ in s])).to(DEV)
        y = torch.from_numpy(np.array([X[i, t + 1] for i, t, _ in s])).to(DEV)
        with torch.autocast(DEV.type, dtype=torch.bfloat16):
            logits = model(x)[0][:, -1]
        out.append(F.cross_entropy(logits.float(), y, reduction="none").cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="name=path/to/checkpoint")
    p.add_argument("--stories", type=int, default=400)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    global DEV
    DEV = torch.device(a.device)
    torch.set_num_threads(a.threads)
    X, rows = build(a.stories)
    t_, d_ = rows[:, 1], rows[:, 2]
    groups = {f"repeat d={lo}-{hi - 1}": (d_ >= lo) & (d_ < hi) for lo, hi in BUCKETS}
    groups["first occurrence"] = d_ == 0
    need_short = (t_ >= W - 1) & ((d_ > W) | (d_ == 0))   # earlier occurrence (if any) lies outside the window
    sel = rows[need_short]
    print(f"{len(X)} stories of >= {L + 1} tokens, {len(rows)} content-word targets "
          f"(the {N_FREQUENT} most frequent tokens excluded); short context = last {W} tokens")
    results = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        model, _ = load(path, device=DEV)
        full = nll_full(model, X)[rows[:, 0], rows[:, 1]]
        short = np.full(len(rows), np.nan)
        short[need_short] = nll_short(model, X, sel)
        r = {}
        for g, m in groups.items():
            e = {"n": int(m.sum()), "full": float(full[m].mean())}
            ms = m & need_short
            if ms.sum() > 0:
                e["short"] = float(short[ms].mean()); e["gain"] = float((short[ms] - full[ms]).mean())
                e["n_short"] = int(ms.sum())
            r[g] = e
        results[name] = r
        print(f"\n{name}  ({path})")
        print(f"  {'target':22s} {'count':>6s} {'full ctx':>9s} {'last 16':>9s} {'gain':>7s}")
        for g, e in r.items():
            extra = f" {e['short']:9.3f} {e['gain']:7.3f}" if "gain" in e else ""
            print(f"  {g:22s} {e['n']:6d} {e['full']:9.3f}{extra}", flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

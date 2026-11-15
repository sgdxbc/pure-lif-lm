"""Long-text evaluation (README 5.22): loss at each position of whole books, read from the start of the book.
On data without a list of books (FineWeb-Edu) the documents are used instead, each from its <|endoftext|>.
For every validation (or test) book of at least --length tokens, the first --length tokens are read in one go:
  spiking network   streamed in chunks of --chunk tokens with the membrane potentials carried over (the same as one
                    pass; only the memory is bounded)
  Transformer       with all earlier tokens of the book as context, or with --window W only the last W tokens
                    (sliding window, W/2 new tokens per step) when the book is longer than it was trained on
and the loss is averaged over position ranges [0, 64), [64, 128), [128, 256), ..., up to --length. If the loss keeps
falling at later positions, the model uses the longer context. From the repository root:
  python3 eval_long.py --device cuda [--ema] [--split test] [--length 8192] runs/X     (writes runs/X/long_{split}.json)
The data directory and tokenizer come from the run's args.json (--data).
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from model import SpikingLM
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def nll_snn(model, x, chunk, dev):
    out, state = [], None
    for i in range(0, x.size(1) - 1, chunk):
        xi = x[:, i:min(i + chunk, x.size(1) - 1)].to(dev)
        yi = x[:, i + 1:i + 1 + xi.size(1)].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            logits, _, state, _ = model(xi, state=state)
        out.append(F.cross_entropy(logits.float().transpose(1, 2), yi, reduction="none").cpu())
    return torch.cat(out, 1)


@torch.no_grad()
def nll_tf(model, x, window, dev):
    n = x.size(1) - 1
    nll = torch.zeros(x.size(0), n)
    W = window if window and window < n else n
    step = max(W // 2, 1)
    start = 0
    while True:
        lo = max(0, start + step - W) if start else 0      # context begins W tokens before the end of the scored part
        hi = min(start + step, n) if start else min(W, n)
        xi, yi = x[:, lo:hi].to(dev), x[:, lo + 1:hi + 1].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            logits = model(xi)[0]
        l = F.cross_entropy(logits.float().transpose(1, 2), yi, reduction="none").cpu()
        first = start - lo if start else 0                   # score only the positions not scored before
        nll[:, lo + first:hi] = l[:, first:]
        if hi >= n:
            return nll
        start = hi


def buckets(n):
    edges = [0, 64]
    while edges[-1] < n:
        edges.append(min(edges[-1] * 2, n))
    return list(zip(edges[:-1], edges[1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory with ckpt.pt and args.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true", help="moving-average weights where the checkpoint has them")
    p.add_argument("--split", default="valid")
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=1024, help="spiking network: tokens per forward pass")
    p.add_argument("--window", type=int, default=0, help="Transformer: at most this many tokens of context (0 = all)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_books", type=int, default=0, help="only the first N qualifying books / documents")
    a = p.parse_args()
    dev = torch.device(a.device)
    args = json.load(open(os.path.join(a.run, "args.json")))
    has_ema = bool(a.ema and args.get("ema", 0))
    model, _ = load(os.path.join(a.run, "ckpt.pt"), ema=has_ema, device=dev)
    data = os.path.join(HERE, args.get("data", "data"))
    arr = np.memmap(os.path.join(data, f"{a.split}.bin"), dtype=np.uint16, mode="r")
    books = os.path.join(data, f"{a.split}_books.npy")
    if os.path.exists(books):
        starts = np.load(books)
    else:   # web documents (FineWeb-Edu): each begins with <|endoftext|>, read as its first token
        from tokenizers import Tokenizer
        eot = Tokenizer.from_file(os.path.join(data, args.get("tokenizer", "tok4096.json"))).token_to_id("<|endoftext|>")
        starts = np.flatnonzero(np.asarray(arr) == eot)
    ends = np.append(starts[1:], len(arr))
    use = [s for s, e in zip(starts, ends) if e - s >= a.length + 1]
    if a.max_books:
        use = use[:a.max_books]
    snn = isinstance(model, SpikingLM)
    if not snn and not a.window:
        assert model.cfg.rope or a.length <= model.cfg.max_len, "learned positions end at max_len: use --window"
    tot = torch.zeros(a.length)
    for i in range(0, len(use), a.batch):
        x = torch.from_numpy(np.stack([arr[s:s + a.length + 1] for s in use[i:i + a.batch]]).astype(np.int64))
        nll = nll_snn(model, x, a.chunk, dev) if snn else nll_tf(model, x, a.window, dev)
        tot += nll.sum(0)
    per_pos = tot / len(use)
    res = {"split": a.split, "books": len(use), "length": a.length, "ema": has_ema, "window": a.window,
           "trained_seq": args.get("seq"), "mean": per_pos.mean().item(),
           "buckets": {f"{lo}-{hi}": per_pos[lo:hi].mean().item() for lo, hi in buckets(a.length)}}
    name = f"long_{a.split}" + (f"_w{a.window}" if a.window else "") + ("_ema" if has_ema else "") + ".json"
    json.dump(res, open(os.path.join(a.run, name), "w"), indent=1)
    print(f"{os.path.basename(a.run.rstrip('/'))}: {len(use)} books of >= {a.length} tokens, mean {res['mean']:.4f}  " +
          "  ".join(f"{k} {v:.3f}" for k, v in res["buckets"].items()), flush=True)


if __name__ == "__main__":
    main()

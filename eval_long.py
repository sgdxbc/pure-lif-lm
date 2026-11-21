"""Long-text evaluation (docs/experiments.md 5.22): loss at each position of whole books, read from the start of the book.
On data without a list of books (FineWeb-Edu) the documents are used instead, each from its <|endoftext|>.
For every validation (or test) book of at least --length tokens, the first --length tokens are read in one go:
  spiking network   streamed in chunks of --chunk tokens with the membrane potentials carried over (the same as one
                    pass; only the memory is bounded)
  Transformer       with all earlier tokens of the book as context, or with --window W only the last W tokens
                    (sliding window, W/2 new tokens per step) when the book is longer than it was trained on
The loss is averaged over position ranges [0, 64), [64, 128), [128, 256), ..., up to --length.
--restart [N] compares the same targets with and without a reset every N input tokens (N defaults to 1024).
The reset occurs before reading the first input of the new segment. The softmax window still caps both arms.
Paired reports go to long_{split}.json; the legacy naming and scoring path apply when --restart is omitted.
--recall adds a seeded planted code probe with the same reset, and reports teacher-forced log-probability and top-1.
  python3 eval_long.py --device cuda --ema --restart --recall runs/X
--warmup N prefills each restarted block with up to N preceding tokens from rest and writes long_{split}_warm.json.
With --recall it also scores a length-matched wrong fact and a 512-token one-word positive control.
For FineWeb with slurm/long_eval.sbatch, LONG_ARGS overrides the script's default test split:
  LONG_ARGS="--split valid --restart 1024 --warmup 1024 --recall" \
    sbatch --gres=gpu:a100-80:1 slurm/long_eval.sbatch runs/X
The data directory and tokenizer come from the run's args.json.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from model import SpikingLM
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
RECALL_DISTANCES = (512, 1024, 2048, 4096, 8192)
RESTART_TOKENS = 1024


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


def softmax_spans(n, window, restart, first, warmup=0):
    """The legacy half-window scoring schedule, truncated at a restart when one falls inside a span."""
    width = min(window or n, n)
    if warmup >= width:
        restart = 0  # identical windows and learned position IDs
    stride = max(width // 2, 1)
    end = 0
    while end < n:
        lo = max(0, end + stride - width) if end else 0
        hi = min(end + stride, n) if end else width
        start = max(end, first)
        while start < hi:
            boundary = start // restart * restart if restart else 0
            stop = min(hi, boundary + restart) if restart else hi
            yield max(lo, boundary - warmup), stop, start, False
            start = stop
        end = hi


def carried_spans(n, chunk, restart, reset, first, warmup=0):
    """Align chunk boundaries in the two arms while allowing the full arm to carry across them."""
    interval = restart or n
    initial = first // interval * interval if reset else 0
    for boundary in range(initial, n, interval):
        prefix = max(0, boundary - warmup) if reset else boundary
        for lo in range(prefix, boundary, chunk):
            hi = min(lo + chunk, boundary)
            yield lo, hi, hi, lo == prefix
        end = min(n, boundary + interval)
        for lo in range(boundary, end, chunk):
            yield lo, min(lo + chunk, end), max(lo, first), reset and lo == prefix


@torch.no_grad()
def score(model, x, chunk, window, dev, restart=0, reset=False, first=0, top1=False, warmup=0):
    """Score inputs from index first, with target x[:, i+1] for input x[:, i]."""
    snn = isinstance(model, SpikingLM)
    linear = getattr(model.cfg, "attn", None) == "linear"
    n = x.size(1) - 1
    spans = (carried_spans(n, chunk, restart, reset, first, warmup) if snn or linear
             else softmax_spans(n, window, restart if reset else 0, first, warmup if reset else 0))
    nll, correct, state = [], [], None
    opt = {"return_state": True} if linear else {}
    for lo, hi, start, clear in spans:
        if clear:
            state = None
        xi = x[:, lo:hi].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            logits, _, state, _ = model(xi, state=state, logit_slice=slice(start - lo, None), **opt)
        if start < hi:
            yi = x[:, start + 1:hi + 1].to(dev)
            nll.append(F.cross_entropy(logits.float().transpose(1, 2), yi, reduction="none").cpu())
            if top1:
                correct.append((logits.argmax(-1) == yi).cpu())
    return torch.cat(nll, 1), torch.cat(correct, 1) if top1 else None


def fact_prompt(tok, name, value, one_token):
    if one_token:
        query = f"\nThe color of {name} is"
        fact = tok.encode(f"{query} {value}.\n", add_special_tokens=False).ids
    else:
        fact = tok.encode(f"\nThe code for {name} is {value}.\n", add_special_tokens=False).ids
        query = f"\nQuestion: What is the code for {name}?\nAnswer:"
    tail = tok.encode(f"{query} {value}", add_special_tokens=False)
    answer = sum(end > len(query) for _, end in tail.offsets)
    if not fact or not 0 < answer < len(tail.ids) or any(lo < len(query) < hi for lo, hi in tail.offsets):
        raise ValueError("tokenizer did not separate the planted answer from the query")
    return fact, tail.ids, answer


def recall_trials(tok, arr, starts, ends, distances, restart, trials, matched=False, one_token=False):
    """Yield distance, inputs, and answer lengths. Pair source text, key and answer across distances."""
    rng = np.random.default_rng(0)
    foil_rng = np.random.default_rng(1)
    colors = ("red", "blue", "green", "black", "white", "yellow", "orange", "purple")
    prompts, foils = [], []
    for i in range(trials):
        name = f"record {i:04d}"
        if one_token:
            candidates = [fact_prompt(tok, name, color, True) for color in colors]
            candidates = [p for p in candidates if p[2] == 1 and
                          sum(len(other[0]) == len(p[0]) and other[2] == 1 for other in candidates) >= 2]
            if not candidates:
                raise ValueError("positive control needs two one-token colors with matching fact lengths")
            fact, tail, answer = candidates[int(rng.integers(len(candidates)))]
            alternatives = [alt for alt, _, _ in candidates if len(alt) == len(fact) and alt != fact]
            foils.append(alternatives[int(foil_rng.integers(len(alternatives)))])
        else:
            fact, tail, answer = fact_prompt(tok, name, int(rng.integers(1000, 10000)), False)
            if matched:
                for value in foil_rng.permutation(9000):
                    alt, _, _ = fact_prompt(tok, name, int(value) + 1000, False)
                    if len(alt) == len(fact) and alt != fact:
                        foils.append(alt)
                        break
                else:
                    raise ValueError("tokenizer has no distinct matching-length fact")
        prompts.append((fact, tail, answer))
    if max(len(tail) for _, tail, _ in prompts) > restart:
        raise ValueError("restart interval is too short to keep the planted query and answer together")
    overhead = max(len(fact) + len(tail) - 1 for fact, tail, _ in prompts)
    # Put the final scored input at the end of a restart segment, so short gaps have a near-context control.
    lengths = {d: (d + overhead + restart - 1) // restart * restart for d in distances}
    longest = max(lengths.values()) + 1 - min(len(fact) + len(tail) for fact, tail, _ in prompts)
    windows = np.cumsum(np.maximum(ends - starts - longest + 1, 0))
    if not len(windows) or not windows[-1]:
        raise ValueError(f"recall needs a document with at least {longest} background tokens")
    picks = rng.choice(int(windows[-1]), trials, replace=trials > windows[-1])
    doc = np.searchsorted(windows, picks, side="right")
    offsets = starts[doc] + picks - np.r_[0, windows][doc]
    counts = torch.tensor([count for _, _, count in prompts])
    for distance, length in lengths.items():
        xs, swapped = [], []
        for i, (start, (fact, tail, _)) in enumerate(zip(offsets, prompts)):
            size = length + 1 - len(fact) - len(tail)
            stop = start + longest
            noise = arr[stop - size:stop]
            before = size - distance
            xs.append(np.concatenate([noise[:before], fact, noise[before:], tail]).astype(np.int64))
            if foils:
                wrong = xs[-1].copy()
                wrong[before:before + len(fact)] = foils[i]
                swapped.append(wrong)
        x = torch.from_numpy(np.stack(xs))
        yield (distance, x, counts, torch.from_numpy(np.stack(swapped))) if foils else (distance, x, counts)


def recall_metrics(nll, correct, counts):
    mask = torch.arange(nll.size(1)) >= nll.size(1) - counts[:, None]
    total = nll.masked_fill(~mask, 0).double().sum().item()
    return {"mean_answer_logprob": -total / len(counts), "mean_token_logprob": -total / counts.sum().item(),
            "token_top1": (correct & mask).sum().item() / counts.sum().item(),
            "answer_top1": (correct | ~mask).all(dim=1).double().mean().item()}


def planted_recall(model, trials, chunk, window, batch, restart, dev):
    results = {}
    for distance, x, counts in trials:
        n = x.size(1) - 1
        if getattr(model, "pos", None) is not None and min(window or n, n) > model.cfg.max_len:
            raise ValueError(f"planted recall needs --window at most {model.cfg.max_len} for learned positions")
        first = n - counts.max().item()
        arms = {}
        for mode in ("full", "restart"):
            batches = [score(model, xb, chunk, window, dev, restart, reset=mode == "restart",
                             first=first, top1=True) for xb in x.split(batch)]
            nll, correct = (torch.cat(parts) for parts in zip(*batches))
            arms[mode] = recall_metrics(nll, correct, counts)
        results[str(distance)] = {"trials": len(counts), "answer_tokens": counts.sum().item(), **arms}
        print(f"recall {distance:4d} tokens  " +
              "  ".join(f"{k} logp {v['mean_answer_logprob']:.3f}, exact top-1 {v['answer_top1']:.1%}"
                        for k, v in arms.items()), flush=True)
    return {"seed": 0, "restart_tokens": restart, "distance": "tokens between fact and query", "by_distance": results}


def paired_bootstrap(delta, seed=0, draws=10000):
    """Resample paired per-trial differences and return a two-sided percentile interval."""
    values = np.asarray(delta, dtype=np.float64)
    indices = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    return {"mean": float(values.mean()), "ci95": np.quantile(values[indices].mean(1), (0.025, 0.975)).tolist()}


def matched_recall(model, trials, chunk, window, batch, restart, warmup, dev):
    results = {}
    for distance, x, counts, swapped in trials:
        n = x.size(1) - 1
        if getattr(model, "pos", None) is not None and min(window or n, n) > model.cfg.max_len:
            raise ValueError(f"planted recall needs --window at most {model.cfg.max_len} for learned positions")
        first = n - counts.max().item()
        mask = torch.arange(n - first) >= n - first - counts[:, None]
        arms, preference = {}, {}
        for mode in ("full", "warm"):
            options = dict(restart=restart, reset=mode == "warm", first=first, warmup=warmup)
            true = [score(model, xb, chunk, window, dev, top1=True, **options) for xb in x.split(batch)]
            nll, correct = (torch.cat(parts) for parts in zip(*true))
            wrong = torch.cat([score(model, xb, chunk, window, dev, **options)[0] for xb in swapped.split(batch)])
            preference[mode] = (wrong.double() - nll.double()).masked_fill(~mask, 0).sum(1)
            arms[mode] = {**recall_metrics(nll, correct, counts),
                          "swapped_fact_mean_answer_logprob": -wrong.double().masked_fill(~mask, 0).sum().item()
                          / len(counts),
                          "true_fact_preference": paired_bootstrap(preference[mode])}
        effect = paired_bootstrap(preference["full"] - preference["warm"])
        results[str(distance)] = {"trials": len(counts), "answer_tokens": counts.sum().item(), **arms,
                                  "full_minus_warm_preference": effect}
        print(f"recall {distance:4d} tokens  full minus warm preference {effect['mean']:+.3f} nats/answer",
              flush=True)
    return results


def buckets(n):
    edges = [0, 64]
    while edges[-1] < n:
        edges.append(min(edges[-1] * 2, n))
    return list(zip(edges[:-1], edges[1:]))


def summarize(per_pos):
    return {"mean": per_pos.mean().item(),
            "buckets": {f"{lo}-{hi}": per_pos[lo:hi].mean().item() for lo, hi in buckets(len(per_pos))}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory with ckpt.pt and args.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true", help="moving-average weights where the checkpoint has them")
    p.add_argument("--split", default="valid")
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=1024, help="carried-state evaluation: tokens per forward pass")
    p.add_argument("--window", type=int, default=0, help="softmax Transformer context cap (0 = all)")
    p.add_argument("--restart", type=int, nargs="?", const=RESTART_TOKENS,
                   help="also score with context reset every N input tokens (N defaults to 1024)")
    p.add_argument("--warmup", type=int, help="prefill each restarted block with N earlier tokens from rest")
    p.add_argument("--recall", action="store_true", help="also measure planted key-value recall")
    p.add_argument("--distances", type=int, nargs="+", default=RECALL_DISTANCES, help="planted fact-to-query token gaps")
    p.add_argument("--max_trials", type=int, default=200, help="planted recall trials per distance")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_books", type=int, default=0, help="only the first N qualifying books / documents")
    a = p.parse_args()
    if min(a.length, a.chunk, a.batch, a.max_trials, *a.distances) < 1 or a.window < 0 or a.max_books < 0:
        p.error("length, chunk, batch, trials and distances must be positive; window and max_books must be nonnegative")
    if a.restart is not None and a.restart < 1:
        p.error("--restart must be positive")
    if a.warmup is not None:
        if a.warmup < 0:
            p.error("--warmup must be nonnegative")
        a.restart = a.restart or RESTART_TOKENS
    dev = torch.device(a.device)
    with open(os.path.join(a.run, "args.json")) as f:
        args = json.load(f)
    has_ema = bool(a.ema and args.get("ema", 0))
    model, _ = load(os.path.join(a.run, "ckpt.pt"), ema=has_ema, device=dev)
    data = os.path.join(HERE, args.get("data", "data"))
    arr = np.memmap(os.path.join(data, f"{a.split}.bin"), dtype=np.uint16, mode="r")
    books = os.path.join(data, f"{a.split}_books.npy")
    tok = None
    if a.recall or not os.path.exists(books):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(data, args.get("tokenizer", "tok4096.json")))
    if os.path.exists(books):
        starts = np.load(books)
    else:   # web documents (FineWeb-Edu): each begins with <|endoftext|>, read as its first token
        eot = tok.token_to_id("<|endoftext|>")
        if eot is None:
            p.error("validation documents need a book index or an <|endoftext|> token")
        starts = np.flatnonzero(np.asarray(arr) == eot)
    ends = np.append(starts[1:], len(arr))
    use = [s for s, e in zip(starts, ends) if e - s >= a.length + 1]
    if a.max_books:
        use = use[:a.max_books]
    if not use:
        p.error(f"no documents have {a.length + 1} tokens")
    snn = isinstance(model, SpikingLM)
    linear = getattr(model.cfg, "attn", None) == "linear"
    if (a.restart or a.recall) and (snn or linear) and a.window:
        p.error("--window is for softmax models in paired evaluation; use --restart for carried state")
    if getattr(model, "pos", None) is not None and min(a.window or a.length, a.length) > model.cfg.max_len:
        p.error(f"learned positions require --window at most {model.cfg.max_len} for this evaluation")
    tot = torch.zeros(a.length)
    restarted = torch.zeros_like(tot) if a.restart else None
    for i in range(0, len(use), a.batch):
        x = torch.from_numpy(np.stack([arr[s:s + a.length + 1] for s in use[i:i + a.batch]]).astype(np.int64))
        if a.restart:
            nll, _ = score(model, x, a.chunk, a.window, dev, a.restart)
            reset_nll, _ = score(model, x, a.chunk, a.window, dev, a.restart, reset=True, warmup=a.warmup or 0)
            restarted += reset_nll.sum(0)
        else:
            nll = nll_snn(model, x, a.chunk, dev) if snn else nll_tf(model, x, a.window, dev)
        tot += nll.sum(0)
    per_pos = tot / len(use)
    res = {"split": a.split, "books": len(use), "length": a.length, "ema": has_ema, "window": a.window,
           "trained_seq": args.get("seq"), **summarize(per_pos)}
    if a.restart:
        restart_pos = restarted / len(use)
        control = {"tokens": a.restart, "full": summarize(per_pos), "gain": summarize(restart_pos - per_pos)}
        if a.warmup is None:
            res["restart_control"] = {"tokens": control["tokens"], "full": control["full"],
                                      "restart": summarize(restart_pos), "gain": control["gain"]}
        else:
            res["warm_restart_control"] = {**control, "warmup_tokens": a.warmup, "warm": summarize(restart_pos)}
    if a.recall:
        interval = a.restart or RESTART_TOKENS
        if a.warmup is None:
            trials = recall_trials(tok, arr, starts, ends, a.distances, interval, a.max_trials)
            res["planted_recall"] = planted_recall(model, trials, a.chunk, a.window, a.batch, interval, dev)
        else:
            trials = recall_trials(tok, arr, starts, ends, a.distances, interval, a.max_trials, matched=True)
            positive = recall_trials(tok, arr, starts, ends, (512,), interval, a.max_trials, one_token=True)
            options = (a.chunk, a.window, a.batch, interval, a.warmup, dev)
            res["planted_recall"] = {
                "seed": 0, "bootstrap_seed": 0, "bootstrap_draws": 10000, "restart_tokens": interval,
                "warmup_tokens": a.warmup, "distance": "tokens between fact and query",
                "by_distance": matched_recall(model, trials, *options),
                "positive_control": {"distance": 512, **matched_recall(model, positive, *options)["512"]},
            }
    suffix = "_warm" if a.warmup is not None else (
        "" if a.restart else (f"_w{a.window}" if a.window else "") + ("_ema" if has_ema else ""))
    name = f"long_{a.split}{suffix}.json"
    with open(os.path.join(a.run, name), "w") as f:
        json.dump(res, f, indent=1)
    print(f"{os.path.basename(a.run.rstrip('/'))}: {len(use)} books of >= {a.length} tokens, mean {res['mean']:.4f}  " +
          "  ".join(f"{k} {v:.3f}" for k, v in res["buckets"].items()), flush=True)
    if a.restart:
        arm = "restart" if a.warmup is None else "warm"
        stats = res["restart_control" if a.warmup is None else "warm_restart_control"]
        print(f"{arm} {a.restart}: loss {stats[arm]['mean']:.4f}, "
              f"gain from carrying context {stats['gain']['mean']:+.4f}", flush=True)


if __name__ == "__main__":
    main()

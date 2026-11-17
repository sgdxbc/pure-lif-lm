"""Word classes for the two-level readout (model.py out_classes, train.py --class_file).

  python3 experiments/word_classes.py --vocab 4096 --classes 64 --method freq --data data/train.bin --out c.npy
  python3 experiments/word_classes.py --vocab 4096 --classes 64 --method kmeans --ckpt runs/main/model.pt --out c.npy

Every class holds at most m = ceil((vocab - short) / classes) words (the readout has m word neurons per class).
  --short S  the S most frequent words in --data form the short list of model.py out_short (class -1, a readout
             neuron each); the classes below are made from the other words.
  freq    by frequency in the training data: class 0 is the m most frequent words, class 1 the next m, ...
  kmeans  by the weights of a trained model: its readout rows (--rows head: words predicted in similar contexts) or
          its input rows (--rows embed: words that drive the network alike),
          balanced k-means on the unit-length rows: each round assigns the (word, class) pairs in order of
          similarity, skipping full classes.
Writes an int array (vocab,) with the class of every word.
"""
import argparse, math
import numpy as np
import torch


def freq_rank(a):
    """Rank of every word by its count in the first --sample tokens of --data (0: the most frequent)."""
    data = np.memmap(a.data, dtype=np.uint16, mode="r")
    n = min(len(data), a.sample)
    counts = np.bincount(np.asarray(data[:n]), minlength=a.vocab)[:a.vocab]
    rank = np.empty(a.vocab, dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(a.vocab)
    print(f"{n:,} tokens counted")
    if a.short:
        print(f"the {a.short} most frequent words (the short list) cover {counts[rank < a.short].sum() / counts.sum():.1%}")
    return rank, counts


def by_freq(a, rank, counts, words):
    m = math.ceil(len(words) / a.classes)
    r = np.argsort(np.argsort(rank[words], kind="stable"), kind="stable")   # rank among these words
    print(f"the first class covers {counts[words[r < m]].sum() / counts.sum():.1%} of the tokens")
    return r // m


def by_kmeans(a, words):
    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck.get("model", ck)
    W = sd[f"{a.rows}.weight"].float()[:a.vocab][torch.from_numpy(words)]
    X = torch.nn.functional.normalize(W, dim=1)
    V, C, m = len(words), a.classes, math.ceil(len(words) / a.classes)
    g = torch.Generator().manual_seed(0)
    cent = X[torch.randperm(V, generator=g)[:C]].clone()
    for it in range(a.iters):
        sim = X @ cent.T                                                  # (V, C)
        K = min(C, 32)                                                    # each word's K most similar classes
        top_v, top_c = sim.topk(K, dim=1)
        order = torch.argsort(top_v.flatten(), descending=True).tolist()
        tc = top_c.flatten().tolist()
        cls = [-1] * V
        fill = [0] * C
        for i in order:   # greedy balanced assignment, most similar pairs first
            w, c = i // K, tc[i]
            if cls[w] < 0 and fill[c] < m:
                cls[w] = c; fill[c] += 1
        for w in range(V):   # words whose K classes all filled up: the most similar class with room
            if cls[w] < 0:
                for c in torch.argsort(sim[w], descending=True).tolist():
                    if fill[c] < m:
                        cls[w] = c; fill[c] += 1
                        break
        cls = torch.tensor(cls)
        mean_sim = sim[torch.arange(V), cls].mean().item()
        cent = torch.nn.functional.normalize(torch.zeros(C, X.size(1)).index_add_(0, cls, X), dim=1)
        print(f"round {it + 1}: mean similarity to the class centre {mean_sim:.3f}")
    return cls.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--classes", type=int, required=True)
    p.add_argument("--method", choices=["freq", "kmeans"], required=True)
    p.add_argument("--data", default="data/train.bin")
    p.add_argument("--sample", type=int, default=200_000_000, help="freq: count over the first this many tokens")
    p.add_argument("--ckpt", default="")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--rows", default="head", choices=["head", "embed"],
                   help="kmeans: cluster the readout rows (for out_classes) or the input rows (for --in_class_file)")
    p.add_argument("--short", type=int, default=0, help="the most frequent this many words in --data are the short "
                   "list (class -1, model.py out_short)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rank = counts = None
    if a.method == "freq" or a.short:
        rank, counts = freq_rank(a)
    words = np.arange(a.vocab) if not a.short else np.flatnonzero(rank >= a.short)   # the words to put in classes
    cls = np.full(a.vocab, -1, dtype=np.int64)
    cls[words] = by_freq(a, rank, counts, words) if a.method == "freq" else by_kmeans(a, words)
    sizes = np.bincount(cls[words], minlength=a.classes)
    print(f"{a.classes} classes, sizes {sizes.min()}-{sizes.max()}" + (f", short list {a.short}" if a.short else "")
          + f"; wrote {a.out}")
    np.save(a.out, cls.astype(np.int64))


if __name__ == "__main__":
    main()

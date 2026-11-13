"""Figures for a trained spiking LM: loss curves, spike raster while reading text, learned time constants."""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))


def read_log(path):
    tr, va = [], []
    for line in open(path):
        r = json.loads(line)
        (va if "val_loss" in r else tr).append(r)
    return tr, va


def plot_curves(runs, out):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for label, path in runs:
        tr, va = read_log(path)
        x = np.array([r["tokens"] for r in tr]) / 1e6
        y = np.array([r["loss"] for r in tr])
        k = 9
        if len(y) > k:
            y = np.convolve(y, np.ones(k) / k, mode="valid"); x = x[k // 2: k // 2 + len(y)]
        l, = ax.plot(x, y, lw=1, alpha=0.6, label=f"{label} (train)")
        ax.plot([r["tokens"] / 1e6 for r in va], [r["val_loss"] for r in va], "o-", ms=3, color=l.get_color(),
                label=f"{label} (val)")
    ax.set_xlabel("training tokens (millions)"); ax.set_ylabel("cross-entropy (nats / token)")
    ax.set_ylim(1.0, 4.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


@torch.no_grad()
def raster(model, tok, text, out, layers=(0, 3, 7), n_neurons=120):
    ids = torch.tensor([tok.encode(text).ids])
    words = [tok.decode([i]) for i in ids[0].tolist()]
    _, _, _, spikes = model(ids, return_spikes=True)
    layers = [l for l in layers if l < len(spikes)]
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 2.3 * len(layers) + 0.6), dpi=150, sharex=True)
    axes = np.atleast_1d(axes)
    L = ids.size(1)
    for ax, li in zip(axes, layers):
        s = spikes[li][0].float().numpy()            # (L, H)
        beta = model.layers[li].beta().detach().numpy()
        order = np.argsort(beta)                      # fast neurons at the bottom, slow at the top
        pick = order[np.linspace(0, len(order) - 1, n_neurons).astype(int)]
        t, n = np.nonzero(s[:, pick])
        ax.scatter(t, n, s=2.5, marker="|", color="black", linewidths=0.8)
        ax.set_ylabel(f"layer {li+1}\nneurons\n(fast→slow)", fontsize=8)
        ax.set_yticks([]); ax.set_xlim(-0.5, L - 0.5)
        ax.set_title(f"layer {li+1}: mean firing rate {s.mean():.3f}", fontsize=8, loc="left")
    axes[-1].set_xticks(range(L)); axes[-1].set_xticklabels(words, rotation=90, fontsize=6)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def time_constants(model, out):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    bins = np.logspace(np.log10(0.3), np.log10(3000), 50)
    for i, layer in enumerate(model.layers):
        b = layer.beta().detach().clamp(1e-6, 1 - 1e-6)
        tau = (-1.0 / torch.log(b)).numpy()
        ax.hist(tau, bins=bins, histtype="step", label=f"layer {i+1}", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("membrane time constant τ = -1/ln β  (tokens)"); ax.set_ylabel("# neurons")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ckpt.pt")
    p.add_argument("--runs", nargs="*", default=[], help="label=path/to/log.jsonl")
    p.add_argument("--out", default="figs")
    p.add_argument("--text", default="Once upon a time, there was a little girl named Lily. She loved to play "
                                     "with her red ball in the park. One day, the ball rolled into the pond.")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.runs:
        plot_curves([r.split("=", 1) for r in a.runs], os.path.join(a.out, "loss_curves.png"))
    if os.path.exists(a.ckpt):
        model, _ = load(a.ckpt)
        tok = Tokenizer.from_file(os.path.join(HERE, "data", "tok4096.json"))
        raster(model, tok, a.text, os.path.join(a.out, "raster.png"),
               layers=(0, model.cfg.n_layers // 2, model.cfg.n_layers - 1))
        time_constants(model, os.path.join(a.out, "time_constants.png"))
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()

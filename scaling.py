"""Validation loss against training compute (README section 0, stage goal 2), for spiking networks and Transformers
trained on the same data. From the repository root:
  python3 scaling.py runs/g13_* [--out runs/scaling_fineweb.png] [--title ...]
writes the figure and, next to it, a JSON table with every point.

Training compute is counted the usual way: 6 floating-point operations per weight use per token (2 for the
multiply-add, times 3 for the forward and backward pass), over the matrix weights a token goes through:
  spiking network   per layer W_in, W_out and the leak-modulating synapses (also the delay and recurrent synapses
                    when used), all of them at each of the token's T simulation steps; plus the readout
  Transformer       per layer the attention and feed-forward matrices (12 d^2) and the attention itself (seq x d
                    multiply-adds per token and layer, the causal average over a training window); plus the readout
The embedding lookup and the neurons' own element-wise updates are left out (small next to the matrices, and the
same convention as for Transformers). A teacher (distillation) costs 2 operations per weight use per token, forward
only; it is reported separately ("teacher_flops") and not included on the x axis.
Each run's curve is its periodic evaluations (40 random validation windows; the moving-average weights where the run
has them); the marker at the end is the final evaluation on the whole validation set, when there is one.
"""
import argparse, glob, json, os

# Hugging Face teachers (--teacher hf:NAME): layers, width, vocabulary (GPT-2-style blocks: 12 d^2 per layer)
HF = {"openai-community/gpt2": (12, 768, 50257), "openai-community/gpt2-medium": (24, 1024, 50257),
      "openai-community/gpt2-large": (36, 1280, 50257), "openai-community/gpt2-xl": (48, 1600, 50257),
      "EleutherAI/gpt-neo-125m": (12, 768, 50257), "EleutherAI/gpt-neo-1.3B": (24, 2048, 50257),
      "EleutherAI/gpt-neo-2.7B": (32, 2560, 50257), "EleutherAI/gpt-j-6b": (28, 4096, 50400)}
for k in list(HF):
    HF[k.split("/")[-1]] = HF[k]


def macs_per_token(a):
    """Forward multiply-adds per token through the matrix weights, from a run's args.json."""
    d, L, V, seq = a["d_model"], a["n_layers"], a["vocab"], a["seq"]
    if a.get("arch", "snn") == "tf":
        return L * (12 * d * d + seq * d) + d * V
    H, T = a["d_hidden"], a.get("steps_per_token", 1)
    per = 2 * d * H
    if a.get("leak_rank", 0):
        per += a["leak_rank"] * (d + H)
    if a.get("delay_k", 0):
        per += a["delay_k"] * a["delay_rank"] * d + a["delay_rank"] * H
    if a.get("rec_rank", 0):
        per += 2 * a["rec_rank"] * H
    return L * T * per + d * V


def teacher_macs(a):
    t = a.get("teacher", "")
    if not t or not a.get("distill_weight", 0):
        return 0
    if t.startswith("hf:"):
        L, d, V = HF[t[3:]]
        return L * (12 * d * d + a["seq"] * d) + d * V
    ta = json.load(open(os.path.join(os.path.dirname(t), "args.json")))
    ta = dict(ta, seq=a["seq"])
    return macs_per_token(ta)


def load_run(path):
    a = json.load(open(os.path.join(path, "args.json")))
    m, mt = macs_per_token(a), teacher_macs(a)
    tok_per_step = a["batch"] * a["seq"]
    pts = []
    for line in open(os.path.join(path, "log.jsonl")):
        r = json.loads(line)
        if "val_loss" not in r:
            continue
        v = r.get("val_loss_ema") or r["val_loss"]
        tokens = r.get("tokens", r["step"] * tok_per_step)
        pts.append({"step": r["step"], "tokens": tokens, "flops": 6 * m * tokens, "val_loss": v})
    final = None
    for f in ("final_eval_ema.json", "final_eval.json"):
        p = os.path.join(path, f)
        if os.path.exists(p):
            e = json.load(open(p))
            tokens = e["step"] * tok_per_step
            final = {"tokens": tokens, "flops": 6 * m * tokens, "val_loss": e["val_loss"], "file": f}
            break
    params = None
    for f in ("hw_cost.json",):
        p = os.path.join(path, f)
        if os.path.exists(p):
            params = json.load(open(p)).get("params")
    return {"run": os.path.basename(path.rstrip("/")), "arch": a.get("arch", "snn"), "params": params,
            "macs_per_token": m, "teacher_macs_per_token": mt, "points": pts, "final": final,
            "teacher_flops": 2 * mt * (final or pts[-1])["tokens"] if pts else 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="runs/scaling.png")
    p.add_argument("--title", default="Validation loss against training compute")
    p.add_argument("--min_tokens", type=float, default=2e8, help="leave out evaluations before this many tokens")
    a = p.parse_args()
    paths = sorted({q for r in a.runs for q in glob.glob(r) if os.path.exists(os.path.join(q, "log.jsonl"))})
    runs = [load_run(q) for q in paths]
    json.dump(runs, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    for r in runs:
        f = r["final"] or r["points"][-1]
        print(f"{r['run']:28s} {r['arch']:4s} {r['macs_per_token'] / 1e6:7.1f}M MAC/token  {f['tokens'] / 1e9:5.2f}B tokens  "
              f"{f['flops']:.2e} FLOPs (teacher {r['teacher_flops']:.1e})  val_loss {f['val_loss']:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # reference palette (dataviz skill): slot 1 blue for spiking networks, slot 2 orange for Transformers
    color = {"snn": "#2a78d6", "tf": "#eb6834"}
    name = {"snn": "spiking", "tf": "Transformer"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    seen = set()
    size_of = {r["run"]: f"{r['params'] / 1e6:.0f}M" if r["params"] else r["run"] for r in runs}
    sizes = list(size_of.values())
    for r in runs:
        pts = [q for q in r["points"] if q["tokens"] >= a.min_tokens]
        if not pts:
            continue
        c = color[r["arch"]]
        label = name[r["arch"]] if r["arch"] not in seen else None
        seen.add(r["arch"])
        ax.plot([q["flops"] for q in pts], [q["val_loss"] for q in pts], color=c, lw=2, label=label)
        end = r["final"] or pts[-1]
        ax.plot([end["flops"]], [end["val_loss"]], "o", ms=8, color=c, mec="#fcfcfb", mew=2)
        size = size_of[r["run"]]
        if sizes.count(size) > 1:   # two runs of the same size: name them
            size = r["run"] if size == r["run"] else f"{size} {r['run']}"
        ax.annotate(size, (end["flops"], end["val_loss"]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#52514e")
    ax.set_xscale("log")
    ax.set_xlabel("training compute (FLOPs, student only)", color="#52514e")
    ax.set_ylabel("validation loss (nats per token)", color="#52514e")
    ax.set_title(a.title, color="#0b0b0b", fontsize=11, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    ax.grid(True, which="minor", axis="x", color="#efeeea", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Where do the gradient spikes of the runs with plastic synapses come from (docs/experiments.md 5.34)? Loads a run's
final weights (the raw ones, which training updates) and, without changing them, computes the training loss (with the
run's distillation when it had one) and its gradient on
  - random training batches (--batches), to see how often a batch gives a gradient far above the median, and
  - the exact batches of logged spikes (--replay P:S ...: the batch of logged step S in the relay part that resumed at
    step P; train.py draws batch offsets from numpy's default_rng(seed + P), one draw per step with --accum 1).
For each batch: the loss, the gradient norm, its split by kind of parameter (mem.* = plastic synapses, pop.* = the
neuron populations) and the largest single parameters. For the batches with the largest gradients, each sequence alone:
its loss, gradient norm, the start of its text, how repetitive it is, and its plastic synapses (the size of the largest
row of S at the end, the fraction of positions where a group's query neurons are silent, the raw read-out r before
normalisation). Writes runs/X/grad_spikes.json. On a GPU from the repository root:
  python3 experiments/grad_spikes.py runs/g18_kdmem212 --batches 300 --replay 0:6000 0:16500 30000:53000
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import model as M
from sample import load

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--batches", type=int, default=300)
p.add_argument("--replay", nargs="*", default=[])
p.add_argument("--rows", type=int, default=4, help="batches analysed sequence by sequence")
p.add_argument("--seed", type=int, default=12345, help="for the random batches")
a = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = json.load(open(os.path.join(a.run, "args.json")))
model, step = load(os.path.join(a.run, "ckpt.pt"), device=dev)
model.train()
teacher = None
if args.get("teacher") and args.get("distill_weight", 0) > 0:
    teacher, _ = load(args["teacher"], device=dev)
    for q in teacher.parameters():
        q.requires_grad_(False)
W, T = args.get("distill_weight", 0.0), args.get("distill_temp", 1.0)
B, L = args["batch"], args["seq"]
arr = np.memmap(os.path.join(args["data"], "train.bin"), dtype=np.uint16, mode="r")
n_off = len(arr) - L - 1
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(args["data"], args["tokenizer"]))
amp = lambda: torch.autocast(dev.type, dtype=torch.bfloat16)
print(f"{a.run}: step {step}, batch {B} x {L}, distillation weight {W if teacher is not None else 0}", flush=True)

stats = []   # filled by the wrapped plastic_scan during a forward pass
_scan = M.plastic_scan


def scan(q, k, v, log_lam, S0=None, chunk=128):
    r, S = _scan(q, k, v, log_lam, S0, chunk)
    with torch.no_grad():
        rr = r.float().pow(2).mean(-1).sqrt()                                       # (B, h, L)
        stats.append({"S_row_max": S.norm(dim=-1).amax(-1).float().cpu(),          # (B, h)
                      "q_silent": (q.sum(-1) == 0).float().mean(-1).cpu(),           # (B, h)
                      "r_rms_med": rr.median(-1).values.cpu(), "r_rms_max": rr.amax(-1).cpu()})
    return r, S


M.plastic_scan = scan


def kind(n):
    s = n.split(".")
    return {"mems": "mem." + ".".join(s[2:]), "layers": "pop." + ".".join(s[2:])}.get(s[0], s[0])


def grad_of(x, y):
    model.zero_grad(set_to_none=True)
    stats.clear()
    with amp():
        logits, ce, _, _ = model(x, y)
    loss, kd = ce, None
    if teacher is not None:
        with torch.no_grad(), amp():
            tl = teacher(x)[0]
        V = logits.size(-1)
        if tl.size(-1) < V:
            tl = F.pad(tl.float(), (0, V - tl.size(-1)), value=-1e4)
        kd = F.kl_div(F.log_softmax(logits.float().view(-1, V) / T, -1), F.log_softmax(tl.float().view(-1, V) / T, -1),
                      log_target=True, reduction="batchmean") * T * T
        loss = (1 - W) * ce + W * kd
    loss.backward()
    names, ps = zip(*[(n, q) for n, q in model.named_parameters() if q.grad is not None])
    norms = torch.stack([q.grad.float().norm() for q in ps]).tolist()
    by = {}
    for n, v in zip(names, norms):
        by[kind(n)] = by.get(kind(n), 0.0) + v * v
    top = sorted(zip(names, norms), key=lambda t: -t[1])[:8]
    return {"ce": ce.item(), "kd": None if kd is None else kd.item(), "gn": sum(v * v for v in norms) ** 0.5,
            "by_kind": {k: v ** 0.5 for k, v in sorted(by.items(), key=lambda t: -t[1])}, "top": top}


def batch_at(ix):
    x = torch.from_numpy(np.stack([arr[i:i + L + 1] for i in ix]).astype(np.int64)).to(dev)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def repetition(ids):
    """Fraction of positions whose token, together with the one before, already occurred in the previous 512."""
    seen, rep = {}, 0
    for t in range(1, len(ids)):
        bg = (ids[t - 1], ids[t])
        if t - seen.get(bg, -10 ** 9) <= 512:
            rep += 1
        seen[bg] = t
    return rep / (len(ids) - 1)


def show(r):
    return (f"gn {r['gn']:8.3f}  ce {r['ce']:.3f}" + (f"  kd {r['kd']:.3f}" if r["kd"] is not None else "") + "  | " +
            " ".join(f"{k} {v:.3f}" for k, v in list(r["by_kind"].items())[:5]))


out = {"run": a.run, "step": step, "random": [], "replay": [], "rows": []}
t0 = time.time()
rng = np.random.default_rng(a.seed)
for b in range(a.batches):
    ix = rng.integers(0, n_off, size=B)
    r = grad_of(*batch_at(ix))
    r["ix"] = ix.tolist()
    out["random"].append(r)
    if b % 50 == 0:
        print(f"random batch {b} {show(r)} ({time.time() - t0:.0f}s)", flush=True)
gns = np.array([r["gn"] for r in out["random"]])
med = float(np.median(gns))
print(f"\n{len(gns)} random batches: gradient norm median {med:.3f}, 90% {np.percentile(gns, 90):.3f}, "
      f"99% {np.percentile(gns, 99):.3f}, max {gns.max():.3f}; above 5x median: {(gns > 5 * med).sum()}, "
      f"above 20x: {(gns > 20 * med).sum()}", flush=True)
kinds = list(out["random"][0]["by_kind"])
print("median by kind: " + " ".join(f"{k} {np.median([r['by_kind'].get(k, 0) for r in out['random']]):.3f}"
                                    for k in kinds[:10]))
for spec in a.replay:
    P, S = map(int, spec.split(":"))
    rr = np.random.default_rng(args.get("seed", 0) + P)
    for _ in range(S - P - 1):
        rr.integers(0, n_off, size=B)
    ix = rr.integers(0, n_off, size=B)
    r = grad_of(*batch_at(ix))
    r.update(ix=ix.tolist(), part_start=P, step=S)
    out["replay"].append(r)
    print(f"replay of step {S} (part from {P}): {show(r)}", flush=True)
cands = sorted(out["random"] + out["replay"], key=lambda r: -r["gn"])[:a.rows]
for r in cands:
    print(f"\nbatch with gn {r['gn']:.3f}" + (f" (replay of step {r['step']})" if "step" in r else "") +
          "; largest parameters: " + ", ".join(f"{n} {v:.3f}" for n, v in r["top"][:6]))
    x, y = batch_at(r["ix"])
    for j in range(B):
        g = grad_of(x[j:j + 1], y[j:j + 1])
        ids = x[j].tolist()
        mem = [{k: v[0].tolist() for k, v in s.items()} for s in stats]
        g.update(batch_gn=r["gn"], row=j, offset=r["ix"][j], repetition=repetition(ids),
                 text=tok.decode(ids[:160]), mem=mem)
        out["rows"].append(g)
    rows = sorted([g for g in out["rows"] if g["batch_gn"] == r["gn"]], key=lambda g: -g["gn"])
    gm = float(np.median([g["gn"] for g in rows]))
    for g in rows[:3]:
        Smax = max(max(m["S_row_max"]) for m in g["mem"]) if g["mem"] else 0
        print(f"  row {g['row']:2d} {show(g)}  (row median gn {gm:.3f})\n"
              f"     repetition {g['repetition']:.2f}, largest S row {Smax:.1f}, "
              f"query silent {np.mean([np.mean(m['q_silent']) for m in g['mem']]):.2f}, "
              f"largest raw read-out {max(max(m['r_rms_max']) for m in g['mem']):.1f}\n"
              f"     top: " + ", ".join(f"{n} {v:.3f}" for n, v in g["top"][:4]) +
              f"\n     text: {g['text'][:300]!r}", flush=True)
    typ = sorted(rows, key=lambda g: g["gn"])[B // 2]
    print(f"  typical row {typ['row']}: {show(typ)}, repetition {typ['repetition']:.2f}, largest S row "
          f"{max(max(m['S_row_max']) for m in typ['mem']):.1f}", flush=True)
with open(os.path.join(a.run, "grad_spikes.json"), "w") as f:
    json.dump(out, f)
print(f"\nwrote {a.run}/grad_spikes.json ({time.time() - t0:.0f}s)")

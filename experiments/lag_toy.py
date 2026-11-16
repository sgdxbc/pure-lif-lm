"""Synthetic lagged copy on the CPU (docs/experiments.md 5.33): can the network say exactly which token it read LAG
positions ago? Sequences of random tokens (NV of them, all equally likely); at every position t >= LAG the target is
the token at t - LAG (chance: ln NV; a Transformer does this easily through its position embedding). This isolates
precise access to the last few tokens, where much of the spiking network's gap to the Transformer lies (5.30, 5.32).
Same small spiking network as recall_toy.py (2 layers, bus 64, 256 neurons per layer). From the repository root:
  python3 experiments/lag_toy.py LAG STEPS [NV] [key=value ...]     e.g.  python3 experiments/lag_toy.py 4 2000 32 mem_heads=2
key=value pairs override the model's Config (numbers are parsed as int or float, true/false as booleans).
"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import Config, SpikingLM

lag, steps = int(sys.argv[1]), int(sys.argv[2])
NV = int(sys.argv[3]) if len(sys.argv) > 3 and "=" not in sys.argv[3] else 32
over = {}
for kv in sys.argv[3:]:
    if "=" in kv:
        k, v = kv.split("=", 1)
        over[k] = {"true": True, "false": False}.get(v.lower(), None)
        if over[k] is None:
            over[k] = float(v) if any(c in v for c in ".e") else int(v)
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
L = 128

torch.manual_seed(0)
cfg = dict(vocab=NV, d_model=64, d_hidden=256, n_layers=2, leak_rank=8, hard_reset=True, reset_learn=True,
           gamma_init_min=0.05, gamma_init=0.999, sg="fast_sigmoid", detach_reset=False,
           mem_keys=32, mem_dv=32, mem_chunk=64)
cfg.update(over)
model = SpikingLM(Config(**cfg))
opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=0.0)
gen = torch.Generator().manual_seed(1)
tag = f"lag {lag} NV {NV} " + " ".join(f"{k}={v}" for k, v in over.items())
t0, accs = time.time(), []
for it in range(steps + 1):
    x = torch.randint(0, NV, (64, L), generator=gen)
    tgt = torch.full_like(x, -100)
    tgt[:, lag:] = x[:, :L - lag]
    logits = model(x)[0]
    loss = F.cross_entropy(logits.reshape(-1, NV), tgt.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    acc = (logits.argmax(-1) == tgt)[tgt >= 0].float().mean().item()
    accs.append(acc)
    if it % 250 == 0:
        print(f"{tag} step {it} loss {loss.item():.3f} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
print(f"{tag} FINAL mean acc over the last 100 steps {sum(accs[-100:]) / 100:.3f}", flush=True)

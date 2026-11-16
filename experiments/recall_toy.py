"""Synthetic associative recall on the CPU (docs/experiments.md 5.32): a quick check of a memory mechanism before a GPU
wave. Each sequence lists NP key-value pairs (keys from NK key tokens, values from NK value tokens), then asks for every
key again in random order; the loss counts only the value after each repeated key (chance: ln NK). A small spiking
network (2 layers, bus 64) with plastic synapses (2 groups of MK key neurons) learns it; without them it stays near
chance. From the repository root:
  python3 experiments/recall_toy.py RULE STEPS [NK NP MK]     e.g.  python3 experiments/recall_toy.py hebb 3000 128 64 16
RULE: none (no plastic synapses), hebb (or another value of --mem_rule, when the model has one).
"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import Config, SpikingLM

rule, steps = sys.argv[1], int(sys.argv[2])
NK, NP, MK = (int(x) for x in (sys.argv[3:6] + ["32", "16", "32"][len(sys.argv[3:6]):]))
V = 2 * NK   # token 0 unused, keys 1..NK, values NK+1..2NK-1
torch.set_num_threads(2)


def batch(B, gen):
    ks = torch.stack([torch.randperm(NK, generator=gen)[:NP] + 1 for _ in range(B)])
    vs = torch.randint(NK + 1, V, (B, NP), generator=gen)
    perm = torch.stack([torch.randperm(NP, generator=gen) for _ in range(B)])
    asked = torch.stack([torch.gather(ks, 1, perm), torch.gather(vs, 1, perm)], -1).view(B, 2 * NP)
    x = torch.cat([torch.stack([ks, vs], -1).view(B, 2 * NP), asked], 1)
    tgt = torch.full_like(x, -100)
    tgt[:, 2 * NP::2] = asked[:, 1::2]   # at each repeated key, predict its value
    return x, tgt


torch.manual_seed(0)
cfg = Config(vocab=V, d_model=64, d_hidden=256, n_layers=2, leak_rank=8, hard_reset=True, reset_learn=True,
             gamma_init_min=0.05, gamma_init=0.999, sg="fast_sigmoid", detach_reset=False,
             mem_heads=0 if rule == "none" else 2, mem_keys=MK, mem_dv=32, mem_tau_min=64, mem_tau_max=1024,
             mem_chunk=64, **({} if rule in ("none", "hebb") else {"mem_rule": rule}))
model = SpikingLM(cfg)
opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=0.0)
gen = torch.Generator().manual_seed(1)
t0 = time.time()
for it in range(steps + 1):
    x, tgt = batch(64, gen)
    logits = model(x)[0]
    loss = F.cross_entropy(logits.view(-1, V), tgt.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if it % 200 == 0:
        acc = (logits.argmax(-1) == tgt)[tgt >= 0].float().mean().item()
        print(f"{rule} NK{NK} NP{NP} MK{MK} step {it} loss {loss.item():.3f} acc {acc:.3f} ({time.time() - t0:.0f}s)",
              flush=True)

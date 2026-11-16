"""GPU benchmark of the plastic synapses' training computation (docs/experiments.md 5.32): model.plastic_scan, which
loops over the chunks, against plastic_scan_batched below, which computes all chunks together (the same arithmetic).
Checks that both give the same result, then times them alone and inside a whole training step of the 138M network
with --mem_heads 8. If the batched one is faster on the GPU, it replaces model.plastic_scan.
From the repository root, on a GPU node:  python3 experiments/bench_scan.py [batch]
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch.nn.functional as F
import model as M


def plastic_scan_loop(q, k, v, log_lam, S0=None, chunk=128):   # = model.plastic_scan on 2026-11-16
    B, h, L, m = q.shape
    S = q.new_zeros(B, h, m, v.size(-1), dtype=torch.float32) if S0 is None else S0.float()
    ll = log_lam.float().view(1, h, 1, 1)
    out = []
    for a in range(0, L, chunk):
        qc, kc, vc = q[:, :, a:a + chunk], k[:, :, a:a + chunk], v[:, :, a:a + chunk]
        n = qc.size(2)
        i = torch.arange(n, device=q.device, dtype=torch.float32)
        diff = i[:, None] - i[None, :]
        D = torch.exp(ll * diff.clamp(min=0)) * (diff >= 0)
        A = (qc @ kc.transpose(-1, -2)) * D.to(qc.dtype)
        inter = (qc * torch.exp(ll * (i + 1).view(1, 1, n, 1)).to(qc.dtype)) @ S.to(qc.dtype)
        out.append(A @ vc + inter)
        w = torch.exp(ll * (n - 1 - i).view(1, 1, n, 1)).to(kc.dtype)
        S = torch.exp(ll * n) * S + ((kc * w).transpose(-1, -2) @ vc).float()
    return torch.cat(out, 2), S


def plastic_scan_batched(q, k, v, log_lam, S0=None, chunk=128):
    """Plastic synapses over a sequence. q, k: (B, h, L, m) spikes; v: (B, h, L, dv); log_lam: (h,) log of the decay
    per step; S0: (B, h, m, dv) synapses before the first position, or None (all zero). Returns
        r[t] = sum_{s<=t} lam^(t-s) (q[t] . k[s]) v[s] + lam^(t+1) S0^T q[t]        (B, h, L, dv)
    and the synapses after the last position, S[L-1] (B, h, m, dv; float32). Chunks of `chunk` positions: within a
    chunk all pairs at once, between chunks through S (the same result as stepping S[t] = lam S[t-1] + k[t] v[t]^T).
    All chunks are computed together (one batched product each for the pairs within chunks, the chunks' writes and the
    reads of the earlier chunks); only the synapses at the chunk boundaries are stepped one chunk at a time."""
    B, h, L, m = q.shape
    dv = v.size(-1)
    C = min(chunk, L)
    nc = -(-L // C)
    pad = nc * C - L                     # the last chunk is padded with silent positions (q = k = 0, v = 0)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    qc, kc, vc = q.view(B, h, nc, C, m), k.view(B, h, nc, C, m), v.view(B, h, nc, C, dv)
    dt = qc.dtype
    ll = log_lam.float().view(1, h, 1, 1, 1)
    i = torch.arange(C, device=q.device, dtype=torch.float32)
    diff = i[:, None] - i[None, :]
    D = torch.exp(ll * diff.clamp(min=0)) * (diff >= 0)                          # (1, h, 1, C, C): lam^(t-s), s <= t
    out = ((qc @ kc.transpose(-1, -2)) * D.to(dt)) @ vc                         # pairs within each chunk
    w = torch.exp(ll * (C - 1 - i).view(1, 1, 1, C, 1))                        # decay of each write to its chunk end
    dS = ((kc * w.to(dt)).transpose(-1, -2) @ vc).float()                       # (B, h, nc, m, dv): each chunk's writes
    S = q.new_zeros(B, h, m, dv, dtype=torch.float32) if S0 is None else S0.float()
    lamC = torch.exp(ll[:, :, 0] * C)
    before = []
    for c in range(nc):                                                          # synapses before each chunk
        before.append(S)
        S = lamC * S + dS[:, :, c]
    Sb = torch.stack(before, 2)
    g = torch.exp(ll * (i + 1).view(1, 1, 1, C, 1))                             # decay from the chunk start
    out = out + (qc * g.to(dt)) @ Sb.to(dt)                                     # reads of the earlier chunks
    if pad:   # the synapses after the last real position: the last chunk's writes decayed only to that position
        n = C - pad
        wl = torch.exp(ll[:, :, 0] * (n - 1 - i).clamp(min=0).view(1, 1, C, 1)) * (i < n).view(1, 1, C, 1)
        S = torch.exp(ll[:, :, 0] * n) * before[-1] + ((kc[:, :, -1] * wl.to(dt)).transpose(-1, -2) @ vc[:, :, -1]).float()
    return out.view(B, h, nc * C, dv)[:, :, :L], S


def timeit(fn, reps=10):
    for _ in range(3):
        fn()
    sync(); t = time.time()
    for _ in range(reps):
        fn()
    sync()
    return (time.time() - t) / reps


B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
L = int(os.environ.get("BENCH_L", 1024))   # shorter only for a quick check on the CPU
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
print(torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "batch", B)
g = torch.Generator(device=dev).manual_seed(0)
q = (torch.rand(B, 8, L, 64, device=dev, generator=g) < 0.1).bfloat16()
k = (torch.rand(B, 8, L, 64, device=dev, generator=g) < 0.1).bfloat16()
v = torch.randn(B, 8, L, 64, device=dev, generator=g).bfloat16()
ll = torch.log(torch.exp(-1.0 / torch.exp(torch.linspace(2.08, 6.93, 8, device=dev))))
a, b = plastic_scan_loop(q, k, v, ll), plastic_scan_batched(q, k, v, ll)
print("same result: r", (a[0].float() - b[0].float()).abs().max().item(), "S", (a[1] - b[1]).abs().max().item())
for name, f in (("loop", plastic_scan_loop), ("batched", plastic_scan_batched)):
    def step():
        qq, kk, vv = (x.clone().requires_grad_() for x in (q, k, v))
        r, S = f(qq, kk, vv, ll)
        (r.float().sum() + S.sum()).backward()
    print(f"plastic_scan {name:8s} forward+backward {1000 * timeit(step):7.1f} ms")
cfg = M.Config(vocab=50304, d_model=768, d_hidden=3072, n_layers=12, leak_rank=96, hard_reset=True, sg="fast_sigmoid",
               detach_reset=False, reset_learn=True, gamma_init=0.999, gamma_init_min=0.05, mem_heads=8)
net = M.SpikingLM(cfg).to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-4)
x = torch.randint(0, 50304, (B, L), device=dev)
for name, f in (("loop", plastic_scan_loop), ("batched", plastic_scan_batched), ("loop", plastic_scan_loop),
                ("batched", plastic_scan_batched)):
    M.plastic_scan = f
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def step():
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            loss = net(x, x)[1]
        opt.zero_grad(); loss.backward(); opt.step()
    dt = timeit(step, 5)
    print(f"training step, {name:8s}: {1000 * dt:7.1f} ms, {B * L / dt / 1e3:6.1f}k tok/s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GB")
del net, opt
torch.cuda.empty_cache() if dev.type == "cuda" else None
net = M.SpikingLM(M.Config(**{**cfg.__dict__, "mem_heads": 0})).to(dev)   # the same network without plastic synapses
opt = torch.optim.AdamW(net.parameters(), 1e-4)
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
dt = timeit(step, 5)
print(f"training step, no plastic synapses: {1000 * dt:7.1f} ms, {B * L / dt / 1e3:6.1f}k tok/s, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GB")

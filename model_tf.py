"""Reference (non-spiking) baseline: a small GPT-style Transformer with the same tokenizer,
used only to put the spiking model's numbers in context.

attn = "linear" (docs/experiments.md 5.47): the softmax attention of every block is replaced by gated linear attention,
a non-spiking network with a fixed-size state like the spiking one, as a control: trained the same way, it shows how
much of the spiking network's gap to the Transformer comes from keeping the past in a fixed-size state (which this
network shares) and how much from the spiking itself (which it does not). Per head (the form of Mamba-2's scalar decay,
or a RetNet whose decay depends on the input):
    S[t] = a[t] S[t-1] + k[t] v[t]^T          a[t] = exp(-exp(A_log) softplus(W_a x[t] + b_a)), one per head and step
    o[t] = S[t]^T q[t]                        (S: head_dim x head_dim per head, the whole memory of the past)
q, k, v pass a causal depthwise convolution over the last 4 positions and SiLU; o is normalised per head (RMSNorm) and
multiplied by a SiLU gate of the input before the output projection. No position embedding (the decay and the
convolution give the order; nothing grows with the text). The spiking network's plastic synapses are the same rule with
a fixed decay per group and 0/1 keys and queries (model.py)."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TFConfig:
    vocab: int = 4096
    d_model: int = 384
    n_layers: int = 5
    n_heads: int = 6
    max_len: int = 1024
    rope: bool = False        # rotary position embedding on queries and keys instead of learned absolute positions
    attn: str = "softmax"     # "linear": gated linear attention instead of softmax attention (module docstring)


def rotary(x, base=10000.0):
    """Rotary position embedding of x (B, heads, L, head_dim): pairs of channels rotated by position-dependent angles."""
    hd, L = x.size(-1), x.size(-2)
    inv = base ** (-torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd)
    ang = torch.outer(torch.arange(L, device=x.device, dtype=torch.float32), inv)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


def gla_scan(q, k, v, log_a, chunk=64, S0=None):
    """Linear attention with a decay per head and step. q, k: (B, h, L, dk); v: (B, h, L, dv); log_a: (B, h, L), <= 0.
    Returns o[t] = sum_{s<=t} exp(A[t] - A[s]) (q[t] . k[s]) v[s] + exp(A[t]) q[t]^T S0, A = cumsum(log_a) (the same
    as stepping S[t] = a[t] S[t-1] + k[t] v[t]^T, o[t] = S[t]^T q[t]), and S after the last position (float32).
    In chunks of `chunk` positions: all pairs within a chunk at once, the earlier chunks through S; every exponent is
    <= 0, so nothing overflows."""
    B, h, L, dk = q.shape
    dv = v.size(-1)
    C = min(chunk, L)
    n = -(-L // C)
    pad = n * C - L                      # padded positions: q = k = v = 0 and no decay (log_a = 0)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        log_a = F.pad(log_a, (0, pad))
    dt = q.dtype
    qc, kc, vc = q.reshape(B, h, n, C, dk), k.reshape(B, h, n, C, dk), v.reshape(B, h, n, C, dv)
    A = log_a.float().reshape(B, h, n, C).cumsum(-1)                             # decay from the chunk start
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    D = torch.exp((A.unsqueeze(-1) - A.unsqueeze(-2)).masked_fill(~causal, float("-inf")))   # exp(A[t] - A[s]), s <= t
    out = ((qc @ kc.transpose(-1, -2)).float() * D).to(dt) @ vc                  # pairs within each chunk
    w = torch.exp(A[..., -1:] - A)                                               # decay of each write to its chunk end
    dS = ((kc * w.unsqueeze(-1).to(dt)).transpose(-1, -2) @ vc).float()          # (B, h, n, dk, dv)
    decay_c = torch.exp(A[..., -1])                                              # (B, h, n): over a whole chunk
    S = q.new_zeros(B, h, dk, dv, dtype=torch.float32) if S0 is None else S0.float()
    before = []
    for i in range(n):                                                           # S before each chunk
        before.append(S)
        S = decay_c[:, :, i, None, None] * S + dS[:, :, i]
    Sb = torch.stack(before, 2)
    out = out + (qc * torch.exp(A).unsqueeze(-1).to(dt)) @ Sb.to(dt)             # reads of the earlier chunks
    return out.reshape(B, h, n * C, dv)[:, :, :L], S


class LinearAttention(nn.Module):
    """Gated linear attention (module docstring), in place of a block's softmax attention."""

    def __init__(self, c):
        super().__init__()
        d, h = c.d_model, c.n_heads
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, kernel_size=4, groups=3 * d, padding=3, bias=False)   # causal: first L
        self.decay = nn.Linear(d, h, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, h)))
        self.gate = nn.Linear(d, d, bias=False)
        self.onorm = nn.RMSNorm(self.hd)
        self.proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():   # softplus(bias) log-uniform in [0.001, 0.1] (Mamba's step sizes): a from ~0.2 to ~1
            dt0 = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
            self.decay.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))   # inverse of softplus

    def forward(self, x):
        B, L, D = x.shape
        qkv = F.silu(self.conv(self.qkv(x).transpose(1, 2))[..., :L]).transpose(1, 2)
        q, k, v = qkv.reshape(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)    # (B, h, L, hd) each
        log_a = -F.softplus(self.decay(x).float()) * self.A_log.exp()           # (B, L, h)
        o, _ = gla_scan(q * self.hd ** -0.5, k, v, log_a.transpose(1, 2))
        o = self.onorm(o).transpose(1, 2).reshape(B, L, D) * F.silu(self.gate(x))
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.rope = c.rope
        self.n1 = nn.RMSNorm(c.d_model)
        self.la = LinearAttention(c) if c.attn == "linear" else None
        if self.la is None:
            self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
            self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.n2 = nn.RMSNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.fc2 = nn.Linear(4 * c.d_model, c.d_model, bias=False)
        self.h = c.n_heads

    def forward(self, x):
        B, L, D = x.shape
        if self.la is not None:
            x = x + self.la(self.n1(x))
            return x + self.fc2(F.gelu(self.fc(self.n2(x))))
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotary(q), rotary(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.fc2(F.gelu(self.fc(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, c: TFConfig):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab, c.d_model)
        assert c.attn in ("softmax", "linear")
        self.pos = None if c.rope or c.attn == "linear" else nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = nn.RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * c.n_layers) if n.endswith(("proj.weight", "fc2.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx, targets=None, state=None, return_spikes=False, hidden=False):
        """Returns logits, loss, None (no carried state), zeros; with hidden=True the third item is instead the list
        of residual streams after each block (used as a teacher's hidden states, train.py --hint_weight)."""
        L = idx.size(1)
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(L, device=idx.device))
        xs = []
        for b in self.blocks:
            x = b(x)
            xs.append(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, xs if hidden else None, torch.zeros(1, device=idx.device)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, eos_id=None, generator=None):
        out = []
        for _ in range(max_new_tokens):
            logits, _, _, _ = self(idx[:, -self.cfg.max_len:])
            l = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(l, top_k)
                l[l < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(l, -1), 1, generator=generator)
            out.append(nxt)
            idx = torch.cat([idx, nxt], 1)
            if eos_id is not None and idx.size(0) == 1 and nxt.item() == eos_id:
                break
        return torch.cat(out, 1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_neurons(self):
        return 0

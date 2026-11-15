"""Reference (non-spiking) baseline: a small GPT-style Transformer with the same tokenizer,
used only to put the spiking model's numbers in context."""
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


def rotary(x, base=10000.0):
    """Rotary position embedding of x (B, heads, L, head_dim): pairs of channels rotated by position-dependent angles."""
    hd, L = x.size(-1), x.size(-2)
    inv = base ** (-torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd)
    ang = torch.outer(torch.arange(L, device=x.device, dtype=torch.float32), inv)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.rope = c.rope
        self.n1 = nn.RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.n2 = nn.RMSNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.fc2 = nn.Linear(4 * c.d_model, c.d_model, bias=False)
        self.h = c.n_heads

    def forward(self, x):
        B, L, D = x.shape
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
        self.pos = None if c.rope else nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = nn.RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * c.n_layers) if n.endswith(("proj.weight", "fc2.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx, targets=None, state=None, return_spikes=False):
        L = idx.size(1)
        x = self.embed(idx)
        if not self.cfg.rope:
            x = x + self.pos(torch.arange(L, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, None, torch.zeros(1, device=idx.device)

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

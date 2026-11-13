"""A language model made only of LIF neurons.

Everything that carries state or applies a nonlinearity is a LIF neuron. Neurons talk to each
other only with binary spikes through linear synapses. Concretely:

  input  : one "input neuron" per vocabulary item; reading token x_t makes neuron x_t emit one
           spike. Its outgoing synapses are the rows of `embed` (so embed[x_t] is the injected current).
  layers : n_layers populations of LIF neurons (d_hidden each). Population l receives synaptic
           current from the input neurons and from the spikes of all earlier populations. Those
           skip synapses are factorised through a shared d_model-wide "current bus":
               bus_l[t] = embed[x_t] + sum_{k<l} W_out_k s_k[t]
               I_l[t]   = W_in_l bus_l[t] + b_l
           The bus is just a sum of weighted spikes (dendritic summation); it has no state and no
           nonlinearity. It plays the role of the residual stream of a Transformer.
  LIF    : V[t] = beta * V[t-1] + I[t];  s[t] = [V[t] >= theta];  reset (soft: V -= theta*s).
           beta (leak) and theta (threshold) are learned per neuron. Memory across tokens lives
           only in the membrane potentials V.
  output : vocab-sized non-spiking readout neurons whose membrane potential is
               logits[t] = W_head bus_L[t]
           (a linear function of the spikes of every layer plus the input spike).
           The next-token distribution is softmax(logits).
No attention, no normalisation layers, no gating/multiplicative interactions.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lif import lif, lif_rec


@dataclass
class Config:
    vocab: int = 4096
    d_model: int = 384
    d_hidden: int = 1536
    n_layers: int = 8
    tau_min: float = 0.5      # membrane time constants (in tokens) are initialised log-uniform
    tau_max: float = 200.0    # in [tau_min, tau_max];  beta = exp(-1/tau)
    hard_reset: bool = False
    detach_reset: bool = True
    sg: str = "sigmoid"
    alpha: float = 4.0
    in_norm: str = "theta"    # keep slow neurons from saturating at init: "theta" raises their
                              # threshold to 1/sqrt(1-beta^2); "weight" shrinks their input synapses
                              # by sqrt(1-beta^2) instead (same function, worse for Adam); "none".
    out_init: float = 0.02
    head_init: float = 0.02
    learn_beta: bool = True
    learn_theta: bool = True
    reset_learn: bool = False # multiplicative reset V <- V*(1 - gamma*s) with a learned gamma per neuron
    gamma_init: float = 0.9   # (only with hard_reset=True; gamma = 1 is the classic reset-to-zero)
    rec_rank: int = 0         # >0: low-rank same-layer synapses R = A @ B acting on the previous step's spikes
    rec_init: float = 1.0     # std of A is rec_init/sqrt(d_hidden); B starts at 0 so R = 0 at init


class LIFPopulation(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, d = cfg.d_hidden, cfg.d_model
        self.w_in = nn.Linear(d, H, bias=True)
        self.w_out = nn.Linear(H, d, bias=False)
        log_tau = torch.empty(H).uniform_(math.log(cfg.tau_min), math.log(cfg.tau_max))
        beta = torch.exp(-1.0 / log_tau.exp())
        self.beta_logit = nn.Parameter(torch.logit(beta), requires_grad=cfg.learn_beta)
        theta0 = torch.ones(H)
        if cfg.in_norm == "theta":
            theta0 = 1.0 / torch.sqrt(1 - beta ** 2)
        self.log_theta = nn.Parameter(theta0.log(), requires_grad=cfg.learn_theta)
        with torch.no_grad():
            nn.init.normal_(self.w_in.weight, std=1.0 / math.sqrt(d))
            if cfg.in_norm == "weight":
                self.w_in.weight.mul_(torch.sqrt(1 - beta ** 2).unsqueeze(1))
            nn.init.zeros_(self.w_in.bias)
            nn.init.normal_(self.w_out.weight, std=cfg.out_init)
        if cfg.reset_learn:
            self.gamma_logit = nn.Parameter(torch.full((H,), math.log(cfg.gamma_init / (1 - cfg.gamma_init))))
        else:
            self.register_buffer("gamma_const", torch.ones(H), persistent=False)
        if cfg.rec_rank > 0:
            self.rec_A = nn.Parameter(torch.randn(H, cfg.rec_rank) * cfg.rec_init / math.sqrt(H))
            self.rec_B = nn.Parameter(torch.zeros(cfg.rec_rank, H))

    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def theta(self):
        return self.log_theta.exp()

    def gamma(self):
        return torch.sigmoid(self.gamma_logit) if self.cfg.reset_learn else self.gamma_const

    def forward(self, bus, state=None):
        """state: (membrane potentials (B,H), spikes of the previous step (B,H)) or None (= at rest)."""
        c = self.cfg
        v0, s0 = (None, None) if state is None else state
        I = self.w_in(bus)
        kw = dict(hard_reset=c.hard_reset, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg)
        if c.rec_rank > 0:
            s, v_last = lif_rec(I, self.beta(), self.theta(), self.rec_A, self.rec_B, self.gamma(), v0, s0, **kw)
        else:
            s, v_last = lif(I, self.beta(), self.theta(), self.gamma(), v0, **kw)
        return self.w_out(s), s, (v_last.detach(), s[:, -1].detach())


class SpikingLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=1.0)
        self.layers = nn.ModuleList([LIFPopulation(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        nn.init.normal_(self.head.weight, std=cfg.head_init)

    def forward(self, idx, targets=None, state=None, return_spikes=False):
        """idx: (B, L) token ids. state: per-layer (membrane potentials, last spikes) or None (at rest).
        Returns logits, loss, new_state, and per-layer firing rates (or spike tensors)."""
        bus = self.embed(idx)
        new_state, rates, spikes = [], [], []
        for i, layer in enumerate(self.layers):
            out, s, layer_state = layer(bus, None if state is None else state[i])
            bus = bus + out
            new_state.append(layer_state)
            rates.append(s.detach().mean())
            if return_spikes:
                spikes.append(s.detach())
        logits = self.head(bus)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        extra = spikes if return_spikes else torch.stack(rates)
        return logits, loss, new_state, extra

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, eos_id=None, generator=None):
        """Streaming generation: the prompt is run once, then one simulation step per new token.
        Carried from step to step: every neuron's membrane potential (and, with same-layer synapses,
        the previous step's spikes, which arrive through those synapses one step later)."""
        logits, _, state, _ = self(idx)
        out = []
        for _ in range(max_new_tokens):
            l = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(l, top_k)
                l[l < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(l, -1), 1, generator=generator)
            out.append(nxt)
            if eos_id is not None and idx.size(0) == 1 and nxt.item() == eos_id:
                break
            logits, _, state, _ = self(nxt, state=state)
        return torch.cat(out, 1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_neurons(self):
        return self.cfg.n_layers * self.cfg.d_hidden

"""A language model made only of LIF neurons (by default; of the options below, only the plastic synapses add state
outside the neurons, and they are marked as a deviation).

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
  LIF    : V[t] = beta * V[t-1] + I[t];  s[t] = [V[t] >= theta];  reset: soft (V -= theta*s), or with hard_reset
           V <- V*(1 - gamma*s) (gamma = 1, or learned per neuron with reset_learn).
           beta (leak) and theta (threshold) are learned per neuron. Memory across tokens lives
           only in the membrane potentials V.
  output : vocab-sized non-spiking readout neurons whose membrane potential is
               logits[t] = W_head bus_L[t]
           (a linear function of the spikes of every layer plus the input spike).
           The next-token distribution is softmax(logits).
No attention, no normalisation layers.
Optional (leak_rank > 0): leak-modulating synapses. Besides the ordinary synapses that inject current,
           a second set of synapses from the same presynaptic spikes changes how fast the membrane leaks
           (as conductance-based synapses do in real neurons):
               beta_l[t] = sigmoid(beta_logit_l + U_l bus_l[t])     (per neuron and per step)
           so the network can make a neuron hold its potential for a long time, or clear it, depending on
           what it reads. U_l is low-rank and starts at zero, i.e. the model starts as the plain LIF network.
Optional (steps_per_token = T > 1): every token is shown for T simulation steps instead of one. Its input neuron
           fires at each of the T steps, the populations run T steps (so a neuron can fire up to T times per token),
           and the readout neurons integrate the bus over the T steps without leaking and are read and cleared at the
           end of the token: logits[t] = W_head mean_tau bus_L[t, tau]. Time constants are initialised T times longer
           in steps, i.e. the same in tokens. Costs T times the neuron steps per token; adds no parameters.
Optional (mem_heads > 0): plastic synapses (a fast Hebbian memory) in front of the population of every mem_every-th
           layer. Two small sets of LIF neurons read the bus: key neurons k and query neurons q, mem_heads groups of
           mem_keys each (short time constants, soft reset). Each group has a matrix of plastic synapses S
           (mem_keys x mem_dv) from its query neurons to mem_dv value channels. When key neuron j fires, the synapses
           of query neuron j are strengthened by the current value v = W_v bus (pre- and postsynaptic activity at the
           same time), and all of them decay back with the group's time constant:
               S[t] = lambda S[t-1] + k[t] v[t]^T            (lambda = sigmoid(decay_beta_logit), per group, learned)
           The query neurons that fire read it out, r[t] = S[t]^T q[t] (the sum of the rows of S whose query neuron
           fired), scaled to unit RMS per group when it is large, and the result enters the bus through ordinary
           synapses:
               bus_l <- bus_l + W_o (g * r / sqrt(mean(r^2) + mem_eps))
           (mem_eps = 1, the size of one value read once at initialisation: at many positions no query neuron meets
           a written key and r = 0, and a tiny eps there multiplies the gradients by 1/sqrt(eps); 1e-6 made training
           diverge.)
           A word that made some key neurons fire can thus be brought back later, by a word that makes the matching
           query neurons fire, as long ago as the decay allows. With binary k and q, writing and reading are additions;
           S is a fixed-size state per stream, like the membrane potentials (no history that grows with the text).
           W_o starts at zero, so the model starts as the network without them. Training computes all positions at
           once, in chunks of mem_chunk (plastic_scan).
Removed options (tried, did not help, deleted to keep the code small; the code is in git history):
           same-layer synapses rec_rank (docs/experiments.md 5.2), delayed synapses delay_k (5.9), the spiking bus
           spike_bus (5.23), the current scaled by (1 - beta) leak_gate (5.26), all last in commit 490e9e6;
           error-correcting writes to the plastic synapses mem_rule = "delta" (5.32), in commit c138343.
           Checkpoints that saved them switched off still load (Config.from_dict); ones that used them need that commit.
"""
import math
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from lif import lif, lif_leak

# Options removed from the code, with the value that meant "off" (see the end of the module docstring).
REMOVED = {"rec_rank": 0, "rec_init": None, "delay_k": 0, "delay_rank": None, "leak_gate": False, "spike_bus": False,
           "bus_tau_min": None, "bus_tau_max": None, "bus_theta": None, "mem_rule": "hebb"}


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
    gamma_init_min: float = -1.0  # >=0: initialise gamma uniformly in [gamma_init_min, gamma_init] instead
    leak_rank: int = 0        # >0: low-rank leak-modulating synapses (see module docstring)
    steps_per_token: int = 1  # >1: simulation steps per token (see module docstring)
    mem_heads: int = 0        # >0: plastic synapses with this many groups (see module docstring)
    mem_keys: int = 64        # key neurons (and query neurons) per group
    mem_dv: int = 64          # value channels per group
    mem_every: int = 1        # plastic synapses in front of every mem_every-th population (the last of each group)
    mem_tau_min: float = 8.0  # their decay time constants (tokens), initialised log-uniform over the groups
    mem_tau_max: float = 1024.0
    mem_theta: float = 1.5    # key and query neurons' threshold at initialisation
    mem_chunk: int = 128      # training: positions per chunk of plastic_scan
    mem_eps: float = 1.0      # read-out scaling r / sqrt(mean(r^2) + mem_eps) (see module docstring)
    mem_clock: int = 0        # >0: the first mem_clock groups are written by position, not content (EXPERIMENTAL)

    @classmethod
    def from_dict(cls, d):
        """Config from a saved dict (checkpoint "cfg"); drops removed options that were off, refuses ones that were on."""
        on = [k for k, off in REMOVED.items() if off is not None and k in d and d[k] != off]
        assert not on, f"this checkpoint uses {on}, removed from the code: check out commit 490e9e6 (c138343 for mem_rule)"
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class LIFPopulation(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, d = cfg.d_hidden, cfg.d_model
        self.w_in = nn.Linear(d, H, bias=True)
        self.w_out = nn.Linear(H, d, bias=False)
        T = cfg.steps_per_token   # tau_min / tau_max are in tokens; beta is per simulation step
        log_tau = torch.empty(H).uniform_(math.log(cfg.tau_min * T), math.log(cfg.tau_max * T))
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
            if cfg.gamma_init_min >= 0:
                g0 = torch.empty(H).uniform_(cfg.gamma_init_min, cfg.gamma_init)
            else:
                g0 = torch.full((H,), cfg.gamma_init)
            self.gamma_logit = nn.Parameter(torch.logit(g0))
        else:
            self.register_buffer("gamma_const", torch.ones(H), persistent=False)
        if cfg.leak_rank > 0:
            self.leak_in = nn.Linear(d, cfg.leak_rank, bias=False)
            self.leak_out = nn.Linear(cfg.leak_rank, H, bias=False)
            with torch.no_grad():
                nn.init.normal_(self.leak_in.weight, std=1.0 / math.sqrt(d))
                nn.init.zeros_(self.leak_out.weight)

    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def theta(self):
        return self.log_theta.exp()

    def gamma(self):
        return torch.sigmoid(self.gamma_logit) if self.cfg.reset_learn else self.gamma_const

    def forward(self, bus, state=None):
        """state: (membrane potentials (B,H), spikes of the previous step (B,H)) or None (= at rest). The spikes are
        carried along but not used (they were the input of the removed same-layer synapses)."""
        c = self.cfg
        v0 = None if state is None else state[0]
        I = self.w_in(bus)
        kw = dict(hard_reset=c.hard_reset, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg)
        if c.leak_rank > 0:   # leak changes per step; beta = sigmoid(beta_logit + M) is formed in the kernel
            M = self.leak_out(self.leak_in(bus))
            s, v_last = lif_leak(I, self.beta_logit, M, self.theta(), self.gamma(), v0, **kw)
        else:
            s, v_last = lif(I, self.beta(), self.theta(), self.gamma(), v0, **kw)
        return self.w_out(s), s, (v_last.detach(), s[:, -1].detach())


def plastic_scan(q, k, v, log_lam, S0=None, chunk=128):
    """Plastic synapses over a sequence. q, k: (B, h, L, m) spikes; v: (B, h, L, dv); log_lam: (h,) log of the decay
    per step; S0: (B, h, m, dv) synapses before the first position, or None (all zero). Returns
        r[t] = sum_{s<=t} lam^(t-s) (q[t] . k[s]) v[s] + lam^(t+1) S0^T q[t]        (B, h, L, dv)
    and the synapses after the last position, S[L-1] (B, h, m, dv; float32). Chunks of `chunk` positions: within a
    chunk all pairs at once, between chunks through S (the same result as stepping S[t] = lam S[t-1] + k[t] v[t]^T).
    All chunks are computed together (one batched product each for the pairs within chunks, the chunks' writes and the
    reads of the earlier chunks); only the synapses at the chunk boundaries are stepped one chunk at a time. On an H100
    this is 2.75x faster than looping over the chunks, with the same result (experiments/bench_scan.py, 5.32)."""
    B, h, L, m = q.shape
    dv = v.size(-1)
    C = min(chunk, L)
    nc = -(-L // C)
    pad = nc * C - L                     # the last chunk is padded with silent positions (q = k = 0, v = 0)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    qc, kc, vc = q.reshape(B, h, nc, C, m), k.reshape(B, h, nc, C, m), v.reshape(B, h, nc, C, dv)
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
    return out.reshape(B, h, nc * C, dv)[:, :, :L], S


class PlasticSynapses(nn.Module):
    """Key and query neurons and the plastic synapses between them and the value channels (module docstring)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        h, m, dv, d = cfg.mem_heads, cfg.mem_keys, cfg.mem_dv, cfg.d_model
        self.kq_in = nn.Linear(d, 2 * h * m, bias=True)    # input synapses of the key and the query neurons
        self.v_in = nn.Linear(d, h * dv, bias=False)       # value channels
        self.w_out = nn.Linear(h * dv, d, bias=False)      # readout into the bus (starts at zero)
        self.gain = nn.Parameter(torch.ones(h * dv))
        tau = torch.exp(torch.linspace(math.log(cfg.mem_tau_min), math.log(cfg.mem_tau_max), h))
        self.decay_beta_logit = nn.Parameter(torch.logit(torch.exp(-1.0 / tau)))
        kq_tau = torch.exp(torch.empty(2 * h * m).uniform_(math.log(0.5), math.log(2.0)))   # short: mostly this token
        self.kq_beta_logit = nn.Parameter(torch.logit(torch.exp(-1.0 / kq_tau)))
        self.kq_log_theta = nn.Parameter(torch.full((2 * h * m,), math.log(cfg.mem_theta)))
        with torch.no_grad():
            nn.init.normal_(self.kq_in.weight, std=1.0 / math.sqrt(d))
            nn.init.zeros_(self.kq_in.bias)
            nn.init.normal_(self.v_in.weight, std=1.0 / math.sqrt(d))
            nn.init.zeros_(self.w_out.weight)
        if cfg.mem_clock:   # clock groups: a ring of m clock neurons, neuron (t mod m) fires at position t and acts as
            G = cfg.mem_clock  # the key (row t mod m is written); clock_q[g, j, i]: synapse clock neuron i -> query j
            self.clock_q = nn.Parameter(torch.zeros(G, m, m))
            with torch.no_grad():   # a row is rewritten every m positions: what was there should have faded by then
                self.decay_beta_logit[:G] = torch.logit(torch.tensor(math.exp(-3.0 / m)))

    def forward(self, bus, state=None):
        """state: (synapses S (B, h, m, dv), key and query potentials (B, 2hm)) or None. Returns the current into the
        bus, the key and query spikes (B, L, 2hm) and the new state."""
        c = self.cfg
        B, L, _ = bus.shape
        h, m, dv = c.mem_heads, c.mem_keys, c.mem_dv
        S0, v0 = (None, None) if state is None else state[:2]
        I = self.kq_in(bus)
        G = c.mem_clock
        if G:
            p0 = 0 if state is None else state[2]
            pos = (p0 + torch.arange(L, device=bus.device)) % m
            clk = F.one_hot(pos, m).to(I.dtype)                                     # (L, m): the clock's spikes
            extra = self.clock_q[:, :, pos].permute(2, 0, 1).reshape(L, G * m)       # into the clock groups' queries
            I = I + F.pad(extra.to(I.dtype), (h * m, h * m - G * m))
        s, v_last = lif(I, torch.sigmoid(self.kq_beta_logit), self.kq_log_theta.exp(), None, v0,
                        hard_reset=False, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg)
        k = s[..., :h * m].reshape(B, L, h, m).transpose(1, 2)
        if G:
            k = torch.cat([clk.to(k.dtype).expand(B, G, L, m), k[:, G:]], 1)
        q = s[..., h * m:].reshape(B, L, h, m).transpose(1, 2)
        val = self.v_in(bus).view(B, L, h, dv).transpose(1, 2)
        r, S = plastic_scan(q.to(val.dtype), k.to(val.dtype), val, F.logsigmoid(self.decay_beta_logit), S0, c.mem_chunk)
        r = r.float()
        r = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + c.mem_eps)).to(val.dtype)
        r = r.transpose(1, 2).reshape(B, L, h * dv) * self.gain.to(val.dtype)
        if G:
            return self.w_out(r), s, (S.detach(), v_last.detach(), (p0 + L) % m)
        return self.w_out(r), s, (S.detach(), v_last.detach())


class SpikingLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=1.0)
        self.layers = nn.ModuleList([LIFPopulation(cfg) for _ in range(cfg.n_layers)])
        self.mem_layers = [i for i in range(cfg.n_layers) if cfg.mem_heads and (i + 1) % cfg.mem_every == 0]
        if self.mem_layers:   # plastic synapses in front of these populations
            self.mems = nn.ModuleDict({str(i): PlasticSynapses(cfg) for i in self.mem_layers})
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        nn.init.normal_(self.head.weight, std=cfg.head_init)
        self.keep_bus = False   # training only (train.py --hint_weight): keep the bus after each layer in last_bus
        self.last_bus = None

    def forward(self, idx, targets=None, state=None, return_spikes=False):
        """idx: (B, L) token ids. state: per-layer (membrane potentials, last spikes) or None (at rest); with plastic
        synapses followed by one more entry, the list of their states.
        Returns logits, loss, new_state, and per-layer firing rates (or spike tensors, (B, L*T, H) with
        T = steps_per_token, followed with plastic synapses by their key and query spikes, (B, L, 2 mem_heads mem_keys)
        per such layer)."""
        T = self.cfg.steps_per_token
        assert T == 1 or not self.mem_layers, "plastic synapses are written once per token: not combined with T > 1"
        mem_state, mem_spikes = [], []
        bus = self.embed(idx)
        if T > 1:   # each token is shown for T steps: its input neuron fires at every one of them
            bus = bus.repeat_interleave(T, dim=1)
        new_state, rates, spikes, buses = [], [], [], []
        for i, layer in enumerate(self.layers):
            if i in self.mem_layers:
                j = self.mem_layers.index(i)
                cur, sk, ms = self.mems[str(i)](bus, None if state is None else state[-1][j])
                bus = bus + cur
                mem_state.append(ms)
                if return_spikes:
                    mem_spikes.append(sk.detach())
            out, s, layer_state = layer(bus, None if state is None else state[i])
            bus = bus + out
            if self.keep_bus:
                buses.append(bus)
            new_state.append(layer_state)
            rates.append(s.detach().mean())
            if return_spikes:
                spikes.append(s.detach())
        if T > 1:   # readout neurons integrate over the token's T steps, then are read and cleared
            bus = bus.view(bus.size(0), -1, T, bus.size(-1)).mean(2)
        if self.keep_bus:
            self.last_bus = [b.view(b.size(0), -1, T, b.size(-1)).mean(2) if T > 1 else b for b in buses]
        logits = self.head(bus)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        extra = spikes + mem_spikes if return_spikes else torch.stack(rates)
        if self.mem_layers:
            new_state.append(mem_state)
        return logits, loss, new_state, extra

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, eos_id=None, generator=None):
        """Streaming generation: the prompt is run once, then one simulation step per new token.
        Carried from step to step: every neuron's membrane potential (and the plastic synapses, when used)."""
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

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
  LIF    : V[t] = beta * V[t-1] + I[t];  s[t] = [V[t] >= theta];  reset (soft: V -= theta*s).
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
Optional (leak_gate, with leak_rank > 0): the current is scaled by (1 - beta):
               V[t] = beta[t] * V[t-1] + (1 - beta[t]) * I[t]
           This is the exact one-step solution of the LIF equation tau dV/dt = -V + I with tau held within each step
           (the input sets the potential the membrane relaxes towards). While tau is fixed, the plain V = beta V + I
           differs from it only by a constant scale of each neuron's current; once the leak-modulating synapses change
           tau from step to step, the two differ.
           A neuron that the leak-modulating synapses make hold its potential (beta near 1) then also takes in little
           new current, so what it holds is not overwritten; one that is made to leak fast takes in the new current
           fully (as the forget and input gates are tied in some linear recurrent networks). The threshold starts
           (1 - beta) times lower, so at initialisation (U_l = 0) the network computes exactly what it does without
           the option; only the trained behaviour differs.
Optional (delay_k > 0): delayed synapses. Spikes also reach the population through synapses whose
           signal arrives k = 1..delay_k steps (tokens) later, with their own weights:
               I_l[t] = W_in_l bus_l[t] + b_l + sum_k D_lk bus_l[t-k]
           so a neuron can respond to "this token now and that one k steps ago" directly, instead of only
           through what is left of the earlier input in its own decaying potential. D_lk = Q_l P_lk is
           low-rank (P_lk: delay_rank x d_model per delay, Q_l shared), Q_l starts at zero.
           For streaming, the last delay_k values of P_lk bus_l are carried along with the potentials.
Optional (steps_per_token = T > 1): every token is shown for T simulation steps instead of one. Its input neuron
           fires at each of the T steps, the populations run T steps (so a neuron can fire up to T times per token),
           and the readout neurons integrate the bus over the T steps without leaking and are read and cleared at the
           end of the token: logits[t] = W_head mean_tau bus_L[t, tau]. Time constants are initialised T times longer
           in steps, i.e. the same in tokens. Costs T times the neuron steps per token; adds no parameters.
Optional (spike_bus): spiking bus. Before each population, the bus passes through a population of relay LIF neurons,
           two per bus channel: an ON neuron driven by +bus_l and an OFF neuron driven by -bus_l, with soft reset
           (V -= theta at a spike, so over time the spike count follows the input) and their own learned leak and
           threshold. The population then reads z_l = s_on - s_off (one of -1, 0, 1 per channel) instead of bus_l:
               I_l[t] = W_in_l z_l[t] + b_l      (and the leak-modulating synapses read z_l too)
           so every synapse into a population is driven by spikes, and on hardware only the channels that fired cost
           anything. The bus itself is still the sum of weighted spikes; the readout still reads bus_L.
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
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lif import lif, lif_leak, lif_rec


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
    rec_rank: int = 0         # >0: low-rank same-layer synapses R = A @ B acting on the previous step's spikes
    rec_init: float = 1.0     # std of A is rec_init/sqrt(d_hidden); B starts at 0 so R = 0 at init
    leak_rank: int = 0        # >0: low-rank leak-modulating synapses (see module docstring)
    leak_gate: bool = False   # with leak_rank > 0: current scaled by (1 - beta[t]) (see module docstring)
    delay_k: int = 0          # >0: delayed synapses with delays 1..delay_k tokens (see module docstring)
    delay_rank: int = 64      # rank of each delay's synapse matrix
    steps_per_token: int = 1  # >1: simulation steps per token (see module docstring)
    spike_bus: bool = False   # relay neurons turn the bus into spikes before each population (see module docstring)
    bus_tau_min: float = 0.5  # relay time constants (tokens), initialised log-uniform in [bus_tau_min, bus_tau_max]
    bus_tau_max: float = 4.0
    bus_theta: float = 1.0    # relay threshold at initialisation (the bus starts with roughly unit scale)
    mem_heads: int = 0        # >0: plastic synapses with this many groups (see module docstring)
    mem_keys: int = 64        # key neurons (and query neurons) per group
    mem_dv: int = 64          # value channels per group
    mem_every: int = 1        # plastic synapses in front of every mem_every-th population (the last of each group)
    mem_tau_min: float = 8.0  # their decay time constants (tokens), initialised log-uniform over the groups
    mem_tau_max: float = 1024.0
    mem_theta: float = 1.5    # key and query neurons' threshold at initialisation
    mem_chunk: int = 128      # training: positions per chunk of plastic_scan
    mem_eps: float = 1.0      # read-out scaling r / sqrt(mean(r^2) + mem_eps) (see module docstring)


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
        if cfg.leak_gate:   # the current is scaled by (1 - beta): the same network at initialisation
            assert cfg.leak_rank > 0, "leak_gate needs the leak-modulating synapses (leak_rank > 0)"
            theta0 = theta0 * (1 - beta)
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
        if cfg.rec_rank > 0:
            self.rec_A = nn.Parameter(torch.randn(H, cfg.rec_rank) * cfg.rec_init / math.sqrt(H))
            self.rec_B = nn.Parameter(torch.zeros(cfg.rec_rank, H))
        if cfg.leak_rank > 0:
            assert cfg.rec_rank == 0, "leak-modulating synapses are implemented for independent neurons only"
            self.leak_in = nn.Linear(d, cfg.leak_rank, bias=False)
            self.leak_out = nn.Linear(cfg.leak_rank, H, bias=False)
            with torch.no_grad():
                nn.init.normal_(self.leak_in.weight, std=1.0 / math.sqrt(d))
                nn.init.zeros_(self.leak_out.weight)
        if cfg.delay_k > 0:
            self.delay_in = nn.Linear(d, cfg.delay_k * cfg.delay_rank, bias=False)   # P_l1 .. P_lK stacked
            self.delay_out = nn.Linear(cfg.delay_rank, H, bias=False)                # Q_l
            with torch.no_grad():
                nn.init.normal_(self.delay_in.weight, std=1.0 / math.sqrt(d))
                nn.init.zeros_(self.delay_out.weight)

    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def theta(self):
        return self.log_theta.exp()

    def gamma(self):
        return torch.sigmoid(self.gamma_logit) if self.cfg.reset_learn else self.gamma_const

    def delayed(self, bus, hist):
        """Current through the delayed synapses. hist: P bus of the last delay_k steps (B, K, K*r) or None."""
        K, r = self.cfg.delay_k, self.cfg.delay_rank
        A = self.delay_in(bus)                                          # (B, L, K*r): P_k bus[t] for every k
        L = A.size(1)
        if hist is None:                                                # before the first token: nothing
            hist = A.new_zeros(A.size(0), K, K * r)
        Af = torch.cat([hist.to(A.dtype), A], 1)                        # Af[:, K + t] = A[:, t]
        D = sum(Af[:, K - k:K - k + L, (k - 1) * r:k * r] for k in range(1, K + 1))   # sum_k P_k bus[t-k]
        return self.delay_out(D), Af[:, -K:].detach()

    def forward(self, bus, state=None):
        """state: (membrane potentials (B,H), spikes of the previous step (B,H)[, delay history]) or None
        (= at rest)."""
        c = self.cfg
        v0, s0 = (None, None) if state is None else state[:2]
        I = self.w_in(bus)
        hist = None
        if c.delay_k > 0:
            Id, hist = self.delayed(bus, None if state is None else state[2])
            I = I + Id
        kw = dict(hard_reset=c.hard_reset, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg)
        if c.rec_rank > 0:
            s, v_last = lif_rec(I, self.beta(), self.theta(), self.rec_A, self.rec_B, self.gamma(), v0, s0, **kw)
        else:
            if c.leak_rank > 0:   # leak changes per step; beta = sigmoid(beta_logit + M) is formed in the kernel
                M = self.leak_out(self.leak_in(bus))
                if c.leak_gate:   # (1 - beta[t]) = sigmoid(-(beta_logit + M)), computed in float32
                    I = (I.float() * torch.sigmoid(-(self.beta_logit + M.float()))).to(I.dtype)
                s, v_last = lif_leak(I, self.beta_logit, M, self.theta(), self.gamma(), v0, **kw)
            else:
                s, v_last = lif(I, self.beta(), self.theta(), self.gamma(), v0, **kw)
        new_state = (v_last.detach(), s[:, -1].detach()) + ((hist,) if hist is not None else ())
        return self.w_out(s), s, new_state


class RelayPopulation(nn.Module):
    """ON/OFF relay neurons for the spiking bus: neuron j is driven by +bus[j], neuron d + j by -bus[j]; soft reset."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, T = cfg.d_model, cfg.steps_per_token
        log_tau = torch.empty(2 * d).uniform_(math.log(cfg.bus_tau_min * T), math.log(cfg.bus_tau_max * T))
        self.beta_logit = nn.Parameter(torch.logit(torch.exp(-1.0 / log_tau.exp())))
        self.log_theta = nn.Parameter(torch.full((2 * d,), math.log(cfg.bus_theta)))

    def forward(self, bus, v0=None):
        """Returns z = s_on - s_off (B, L, d), the spikes (B, L, 2d) and the final potentials (B, 2d)."""
        c = self.cfg
        s, v = lif(torch.cat([bus, -bus], -1), torch.sigmoid(self.beta_logit), self.log_theta.exp(), None, v0,
                   hard_reset=False, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg)
        d = bus.size(-1)
        return s[..., :d] - s[..., d:], s, v.detach()


def plastic_scan(q, k, v, log_lam, S0=None, chunk=128):
    """Plastic synapses over a sequence. q, k: (B, h, L, m) spikes; v: (B, h, L, dv); log_lam: (h,) log of the decay
    per step; S0: (B, h, m, dv) synapses before the first position, or None (all zero). Returns
        r[t] = sum_{s<=t} lam^(t-s) (q[t] . k[s]) v[s] + lam^(t+1) S0^T q[t]        (B, h, L, dv)
    and the synapses after the last position, S[L-1] (B, h, m, dv; float32). Chunks of `chunk` positions: within a
    chunk all pairs at once, between chunks through S (the same result as stepping S[t] = lam S[t-1] + k[t] v[t]^T)."""
    B, h, L, m = q.shape
    S = q.new_zeros(B, h, m, v.size(-1), dtype=torch.float32) if S0 is None else S0.float()
    ll = log_lam.float().view(1, h, 1, 1)
    out = []
    for a in range(0, L, chunk):
        qc, kc, vc = q[:, :, a:a + chunk], k[:, :, a:a + chunk], v[:, :, a:a + chunk]
        n = qc.size(2)
        i = torch.arange(n, device=q.device, dtype=torch.float32)
        diff = i[:, None] - i[None, :]
        D = torch.exp(ll * diff.clamp(min=0)) * (diff >= 0)                       # (1, h, n, n): lam^(t-s), s <= t
        A = (qc @ kc.transpose(-1, -2)) * D.to(qc.dtype)
        inter = (qc * torch.exp(ll * (i + 1).view(1, 1, n, 1)).to(qc.dtype)) @ S.to(qc.dtype)
        out.append(A @ vc + inter)
        w = torch.exp(ll * (n - 1 - i).view(1, 1, n, 1)).to(kc.dtype)           # decay of each write to the chunk end
        S = torch.exp(ll * n) * S + ((kc * w).transpose(-1, -2) @ vc).float()
    return torch.cat(out, 2), S


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

    def forward(self, bus, state=None):
        """state: (synapses S (B, h, m, dv), key and query potentials (B, 2hm)) or None. Returns the current into the
        bus, the key and query spikes (B, L, 2hm) and the new state."""
        c = self.cfg
        B, L, _ = bus.shape
        h, m, dv = c.mem_heads, c.mem_keys, c.mem_dv
        S0, v0 = (None, None) if state is None else state
        s, v_last = lif(self.kq_in(bus), torch.sigmoid(self.kq_beta_logit), self.kq_log_theta.exp(), None, v0,
                        hard_reset=False, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg)
        k = s[..., :h * m].reshape(B, L, h, m).transpose(1, 2)
        q = s[..., h * m:].reshape(B, L, h, m).transpose(1, 2)
        val = self.v_in(bus).view(B, L, h, dv).transpose(1, 2)
        r, S = plastic_scan(q.to(val.dtype), k.to(val.dtype), val, F.logsigmoid(self.decay_beta_logit), S0, c.mem_chunk)
        r = r.float()
        r = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + c.mem_eps)).to(val.dtype)
        r = r.transpose(1, 2).reshape(B, L, h * dv) * self.gain.to(val.dtype)
        return self.w_out(r), s, (S.detach(), v_last.detach())


class SpikingLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.steps_per_token == 1 or cfg.delay_k == 0, "delays are counted in tokens; not combined with T > 1"
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=1.0)
        self.layers = nn.ModuleList([LIFPopulation(cfg) for _ in range(cfg.n_layers)])
        self.mem_layers = [i for i in range(cfg.n_layers) if cfg.mem_heads and (i + 1) % cfg.mem_every == 0]
        if self.mem_layers:   # plastic synapses in front of these populations
            self.mems = nn.ModuleDict({str(i): PlasticSynapses(cfg) for i in self.mem_layers})
        if cfg.spike_bus:
            self.relays = nn.ModuleList([RelayPopulation(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        nn.init.normal_(self.head.weight, std=cfg.head_init)
        self.keep_bus = False   # training only (train.py --hint_weight): keep the bus after each layer in last_bus
        self.last_bus = None

    def forward(self, idx, targets=None, state=None, return_spikes=False):
        """idx: (B, L) token ids. state: per-layer (membrane potentials, last spikes) or None (at rest); with plastic
        synapses followed by one more entry, the list of their states.
        Returns logits, loss, new_state, and per-layer firing rates (or spike tensors, (B, L*T, H) with
        T = steps_per_token, followed with spike_bus by the relay spikes, (B, L*T, 2 d_model) per layer, and with
        plastic synapses by their key and query spikes, (B, L, 2 mem_heads mem_keys) per such layer)."""
        T = self.cfg.steps_per_token
        assert T == 1 or not self.mem_layers, "plastic synapses are written once per token: not combined with T > 1"
        mem_state, mem_spikes = [], []
        bus = self.embed(idx)
        if T > 1:   # each token is shown for T steps: its input neuron fires at every one of them
            bus = bus.repeat_interleave(T, dim=1)
        new_state, rates, spikes, relay_spikes, buses = [], [], [], [], []
        for i, layer in enumerate(self.layers):
            st = None if state is None else state[i]
            if i in self.mem_layers:
                j = self.mem_layers.index(i)
                cur, sk, ms = self.mems[str(i)](bus, None if state is None else state[-1][j])
                bus = bus + cur
                mem_state.append(ms)
                if return_spikes:
                    mem_spikes.append(sk.detach())
            if self.cfg.spike_bus:   # state[i] = (relay potentials,) + the population's state
                z, sz, vz = self.relays[i](bus, None if st is None else st[0])
                out, s, layer_state = layer(z, None if st is None else st[1:])
                layer_state = (vz,) + layer_state
                if return_spikes:
                    relay_spikes.append(sz.detach())
            else:
                out, s, layer_state = layer(bus, st)
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
        extra = spikes + relay_spikes + mem_spikes if return_spikes else torch.stack(rates)
        if self.mem_layers:
            new_state.append(mem_state)
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

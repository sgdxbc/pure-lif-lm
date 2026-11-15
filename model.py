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
No attention, no normalisation layers.
Optional (leak_rank > 0): leak-modulating synapses. Besides the ordinary synapses that inject current,
           a second set of synapses from the same presynaptic spikes changes how fast the membrane leaks
           (as conductance-based synapses do in real neurons):
               beta_l[t] = sigmoid(beta_logit_l + U_l bus_l[t])     (per neuron and per step)
           so the network can make a neuron hold its potential for a long time, or clear it, depending on
           what it reads. U_l is low-rank and starts at zero, i.e. the model starts as the plain LIF network.
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
    delay_k: int = 0          # >0: delayed synapses with delays 1..delay_k tokens (see module docstring)
    delay_rank: int = 64      # rank of each delay's synapse matrix
    steps_per_token: int = 1  # >1: simulation steps per token (see module docstring)


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
                s, v_last = lif_leak(I, self.beta_logit, M, self.theta(), self.gamma(), v0, **kw)
            else:
                s, v_last = lif(I, self.beta(), self.theta(), self.gamma(), v0, **kw)
        new_state = (v_last.detach(), s[:, -1].detach()) + ((hist,) if hist is not None else ())
        return self.w_out(s), s, new_state


class SpikingLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.steps_per_token == 1 or cfg.delay_k == 0, "delays are counted in tokens; not combined with T > 1"
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=1.0)
        self.layers = nn.ModuleList([LIFPopulation(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        nn.init.normal_(self.head.weight, std=cfg.head_init)

    def forward(self, idx, targets=None, state=None, return_spikes=False):
        """idx: (B, L) token ids. state: per-layer (membrane potentials, last spikes) or None (at rest).
        Returns logits, loss, new_state, and per-layer firing rates (or spike tensors, (B, L*T, H) with
        T = steps_per_token)."""
        T = self.cfg.steps_per_token
        bus = self.embed(idx)
        if T > 1:   # each token is shown for T steps: its input neuron fires at every one of them
            bus = bus.repeat_interleave(T, dim=1)
        new_state, rates, spikes = [], [], []
        for i, layer in enumerate(self.layers):
            out, s, layer_state = layer(bus, None if state is None else state[i])
            bus = bus + out
            new_state.append(layer_state)
            rates.append(s.detach().mean())
            if return_spikes:
                spikes.append(s.detach())
        if T > 1:   # readout neurons integrate over the token's T steps, then are read and cleared
            bus = bus.view(bus.size(0), -1, T, bus.size(-1)).mean(2)
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

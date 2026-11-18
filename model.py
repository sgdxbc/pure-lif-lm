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
No attention, no normalisation layers (by default; see bus_norm below).
Optional (leak_rank > 0): leak-modulating synapses. Besides the ordinary synapses that inject current,
           a second set of synapses from the same presynaptic spikes changes how fast the membrane leaks
           (as conductance-based synapses do in real neurons):
               beta_l[t] = sigmoid(beta_logit_l + U_l bus_l[t])     (per neuron and per step)
           so the network can make a neuron hold its potential for a long time, or clear it, depending on
           what it reads. U_l is low-rank and starts at zero, i.e. the model starts as the plain LIF network.
Optional (in_groups = k > 1): a word is coded by k spikes instead of one. There are k groups of n = ceil(vocab^(1/k))
           input neurons; word x makes neuron (x // n^j) % n of group j fire, for j = 0..k-1 (the k base-n digits of
           its number, so every word has its own combination; or, k = 2 with train.py --in_class_file, its class and
           its position in the class), and the injected current is the sum of their k synapse rows. k * n rows instead of vocab (k = 2: 2 * 225 = 450 for 50,304 words): the input synapses
           almost vanish from the storage, at the price of words that share a digit sharing part of their current.
Optional (out_classes = C > 0): two-level readout. The words are split into C classes of at most m = ceil(vocab / C)
           (word_cls, set by train.py --class_file, or by number). C class readout neurons give the probability
           of each class, P(c) = softmax(W_c bus); each class has m word readout neurons giving the probability of
           each of its words, P(w | c) = softmax over the class of (W_w bus). P(w) = P(c(w)) P(w | c(w)) is a full
           distribution over the vocabulary (training and evaluation are unchanged; forward returns log P(w) in
           place of the logits). Generation needs only the C class neurons and the m word neurons of the class drawn,
           (C + m) * d_model readout weights per token instead of vocab * d_model.
           With out_short = S > 0 (an adaptive softmax), S words (the most frequent, set by the class file) are
           not in a class: each has its own neuron beside the C class neurons, P(w) = softmax over the S + C of
           them, and only the other vocab - S words are split into the C classes (m = ceil((vocab - S) / C)).
           (S + C + m) * d_model readout weights per token.
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
           mem_write_avg (5.34): write (1 - lambda) v instead of v, so that S is a running average of k v^T rather
           than a sum (a row written at every token grows to about tau times a value without it, and the gradient of
           a query neuron is its row of S). lambda is fixed after training, so this is folded into W_v: the same
           network, only trained in a different parametrisation.
           A word that made some key neurons fire can thus be brought back later, by a word that makes the matching
           query neurons fire, as long ago as the decay allows. With binary k and q, writing and reading are additions;
           S is a fixed-size state per stream, like the membrane potentials (no history that grows with the text).
           W_o starts at zero, so the model starts as the network without them. Training computes all positions at
           once, in chunks of mem_chunk (plastic_scan).
Optional (in_rank = r > 0, docs/experiments.md 5.46): each population reads the bus through a rank-r W_in = A_l B_l
           (B_l: r x d_model, A_l: d_hidden x r), r (d_model + d_hidden) dense multiply-adds per step instead of
           d_model d_hidden: a cheaper reading of the dense, real-valued bus.
Optional (mem_src = "spikes", 5.46): the key, query and value synapses of the plastic synapses in front of population
           l > 0 read the spikes of population l - 1 (d_hidden binary inputs firing at a few percent) instead of the
           bus: additions of the synapse rows of the neurons that fired instead of dense multiply-adds (the first
           population's still read the bus). W_o, the readout of S into the bus, is unchanged.
Optional (spike_soft = rho > 0, docs/experiments.md 5.48): every LIF neuron (the populations' and the key and query
           neurons) outputs (1 - rho) [V >= theta] + rho f((V - theta) / theta) instead of a 0/1 spike, and the reset
           uses that output; f rises smoothly from 0 to 1 around the threshold (lif.py soft_fn). rho = 1 is a
           non-spiking twin of the network with exact gradients (a diagnostic: what the 0/1 signals and the surrogate
           gradient cost); train.py --soft_hold/--soft_end lower rho to 0 during training, so the trained network
           spikes (soft-to-hard training). Not a hardware network while rho > 0.
Optional (bus_norm, docs/experiments.md 5.49): every reader of the bus (each population's W_in and leak synapses, the
           plastic synapses' key, query and value synapses, the readout) reads it divided by its RMS over the d_model
           channels of the token, as a Transformer's pre-norm does (no gain: the reader's own synapses can scale it).
           The bus itself is unchanged. A deviation: one division per reader and token (a divisive gain control), but
           the neurons still exchange only spikes. At initialisation the bus has about unit RMS, so this changes
           little at first; it makes what the readers see independent of how large the bus grows during training.
Low-precision weights (wbits, docs/experiments.md 5.38): every weight matrix (the 2-D parameters: input rows, W_in, W_out,
           leak synapses, plastic-synapse inputs and outputs, readout) is meant to be stored as symmetric wbits-bit
           integers with one scale per group of wgroup inputs (quantize_weight). The forward pass does not round:
           train.py --wbits trains with the matrices rounded (quantization-aware training: forward and backward see
           the rounded values, the update goes to the full-precision ones, a straight-through estimate), and
           sample.load rounds them when it loads such a checkpoint, so every evaluation sees the rounded network.
           wbits8: matrices whose names match (e.g. "^head,^embed": the readout and the input rows) get 8 bits instead.
Removed options (tried, did not help, deleted to keep the code small; the code is in git history):
           same-layer synapses rec_rank (docs/experiments.md 5.2), delayed synapses delay_k (5.9), the spiking bus
           spike_bus (5.23), the current scaled by (1 - beta) leak_gate (5.26), all last in commit 490e9e6;
           error-correcting writes to the plastic synapses mem_rule = "delta" (5.32), in commit c138343; groups of
           plastic synapses written by position instead of content, mem_clock (5.33), in commit c0868f9; rows of the
           plastic synapses read divided by their write count, mem_count_norm (5.34), in commit bf7fed7.
           Checkpoints that saved them switched off still load (Config.from_dict); ones that used them need that commit.
"""
import math
import re
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from lif import lif, lif_leak

# Options removed from the code, with the value that meant "off" (see the end of the module docstring).
REMOVED = {"rec_rank": 0, "rec_init": None, "delay_k": 0, "delay_rank": None, "leak_gate": False, "spike_bus": False,
           "bus_tau_min": None, "bus_tau_max": None, "bus_theta": None, "mem_rule": "hebb", "mem_clock": 0,
           "mem_count_norm": False}


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
    in_groups: int = 0        # >1: each word is coded by this many input spikes (module docstring)
    in_short: int = 0         # with in_groups: this many words (a short list) get an input neuron of their own
    out_classes: int = 0      # >0: two-level readout over this many word classes (module docstring)
    out_short: int = 0        # with out_classes: this many words (a short list) get a readout neuron of their own
    learn_beta: bool = True
    learn_theta: bool = True
    reset_learn: bool = False # multiplicative reset V <- V*(1 - gamma*s) with a learned gamma per neuron
    gamma_init: float = 0.9   # (only with hard_reset=True; gamma = 1 is the classic reset-to-zero)
    gamma_init_min: float = -1.0  # >=0: initialise gamma uniformly in [gamma_init_min, gamma_init] instead
    leak_rank: int = 0        # >0: low-rank leak-modulating synapses (see module docstring)
    in_rank: int = 0          # >0: each population reads the bus through a rank-in_rank W_in (module docstring)
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
    mem_write_avg: bool = False   # write (1 - lambda) v: S is a running average, not a sum (module docstring)
    mem_src: str = "bus"      # "spikes": key, query and value synapses read the previous population's spikes
    spike_soft: float = 0.0   # >0: neurons output (1 - rho) * spike + rho * a smooth function (module docstring);
                              # train.py --soft_hold/--soft_end lower it during training
    bus_norm: bool = False    # every reader of the bus sees it scaled to unit RMS per token (module docstring)
    wbits: int = 0            # >0: the weight matrices are used rounded to this many bits (module docstring)
    wgroup: int = 128         # one scale per this many inputs of a weight matrix (per row if it does not divide)
    wbits8: str = ""          # with wbits: comma-separated regular expressions; matrices whose names match get 8 bits

    @classmethod
    def from_dict(cls, d):
        """Config from a saved dict (checkpoint "cfg"); drops removed options that were off, refuses ones that were on."""
        on = [k for k, off in REMOVED.items() if off is not None and k in d and d[k] != off]
        assert not on, (f"this checkpoint uses {on}, removed from the code: check out commit 490e9e6 "
                        "(c138343 for mem_rule, c0868f9 for mem_clock, bf7fed7 for mem_count_norm)")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def quantize_weight(w, bits, group=0):
    """Symmetric `bits`-bit integers times a scale: one scale per group of `group` inputs (columns), or per output
    row when `group` is 0 or does not divide the input width. The scale makes the largest |w| of its group the
    largest integer. bits = 0: w unchanged."""
    if not bits:
        return w
    qmax = 2 ** (bits - 1) - 1
    o, i = w.shape
    if group and i % group == 0:
        wg = w.reshape(o, i // group, group)
        s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
        return (torch.round(wg / s).clamp(-qmax, qmax) * s).reshape(o, i)
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax
    return torch.round(w / s).clamp(-qmax, qmax) * s


@torch.no_grad()
def round_weights_(model, bits, group=128, bits8=""):
    """Round the model's weight matrices (its 2-D parameters) in place, to 8 bits those whose names match one of the
    comma-separated regular expressions bits8; returns their previous values, which restore_weights_ puts back."""
    pats = [re.compile(x) for x in bits8.split(",") if x]
    saved = []
    for n, p in model.named_parameters():
        if p.dim() == 2 and p.is_floating_point():
            saved.append(p.detach().clone())
            b = 8 if any(x.search(n) for x in pats) else bits
            p.copy_(quantize_weight(p.float(), b, group).to(p.dtype))
    return saved


@torch.no_grad()
def restore_weights_(model, saved):
    it = iter(saved)
    for p in model.parameters():
        if p.dim() == 2 and p.is_floating_point():
            p.copy_(next(it))


class LIFPopulation(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, d = cfg.d_hidden, cfg.d_model
        assert cfg.in_rank == 0 or cfg.in_norm != "weight", "in_rank with in_norm 'weight' is not implemented"
        if cfg.in_rank > 0:   # W_in = w_in . w_in_low, rank in_rank
            self.w_in_low = nn.Linear(d, cfg.in_rank, bias=False)
        self.w_in = nn.Linear(cfg.in_rank or d, H, bias=True)
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
            nn.init.normal_(self.w_in.weight, std=1.0 / math.sqrt(cfg.in_rank or d))
            if cfg.in_rank > 0:   # the same variance of the current at initialisation as the full-rank W_in
                nn.init.normal_(self.w_in_low.weight, std=1.0 / math.sqrt(d))
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
        I = self.w_in(self.w_in_low(bus)) if c.in_rank > 0 else self.w_in(bus)
        kw = dict(hard_reset=c.hard_reset, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg, soft=c.spike_soft)
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

    def __init__(self, cfg: Config, src_dim=0):
        """src_dim > 0: the key, query and value synapses read spikes of that many neurons (mem_src "spikes": the
        previous population) instead of the bus."""
        super().__init__()
        self.cfg = cfg
        h, m, dv, d = cfg.mem_heads, cfg.mem_keys, cfg.mem_dv, cfg.d_model
        n_in = src_dim or d
        # spikes are 0/1 at a rate of a few percent: scale the initial synapses so that the input current has about
        # the variance it has from the bus (unit-variance entries), assuming 5% firing
        std_in = 1.0 / math.sqrt(0.05 * src_dim) if src_dim else 1.0 / math.sqrt(d)
        self.kq_in = nn.Linear(n_in, 2 * h * m, bias=True)    # input synapses of the key and the query neurons
        self.v_in = nn.Linear(n_in, h * dv, bias=False)       # value channels
        self.w_out = nn.Linear(h * dv, d, bias=False)      # readout into the bus (starts at zero)
        self.gain = nn.Parameter(torch.ones(h * dv))
        tau = torch.exp(torch.linspace(math.log(cfg.mem_tau_min), math.log(cfg.mem_tau_max), h))
        self.decay_beta_logit = nn.Parameter(torch.logit(torch.exp(-1.0 / tau)))
        kq_tau = torch.exp(torch.empty(2 * h * m).uniform_(math.log(0.5), math.log(2.0)))   # short: mostly this token
        self.kq_beta_logit = nn.Parameter(torch.logit(torch.exp(-1.0 / kq_tau)))
        self.kq_log_theta = nn.Parameter(torch.full((2 * h * m,), math.log(cfg.mem_theta)))
        with torch.no_grad():
            nn.init.normal_(self.kq_in.weight, std=std_in)
            nn.init.zeros_(self.kq_in.bias)
            nn.init.normal_(self.v_in.weight, std=std_in)
            nn.init.zeros_(self.w_out.weight)

    def forward(self, bus, state=None, src=None):
        """state: (synapses S (B, h, m, dv), key and query potentials (B, 2hm)) or None; src: the spikes the key, query
        and value synapses read (B, L, src_dim), or None for the bus. Returns the current into the bus, the key and
        query spikes (B, L, 2hm) and the new state."""
        c = self.cfg
        B, L, _ = bus.shape
        h, m, dv = c.mem_heads, c.mem_keys, c.mem_dv
        S0, v0 = (None, None) if state is None else state
        x = bus if src is None else src.to(bus.dtype)
        s, v_last = lif(self.kq_in(x), torch.sigmoid(self.kq_beta_logit), self.kq_log_theta.exp(), None, v0,
                        hard_reset=False, alpha=c.alpha, detach_reset=c.detach_reset, sg=c.sg, soft=c.spike_soft)
        k = s[..., :h * m].reshape(B, L, h, m).transpose(1, 2)
        q = s[..., h * m:].reshape(B, L, h, m).transpose(1, 2)
        val = self.v_in(x).view(B, L, h, dv).transpose(1, 2)
        if c.mem_write_avg:   # (1 - lambda) = sigmoid(-decay_beta_logit), per group
            val = val * torch.sigmoid(-self.decay_beta_logit).to(val.dtype).view(1, h, 1, 1)
        r, S = plastic_scan(q.to(val.dtype), k.to(val.dtype), val, F.logsigmoid(self.decay_beta_logit), S0, c.mem_chunk)
        r = r.float()
        r = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + c.mem_eps)).to(val.dtype)
        r = r.transpose(1, 2).reshape(B, L, h * dv) * self.gain.to(val.dtype)
        return self.w_out(r), s, (S.detach(), v_last.detach())


class CodedInput(nn.Module):
    """in_groups: word x makes one input neuron fire in each of k groups of n; the current is the sum of their
    synapse rows. code[x] (k,) is the neuron of each group: by default the base-n digits of x; with set_classes
    (k = 2), its class and its position in the class. With in_short S, S words of a short list instead have an input
    neuron of their own (one row; code[x] = -1 - its index). weight is the table of all rows: the S own rows, the
    k * n rows of the groups, and (S > 0) a zero row that a short-list word uses in place of its other k - 1 spikes."""

    def __init__(self, vocab, d, k, S=0):
        super().__init__()
        self.k, self.vocab, self.S = k, vocab, S
        self.n = math.ceil((vocab - S) ** (1 / k) - 1e-9)
        while self.n ** k < vocab - S:
            self.n += 1
        self.pad = S + k * self.n                                               # the zero row (S > 0)
        w = torch.randn(S + k * self.n + (1 if S else 0), d)
        w[S:S + k * self.n] /= math.sqrt(k)                                     # a sum of k rows has unit variance
        if S:
            w[self.pad] = 0
        self.weight = nn.Parameter(w)
        x = torch.arange(vocab).unsqueeze(-1)
        digits = ((x - S) // self.n ** torch.arange(k)) % self.n
        self.register_buffer("code", torch.where(x < S, -1 - x, digits))       # saved with the weights
        self.register_buffer("offset", S + self.n * torch.arange(k), persistent=False)

    @torch.no_grad()
    def set_classes(self, word_cls):
        """k = 2: word x fires neuron cls(x) of the first group and its position within the class in the second;
        the words with class -1 are the short list (in_short of them), each firing its own neuron."""
        word_cls = torch.as_tensor(word_cls, dtype=torch.long, device=self.code.device)
        assert self.k == 2 and word_cls.shape == (self.vocab,)
        short = word_cls < 0
        assert int(short.sum()) == self.S, f"{int(short.sum())} words have class -1, in_short is {self.S}"
        counts = torch.bincount(word_cls[~short])
        assert len(counts) <= self.n and int(counts.max()) <= self.n, "classes must fit the n neurons of each group"
        key = torch.where(short, len(counts), word_cls)
        order = torch.argsort(key, stable=True)                                 # by class, short list last
        start = torch.cumsum(counts, 0) - counts
        pos = torch.empty_like(word_cls)
        tail = order[:self.vocab - self.S]
        pos[tail] = torch.arange(len(tail), device=word_cls.device) - start[word_cls[tail]]
        pos[order[self.vocab - self.S:]] = torch.arange(self.S, device=word_cls.device)
        own = (-1 - pos).unsqueeze(-1).expand(-1, 2)
        self.code.copy_(torch.where(short.unsqueeze(-1), own, torch.stack([word_cls, pos], -1)))

    def rows(self, idx):   # (..., k): the rows of the input neurons that fire for each word
        c = self.code[idx]
        if not self.S:
            return c + self.offset
        r = torch.where(c < 0, self.pad, c + self.offset)
        r[..., 0] = torch.where(c[..., 0] < 0, -1 - c[..., 0], r[..., 0])
        return r

    def forward(self, idx):
        return F.embedding(self.rows(idx), self.weight, padding_idx=self.pad if self.S else None).sum(-2)


class ClassReadout(nn.Module):
    """out_classes: P(w) = P(class of w) P(w | its class); with out_short S, S words of a short list have their own
    neurons beside the class neurons (P(w) directly). word_cls (vocab,) is the class of every word, -1 for the short
    list (by default by number: the first S words are the short list, then word w in class (w - S) // m); forward
    returns log P(w) for every word, (..., vocab)."""

    def __init__(self, vocab, d, C, init, S=0):
        super().__init__()
        self.C, self.S, self.m, self.vocab = C, S, math.ceil((vocab - S) / C), vocab
        self.w_cls = nn.Parameter(torch.randn(S + C, d) * init)        # short-list neurons, then class neurons
        self.w_word = nn.Parameter(torch.randn(C * self.m, d) * init)  # word readout neurons, class by class
        # saved with the weights, so a checkpoint keeps its classes
        w = torch.arange(vocab)
        self.register_buffer("word_cls", torch.where(w < S, -1, (w - S) // self.m))
        self.register_buffer("slot", torch.zeros(vocab, dtype=torch.long))
        self.register_buffer("used", torch.zeros(C * self.m, dtype=torch.bool))
        self.set_classes(self.word_cls.clone())

    @torch.no_grad()
    def set_classes(self, word_cls):
        """word_cls (vocab,): class of every word, each class holding at most m words; -1 for the S short-list words."""
        word_cls = torch.as_tensor(word_cls, dtype=torch.long, device=self.word_cls.device)
        assert word_cls.shape == (self.vocab,) and int(word_cls.min()) >= -1 and int(word_cls.max()) < self.C
        short = word_cls < 0
        assert int(short.sum()) == self.S, f"{int(short.sum())} words have class -1, out_short is {self.S}"
        counts = torch.bincount(word_cls[~short], minlength=self.C)
        assert int(counts.max()) <= self.m, f"a class has {int(counts.max())} words, more than m = {self.m}"
        key = torch.where(short, self.C, word_cls)
        order = torch.argsort(key, stable=True)                             # words grouped by class, short list last
        start = torch.cumsum(counts, 0) - counts
        pos = torch.empty_like(word_cls)
        tail = order[:self.vocab - self.S]
        pos[tail] = torch.arange(len(tail), device=word_cls.device) - start[word_cls[tail]]
        pos[order[self.vocab - self.S:]] = torch.arange(self.S, device=word_cls.device)
        self.word_cls.copy_(word_cls)
        # index of w in [short-list neurons (S), word neurons class by class (C*m)]
        self.slot.copy_(torch.where(short, pos, self.S + word_cls.clamp(min=0) * self.m + pos))
        self.used.zero_(); self.used[self.slot[~short] - self.S] = True

    def forward(self, x):
        top = F.log_softmax(F.linear(x, self.w_cls).float(), -1)                                  # (..., S + C)
        lc = top[..., self.S:]
        lw = F.linear(x, self.w_word).float().masked_fill(~self.used, float("-inf"))              # (..., C*m)
        lw = (F.log_softmax(lw.view(*lw.shape[:-1], self.C, self.m), -1) + lc.unsqueeze(-1)).flatten(-2)  # per slot
        if self.S:
            lw = torch.cat([top[..., :self.S], lw], -1)
        # to word order; index_select (no repeated index) rather than advanced indexing, whose backward is slow on GPUs
        return lw.index_select(-1, self.slot)


class SpikingLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.in_groups > 1:
            self.embed = CodedInput(cfg.vocab, cfg.d_model, cfg.in_groups, cfg.in_short)
        else:
            self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
            nn.init.normal_(self.embed.weight, std=1.0)
        self.layers = nn.ModuleList([LIFPopulation(cfg) for _ in range(cfg.n_layers)])
        self.mem_layers = [i for i in range(cfg.n_layers) if cfg.mem_heads and (i + 1) % cfg.mem_every == 0]
        if self.mem_layers:   # plastic synapses in front of these populations
            assert cfg.mem_src in ("bus", "spikes")
            # mem_src "spikes": in front of population i > 0 they read population i - 1's spikes (in front of the
            # first population, which has none before it, the bus)
            src_dim = lambda i: cfg.d_hidden if cfg.mem_src == "spikes" and i > 0 else 0
            self.mems = nn.ModuleDict({str(i): PlasticSynapses(cfg, src_dim(i)) for i in self.mem_layers})
        if cfg.out_classes > 0:
            self.head = ClassReadout(cfg.vocab, cfg.d_model, cfg.out_classes, cfg.head_init, cfg.out_short)
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
            nn.init.normal_(self.head.weight, std=cfg.head_init)
        self.keep_bus = False   # training only (train.py --hint_weight): keep the bus after each layer in last_bus
        self.last_bus = None

    def read_bus(self, bus):
        """What a population, a set of plastic synapses or the readout reads: the bus, or with bus_norm the bus divided
        by its RMS over the d_model channels of each token (one sum of squares and one reciprocal square root per
        reader and token)."""
        if not self.cfg.bus_norm:
            return bus
        return bus * torch.rsqrt(bus.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(bus.dtype)

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
        s = None   # the previous population's spikes
        for i, layer in enumerate(self.layers):
            if i in self.mem_layers:
                j = self.mem_layers.index(i)
                src = s if self.cfg.mem_src == "spikes" and i > 0 else None
                cur, sk, ms = self.mems[str(i)](self.read_bus(bus), None if state is None else state[-1][j], src)
                bus = bus + cur
                mem_state.append(ms)
                if return_spikes:
                    mem_spikes.append(sk.detach())
            out, s, layer_state = layer(self.read_bus(bus), None if state is None else state[i])
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
        logits = self.head(self.read_bus(bus))
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

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
a fixed decay per group and 0/1 keys and queries (model.py).

mlp = "spike" (docs/experiments.md 5.49): the MLP's hidden units are 0/1 instead of GELU, h = W1 RMSNorm(x),
s = [h >= theta] (theta learned per unit, positive), x += W2 s; trained with the spiking network's surrogate gradient
(the fast sigmoid of lif.py, ds/dh = 1 / (theta (1 + alpha |z|)^2), z = (h - theta) / theta, alpha = 4). These units
have no membrane (no memory, no reset): only the 0/1 output is taken from the spiking network, with everything else
(normalisation, attention, the rest of the block) kept, so it measures what 0/1 hidden units cost in an otherwise
ordinary network.

mlp = "lif" (5.50): the MLP's hidden units are the spiking network's LIF neurons, V[t] = beta V[t-1] + h[t],
s[t] = [V[t] >= theta], reset V <- V (1 - gamma s) (beta, theta, gamma learned per unit; time constants log-uniform in
0.5-200 tokens, gamma uniform in 0.05-0.999, theta one standard deviation of V at initialisation, as the spiking
network's populations are initialised). The next rung after "spike": the same 0/1 units with a membrane, i.e. memory
and reset.

relay = "lif" (5.51, step 3): every matrix of a block reads spikes instead of real numbers. In front of it, a pair of
LIF relay neurons per input channel, one driven by the channel's value x and one by -x, with a subtractive reset
(V <- V - theta s: what is not sent stays in the membrane and goes out with a later spike); the matrix reads
theta (s+ - s-), so it only adds or subtracts the rows of the channels that fired (theta folds into its columns).
beta and theta are learned per channel (time constants 1-4 tokens, theta at initialisation about one RMS of the
input: 1 for the normalised stream, 0.01 sqrt(d) for the gated attention output). Relays sit in front of q/k/v/gate/decay (reading the normalised stream), of the output projection
(reading the gated attention output) and of the MLP's first matrix; with mlp spike/lif the second one already reads
0/1 units. relay = "quant": the same 3-valued output theta * round(x / theta) in {-theta, 0, theta}, without a
membrane (a straight-through gradient within |x| <= 1.5 theta): what the membrane adds. relay_levels K (quant only;
5.52): up to K spikes per channel and token, theta * clip(round(x / theta), -K, K) (straight-through within
|x| <= (K + 1/2) theta): an integrate-and-fire neuron that receives |x| + theta / 2 at the start of the token,
fires (subtracting theta) while its potential is at least theta, at most K times, and is reset at the token's end.
With theta still about one RMS at initialisation, the extra spikes go to the large inputs that one spike clips.
The matrix adds the row of a channel once per spike. K = 1 is the 3-valued relay. GPT.relay_spikes: the spikes per
channel and token of all relays, each weighted by the columns of the matrix it feeds (0.5: the block matrices add
half as many rows as they would multiply densely); train.py --relay_target holds it to a target (5.52).
For quant relays, the count penalty can use a smooth gradient at the K thresholds in each direction (docs 5.57),
so it can act on silent and saturated inputs while the matrices and the spike budget still see hard spikes.
wq ="ternary": the blocks' matrices use weights in {-1, 0, +1} times one scale per matrix (the mean |w|; BitNet
b1.58's rounding, straight-through gradient), the input rows and the readout stay real: the non-spiking way to
turn the matrices' multiplications into additions, as a control.
out_classes > 0: the spiking network's two-level readout (model.ClassReadout, docs 5.36), with out_short words
scored directly alongside the class scores. The default head is a single full-vocabulary projection."""
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
    mlp: str = "gelu"         # "spike": 0/1 hidden units with a surrogate gradient; "lif": LIF neurons (module docstring)
    relay: str = "none"       # "lif": LIF relay neurons in front of every matrix; "quant": their output without a membrane
    relay_levels: int = 1     # quant: up to this many spikes per channel and token (module docstring)
    wq: str = "none"          # "ternary": the blocks' matrices with weights in {-1, 0, +1} times a scale (module docstring)
    mlp_mult: int = 4         # the MLP's hidden units per channel (4: GPT-2's); wider with sparse units: docs 5.56
    out_classes: int = 0      # >0: two-level readout over this many word classes
    out_short: int = 0        # number of words scored directly in the two-level readout's shortlist


class _Tern(torch.autograd.Function):
    """sign(x) min(floor(|x| + 1/2), K): rounding of x (in units of theta) to the 2K + 1 values -K..K, ties away from
    0 (K = 1: sign(x) [|x| >= 1/2], 3-valued); straight-through gradient for |x| <= K + 1/2."""

    @staticmethod
    def forward(ctx, x, K=1):
        ctx.save_for_backward(x)
        ctx.K = K
        if K == 1:
            return ((x >= 0.5).to(x.dtype) - (x <= -0.5).to(x.dtype))
        return torch.sign(x) * torch.floor(x.abs() + 0.5).clamp(max=K)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * (x.abs() <= ctx.K + 0.5).to(g.dtype), None


class _RelayCount(torch.autograd.Function):
    """Hard mean count in forward, gradient of a sum of sigmoids at ±(k - 1/2) in backward (docs 5.57).
    With z = x/theta and width in units of theta, each threshold t contributes sigmoid((z-t)/width) and
    sigmoid((-z-t)/width). The tails give the count penalty a gradient for silent and saturated relays.
    Save z once and compute the slopes in backward to avoid saving a tensor for every threshold."""

    @staticmethod
    def forward(ctx, z, count, K, width):
        ctx.save_for_backward(z)
        ctx.K, ctx.width = K, width
        return count

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        slope = torch.zeros_like(z)
        for k in range(ctx.K):
            t = k + 0.5
            # The sigmoid slope is even. Use its negative tail to avoid rounding the sigmoid to 1.
            p = torch.sigmoid(-(z - t).abs() / ctx.width)
            n = torch.sigmoid(-(z + t).abs() / ctx.width)
            slope = slope + (p * (1 - p) - n * (1 - n))
        return slope * (g / (ctx.width * z.numel())), None, None, None


class _TernW(torch.autograd.Function):
    """BitNet b1.58's weight rounding: round(w / a) clipped to {-1, 0, 1}, times a = mean |w|; straight-through."""

    @staticmethod
    def forward(ctx, w):
        a = w.abs().mean().clamp(min=1e-8)
        return (w / a).round().clamp(-1, 1) * a

    @staticmethod
    def backward(ctx, g):
        return g


class QLinear(nn.Linear):
    """nn.Linear whose weight is used rounded to ternary values when ternary is set (TFConfig.wq)."""
    ternary = False

    def forward(self, x):
        return F.linear(x, _TernW.apply(self.weight) if self.ternary else self.weight, self.bias)


class Relay(nn.Module):
    """LIF relay neurons in front of a matrix (module docstring, relay): (..., D) real -> theta * (s+ - s-).
    rate: spikes per channel and token (with one spike at most: the share of nonzero outputs). fan: the columns of
    the matrix it feeds (additions per spike). count: rate with its gradient, from forwards in training mode while
    keep_count is set (GPT.relay_spikes, train.py --relay_target); else None."""

    def __init__(self, D, kind, theta0=1.0, levels=1, fan=1):
        super().__init__()
        assert levels >= 1 and (levels == 1 or kind == "quant"), "relay_levels > 1: quant only"
        self.D, self.kind, self.levels = D, kind, levels
        self.fan, self.keep_count, self.count = fan, False, None
        self.count_grad_width = 0.0   # nonzero: smooth the count gradient (quant only; train.py --relay_count_grad)
        self.log_theta = nn.Parameter(torch.full((D,), math.log(theta0)))          # about one RMS of the input
        if kind == "lif":
            tau = torch.exp(torch.empty(D).uniform_(0.0, math.log(4.0)))
            self.beta_logit = nn.Parameter(torch.logit(torch.exp(-1.0 / tau)))
        self.rate = None

    def forward(self, x):
        th = self.log_theta.exp()
        if self.kind == "quant":
            u = _Tern.apply(x / th.to(x.dtype), self.levels)
        else:
            from lif import lif
            s, _ = lif(torch.cat([x, -x], -1), torch.sigmoid(self.beta_logit).repeat(2), th.repeat(2), None, None,
                       hard_reset=False, alpha=4.0, detach_reset=False, sg="fast_sigmoid")
            s = s.to(x.dtype)
            u = s[..., :self.D] - s[..., self.D:]
        self.rate = u.detach().abs().float().mean()
        self.count = None
        if self.keep_count and self.training:
            if self.count_grad_width:
                self.count = _RelayCount.apply(x.float() / th, self.rate, self.levels, self.count_grad_width)
            else:
                self.count = u.abs().float().mean()
        return u * th.to(x.dtype)


class _Step(torch.autograd.Function):
    """s = [h >= theta]; backward: the fast-sigmoid surrogate of the spiking network (lif.py, sg "fast_sigmoid"),
    in float32. h: (..., H); theta: (H,) float32."""

    @staticmethod
    def forward(ctx, h, theta, alpha):
        ctx.save_for_backward(h, theta)
        ctx.alpha = alpha
        return (h >= theta).to(h.dtype)

    @staticmethod
    def backward(ctx, g):
        h, theta = ctx.saved_tensors
        hf = h.float()
        d = 1.0 / (theta * (1 + ctx.alpha * ((hf - theta) / theta).abs()) ** 2)    # ds/dh
        gh = g.float() * d
        gth = -(gh * hf / theta).reshape(-1, h.size(-1)).sum(0)                      # ds/dtheta = -ds/dh h / theta
        return gh.to(h.dtype), gth, None


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
    """Gated linear attention (module docstring), in place of a block's softmax attention.
    c.ablate (model.py --mem_gla_ablate, docs 5.53; a comma-separated subset), each removing one of its features:
      fixdecay  the decay per head is a learned constant, not a function of the input
      noconv    no short convolution in front of q, k, v
      nogate    no output gate: the normalised read times a learned gain per channel, 0.3 at initialisation (the
                gate's RMS there in model.py's network), so only its dependence on the input is removed
      spkkq     q and k are 0/1: [silu(conv) >= theta], a threshold per channel (learned; at initialisation about
                one standard deviation of silu(conv), 0.16, or of silu(qkv(x)) with noconv, 0.29, as measured in
                model.py's network), the spiking network's fast-sigmoid surrogate (alpha c.alpha); v stays real.
                The read's normalisation then uses eps 1, like the plastic synapses' (model.py mem_eps): with 0/1
                queries and keys a read is often exactly 0 (no spike in common), where an RMS norm with a tiny eps
                has huge gradients (without the output gate, which damps them, training blew up: docs 5.53)"""

    def __init__(self, c):
        super().__init__()
        d, h = c.d_model, c.n_heads
        self.h, self.hd = h, d // h
        ab = set(filter(None, getattr(c, "ablate", "").split(",")))
        assert ab <= {"fixdecay", "noconv", "nogate", "spkkq"}, f"unknown ablation: {ab}"
        self.ab, self.alpha = ab, getattr(c, "alpha", 4.0)
        self.qkv = QLinear(d, 3 * d, bias=False)
        self.conv = None if "noconv" in ab else nn.Conv1d(3 * d, 3 * d, kernel_size=4, groups=3 * d, padding=3,
                                                          bias=False)   # causal: its first L outputs
        self.decay = nn.Linear(d, h, bias=True)                      # fixdecay: only its bias is used
        if "fixdecay" in ab:
            self.decay.weight.requires_grad_(False)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, h)))
        self.gate = None if "nogate" in ab else QLinear(d, d, bias=False)
        if "nogate" in ab:
            self.out_gain = nn.Parameter(torch.full((d,), 0.3))
        if "spkkq" in ab:
            self.kq_log_theta = nn.Parameter(torch.full((2 * d,), math.log(0.29 if "noconv" in ab else 0.16)))
        self.onorm = nn.RMSNorm(self.hd, eps=1.0 if "spkkq" in ab else None)   # spkkq: as mem_eps
        self.proj = QLinear(d, d, bias=False)
        relay = getattr(c, "relay", "none")
        K = getattr(c, "relay_levels", 1)
        fan_in = 3 * d + (d if self.gate is not None else 0) + h
        self.r_in = Relay(d, relay, levels=K, fan=fan_in) if relay != "none" else None   # in front of q, k, v, gate, decay
        # the gated output starts at about 0.01 sqrt(d) RMS (gate weights 0.02, silu(g) ~ g / 2 near 0)
        self.r_out = Relay(d, relay, 0.01 * math.sqrt(d), K, fan=d) if relay != "none" else None   # in front of the output projection
        with torch.no_grad():   # softplus(bias) log-uniform in [0.001, 0.1] (Mamba's step sizes): a from ~0.2 to ~1
            dt0 = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
            self.decay.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))   # inverse of softplus

    def forward(self, x, state=None, return_state=False):
        """state: (S (B, h, hd, hd) float32, the last 3 inputs of the convolution (B, 3d, 3)) from the previous call,
        or None (nothing before). return_state: also return the state after the last position."""
        B, L, D = x.shape
        if self.r_in is not None:
            x = self.r_in(x)
        u = self.qkv(x).transpose(1, 2)                                             # (B, 3d, L): the conv's input
        S0, prev = (None, None) if state is None else state
        if self.conv is None:
            c = u
        elif state is None:
            c = self.conv(u)[..., :L]
        else:
            c = F.conv1d(torch.cat([prev.to(u.dtype), u], -1), self.conv.weight.to(u.dtype), groups=3 * D)
        qkv = F.silu(c).transpose(1, 2)
        if "spkkq" in self.ab:
            kq = _Step.apply(qkv[..., :2 * D].contiguous(), self.kq_log_theta.exp().float(), self.alpha)
            qkv = torch.cat([kq, qkv[..., 2 * D:]], -1)
        q, k, v = qkv.reshape(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)    # (B, h, L, hd) each
        if "fixdecay" in self.ab:
            log_a = (-F.softplus(self.decay.bias.float()) * self.A_log.exp()).expand(B, L, self.h)
        else:
            log_a = -F.softplus(self.decay(x).float()) * self.A_log.exp()       # (B, L, h)
        o, S = gla_scan(q * self.hd ** -0.5, k, v, log_a.transpose(1, 2), S0=S0)
        o = self.onorm(o).transpose(1, 2).reshape(B, L, D)
        o = o * (F.silu(self.gate(x)) if self.gate is not None else self.out_gain.to(o.dtype))
        if self.r_out is not None:
            o = self.r_out(o)
        if not return_state:
            return self.proj(o)
        last = (F.pad(u, (3, 0)) if state is None else torch.cat([prev.to(u.dtype), u], -1))[..., -3:]
        return self.proj(o), (S.detach(), last.detach())


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.rope = c.rope
        self.n1 = nn.RMSNorm(c.d_model)
        self.la = LinearAttention(c) if c.attn == "linear" else None
        if self.la is None:
            self.qkv = QLinear(c.d_model, 3 * c.d_model, bias=False)   # QLinear: ternary with --wq ternary, like the
            self.proj = QLinear(c.d_model, c.d_model, bias=False)      # other block matrices (fpga_est.tf_hw counts them so)
        self.n2 = nn.RMSNorm(c.d_model)
        hid = getattr(c, "mlp_mult", 4) * c.d_model   # the MLP's hidden units
        self.fc = QLinear(c.d_model, hid, bias=False)
        self.fc2 = QLinear(hid, c.d_model, bias=False)
        self.h = c.n_heads
        relay = getattr(c, "relay", "none")
        assert relay in ("none", "lif", "quant") and (relay == "none" or c.attn == "linear"), "relays: linear attention"
        self.r_mlp = (Relay(c.d_model, relay, levels=getattr(c, "relay_levels", 1), fan=hid)
                      if relay != "none" else None)
        assert getattr(c, "wq", "none") in ("none", "ternary")
        if getattr(c, "wq", "none") == "ternary":
            for m in self.modules():
                if isinstance(m, QLinear):
                    m.ternary = True
        assert c.mlp in ("gelu", "spike", "lif")
        self.mlp = c.mlp
        H, sd = hid, 0.02 * math.sqrt(c.d_model)   # sd: standard deviation of h at initialisation
        if c.mlp == "spike":   # threshold at one standard deviation of h: ~16% fire
            self.log_theta = nn.Parameter(torch.full((H,), math.log(sd)))
        if c.mlp == "lif":     # as model.LIFPopulation with hard_reset, reset_learn and in_norm "theta"
            beta = torch.exp(-1.0 / torch.exp(torch.empty(H).uniform_(math.log(0.5), math.log(200.0))))
            self.beta_logit = nn.Parameter(torch.logit(beta))
            self.log_theta = nn.Parameter(torch.log(sd / torch.sqrt(1 - beta ** 2)))
            self.gamma_logit = nn.Parameter(torch.logit(torch.empty(H).uniform_(0.05, 0.999)))
        self.rate = None

    def ff(self, x):
        x = self.n2(x)
        h = self.fc(self.r_mlp(x) if self.r_mlp is not None else x)
        if self.mlp == "gelu":
            return self.fc2(F.gelu(h))
        if self.mlp == "spike":
            s = _Step.apply(h, self.log_theta.exp().float(), 4.0)
        else:
            from lif import lif
            s, _ = lif(h, torch.sigmoid(self.beta_logit), self.log_theta.exp(), torch.sigmoid(self.gamma_logit), None,
                       hard_reset=True, alpha=4.0, detach_reset=False, sg="fast_sigmoid")
            s = s.to(h.dtype)
        self.rate = s.detach().float().mean()
        return self.fc2(s)

    def forward(self, x):
        B, L, D = x.shape
        if self.la is not None:
            x = x + self.la(self.n1(x))
            return x + self.ff(x)
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotary(q), rotary(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(x)


class GPT(nn.Module):
    def __init__(self, c: TFConfig):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab, c.d_model)
        assert c.attn in ("softmax", "linear")
        self.pos = None if c.rope or c.attn == "linear" else nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = nn.RMSNorm(c.d_model)
        if c.out_classes:
            from model import ClassReadout
            self.head = ClassReadout(c.vocab, c.d_model, c.out_classes, 0.02, c.out_short)
        else:
            if c.out_short:
                raise ValueError("out_short needs out_classes")
            self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * c.n_layers) if n.endswith(("proj.weight", "fc2.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def _features(self, idx):
        L = idx.size(1)
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(L, device=idx.device))
        xs = []
        for b in self.blocks:
            x = b(x)
            xs.append(x)
        return self.norm(x), xs

    def forward(self, idx, targets=None, state=None, return_spikes=False, hidden=False):
        """Returns logits (word log-probabilities with the two-level head), loss, None, and firing rates.
        With hidden=True the third item holds the residual streams after each block for hidden-state distillation."""
        x, xs = self._features(idx)
        logits = self.head(x)
        loss = None
        if targets is not None:
            if self.cfg.out_classes:
                loss = F.nll_loss(logits.flatten(0, 1), targets.reshape(-1))
            else:
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        rates = self.rates(idx.device)
        return logits, loss, xs if hidden else None, rates

    def rates(self, device):
        """Share of 1s of the 0/1 MLP units per block, then with relays their spikes per channel and token (with one
        spike at most: the share of nonzero outputs; with relay_levels > 1 it can exceed 1) in front of q/k/v, of the
        output projection and of the MLP, per block (n_layers values for the MLP units, then 3 x n_layers for the
        relays); zeros(1) if none."""
        z = torch.zeros((), device=device)
        r = [b.rate if b.rate is not None else z for b in self.blocks] if self.cfg.mlp != "gelu" else []
        if getattr(self.cfg, "relay", "none") != "none":
            r += [b.rate if b.rate is not None else z for blk in self.blocks for b in (blk.la.r_in, blk.la.r_out, blk.r_mlp)]
        return torch.stack(r) if r else torch.zeros(1, device=device)

    def relays(self):
        """The relays in front of q/k/v, of the output projection and of the MLP, block by block ([] if none)."""
        if getattr(self.cfg, "relay", "none") == "none":
            return []
        return [r for blk in self.blocks for r in (blk.la.r_in, blk.la.r_out, blk.r_mlp)]

    def relay_spikes(self):
        """Spikes per channel and token of all relays, each weighted by the columns of the matrix it feeds: the block
        matrices' additions over their multiply-adds read densely (fpga_est.tf_hw: 0.5, half as many). The value is the
        hard count with the configured penalty gradient, from the last training forward with keep_count set."""
        rs = self.relays()
        return sum(r.fan * r.count for r in rs) / sum(r.fan for r in rs)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, eos_id=None, generator=None):
        """At temperature 1 with no top-k cutoff, the two-level head samples a shortlist word or one class.
        Global temperature scaling and top-k filtering use full word scores."""
        out = []
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.max_len:]
            if self.cfg.out_classes:
                x, _ = self._features(context)
                if temperature == 1.0 and not top_k:
                    nxt = self.head.sample(x[:, -1], generator=generator)
                else:
                    nxt = self._sample_logits(self.head(x[:, -1]), temperature, top_k, generator)
            else:
                logits, _, _, _ = self(context)
                nxt = self._sample_logits(logits[:, -1], temperature, top_k, generator)
            out.append(nxt)
            idx = torch.cat([idx, nxt], 1)
            if eos_id is not None and idx.size(0) == 1 and nxt.item() == eos_id:
                break
        return torch.cat(out, 1)

    @staticmethod
    def _sample_logits(logits, temperature, top_k, generator):
        l = logits.float() / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(l, top_k)
            l[l < v[:, [-1]]] = -float("inf")
        return torch.multinomial(F.softmax(l, -1), 1, generator=generator)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_neurons(self):
        return 0

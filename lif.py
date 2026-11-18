"""LIF neurons: autograd wrappers around the scan kernels, plus a slow pure-PyTorch reference.

Tensors on the CPU use the C++/OpenMP kernels (lif_kernel.cpp, compiled on first use); tensors on an NVIDIA GPU
use the Triton kernels (lif_triton.py). LIF_BACKEND=triton forces the Triton kernels also for CPU tensors
(only useful for testing, together with TRITON_INTERPRET=1).

soft = rho in [0, 1] (model.py spike_soft): each neuron outputs (1 - rho) * step(z) + rho * soft_fn(z) instead of a
0/1 spike, z = (V - theta) / theta, and the reset uses that output. soft_fn is the smooth function whose derivative
has the surrogate's shape; the gradient is the surrogate for the step part and exact for the smooth part. rho = 1 is
a non-spiking twin of the network (real-valued outputs, exact gradients), rho = 0 the spiking neuron. Only the Triton
kernels implement it; on the CPU, rho > 0 falls back to the slow reference.
"""
import os
import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))


class _LazyExt:
    """Compiles / loads the C++ extension the first time a CPU kernel is needed."""
    _mod = None

    def __getattr__(self, name):
        if _LazyExt._mod is None:
            _LazyExt._mod = load(
                name="lif_ext_v3",
                sources=[os.path.join(_here, "lif_kernel.cpp")],
                extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
                extra_ldflags=["-fopenmp"],
                verbose=False,
            )
        return getattr(_LazyExt._mod, name)


_ext = _LazyExt()


def _use_triton(I):
    return I.is_cuda or os.environ.get("LIF_BACKEND") == "triton"


SG_TYPES = {"sigmoid": 0, "fast_sigmoid": 1, "triangle": 2}


def _prep(I, beta, theta, gamma, v0):
    if I.dtype not in (torch.float32, torch.bfloat16):
        I = I.float()
    f = lambda t: t.float().contiguous()
    return I.contiguous(), f(beta), f(theta), f(gamma), f(v0)


class LIFFunction(torch.autograd.Function):
    """Independent LIF neurons (no same-layer synapses)."""

    @staticmethod
    def forward(ctx, I, beta, theta, gamma, v0, hard_reset, alpha, detach_reset, sg_type):
        I, beta, theta, gamma, v0 = _prep(I, beta, theta, gamma, v0)
        S, Vp, vlast = _ext.forward(I, beta, theta, gamma, v0, int(hard_reset))
        ctx.save_for_backward(Vp, beta, theta, gamma, v0)
        ctx.cfg = (int(hard_reset), float(alpha), int(detach_reset), int(sg_type))
        ctx.in_dtype = I.dtype
        return S, vlast

    @staticmethod
    def backward(ctx, gS, gvlast):
        Vp, beta, theta, gamma, v0 = ctx.saved_tensors
        hard, alpha, detach, sg = ctx.cfg
        if gS is None:
            gS = torch.zeros_like(Vp, dtype=ctx.in_dtype)
        if gvlast is None:
            gvlast = torch.zeros_like(v0)
        gI, gb, gt, gg, gv0 = _ext.backward(gS.to(ctx.in_dtype).contiguous(), Vp, beta, theta, gamma, v0,
                                            gvlast.float().contiguous(), hard, alpha, detach, sg)
        return gI, gb.sum(0), gt.sum(0), gg.sum(0), gv0, None, None, None, None


class LIFLeakFunction(torch.autograd.Function):
    """Independent LIF neurons whose leak changes from step to step: beta = sigmoid(beta_logit + M),
    with M (B, L, D) given per step (from the leak-modulating synapses)."""

    @staticmethod
    def forward(ctx, I, M, beta_logit, theta, gamma, v0, hard_reset, alpha, detach_reset, sg_type):
        I, beta_logit, theta, gamma, v0 = _prep(I, beta_logit, theta, gamma, v0)
        M = M.to(I.dtype).contiguous()
        S, Vp, vlast = _ext.tv_forward(I, M, beta_logit, theta, gamma, v0, int(hard_reset))
        ctx.save_for_backward(Vp, M, beta_logit, theta, gamma, v0)
        ctx.cfg = (int(hard_reset), float(alpha), int(detach_reset), int(sg_type))
        ctx.in_dtype = I.dtype
        return S, vlast

    @staticmethod
    def backward(ctx, gS, gvlast):
        Vp, M, beta_logit, theta, gamma, v0 = ctx.saved_tensors
        hard, alpha, detach, sg = ctx.cfg
        if gS is None:
            gS = torch.zeros_like(Vp, dtype=ctx.in_dtype)
        if gvlast is None:
            gvlast = torch.zeros_like(v0)
        gI, gM, gbl, gt, gg, gv0 = _ext.tv_backward(gS.to(ctx.in_dtype).contiguous(), Vp, M, beta_logit, theta,
                                                    gamma, v0, gvlast.float().contiguous(), hard, alpha,
                                                    detach, sg)
        return gI, gM, gbl.sum(0), gt.sum(0), gg.sum(0), gv0, None, None, None, None


def soft_fn(z, alpha, sg):
    """The smooth output for soft > 0: rises from 0 to 1 around z = 0, with derivative (dz) of the surrogate's shape:
    sigmoid: sigmoid(alpha z) (derivative = the surrogate); fast_sigmoid: 0.5 + 0.5 alpha z / (1 + alpha |z|)
    (derivative = alpha / 2 x the surrogate); triangle: the integral of the triangle (derivative = the surrogate)."""
    if sg == "sigmoid":
        return torch.sigmoid(alpha * z)
    if sg == "fast_sigmoid":
        return 0.5 + 0.5 * alpha * z / (1 + alpha * z.abs())
    u = (alpha * z).clamp(-1, 1)
    return torch.where(u <= 0, 0.5 * (1 + u) ** 2, 1 - 0.5 * (1 - u) ** 2)


def _defaults(I, gamma, v0):
    B, L, D = I.shape
    if gamma is None:
        gamma = torch.ones(D, dtype=torch.float32, device=I.device)
    if v0 is None:
        v0 = torch.zeros(B, D, dtype=torch.float32, device=I.device)
    return gamma, v0


def lif(I, beta, theta, gamma=None, v0=None, hard_reset=False, alpha=4.0, detach_reset=True, sg="sigmoid",
        soft=0.0):
    """I: (B, L, D) input current. Returns spikes (B, L, D) in {0,1} and final membrane (B, D).
    beta: per-neuron leak (D).
    gamma: per-neuron reset fraction for the multiplicative (hard_reset=True) reset; None = 1.
    soft > 0: real-valued outputs (module docstring)."""
    gamma, v0 = _defaults(I, gamma, v0)
    if _use_triton(I):
        from lif_triton import TritonLIF
        return TritonLIF.apply(I, beta, theta, gamma, v0, hard_reset, alpha, detach_reset, SG_TYPES[sg], float(soft))
    if soft > 0:
        return lif_reference(I, beta, theta, gamma, v0, hard_reset, alpha, detach_reset, sg, soft)
    return LIFFunction.apply(I, beta, theta, gamma, v0, hard_reset, alpha, detach_reset, SG_TYPES[sg])


def lif_leak(I, beta_logit, M, theta, gamma=None, v0=None, hard_reset=False, alpha=4.0, detach_reset=True,
             sg="sigmoid", soft=0.0):
    """Like lif(), but the leak changes every step: beta[b, t, d] = sigmoid(beta_logit[d] + M[b, t, d]).
    beta is formed inside the kernel in float32; M may be bfloat16 (it is cast to the dtype of I)."""
    gamma, v0 = _defaults(I, gamma, v0)
    if _use_triton(I):
        from lif_triton import TritonLIFLeak
        return TritonLIFLeak.apply(I, M, beta_logit, theta, gamma, v0, hard_reset, alpha, detach_reset,
                                   SG_TYPES[sg], float(soft))
    if soft > 0:
        return lif_reference(I, torch.sigmoid(beta_logit + M.float()), theta, gamma, v0, hard_reset, alpha,
                             detach_reset, sg, soft)
    return LIFLeakFunction.apply(I, M, beta_logit, theta, gamma, v0, hard_reset, alpha, detach_reset,
                                 SG_TYPES[sg])


# ---------------------------------------------------------------- reference (slow, for testing)
class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, alpha, sg_type):
        ctx.save_for_backward(x, theta)
        ctx.alpha, ctx.sg = alpha, sg_type
        return (x >= theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        z = (x - theta) / theta
        a = ctx.alpha
        if ctx.sg == 0:
            s = torch.sigmoid(a * z); sgz = a * s * (1 - s)
        elif ctx.sg == 1:
            sgz = 1.0 / (1 + a * z.abs()) ** 2
        else:
            sgz = (a * (1 - a * z.abs())).clamp(min=0)
        dsdx = sgz / theta
        gx = g * dsdx
        gth = (g * (-dsdx * x / theta))
        return gx, gth.sum(dim=0) if gth.dim() > 1 else gth, None, None


def lif_reference(I, beta, theta, gamma=None, v0=None, hard_reset=False, alpha=4.0, detach_reset=True,
                  sg="sigmoid", soft=0.0):
    B, L, D = I.shape
    I = I.float()
    v = I.new_zeros(B, D) if v0 is None else v0
    g = I.new_ones(D) if gamma is None else gamma
    out = []
    for t in range(L):
        x = (beta[:, t] if beta.dim() == 3 else beta) * v + I[:, t]
        s = _Spike.apply(x, theta, alpha, SG_TYPES[sg])
        if soft > 0:
            s = (1 - soft) * s + soft * soft_fn((x - theta) / theta, alpha, sg)
        sr = s.detach() if detach_reset else s
        v = x * (1 - g * sr) if hard_reset else x - theta * sr
        out.append(s)
    return torch.stack(out, 1), v

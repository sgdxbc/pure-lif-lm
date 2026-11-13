"""LIF neuron: autograd wrapper around the C++ scan kernel, plus a slow pure-PyTorch reference."""
import os
import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="lif_ext",
    sources=[os.path.join(_here, "lif_kernel.cpp")],
    extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
    extra_ldflags=["-fopenmp"],
    build_directory=os.path.join(_here, "build") if os.path.isdir(os.path.join(_here, "build")) else None,
    verbose=False,
)

SG_TYPES = {"sigmoid": 0, "fast_sigmoid": 1, "triangle": 2}


class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, beta, theta, v0, hard_reset, alpha, detach_reset, sg_type):
        if I.dtype not in (torch.float32, torch.bfloat16):
            I = I.float()
        I = I.contiguous()
        ctx.in_dtype = I.dtype
        beta = beta.float().contiguous()
        theta = theta.float().contiguous()
        v0 = v0.float().contiguous()
        S, Vp, vlast = _ext.forward(I, beta, theta, v0, int(hard_reset))
        ctx.save_for_backward(Vp, beta, theta, v0)
        ctx.cfg = (int(hard_reset), float(alpha), int(detach_reset), int(sg_type))
        return S, vlast

    @staticmethod
    def backward(ctx, gS, gvlast):
        Vp, beta, theta, v0 = ctx.saved_tensors
        hard, alpha, detach, sg = ctx.cfg
        if gS is None:
            gS = torch.zeros_like(Vp, dtype=ctx.in_dtype)
        if gvlast is None:
            gvlast = torch.zeros_like(v0)
        gI, gb, gt, gv0 = _ext.backward(gS.to(ctx.in_dtype).contiguous(), Vp, beta, theta, v0,
                                        gvlast.float().contiguous(), hard, alpha, detach, sg)
        return gI, gb.sum(0), gt.sum(0), gv0, None, None, None, None


def lif(I, beta, theta, v0=None, hard_reset=False, alpha=4.0, detach_reset=True, sg="sigmoid"):
    """I: (B, L, D) input current. Returns spikes (B, L, D) in {0,1} and final membrane (B, D)."""
    if v0 is None:
        v0 = I.new_zeros(I.shape[0], I.shape[2], dtype=torch.float32)
    return LIFFunction.apply(I, beta, theta, v0, hard_reset, alpha, detach_reset, SG_TYPES[sg])


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


def lif_reference(I, beta, theta, v0=None, hard_reset=False, alpha=4.0, detach_reset=True, sg="sigmoid"):
    B, L, D = I.shape
    v = I.new_zeros(B, D) if v0 is None else v0
    out = []
    for t in range(L):
        x = beta * v + I[:, t]
        s = _Spike.apply(x, theta, alpha, SG_TYPES[sg])
        sr = s.detach() if detach_reset else s
        if hard_reset:
            v = x * (1 - sr)
        else:
            v = x - theta * sr
        out.append(s)
    return torch.stack(out, 1), v

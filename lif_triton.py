"""LIF scan kernels for NVIDIA GPUs, written in Triton. Same equations and same gradients as the C++ kernels
in lif_kernel.cpp (see the comment at the top of that file); test_lif_gpu.py checks them against each other.

Each Triton program handles BLOCK neurons of one sequence and walks through time, so the work is spread over
(batch, neuron-blocks) exactly like the C++ kernel. Two variants:
  plain : per-neuron leak beta (D)
  leak  : leak changes every step, beta[b, t, d] = sigmoid(beta_logit[d] + M[b, t, d])

Without a GPU the kernels can still be run (slowly) by Triton's interpreter: TRITON_INTERPRET=1.
"""
import torch
import triton
import triton.language as tl

BLOCK = 128


@triton.jit
def _sigmoid(z):
    u = tl.minimum(tl.maximum(-z, -30.0), 30.0)
    return 1.0 / (1.0 + tl.exp(u))


@triton.jit
def _surrogate(z, alpha, SG: tl.constexpr):
    if SG == 0:     # derivative of sigmoid(alpha * z)
        s = _sigmoid(alpha * z)
        r = alpha * s * (1.0 - s)
    elif SG == 1:   # fast sigmoid: 1 / (1 + alpha |z|)^2
        a = 1.0 + alpha * tl.abs(z)
        r = 1.0 / (a * a)
    else:           # triangle
        r = tl.maximum(alpha * (1.0 - alpha * tl.abs(z)), 0.0)
    return r


@triton.jit
def _fwd_kernel(I, M, BP, TH, GM, V0, S, VP, VL, L, D,
                HARD: tl.constexpr, LEAK: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    th = tl.load(TH + d, mask=m, other=1.0)
    gm = tl.load(GM + d, mask=m, other=1.0)
    bp = tl.load(BP + d, mask=m, other=0.0)          # beta (plain) or beta_logit (leak)
    v = tl.load(V0 + b * D + d, mask=m, other=0.0)
    base = b * L * D
    for t in range(0, L):
        off = base + t * D + d
        x_in = tl.load(I + off, mask=m, other=0.0).to(tl.float32)
        if LEAK:
            beta = _sigmoid(bp + tl.load(M + off, mask=m, other=0.0).to(tl.float32))
        else:
            beta = bp
        x = beta * v + x_in
        tl.store(VP + off, x, mask=m)
        s = tl.where(x >= th, 1.0, 0.0)
        tl.store(S + off, s.to(S.dtype.element_ty), mask=m)
        if HARD:
            v = x * (1.0 - gm * s)
        else:
            v = x - th * s
    tl.store(VL + b * D + d, v, mask=m)


@triton.jit
def _bwd_kernel(GS, VP, M, BP, TH, GM, V0, GVL, GI, GMOD, GB, GT, GG, GV0, L, D, alpha,
                HARD: tl.constexpr, DETACH: tl.constexpr, SG: tl.constexpr, LEAK: tl.constexpr,
                BLOCK: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    th = tl.load(TH + d, mask=m, other=1.0)
    gk = tl.load(GM + d, mask=m, other=1.0)
    bp = tl.load(BP + d, mask=m, other=0.0)
    v0 = tl.load(V0 + b * D + d, mask=m, other=0.0)
    gpost = tl.load(GVL + b * D + d, mask=m, other=0.0)
    inv = 1.0 / th
    accb = tl.zeros([BLOCK], dtype=tl.float32)
    acct = tl.zeros([BLOCK], dtype=tl.float32)
    accg = tl.zeros([BLOCK], dtype=tl.float32)
    base = b * L * D
    for i in range(0, L):
        t = L - 1 - i
        off = base + t * D + d
        x = tl.load(VP + off, mask=m, other=0.0)
        s = tl.where(x >= th, 1.0, 0.0)
        sg = _surrogate((x - th) * inv, alpha, SG) * inv           # dS/dV
        if HARD:
            dx = 1.0 - gk * s
            dth = tl.zeros([BLOCK], dtype=tl.float32)
            dgm = -s * x
            if not DETACH:
                dx = dx - gk * x * sg
                dth = gk * x * sg * x * inv
        else:
            dx = tl.full([BLOCK], 1.0, tl.float32)
            dth = -s
            dgm = tl.zeros([BLOCK], dtype=tl.float32)
            if not DETACH:
                dx = dx - th * sg
                dth = dth + sg * x
        g_s = tl.load(GS + off, mask=m, other=0.0).to(tl.float32)
        gx = g_s * sg + gpost * dx                                  # dL/dV_pre[t] = dL/dI[t]
        tl.store(GI + off, gx.to(GI.dtype.element_ty), mask=m)
        has_prev = (d * 0 + t) > 0                                  # t > 0, as a vector (for the mask)
        xp = tl.load(VP + off - D, mask=m & has_prev, other=0.0)
        sp = tl.where(xp >= th, 1.0, 0.0)
        if HARD:
            rp = xp * (1.0 - gk * sp)
        else:
            rp = xp - th * sp
        vprev = tl.where(has_prev, rp, v0)
        acct += g_s * (-sg * x * inv) + gpost * dth
        accg += gpost * dgm
        if LEAK:
            beta = _sigmoid(bp + tl.load(M + off, mask=m, other=0.0).to(tl.float32))
            gz = gx * vprev * beta * (1.0 - beta)                    # dL/dbeta[t] -> dL/dlogit
            tl.store(GMOD + off, gz.to(GMOD.dtype.element_ty), mask=m)
            accb += gz
        else:
            beta = bp
            accb += gx * vprev
        gpost = beta * gx
    tl.store(GB + b * D + d, accb, mask=m)
    tl.store(GT + b * D + d, acct, mask=m)
    tl.store(GG + b * D + d, accg, mask=m)
    tl.store(GV0 + b * D + d, gpost, mask=m)


def _forward(I, M, bp, theta, gamma, v0, hard, leak):
    B, L, D = I.shape
    S = torch.empty_like(I)
    Vp = torch.empty(B, L, D, dtype=torch.float32, device=I.device)
    vl = torch.empty(B, D, dtype=torch.float32, device=I.device)
    grid = (B, triton.cdiv(D, BLOCK))
    _fwd_kernel[grid](I, M if leak else I, bp, theta, gamma, v0, S, Vp, vl, L, D,
                      HARD=bool(hard), LEAK=bool(leak), BLOCK=BLOCK)
    return S, Vp, vl


def _backward(gS, Vp, M, bp, theta, gamma, v0, gvl, hard, alpha, detach, sg, leak):
    B, L, D = Vp.shape
    gI = torch.empty_like(gS)
    gM = torch.empty_like(gS) if leak else gI
    f = dict(dtype=torch.float32, device=Vp.device)
    gb, gt, gg, gv0 = (torch.empty(B, D, **f) for _ in range(4))
    grid = (B, triton.cdiv(D, BLOCK))
    _bwd_kernel[grid](gS, Vp, M if leak else Vp, bp, theta, gamma, v0, gvl, gI, gM, gb, gt, gg, gv0, L, D,
                      float(alpha), HARD=bool(hard), DETACH=bool(detach), SG=int(sg), LEAK=bool(leak),
                      BLOCK=BLOCK)
    return gI, (gM if leak else None), gb, gt, gg, gv0


def _prep(I, *params):
    if I.dtype not in (torch.float32, torch.bfloat16):
        I = I.float()
    return (I.contiguous(),) + tuple(p.float().contiguous() for p in params)


class TritonLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, beta, theta, gamma, v0, hard_reset, alpha, detach_reset, sg_type):
        I, beta, theta, gamma, v0 = _prep(I, beta, theta, gamma, v0)
        S, Vp, vl = _forward(I, None, beta, theta, gamma, v0, hard_reset, False)
        ctx.save_for_backward(Vp, beta, theta, gamma, v0)
        ctx.cfg = (hard_reset, alpha, detach_reset, sg_type)
        ctx.in_dtype = I.dtype
        return S, vl

    @staticmethod
    def backward(ctx, gS, gvl):
        Vp, beta, theta, gamma, v0 = ctx.saved_tensors
        hard, alpha, detach, sg = ctx.cfg
        gS = torch.zeros_like(Vp, dtype=ctx.in_dtype) if gS is None else gS.to(ctx.in_dtype).contiguous()
        gvl = torch.zeros_like(v0) if gvl is None else gvl.float().contiguous()
        gI, _, gb, gt, gg, gv0 = _backward(gS, Vp, None, beta, theta, gamma, v0, gvl, hard, alpha, detach, sg,
                                           False)
        return gI, gb.sum(0), gt.sum(0), gg.sum(0), gv0, None, None, None, None


class TritonLIFLeak(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, M, beta_logit, theta, gamma, v0, hard_reset, alpha, detach_reset, sg_type):
        I, beta_logit, theta, gamma, v0 = _prep(I, beta_logit, theta, gamma, v0)
        M = M.to(I.dtype).contiguous()
        S, Vp, vl = _forward(I, M, beta_logit, theta, gamma, v0, hard_reset, True)
        ctx.save_for_backward(Vp, M, beta_logit, theta, gamma, v0)
        ctx.cfg = (hard_reset, alpha, detach_reset, sg_type)
        ctx.in_dtype = I.dtype
        return S, vl

    @staticmethod
    def backward(ctx, gS, gvl):
        Vp, M, beta_logit, theta, gamma, v0 = ctx.saved_tensors
        hard, alpha, detach, sg = ctx.cfg
        gS = torch.zeros_like(Vp, dtype=ctx.in_dtype) if gS is None else gS.to(ctx.in_dtype).contiguous()
        gvl = torch.zeros_like(v0) if gvl is None else gvl.float().contiguous()
        gI, gM, gbl, gt, gg, gv0 = _backward(gS, Vp, M, beta_logit, theta, gamma, v0, gvl, hard, alpha, detach,
                                             sg, True)
        return gI, gM, gbl.sum(0), gt.sum(0), gg.sum(0), gv0, None, None, None, None

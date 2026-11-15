"""Check the C++ LIF kernels against the pure-PyTorch reference (values and every gradient), then time them."""
import itertools, time
import torch
from lif import lif, lif_leak, lif_reference

torch.manual_seed(0)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-8)


def check(tv=False):
    B, L, D = 3, 37, 300
    worst = 0.0
    for hard, detach, sg in itertools.product([False, True], [True, False], ["sigmoid", "fast_sigmoid", "triangle"]):
        I = (torch.randn(B, L, D) * 0.8 + 0.3).requires_grad_()
        if tv:   # leak per step: beta = sigmoid(beta_logit + M)
            beta_logit = (torch.randn(D) * 2).requires_grad_()
            M = (torch.randn(B, L, D) * 1.5).requires_grad_()
        else:
            beta = torch.rand(D).requires_grad_()
        theta = (torch.rand(D) + 0.5).requires_grad_()
        gamma = (torch.rand(D) * 0.8 + 0.2).requires_grad_()
        v0 = torch.randn(B, D).requires_grad_()
        params = [I, M, beta_logit, theta, gamma, v0] if tv else [I, beta, theta, gamma, v0]
        w = torch.randn(B, L, D); wv = torch.randn(B, D)
        outs = []
        for impl in ("kernel", "reference"):
            for p in params:
                p.grad = None
            kw = dict(hard_reset=hard, alpha=4.0, detach_reset=detach, sg=sg)
            if impl == "kernel":
                if tv:
                    S, vl = lif_leak(I, beta_logit, M, theta, gamma, v0, **kw)
                else:
                    S, vl = lif(I, beta, theta, gamma, v0, **kw)
            else:
                if tv:
                    S, vl = lif_reference(I, torch.sigmoid(beta_logit + M), theta, gamma, v0, **kw)
                else:
                    S, vl = lif_reference(I, beta, theta, gamma, v0, **kw)
            ((S * w).sum() + (vl * wv).sum()).backward()
            # gamma only matters for the multiplicative reset; in soft mode the reference gives it no grad
            g = [p.grad.clone() if p.grad is not None else torch.zeros(1) for p in params]
            if not hard:
                g[len(params) - 2] = torch.zeros(1)
            outs.append([S.detach().clone(), vl.detach().clone()] + g)
        errs = [rel(a, b) for a, b in zip(*outs)]
        worst = max(worst, max(errs[1:]))
        assert torch.equal(outs[0][0], outs[1][0]), "spikes differ"
        names = "v,gI,gM,gbeta_logit,gtheta,ggamma,gv0" if tv else "v,gI,gbeta,gtheta,ggamma,gv0"
        print(f"  tv={tv!s:5} hard={hard!s:5} detach={detach!s:5} sg={sg:12} spikes equal; max rel.err "
              f"[{names}] = {max(errs[1:]):.1e}")
    return worst


print("independent neurons:")
w1 = check()
print("with input-controlled leak (beta = sigmoid(beta_logit + M) changes every step):")
w3 = check(tv=True)
print(f"worst relative error over all checks: {max(w1, w3):.1e}")

# bf16 path (independent neurons): identical spikes on bf16-representable inputs
B, L, D = 4, 50, 700
I32 = torch.randn(B, L, D).bfloat16().float().requires_grad_()
I16 = I32.detach().bfloat16().requires_grad_()
beta = torch.rand(D); theta = torch.rand(D) + 0.5; gamma = torch.rand(D)
S32, _ = lif(I32, beta, theta, gamma, hard_reset=True)
S16, _ = lif(I16, beta, theta, gamma, hard_reset=True)
print("bf16 path, independent neurons: spikes equal:", torch.equal(S32, S16.float()))

# timing at training size
B, L, H = 32, 256, 1536
I = torch.randn(B, L, H).bfloat16().requires_grad_()
beta = torch.rand(H); theta = torch.ones(H); gamma = torch.full((H,), 0.9, requires_grad=True)


def bench(f, n=3):
    f()
    t = time.time()
    for _ in range(n):
        f()
    return (time.time() - t) / n * 1000


def run_plain():
    S, _ = lif(I, beta, theta, gamma, hard_reset=True, sg="fast_sigmoid", detach_reset=False)
    S.float().sum().backward()


print(f"fwd+bwd {B}x{L}x{H} bf16: {bench(run_plain):.1f} ms")

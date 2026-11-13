"""Check the C++ LIF kernels against the pure-PyTorch reference (values and every gradient), then time them."""
import itertools, time
import torch
from lif import lif, lif_rec, lif_reference

torch.manual_seed(0)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-8)


def check(rec):
    B, L, D, r = 3, 37, 300, 16
    worst = 0.0
    for hard, detach, sg in itertools.product([False, True], [True, False], ["sigmoid", "fast_sigmoid", "triangle"]):
        I = (torch.randn(B, L, D) * 0.8 + 0.3).requires_grad_()
        beta = torch.rand(D).requires_grad_()
        theta = (torch.rand(D) + 0.5).requires_grad_()
        gamma = (torch.rand(D) * 0.8 + 0.2).requires_grad_()
        v0 = torch.randn(B, D).requires_grad_()
        params = [I, beta, theta, gamma, v0]
        if rec:
            A = (torch.randn(D, r) * 0.3).requires_grad_()
            Bm = (torch.randn(r, D) * 0.3).requires_grad_()
            s0 = (torch.rand(B, D) < 0.3).float().requires_grad_()
            params += [A, Bm, s0]
        w = torch.randn(B, L, D); wv = torch.randn(B, D)
        outs = []
        for impl in ("kernel", "reference"):
            for p in params:
                p.grad = None
            kw = dict(hard_reset=hard, alpha=4.0, detach_reset=detach, sg=sg)
            if impl == "kernel":
                S, vl = (lif_rec(I, beta, theta, A, Bm, gamma, v0, s0, **kw) if rec
                         else lif(I, beta, theta, gamma, v0, **kw))
            else:
                S, vl = (lif_reference(I, beta, theta, gamma, v0, A=A, Bm=Bm, s0=s0, **kw) if rec
                         else lif_reference(I, beta, theta, gamma, v0, **kw))
            ((S * w).sum() + (vl * wv).sum()).backward()
            # gamma only matters for the multiplicative reset; in soft mode the reference gives it no grad
            g = [p.grad.clone() if p.grad is not None else torch.zeros(1) for p in params]
            if not hard:
                g[3] = torch.zeros(1)
            outs.append([S.detach().clone(), vl.detach().clone()] + g)
        errs = [rel(a, b) for a, b in zip(*outs)]
        worst = max(worst, max(errs[1:]))
        assert torch.equal(outs[0][0], outs[1][0]), "spikes differ"
        names = "v,gI,gbeta,gtheta,ggamma,gv0" + (",gA,gB,gs0" if rec else "")
        print(f"  rec={rec!s:5} hard={hard!s:5} detach={detach!s:5} sg={sg:12} spikes equal; max rel.err "
              f"[{names}] = {max(errs[1:]):.1e}")
    return worst


print("independent neurons:")
w1 = check(rec=False)
print("with same-layer synapses:")
w2 = check(rec=True)
print(f"worst relative error over all checks: {max(w1, w2):.1e}")

# bf16 path (independent neurons): identical spikes on bf16-representable inputs
B, L, D = 4, 50, 700
I32 = torch.randn(B, L, D).bfloat16().float().requires_grad_()
I16 = I32.detach().bfloat16().requires_grad_()
beta = torch.rand(D); theta = torch.rand(D) + 0.5; gamma = torch.rand(D)
S32, _ = lif(I32, beta, theta, gamma, hard_reset=True)
S16, _ = lif(I16, beta, theta, gamma, hard_reset=True)
print("bf16 path, independent neurons: spikes equal:", torch.equal(S32, S16.float()))

# timing at training size
B, L, H, r = 32, 256, 1536, 128
I = torch.randn(B, L, H).bfloat16().requires_grad_()
beta = torch.rand(H); theta = torch.ones(H); gamma = torch.full((H,), 0.9, requires_grad=True)
A = (torch.randn(H, r) * 0.05).requires_grad_(); Bm = (torch.randn(r, H) * 0.05).requires_grad_()


def bench(f, n=3):
    f()
    t = time.time()
    for _ in range(n):
        f()
    return (time.time() - t) / n * 1000


def run_plain():
    S, _ = lif(I, beta, theta, gamma, hard_reset=True, sg="fast_sigmoid", detach_reset=False)
    S.float().sum().backward()


def run_rec():
    S, _ = lif_rec(I, beta, theta, A, Bm, gamma, hard_reset=True, sg="fast_sigmoid", detach_reset=False)
    S.float().sum().backward()


print(f"fwd+bwd {B}x{L}x{H} bf16: independent {bench(run_plain):.1f} ms, "
      f"same-layer rank {r}: {bench(run_rec):.1f} ms")

"""Check the Triton LIF kernels (lif_triton.py) against the pure-PyTorch reference: spikes, final membrane and
every gradient, for all reset / surrogate settings, with a fixed leak and with a leak that changes every step.
On a machine with an NVIDIA GPU it runs there and then times the kernels; without one, run it through Triton's
interpreter on the CPU (slow, correctness only):

  python3 test_lif_gpu.py                       # on a GPU machine
  TRITON_INTERPRET=1 python3 test_lif_gpu.py    # anywhere
"""
import itertools, os, sys, time
import torch
from lif import lif_reference, SG_TYPES

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    sys.exit("no GPU found: run with TRITON_INTERPRET=1 to check the kernels on the CPU")
from lif_triton import TritonLIF, TritonLIFLeak

torch.manual_seed(0)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-8)


def check(leak):
    B, L, D = 3, 37, 300          # D not a multiple of the block size, to exercise the masks
    worst = 0.0
    for hard, detach, sg in itertools.product([False, True], [True, False], ["sigmoid", "fast_sigmoid", "triangle"]):
        I = (torch.randn(B, L, D, device=dev) * 0.8 + 0.3).requires_grad_()
        if leak:
            bl = (torch.randn(D, device=dev) * 2).requires_grad_()
            M = (torch.randn(B, L, D, device=dev) * 1.5).requires_grad_()
        else:
            beta = torch.rand(D, device=dev).requires_grad_()
        theta = (torch.rand(D, device=dev) + 0.5).requires_grad_()
        gamma = (torch.rand(D, device=dev) * 0.8 + 0.2).requires_grad_()
        v0 = torch.randn(B, D, device=dev).requires_grad_()
        params = [I, M, bl, theta, gamma, v0] if leak else [I, beta, theta, gamma, v0]
        w = torch.randn(B, L, D, device=dev); wv = torch.randn(B, D, device=dev)
        outs = []
        for impl in ("triton", "reference"):
            for p in params:
                p.grad = None
            if impl == "triton":
                if leak:
                    S, vl = TritonLIFLeak.apply(I, M, bl, theta, gamma, v0, hard, 4.0, detach, SG_TYPES[sg])
                else:
                    S, vl = TritonLIF.apply(I, beta, theta, gamma, v0, hard, 4.0, detach, SG_TYPES[sg])
            else:
                b_ = torch.sigmoid(bl + M) if leak else beta
                S, vl = lif_reference(I, b_, theta, gamma, v0, hard_reset=hard, alpha=4.0, detach_reset=detach,
                                      sg=sg)
            ((S * w).sum() + (vl * wv).sum()).backward()
            g = [p.grad.clone() if p.grad is not None else torch.zeros(1, device=dev) for p in params]
            if not hard:   # gamma only matters for the multiplicative reset
                g[len(params) - 2] = torch.zeros(1, device=dev)
            outs.append([S.detach().clone(), vl.detach().clone()] + g)
        n_diff = (outs[0][0] != outs[1][0]).sum().item()
        assert n_diff == 0, f"{n_diff} spikes differ"
        errs = [rel(a, b) for a, b in zip(*outs)]
        worst = max(worst, max(errs[1:]))
        names = "v,gI,gM,gbeta_logit,gtheta,ggamma,gv0" if leak else "v,gI,gbeta,gtheta,ggamma,gv0"
        print(f"  leak={leak!s:5} hard={hard!s:5} detach={detach!s:5} sg={sg:12} spikes equal; max rel.err "
              f"[{names}] = {max(errs[1:]):.1e}", flush=True)
    return worst


print(f"device: {dev}")
print("fixed leak:")
w1 = check(False)
print("leak changes every step:")
w2 = check(True)
print(f"worst relative error: {max(w1, w2):.1e}")
assert max(w1, w2) < 1e-3, "gradients disagree with the reference"

# bf16 input current (as under autocast): spikes must equal those of the float32 path on the same values
B, L, D = 4, 64, 512
I16 = (torch.randn(B, L, D, device=dev) * 0.8 + 0.3).bfloat16()
beta = torch.rand(D, device=dev); theta = torch.rand(D, device=dev) + 0.5; gamma = torch.rand(D, device=dev)
v0 = torch.zeros(B, D, device=dev)
S16, _ = TritonLIF.apply(I16, beta, theta, gamma, v0, True, 4.0, False, 1)
S32, _ = TritonLIF.apply(I16.float(), beta, theta, gamma, v0, True, 4.0, False, 1)
assert S16.dtype == torch.bfloat16 and torch.equal(S16.float(), S32)
print("bf16 path: spikes equal to float32 path")

if dev == "cuda":
    for H in (1536, 2048, 4096):
        B, L = 32, 256
        I = (torch.randn(B, L, H, device=dev) * 0.8).bfloat16().requires_grad_()
        M = torch.randn(B, L, H, device=dev).bfloat16().requires_grad_()
        args = (torch.rand(H, device=dev), torch.rand(H, device=dev) + 0.5, torch.rand(H, device=dev),
                torch.zeros(B, H, device=dev))
        for name, fn in (("fixed leak", lambda: TritonLIF.apply(I, *args, True, 4.0, False, 1)),
                         ("leak per step", lambda: TritonLIFLeak.apply(I, M, *args, True, 4.0, False, 1))):
            for i in range(13):
                if i == 3:
                    torch.cuda.synchronize(); t = time.time()
                S, vl = fn()
                S.float().sum().backward()
            torch.cuda.synchronize()
            print(f"fwd+bwd {B}x{L}x{H} bf16, {name}: {(time.time() - t) / 10 * 1e3:.2f} ms")

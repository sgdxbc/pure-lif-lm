import torch, itertools, time
from lif import lif, lif_reference
torch.manual_seed(0)
B, L, D = 3, 37, 300
for hard, detach, sg in itertools.product([False, True], [True, False], ["sigmoid", "fast_sigmoid", "triangle"]):
    I = (torch.randn(B, L, D) * 0.8 + 0.3).requires_grad_()
    beta = torch.rand(D).requires_grad_()
    theta = (torch.rand(D) + 0.5).requires_grad_()
    v0 = torch.randn(B, D).requires_grad_()
    w = torch.randn(B, L, D); wv = torch.randn(B, D)
    outs = []
    for fn in (lif, lif_reference):
        for p in (I, beta, theta, v0): p.grad = None
        S, vl = fn(I, beta, theta, v0, hard_reset=hard, alpha=4.0, detach_reset=detach, sg=sg)
        ((S * w).sum() + (vl * wv).sum()).backward()
        outs.append([S.detach().clone(), vl.detach().clone()] + [p.grad.clone() for p in (I, beta, theta, v0)])
    errs = [(a - b).abs().max().item() / (b.abs().max().item() + 1e-8) for a, b in zip(*outs)]
    print(f"hard={hard!s:5} detach={detach!s:5} sg={sg:12} rel.err S,v,gI,gbeta,gtheta,gv0 = " + " ".join(f"{e:.1e}" for e in errs))
# speed
B, L, D = 64, 256, 1536
I = torch.randn(B, L, D).requires_grad_(); beta = torch.rand(D).requires_grad_(); theta = torch.ones(D).requires_grad_()
for _ in range(2):
    S, _ = lif(I, beta, theta); S.sum().backward()
t = time.time()
for _ in range(5):
    S, _ = lif(I, beta, theta)
t1 = time.time()
for _ in range(5):
    S, _ = lif(I, beta, theta); S.sum().backward()
t2 = time.time()
print(f"fwd {(t1-t)/5*1000:.1f} ms, fwd+bwd {(t2-t1)/5*1000:.1f} ms for {B}x{L}x{D}")
# bf16 path must give identical spikes and (bf16-rounded) identical grads to the fp32 path on bf16-representable inputs
torch.manual_seed(1)
B, L, D = 4, 50, 700
I32 = torch.randn(B, L, D).bfloat16().float().requires_grad_()
I16 = I32.detach().bfloat16().requires_grad_()
beta = torch.rand(D); theta = torch.rand(D) + 0.5
w = torch.randn(B, L, D).bfloat16()
S32, v32 = lif(I32, beta, theta); (S32 * w.float()).sum().backward()
S16, v16 = lif(I16, beta, theta); (S16 * w).sum().backward()
print("bf16 path: S equal:", torch.equal(S32, S16.float()), " v diff:", (v32 - v16).abs().max().item(),
      " gI rel diff:", ((I32.grad - I16.grad.float()).abs().max() / I32.grad.abs().max()).item(), " S dtype", S16.dtype, " gI dtype", I16.grad.dtype)
I = torch.randn(64, 256, 1536).bfloat16().requires_grad_(); beta = torch.rand(1536); theta = torch.ones(1536)
for _ in range(2):
    S, _ = lif(I, beta, theta); S.float().sum().backward()
t = time.time()
for _ in range(5):
    S, _ = lif(I, beta, theta); S.sum().backward()
print(f"bf16 fwd+bwd {(time.time()-t)/5*1000:.1f} ms")

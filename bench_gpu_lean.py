"""Transformer decoding on the GPU with a minimal program, as a bound on what the GPU itself can do for our small
Transformers (docs/experiments.md 5.43). bench_gpu.py measures them with vLLM, the mature serving software the
comparison of README section 0 (stage goal 3) prescribes; for these small models vLLM's time per step is the same for
every model size (about 1.0 ms at 1 stream, 3.7 ms at 256, 50 ms at 4,096 streams: a fixed cost per step and per
stream outside the GPU's arithmetic), so it says little about the GPU's own limit. Here nothing but the model runs:
  - our own Transformer (model_tf.py: learned positions, RMSNorm, 4x MLP, no biases, an untied 50,304-word readout),
    bf16, random weights (the time and energy of decoding do not depend on the weights' values);
  - a static key/value cache of context + gen positions for B streams, filled with random values to stand for the
    context of C tokens (decoding reads it the same way whatever it holds), so no prefill is measured;
  - one decoding step = embedding, every block with attention over the cache (a mask hides the positions not yet
    written: attention is computed over the whole cache, slightly more work than needed), the readout, greedy argmax,
    the chosen token fed back and the position advanced on the GPU; CHUNK steps are captured as one CUDA graph and
    replayed, so the host does nothing per step but launch a graph.
Energy from NVML's board energy counter as in bench_gpu.py (whole board; also above the idle power), each measurement
repeated for at least --min_seconds. From the repository root, on a GPU node (the training venv suffices: no vLLM):
  python3 bench_gpu_lean.py --shape 256,6,4 [--batches 1,16,256,1024,4096] [--contexts 128,896] [--gen 128]
writes runs/bench_gpu/<gpu>_lean_dD_lL_hH.json (the same row fields as bench_gpu.py's).
  python3 bench_gpu_lean.py --check     (CPU, fp32) the cached step reproduces GPT.forward's logits position by position
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from model_tf import GPT, TFConfig

HERE = os.path.dirname(os.path.abspath(__file__))


class Decoder:
    """B streams decoding with a static cache of T positions; step() advances every stream by one token."""

    def __init__(self, m, B, T, dev, dtype):
        c = m.cfg
        self.m, self.B, self.T, self.H, self.D = m, B, T, c.n_heads, c.d_model
        hd = c.d_model // c.n_heads
        shape = (len(m.blocks), B, c.n_heads, T, hd)
        self.kc = torch.randn(shape, device=dev, dtype=dtype)
        self.vc = torch.randn(shape, device=dev, dtype=dtype)
        self.tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.ar = torch.arange(T, device=dev)
        self.logits = None

    def step(self):
        m, B, H, D = self.m, self.B, self.H, self.D
        x = m.embed(self.tok) + m.pos(self.pos)                        # (B, D)
        mask = (self.ar <= self.pos).view(1, 1, 1, -1)                  # positions written so far
        for l, b in enumerate(m.blocks):
            q, k, v = b.qkv(b.n1(x)).view(B, 3, H, D // H).unbind(1)    # (B, H, hd) each
            self.kc[l].index_copy_(2, self.pos, k.unsqueeze(2))
            self.vc[l].index_copy_(2, self.pos, v.unsqueeze(2))
            a = F.scaled_dot_product_attention(q.unsqueeze(2), self.kc[l], self.vc[l], attn_mask=mask)
            x = x + b.proj(a.reshape(B, D))
            x = x + b.fc2(F.gelu(b.fc(b.n2(x))))
        self.logits = m.head(m.norm(x))
        self.tok.copy_(self.logits.argmax(-1))
        self.pos.add_(1)


def check():
    """The cached step gives GPT.forward's logits at every position (CPU, fp32, a small random model)."""
    torch.manual_seed(0)
    m = GPT(TFConfig(vocab=500, d_model=64, n_layers=3, n_heads=4, max_len=64)).eval()
    B, L = 3, 20
    idx = torch.randint(0, 500, (B, L))
    with torch.no_grad():
        ref = m(idx)[0]
        d = Decoder(m, B, 32, "cpu", torch.float32)
        err = 0.0
        for t in range(L):
            d.tok.copy_(idx[:, t])
            d.step()
            err = max(err, (d.logits - ref[:, t]).abs().max().item())
    print(f"cached decoding vs GPT.forward, {B} streams x {L} positions: max |logit difference| {err:.2e}")
    assert err < 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", help="D,L,H: width, blocks, heads")
    p.add_argument("--batches", default="1,16,256,1024,4096")
    p.add_argument("--contexts", default="128,896")
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--chunk", type=int, default=8, help="decoding steps captured in one CUDA graph")
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--min_seconds", type=float, default=10.0)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        return check()
    import pynvml
    pynvml.nvmlInit()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    idx = 0 if pynvml.nvmlDeviceGetCount() == 1 else (int(cvd) if cvd.isdigit() else 0)
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    gpu = pynvml.nvmlDeviceGetName(h)
    gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
    energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0   # joules
    D, L, H = (int(x) for x in a.shape.split(","))
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    m = GPT(TFConfig(vocab=a.vocab, d_model=D, n_layers=L, n_heads=H, max_len=1024)).to(dev, dt).eval()
    contexts = [int(c) for c in a.contexts.split(",")]
    batches = [int(b) for b in a.batches.split(",")]
    assert a.gen % a.chunk == 0 and max(contexts) + a.gen <= 1024
    e0, t0 = energy(), time.perf_counter()
    time.sleep(10)
    idle_w = (energy() - e0) / (time.perf_counter() - t0)
    rows = []
    for C in contexts:
        for B in batches:
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    d = Decoder(m, B, C + a.gen, dev, dt)
                    d.tok.random_(0, a.vocab)
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):                     # warm-up (kernel selection) before capture
                        for _ in range(3):
                            d.pos.fill_(C)
                            d.step()
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(a.chunk):
                            d.step()

                    def gen():                                      # one generation of a.gen tokens per stream
                        d.pos.fill_(C)
                        for _ in range(a.gen // a.chunk):
                            g.replay()

                    gen()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    gen()
                    torch.cuda.synchronize()
                    n = max(2, int(min(a.min_seconds / max(time.perf_counter() - t0, 1e-4), 100000)))
                    e0, t0 = energy(), time.perf_counter()
                    for _ in range(n):
                        gen()
                    torch.cuda.synchronize()
                    dt_s, de = time.perf_counter() - t0, energy() - e0
            except torch.cuda.OutOfMemoryError:
                print(f"{gpu} lean {a.shape} streams {B} context {C}: out of memory", flush=True)
                continue
            steps = n * a.gen
            r = {"streams": B, "context": C, "gen": a.gen, "decode_tok_s": B * steps / dt_s,
                 "decode_tok_s_per_stream": steps / dt_s, "ms_per_token": dt_s / steps * 1000,
                 "decode_J_per_token": de / (B * steps), "decode_avg_W": de / dt_s,
                 "decode_J_per_token_above_idle": (de - idle_w * dt_s) / (B * steps), "calls": n}
            rows.append(r)
            print(f"{gpu} lean {a.shape} streams {B:5d} context {C:4d}: {r['decode_tok_s']:12,.0f} tok/s "
                  f"({r['ms_per_token']:.3f} ms/token per stream), {r['decode_J_per_token'] * 1000:8.3f} mJ/token "
                  f"(above idle {r['decode_J_per_token_above_idle'] * 1000:.3f}), {r['decode_avg_W']:.0f} W", flush=True)
            del d, g
    out = os.path.join(HERE, "runs", "bench_gpu")
    os.makedirs(out, exist_ok=True)
    name = f"{gpu.replace(' ', '_')}_lean_d{D}_l{L}_h{H}.json"
    json.dump({"shape": a.shape, "gpu": gpu, "idle_W": idle_w, "vocab": a.vocab, "chunk": a.chunk, "rows": rows,
               "torch": torch.__version__}, open(os.path.join(out, name), "w"), indent=1)
    print("wrote", os.path.join(out, name), f"(idle {idle_w:.0f} W)")


if __name__ == "__main__":
    main()

"""Transformer inference on the hardware it suits best (README section 0, stage goal 3): text generation with vLLM on
a GPU, for the reference models the spiking network is compared with. For every number of concurrent streams B and
context length C (tokens already in each stream before generation):
  - decode throughput: generated tokens per second over all streams, and per stream;
  - latency: milliseconds per generated token in each stream;
  - energy: GPU board energy (NVML's energy counter) per generated token, and the average board power.
The prefill of the C context tokens is measured separately (a run that generates one token) and subtracted, so the
decode numbers are for generation only. Each of the two is repeated for at least --min_seconds (10 s), because NVML's
energy counter updates only every few tens of milliseconds and a single short run gives noisy, even negative,
differences. Prompts are random token ids, different in every call, with prefix caching off. Energy is the whole
board's (also reported above the idle power); the GPU only, no CPU, host memory or cooling.
Needs vllm and nvidia-ml-py (a separate venv, slurm/bench_gpu.sbatch). From the repository root:
  python3 bench_gpu.py --model openai-community/gpt2 [--batches 1,4,16,64,256] [--contexts 128,896] [--gen 128] [--suffix S]
writes runs/bench_gpu/<gpu>_<model>[S].json
  python3 bench_gpu.py --model shape:D,L,H
measures a GPT-2 of that shape (width D, L blocks, H heads, GPT-2's 50,257-word vocabulary, 1,024 positions) with
random weights (vLLM load_format "dummy"; GPT-2's tokenizer, which vLLM requires although only token ids go in and
out): the time and energy of generation do not depend on the weights' values (ignore_eos, fixed lengths), so this is
the cost of our own Transformers (model_tf.py: the same matrices, a 50,304-word vocabulary; RMSNorm and no biases
instead of LayerNorm and biases, a negligible difference) without converting them. Writes
runs/bench_gpu/<gpu>_gpt2shape_dD_lL_hH.json.
"""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batches", default="1,4,16,64,256")
    p.add_argument("--contexts", default="", help="default: 128 and the longest that fits with --gen")
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--suffix", default="", help="appended to the output file's name (e.g. for a second set of batches)")
    p.add_argument("--min_seconds", type=float, default=10.0,
                   help="each measurement repeats the generation for at least this long (energy counter resolution)")
    a = p.parse_args()
    import pynvml
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    pynvml.nvmlInit()
    # the GPU this job was given: the only one NVML sees (SLURM device cgroups), else CUDA_VISIBLE_DEVICES' first
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    idx = 0 if pynvml.nvmlDeviceGetCount() == 1 else (int(cvd) if cvd.isdigit() else 0)
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    gpu = pynvml.nvmlDeviceGetName(h)
    gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
    batches = [int(b) for b in a.batches.split(",")]
    model, extra, tag = a.model, {}, a.model.split("/")[-1]
    if a.model.startswith("shape:"):   # a GPT-2 of this shape with random weights (see the top of this file)
        import tempfile
        from transformers import GPT2Config
        D, L, H = (int(x) for x in a.model[len("shape:"):].split(","))
        model = tempfile.mkdtemp(prefix="gpt2shape_")
        # architectures: without it vLLM 0.33 picks an embedding ("pooling") runner and refuses generate()
        GPT2Config(vocab_size=50257, n_positions=1024, n_embd=D, n_layer=L, n_head=H, architectures=["GPT2LMHeadModel"],
                   bos_token_id=50256, eos_token_id=50256).save_pretrained(model)
        # vLLM 0.33 needs a tokenizer even when only token ids are used (skip_tokenizer_init fails in its constructor)
        extra, tag = dict(load_format="dummy", tokenizer="openai-community/gpt2"), f"gpt2shape_d{D}_l{L}_h{H}"
    if max(batches) > 2048:   # let one decode step hold every stream (vLLM's default token budget per step is smaller)
        extra["max_num_batched_tokens"] = max(16384, max(batches))
    cfg = AutoConfig.from_pretrained(model)
    max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 1024)
    vocab = cfg.vocab_size
    llm = LLM(model=model, dtype="bfloat16", seed=0, enable_prefix_caching=False, max_num_seqs=max(batches),
              max_model_len=max_len, gpu_memory_utilization=0.85, **extra)
    contexts = [int(c) for c in a.contexts.split(",")] if a.contexts else [128, max_len - a.gen]
    rng = np.random.default_rng(0)

    def prompts_for(B, C):
        return [{"prompt_token_ids": rng.integers(100, min(vocab, 50000), C).tolist()} for _ in range(B)]

    def energy():
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0   # joules since the driver loaded

    def run(B, C, gen, min_s):
        """Repeat the generation (fresh prompts each time, made beforehand) for at least min_s seconds, so the
        energy counter, which updates only every few tens of milliseconds, is read over a long window.
        Returns seconds and joules per call, and the number of calls."""
        sp = SamplingParams(max_tokens=gen, temperature=0.0, ignore_eos=True, detokenize=False)
        t0 = time.perf_counter()
        llm.generate(prompts_for(B, C), sp, use_tqdm=False)          # also a warm-up for this shape
        n = max(1, int(np.ceil(min_s / max(time.perf_counter() - t0, 1e-3))))
        batches_ = [prompts_for(B, C) for _ in range(n)]
        e0, t0 = energy(), time.perf_counter()
        for ps in batches_:
            outs = llm.generate(ps, sp, use_tqdm=False)
        t, e = time.perf_counter() - t0, energy() - e0
        assert all(len(o.outputs[0].token_ids) == gen for o in outs)
        return t / n, e / n, n

    # idle board power, for reference
    e0, t0 = energy(), time.perf_counter()
    time.sleep(10)
    idle_w = (energy() - e0) / (time.perf_counter() - t0)
    run(4, 64, 16, 1.0)   # warm-up (compilation, CUDA graphs)
    rows = []
    for C in contexts:
        for B in batches:
            t1, e1, n1 = run(B, C, 1, a.min_seconds)        # prefill (and one generated token)
            tg, eg, ng = run(B, C, a.gen, a.min_seconds)    # prefill and a.gen generated tokens
            dt, de, n = tg - t1, eg - e1, B * (a.gen - 1)
            r = {"streams": B, "context": C, "gen": a.gen, "decode_tok_s": n / dt, "decode_tok_s_per_stream": (a.gen - 1) / dt,
                 "ms_per_token": dt / (a.gen - 1) * 1000, "decode_J_per_token": de / n, "decode_avg_W": de / dt,
                 "decode_J_per_token_above_idle": (de - idle_w * dt) / n,
                 "prefill_s": t1, "prefill_J": e1, "calls": [n1, ng]}
            rows.append(r)
            print(f"{a.model} {gpu} streams {B:4d} context {C:5d}: {r['decode_tok_s']:10,.0f} tok/s "
                  f"({r['ms_per_token']:.2f} ms/token per stream), {r['decode_J_per_token'] * 1000:8.2f} mJ/token "
                  f"(above idle {r['decode_J_per_token_above_idle'] * 1000:.2f}), {r['decode_avg_W']:.0f} W", flush=True)
    out = os.path.join(HERE, "runs", "bench_gpu")
    os.makedirs(out, exist_ok=True)
    name = f"{gpu.replace(' ', '_')}_{tag}{a.suffix}.json"
    json.dump({"model": a.model, "gpu": gpu, "idle_W": idle_w, "max_model_len": max_len, "rows": rows,
               "vllm": __import__("vllm").__version__}, open(os.path.join(out, name), "w"), indent=1)
    print("wrote", os.path.join(out, name), f"(idle {idle_w:.0f} W)")


if __name__ == "__main__":
    main()

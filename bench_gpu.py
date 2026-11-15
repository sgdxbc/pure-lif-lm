"""Transformer inference on the hardware it suits best (README section 0, stage goal 3): text generation with vLLM on
a GPU, for the reference models the spiking network is compared with. For every number of concurrent streams B and
context length C (tokens already in each stream before generation):
  - decode throughput: generated tokens per second over all streams, and per stream;
  - latency: milliseconds per generated token in each stream;
  - energy: GPU board energy (NVML's energy counter) per generated token, and the average board power.
The prefill of the C context tokens is measured separately (a run that generates one token) and subtracted, so the
decode numbers are for generation only. Prompts are random token ids, different in every run, with prefix caching
off. The GPU's own energy only (no CPU, memory or cooling).
Needs vllm and nvidia-ml-py (a separate venv, slurm/bench_gpu.sbatch). From the repository root:
  python3 bench_gpu.py --model openai-community/gpt2 [--batches 1,4,16,64,256] [--contexts 128,896] [--gen 128]
writes runs/bench_gpu/<gpu>_<model>.json
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
    p.add_argument("--repeats", type=int, default=3, help="measurements per setting; the median is kept")
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
    cfg = AutoConfig.from_pretrained(a.model)
    max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 1024)
    vocab = cfg.vocab_size
    llm = LLM(model=a.model, dtype="bfloat16", seed=0, enable_prefix_caching=False, max_num_seqs=max(batches),
              max_model_len=max_len, gpu_memory_utilization=0.85)
    contexts = [int(c) for c in a.contexts.split(",")] if a.contexts else [128, max_len - a.gen]
    rng = np.random.default_rng(0)

    def run(B, C, gen):
        prompts = [{"prompt_token_ids": rng.integers(100, min(vocab, 50000), C).tolist()} for _ in range(B)]
        sp = SamplingParams(max_tokens=gen, temperature=0.0, ignore_eos=True)
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)   # millijoules
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        t = time.perf_counter() - t0
        e = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1000.0
        assert all(len(o.outputs[0].token_ids) == gen for o in outs)
        return t, e

    # idle board power, for reference
    e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
    time.sleep(5)
    idle_w = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1000.0 / (time.perf_counter() - t0)
    run(4, 64, 16)   # warm-up (compilation, CUDA graphs)
    rows = []
    for C in contexts:
        for B in batches:
            res = []
            for _ in range(a.repeats):
                t1, e1 = run(B, C, 1)
                tg, eg = run(B, C, a.gen)
                res.append((tg - t1, eg - e1, t1, e1))
            dt, de, t1, e1 = sorted(res)[len(res) // 2]
            n = B * (a.gen - 1)
            r = {"streams": B, "context": C, "gen": a.gen, "decode_tok_s": n / dt, "decode_tok_s_per_stream": (a.gen - 1) / dt,
                 "ms_per_token": dt / (a.gen - 1) * 1000, "decode_J_per_token": de / n, "decode_avg_W": de / dt,
                 "prefill_s": t1, "prefill_J": e1}
            rows.append(r)
            print(f"{a.model} {gpu} streams {B:4d} context {C:5d}: {r['decode_tok_s']:10,.0f} tok/s "
                  f"({r['ms_per_token']:.2f} ms/token per stream), {r['decode_J_per_token'] * 1000:8.2f} mJ/token, "
                  f"{r['decode_avg_W']:.0f} W", flush=True)
    out = os.path.join(HERE, "runs", "bench_gpu")
    os.makedirs(out, exist_ok=True)
    name = f"{gpu.replace(' ', '_')}_{a.model.split('/')[-1]}.json"
    json.dump({"model": a.model, "gpu": gpu, "idle_W": idle_w, "max_model_len": max_len, "rows": rows,
               "vllm": __import__("vllm").__version__}, open(os.path.join(out, name), "w"), indent=1)
    print("wrote", os.path.join(out, name), f"(idle {idle_w:.0f} W)")


if __name__ == "__main__":
    main()

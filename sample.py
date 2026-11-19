"""Text completion with a trained spiking LM.

  python3 sample.py --ckpt runs/main/ckpt.pt --prompt "Once upon a time" --n 200
  python3 sample.py --ckpt runs/main/ckpt.pt -i        # interactive
"""
import argparse, json, os, time
import torch
from tokenizers import Tokenizer
from model import Config, SpikingLM, round_weights_

HERE = os.path.dirname(os.path.abspath(__file__))


def load(ckpt_path, ema=False, device="cpu", wbits=None):
    """ema=True: use the moving-average weights saved by train.py --ema (if the checkpoint has them).
    wbits: round the weight matrices to this many bits (model.py quantize_weight); None = the network's own wbits
    (set by train.py --wbits; 0 for a network trained at full precision), 0 = do not round."""
    ck = torch.load(ckpt_path, map_location="cpu")
    if ck.get("arch") == "tf":
        from model_tf import GPT, TFConfig
        model = GPT(TFConfig(**ck["cfg"]))
    else:
        model = SpikingLM(Config.from_dict(ck["cfg"]))
    sd = ck["model"]
    if ema:
        assert "ema" in ck, f"{ckpt_path} has no EMA weights"
        sd = {k: (ck["ema"][k].to(v.dtype) if k in ck["ema"] else v) for k, v in sd.items()}
    model.load_state_dict(sd)
    if ck.get("arch") != "tf":
        bits = model.cfg.wbits if wbits is None else wbits
        if bits:
            round_weights_(model, bits, model.cfg.wgroup, model.cfg.wbits8)
    model.to(device).eval()
    return model, ck.get("step")


def complete(model, tok, prompt, n=200, temperature=0.8, top_k=50, seed=None):
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
    ids = torch.tensor([tok.encode(prompt).ids], device=dev)
    eos = tok.token_to_id("<|endoftext|>")   # 0 in our own tokenizers, 50256 in GPT-2's
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        y = model.generate(ids, n, temperature=temperature, top_k=top_k, eos_id=eos, generator=g)
    return tok.decode([t for t in y[0].tolist() if t != eos])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ckpt.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("--ema", action="store_true", help="use the moving-average weights (train.py --ema)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tokenizer", default="", help="tokenizer file; default: the run's (args.json next to the checkpoint)")
    a = p.parse_args()
    torch.set_num_threads(4)
    model, step = load(a.ckpt, ema=a.ema, device=a.device)
    args_path = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "args.json")   # the run's data and tokenizer
    ra = json.load(open(args_path)) if os.path.exists(args_path) else {}
    tok = Tokenizer.from_file(a.tokenizer or os.path.join(HERE, ra.get("data", "data"), ra.get("tokenizer", "tok4096.json")))
    print(f"[loaded step {step}, {model.num_params()/1e6:.1f}M params, {model.num_neurons():,} LIF neurons]")
    if a.interactive:
        while True:
            try:
                pr = input("\nprompt> ")
            except EOFError:
                break
            print(pr + complete(model, tok, pr, a.n, a.temperature, a.top_k, a.seed))
        return
    for i in range(a.num):
        t = time.time()
        out = complete(model, tok, a.prompt, a.n, a.temperature, a.top_k,
                       None if a.seed is None else a.seed + i)
        print(f"--- sample {i+1} ({time.time()-t:.1f}s)\n{a.prompt}{out}\n")


if __name__ == "__main__":
    main()

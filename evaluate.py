"""Full-validation-set evaluation + sample generation + streaming-speed measurement.
The data directory, tokenizer and (for general text) window length come from the run's args.json next to the
checkpoint; TinyStories runs keep 256-token windows and the story prompts, as before."""
import argparse, json, math, os, time
import numpy as np
import torch
from tokenizers import Tokenizer
from sample import load, complete

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS_PER_TOKEN_VALID = None   # computed from the raw text below

PROMPTS = [
    "Once upon a time",
    "Lily wanted to go to the park, but",
    "The little dog was sad because",
    "Tom and his sister found a big box in the garden. Inside the box,",
    "One day, a little bird could not fly. It",
    "Sara had a red ball. She",
]
GENERAL_PROMPTS = [   # PG-19, FineWeb-Edu
    "The history of the city",
    "In mathematics, a prime number is",
    "Plants need sunlight because",
    "The French Revolution began in",
    "To make bread, you first",
    "Water boils at a lower temperature on a mountain because",
]


@torch.no_grad()
def full_val_loss(model, seq=256, batch=32, data="data"):
    """Non-overlapping windows over the whole validation stream; each window starts from rest (V=0)."""
    dev = next(model.parameters()).device
    arr = np.memmap(os.path.join(HERE, data, "valid.bin"), dtype=np.uint16, mode="r")
    n = (len(arr) - 1) // seq
    tot, cnt = 0.0, 0
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n)) * seq
        x = torch.from_numpy(np.stack([arr[j:j + seq] for j in idx]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([arr[j + 1:j + seq + 1] for j in idx]).astype(np.int64)).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            _, loss, _, _ = model(x, y)
        tot += loss.item() * y.numel(); cnt += y.numel()
    return tot / cnt, cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ckpt.pt")
    p.add_argument("--out", default="runs/main/final_eval.json")
    p.add_argument("--samples_per_prompt", type=int, default=3)
    p.add_argument("--ema", action="store_true", help="evaluate the moving-average weights (train.py --ema)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--wbits", type=int, default=-1, help="round the weight matrices to this many bits (model.py "
                   "quantize_weight; 0 = full precision); default: as the network was trained (train.py --wbits)")
    a = p.parse_args()
    torch.set_num_threads(4)
    model, step = load(a.ckpt, ema=a.ema, device=a.device, wbits=None if a.wbits < 0 else a.wbits)
    dev = torch.device(a.device)
    args_path = os.path.join(os.path.dirname(a.ckpt), "args.json")
    args = json.load(open(args_path)) if os.path.exists(args_path) else {}
    data = args.get("data", "data")
    tok = Tokenizer.from_file(os.path.join(HERE, data, args.get("tokenizer", "tok4096.json")))
    valid = np.memmap(os.path.join(HERE, data, "valid.bin"), dtype=np.uint16, mode="r")
    if data == "data":   # TinyStories: characters of the raw validation text
        raw = open(os.path.join(HERE, "data", "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
        stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
        n_chars = sum(len(s) for s in stories) + len(stories)   # +1 per story for the EOS token
        seq, prompts = 256, PROMPTS
    else:                # general text: characters of the decoded validation tokens, +1 per end-of-text token
        eot = tok.token_to_id("<|endoftext|>")
        n_chars = len(tok.decode(valid.tolist())) + int((valid == eot).sum())
        seq, prompts = args.get("seq", 1024), GENERAL_PROMPTS
    cpt = n_chars / len(valid)

    t = time.time()
    vl, cnt = full_val_loss(model, seq, max(1, 8192 // seq), data)
    wb = getattr(getattr(model, "cfg", None), "wbits", 0) if a.wbits < 0 else a.wbits
    res = {"step": step, "wbits": wb, "data": data, "seq": seq, "val_loss": vl, "val_ppl": math.exp(vl), "bits_per_char": vl / math.log(2) / cpt,
           "chars_per_token": cpt, "eval_tokens": cnt, "eval_seconds": time.time() - t}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)   # saved now as well, in case generation below is interrupted

    # streaming generation speed (batch 1, one simulation step per token)
    ids = torch.tensor([tok.encode("Once upon a time").ids], device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        y = model.generate(ids, 300, temperature=0.8, top_k=50, eos_id=None)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    res["gen_tokens_per_s"] = y.size(1) / (time.time() - t)
    res["device"] = a.device

    res["samples"] = {}
    for pr in prompts:
        res["samples"][pr] = [complete(model, tok, pr, 250, 0.8, 50, seed=s) for s in range(a.samples_per_prompt)]
        print(f"\n### {pr}")
        for s in res["samples"][pr]:
            print("  ->", (pr + s).replace("\n", " / "))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

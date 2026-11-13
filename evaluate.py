"""Full-validation-set evaluation + sample generation + streaming-speed measurement."""
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


@torch.no_grad()
def full_val_loss(model, seq=256, batch=32):
    """Non-overlapping windows over the whole validation stream; each window starts from rest (V=0)."""
    arr = np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r")
    n = (len(arr) - 1) // seq
    tot, cnt = 0.0, 0
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n)) * seq
        x = torch.from_numpy(np.stack([arr[j:j + seq] for j in idx]).astype(np.int64))
        y = torch.from_numpy(np.stack([arr[j + 1:j + seq + 1] for j in idx]).astype(np.int64))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss, _, _ = model(x, y)
        tot += loss.item() * y.numel(); cnt += y.numel()
    return tot / cnt, cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ckpt.pt")
    p.add_argument("--out", default="runs/main/final_eval.json")
    p.add_argument("--samples_per_prompt", type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, step = load(a.ckpt)
    tok = Tokenizer.from_file(os.path.join(HERE, "data", "tok4096.json"))
    raw = open(os.path.join(HERE, "data", "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
    n_chars = sum(len(s) for s in stories) + len(stories)   # +1 per story for the EOS token
    n_tok = len(np.memmap(os.path.join(HERE, "data", "valid.bin"), dtype=np.uint16, mode="r"))
    cpt = n_chars / n_tok

    t = time.time()
    vl, cnt = full_val_loss(model)
    res = {"step": step, "val_loss": vl, "val_ppl": math.exp(vl), "bits_per_char": vl / math.log(2) / cpt,
           "chars_per_token": cpt, "eval_tokens": cnt, "eval_seconds": time.time() - t}
    print(json.dumps(res, indent=1))

    # streaming generation speed (batch 1, one simulation step per token)
    ids = torch.tensor([tok.encode("Once upon a time").ids])
    t = time.time()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model.generate(ids, 300, temperature=0.8, top_k=50, eos_id=None)
    res["gen_tokens_per_s"] = y.size(1) / (time.time() - t)

    res["samples"] = {}
    for pr in PROMPTS:
        res["samples"][pr] = [complete(model, tok, pr, 250, 0.8, 50, seed=s) for s in range(a.samples_per_prompt)]
        print(f"\n### {pr}")
        for s in res["samples"][pr]:
            print("  ->", (pr + s).replace("\n", " / "))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

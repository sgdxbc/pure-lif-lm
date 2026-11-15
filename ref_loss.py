"""Validation loss of Hugging Face models that use GPT-2's tokenizer, on our FineWeb-Edu validation set, computed exactly
as evaluate.py does for our runs (non-overlapping 1024-token windows over the whole data/fineweb/valid.bin, each read
from scratch). Two uses: the classic small models' loss on this data is a direct target for stage goal 1, and the
numbers rank candidate teachers for distillation. A model is accepted only if its tokenizer gives every token of
data/fineweb/gpt2.json the same id. From the repository root (needs transformers):
  python3 ref_loss.py --device cuda openai-community/gpt2 openai-community/gpt2-xl ...   (writes runs/ref_loss.json)
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch_tokens", type=int, default=32768)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ours = json.load(open(os.path.join(HERE, "data", "fineweb", "gpt2.json")))["model"]["vocab"]
    arr = np.memmap(os.path.join(HERE, "data", "fineweb", "valid.bin"), dtype=np.uint16, mode="r")
    n = (len(arr) - 1) // a.seq
    out_path = os.path.join(HERE, "runs", "ref_loss.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    dev = torch.device(a.device)
    for name in a.models:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name)
        v = tok.get_vocab()
        bad = sum(1 for t, i in ours.items() if v.get(t) != i)
        if bad:
            print(f"{name}: tokenizer differs from GPT-2's on {bad} tokens; skipped", flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev).eval()
        params = sum(q.numel() for q in model.parameters())
        B = max(1, a.batch_tokens // a.seq)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, n, B):
                idx = np.arange(i, min(i + B, n)) * a.seq
                x = torch.from_numpy(np.stack([arr[j:j + a.seq] for j in idx]).astype(np.int64)).to(dev)
                y = torch.from_numpy(np.stack([arr[j + 1:j + a.seq + 1] for j in idx]).astype(np.int64)).to(dev)
                logits = model(x).logits.float()
                tot += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
                cnt += y.numel()
        res[name] = {"val_loss": tot / cnt, "params": params, "tokens": cnt, "seq": a.seq}
        print(f"{name:40s} {params / 1e6:7.0f}M  val_loss {tot / cnt:.4f}  ppl {math.exp(tot / cnt):7.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

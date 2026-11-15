"""Why LAMBADA is so hard for our models (README 5.29). lm-evaluation-harness's lambada_openai asks for the last word
of a passage of fiction; most of these words (names above all) already appear earlier in the passage, so the task
mostly measures whether a model can bring back a word it read tens of tokens before. For every passage the last word's
log-probability is computed in two conditions:
  full    after <|endoftext|> and the whole passage (what eval_lm.py scores)
  short   after only the last --short tokens of the passage (the spiking network starting from rest)
and the passages are split by whether the word appears earlier in the passage. On those where it does, short - full is
what the earlier text (the earlier occurrence above all) is worth; on the others it can only help through the general
sense of the passage (the control). Also reported: the mean loss on the passage's own tokens (how familiar this kind of
text is to the model), and with --check N, the first N passages scored again one at a time in float32 (the spiking
network also token by token, carrying its membrane potentials), to make sure batching and bfloat16 do not change the
scores. Run directories and Hugging Face ids (GPT-2's tokenizer) can be mixed. From the repository root:
  python3 probe_lambada.py --device cuda --ema --check 20 runs/g13_snn212 runs/g13_tf163 openai-community/gpt2
(writes runs/lambada_probe.json; needs the datasets package, installed with lm_eval)
"""
import argparse, json, math, os, re
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def passages(limit=0):
    """(context, target) as lm_eval's lambada_openai builds them: the text without its last word, and ' ' + that word."""
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    out = []
    for d in ds:
        words = d["text"].split(" ")
        out.append((" ".join(words[:-1]), " " + words[-1]))
        if limit and len(out) >= limit:
            break
    return out


def load_any(name, ema, dev):
    """(forward function idx -> logits, is_spiking, model, label) for a run directory or a Hugging Face id."""
    if os.path.isdir(name):
        from sample import load
        from model import SpikingLM
        args = json.load(open(os.path.join(name, "args.json")))
        use_ema = bool(ema and args.get("ema", 0))
        model, _ = load(os.path.join(name, "ckpt.pt"), ema=use_ema, device=dev)
        label = os.path.basename(name.rstrip("/")) + (" (EMA)" if use_ema else "")
        return (lambda x: model(x)[0]), isinstance(model, SpikingLM), model, label
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
    return (lambda x: model(x).logits), False, model, name.split("/")[-1]


@torch.no_grad()
def score(fwd, seqs, dev, eot, batch_tokens, bf16=True):
    """Per sequence: (negative log-probabilities of tokens 1.., whether each was the most likely one)."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    out, j = [None] * len(seqs), 0
    while j < len(order):
        idx = order[j:j + max(1, batch_tokens // len(seqs[order[j]]))]
        x = torch.full((len(idx), len(seqs[idx[0]])), eot, dtype=torch.long)
        for b, i in enumerate(idx):
            x[b, :len(seqs[i])] = torch.tensor(seqs[i])   # right padding: causal models, earlier positions unaffected
        x = x.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda"):
            logits = fwd(x[:, :-1])
        lp = F.log_softmax(logits.float(), -1)
        nll = -lp.gather(-1, x[:, 1:, None])[..., 0]
        top = lp.argmax(-1) == x[:, 1:]
        for b, i in enumerate(idx):
            n = len(seqs[i]) - 1
            out[i] = (nll[b, :n].cpu(), top[b, :n].cpu())
        j += len(idx)
    return out


@torch.no_grad()
def stepwise(model, seq, dev):
    """Spiking network, one token per call with the membrane potentials carried over, float32."""
    state, nll = None, []
    for t in range(len(seq) - 1):
        logits, _, state, _ = model(torch.tensor([[seq[t]]], device=dev), state=state)
        nll.append(-F.log_softmax(logits[0, -1].float(), -1)[seq[t + 1]].item())
    return torch.tensor(nll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="run directories and/or Hugging Face ids")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ema", action="store_true", help="moving-average weights where a run has them")
    p.add_argument("--short", type=int, default=16, help="tokens of the passage read in the short condition")
    p.add_argument("--limit", type=int, default=0, help="only the first N passages")
    p.add_argument("--check", type=int, default=0, help="score the first N passages again one at a time in float32")
    p.add_argument("--batch_tokens", type=int, default=8192)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "lambada_probe.json"))
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, "data", "fineweb", "gpt2.json"))
    eot = tok.token_to_id("<|endoftext|>")
    enc = lambda s: tok.encode(s).ids
    docs = passages(a.limit)
    full, short, cont_len, in_ctx = [], [], [], []
    for ctx, word in docs:
        c = [eot] + enc(ctx)
        whole = [eot] + enc(ctx + word)
        w = whole[len(c):] if whole[:len(c)] == c else enc(word)
        full.append(c + w)
        short.append(c[1:][-a.short:] + w)
        cont_len.append(len(w))
        in_ctx.append(re.search(r"(?<!\w)" + re.escape(word.strip()) + r"(?!\w)", ctx) is not None)
    N, share = len(docs), sum(in_ctx) / len(docs)
    print(f"{N} passages, {share:.1%} of the last words appear earlier in the passage; "
          f"{sum(map(len, full)) / N:.0f} tokens per passage on average", flush=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    dev = torch.device(a.device)
    for name in a.models:
        fwd, snn, model, label = load_any(name, a.ema, dev)
        sf, ss = score(fwd, full, dev, eot, a.batch_tokens), score(fwd, short, dev, eot, a.batch_tokens)
        tf = [sf[i][0][-k:].sum().item() for i, k in enumerate(cont_len)]     # target loss, full context
        ts = [ss[i][0][-k:].sum().item() for i, k in enumerate(cont_len)]     # target loss, short context
        hit = [bool(sf[i][1][-k:].all()) for i, k in enumerate(cont_len)]
        mean = lambda v, sel: sum(x for x, s in zip(v, sel) if s) / max(1, sum(sel))
        yes, no = in_ctx, [not s for s in in_ctx]
        allp = [True] * N
        passage = sum(sf[i][0][:-k].sum().item() for i, k in enumerate(cont_len)) / \
            sum(len(sf[i][0]) - k for i, k in enumerate(cont_len))
        r = {"passages": N, "in_context_share": share, "short": a.short,
             "acc": mean(hit, allp), "ppl": math.exp(mean(tf, allp)), "loss": mean(tf, allp),
             "acc_in": mean(hit, yes), "acc_out": mean(hit, no),
             "loss_in": mean(tf, yes), "loss_out": mean(tf, no),
             "loss_in_short": mean(ts, yes), "loss_out_short": mean(ts, no),
             "passage_loss": passage}
        r["gain_in"] = r["loss_in_short"] - r["loss_in"]
        r["gain_out"] = r["loss_out_short"] - r["loss_out"]
        r["net_gain"] = r["gain_in"] - r["gain_out"]
        if a.check:
            n = min(a.check, N)
            one = score(fwd, full[:n], dev, eot, 1, bf16=False)
            d1 = max(abs(one[i][0][-cont_len[i]:].sum().item() - tf[i]) for i in range(n))
            r["check"] = {"passages": n, "max_diff_single_fp32": d1}
            if snn:
                d2 = max(abs(stepwise(model, full[i], dev)[-cont_len[i]:].sum().item() - tf[i]) for i in range(n))
                r["check"]["max_diff_stepwise_fp32"] = d2
        res[label] = r
        print(f"{label:24s} acc {r['acc']:.3f} ppl {r['ppl']:9.1f} | word earlier in passage ({share:.0%}): "
              f"acc {r['acc_in']:.3f} loss {r['loss_in']:.2f} (short {r['loss_in_short']:.2f}, gain {r['gain_in']:+.2f}) "
              f"| not: acc {r['acc_out']:.3f} loss {r['loss_out']:.2f} (gain {r['gain_out']:+.2f}) "
              f"| net gain {r['net_gain']:+.2f} | passage loss {r['passage_loss']:.3f}"
              + (f" | check {r['check']}" if a.check else ""), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del model, fwd
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

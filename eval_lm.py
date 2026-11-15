"""Zero-shot benchmarks of classic small language models, run with EleutherAI's lm-evaluation-harness, so the numbers
are comparable with those published for GPT-2, Pythia and similar models (README section 0, stage goal 1).
From the repository root:
  python3 eval_lm.py --device cuda [--ema] runs/X              # our spiking network or Transformer -> runs/X/zeroshot[_ema].json
  python3 eval_lm.py --device cuda --hf EleutherAI/pythia-160m   # a reference model, with the harness's own Hugging Face
                                                                 # wrapper (needs transformers) -> runs/ref_pythia-160m/zeroshot.json
Default tasks: hellaswag, lambada_openai, piqa, arc_easy, arc_challenge, winogrande (all zero-shot; --limit N for a
quick check on N examples per task). Needs `pip install lm_eval`; the datasets are downloaded from Hugging Face.

How our models are scored: each (context, continuation) pair from the harness is tokenized as the harness does for
Hugging Face models (trailing spaces of the context move to the continuation), preceded by <|endoftext|>, which
begins every document in our training data. The spiking network reads it from rest; the sum of the log-probabilities
of the continuation's tokens is the score, and the pair counts as greedy when every continuation token is the most
likely one. Rolling log-likelihoods (perplexity tasks) are computed over the whole text: the spiking network carries
its membrane potentials from chunk to chunk, the Transformer uses disjoint windows of its training length.
Tasks that need free generation are not supported yet.
"""
import argparse, json, os
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = "hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande"


def make_lm(model, tok, dev, max_len, batch_tokens, prefix_eot=True):
    from lm_eval.api.model import LM
    from model import SpikingLM
    snn = isinstance(model, SpikingLM)
    eot = tok.token_to_id("<|endoftext|>")
    assert eot is not None, "the tokenizer has no <|endoftext|>"

    def enc(s):
        return tok.encode(s).ids

    @torch.no_grad()
    def logprobs(seqs):
        """Log-probabilities of tokens 1.. of each sequence given the earlier ones; sequences of any length."""
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), eot, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)   # right padding: earlier positions are unaffected (causal model)
        x = x.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(x[:, :-1])[0]
        return F.log_softmax(logits.float(), -1), x[:, 1:]

    class Wrapped(LM):
        def loglikelihood(self, requests):
            items = []
            for i, r in enumerate(requests):
                context, cont = r.args
                n = len(context) - len(context.rstrip())
                if n:
                    context, cont = context[:-n], context[-n:] + cont
                pre = [eot] if prefix_eot or not context else []
                ctx = pre + enc(context)
                whole = pre + enc(context + cont)
                cont_ids = whole[len(ctx):] if whole[:len(ctx)] == ctx else enc(cont)
                seq = (ctx + cont_ids)[-(max_len + 1):]
                items.append((i, seq, len(cont_ids)))
            items.sort(key=lambda t: -len(t[1]))
            out = [None] * len(requests)
            j = 0
            while j < len(items):
                n = max(1, batch_tokens // len(items[j][1]))
                chunk = items[j:j + n]
                lp, tgt = logprobs([s for _, s, _ in chunk])
                for b, (i, s, k) in enumerate(chunk):
                    T = len(s) - 1
                    l, t = lp[b, T - k:T], tgt[b, T - k:T]
                    out[i] = (l.gather(-1, t[:, None]).sum().item(), bool((l.argmax(-1) == t).all()))
                j += n
            return out

        def loglikelihood_rolling(self, requests):
            out = []
            for r in requests:
                ids = [eot] + enc(r.args[0])   # (the harness's Hugging Face wrapper also starts rolling windows with it)
                total, state = 0.0, None
                for a in range(0, len(ids) - 1, max_len):
                    if snn:      # carry the membrane potentials: the same as reading the text in one pass
                        x = torch.tensor([ids[a:a + max_len + 1]], device=dev)
                        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                            logits, _, state, _ = model(x[:, :-1], state=state)
                        total += F.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).sum().item()
                    else:        # disjoint windows of the training length
                        lp, tgt = logprobs([ids[a:a + max_len + 1]])
                        total += lp.gather(-1, tgt[..., None]).sum().item()
                out.append(total)
            return out

        def generate_until(self, requests):
            raise NotImplementedError("tasks with free generation are not supported yet")

    return Wrapped()


def summarise(results):
    """task -> {metric: value} for the usual metrics (acc, acc_norm, perplexity), with their standard errors."""
    out = {}
    for task, r in results["results"].items():
        out[task] = {k.split(",")[0] + ("_stderr" if "stderr" in k else ""): v for k, v in r.items()
                     if isinstance(v, (int, float)) and k.split(",")[0].replace("_stderr", "") in
                     ("acc", "acc_norm", "perplexity", "word_perplexity", "bits_per_byte")}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", help="run directory with ckpt.pt and args.json")
    p.add_argument("--hf", default="", help="instead of a run: a Hugging Face model id, scored by the harness itself")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true", help="moving-average weights where the checkpoint has them")
    p.add_argument("--tasks", default=TASKS)
    p.add_argument("--limit", type=int, default=0, help="only the first N examples of each task (a quick check)")
    p.add_argument("--batch_tokens", type=int, default=16384, help="tokens per forward pass")
    p.add_argument("--max_len", type=int, default=0, help="longest input (default: the run's training length)")
    a = p.parse_args()
    import lm_eval
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    kw = dict(tasks=tasks, num_fewshot=0, limit=a.limit or None)
    if a.hf:
        res = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={a.hf},dtype=bfloat16", device=a.device,
                                      batch_size=16, **kw)
        out_dir, name = os.path.join(HERE, "runs", "ref_" + a.hf.split("/")[-1]), "zeroshot"
    else:
        from tokenizers import Tokenizer
        from sample import load
        args = json.load(open(os.path.join(a.run, "args.json")))
        has_ema = bool(a.ema and args.get("ema", 0))
        dev = torch.device(a.device)
        model, _ = load(os.path.join(a.run, "ckpt.pt"), ema=has_ema, device=dev)
        tok = Tokenizer.from_file(os.path.join(HERE, args.get("data", "data"), args.get("tokenizer", "tok4096.json")))
        lm = make_lm(model, tok, dev, a.max_len or args.get("seq", 256), a.batch_tokens)
        res = lm_eval.simple_evaluate(model=lm, **kw)
        out_dir, name = a.run, "zeroshot" + ("_ema" if has_ema else "")
    os.makedirs(out_dir, exist_ok=True)
    summary = {"limit": a.limit, "tasks": summarise(res),
               "versions": {k: str(v) for k, v in res.get("versions", {}).items()}}
    path = os.path.join(out_dir, name + (f"_limit{a.limit}" if a.limit else "") + ".json")
    json.dump(summary, open(path, "w"), indent=1)
    for t, m in summary["tasks"].items():
        print(f"{t:16s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items() if not k.endswith("_stderr")), flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Pretrain the all-LIF spiking LM on TinyStories (CPU, bf16 matmuls via autocast)."""
import argparse, json, math, os, time
from dataclasses import asdict, fields

import numpy as np
import torch

from model import Config, SpikingLM

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args():
    p = argparse.ArgumentParser()
    for f in fields(Config):
        if f.type is bool or isinstance(f.default, bool):
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    p.add_argument("--arch", default="snn", choices=["snn", "tf"], help="tf = non-spiking Transformer baseline")
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample_every", type=int, default=0, help="0: sample at every eval")
    return p.parse_args()


class Data:
    def __init__(self, split, seq, seed):
        self.arr = np.memmap(os.path.join(HERE, "data", f"{split}.bin"), dtype=np.uint16, mode="r")
        self.seq = seq
        self.rng = np.random.default_rng(seed)

    def batch(self, B, rng=None):
        rng = rng or self.rng
        ix = rng.integers(0, len(self.arr) - self.seq - 1, size=B)
        x = np.stack([self.arr[i:i + self.seq + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, val, a):
    model.eval()
    rng = np.random.default_rng(1234)   # same eval batches every time
    tot, rates = 0.0, 0.0
    for _ in range(a.eval_batches):
        x, y = val.batch(a.batch, rng)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss, _, r = model(x, y)
        tot += loss.item()
        rates = rates + r.float()
    model.train()
    return tot / a.eval_batches, (rates / a.eval_batches).tolist()


def sample(model, tok, prompts, n=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    outs = []
    model.eval()
    for pr in prompts:
        ids = torch.tensor([tok.encode(pr).ids])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = model.generate(ids, n, temperature=0.8, top_k=50, eos_id=0, generator=g)
        outs.append(pr + "|" + tok.decode(y[0].tolist()))
    model.train()
    return outs


def main():
    a = get_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.arch == "tf":
        from model_tf import GPT, TFConfig
        cfg = TFConfig(vocab=a.vocab, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads)
        model = GPT(cfg)
    else:
        cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
        model = SpikingLM(cfg)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    step = 0
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed from step {step}")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, "data", "tok4096.json"))
    train, val = Data("train", a.seq, a.seed + step), Data("valid", a.seq, 0)
    print(f"params {model.num_params()/1e6:.2f}M, LIF neurons {model.num_neurons():,}, "
          f"train tokens {len(train.arr)/1e6:.0f}M, tokens/step {a.batch*a.seq:,}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    prompts = ["Once upon a time", "Lily wanted to go to the park, but", "The little dog was sad because"]

    t0 = time.time(); tl = time.time(); tokens_seen = step * a.batch * a.seq
    while step < a.steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train.batch(a.batch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss, _, rates = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step()
        step += 1
        tokens_seen += a.batch * a.seq
        if step % a.log_every == 0 or step == 1:
            dt = time.time() - tl; tl = time.time()
            tps = a.batch * a.seq * (a.log_every if step > 1 else 1) / dt
            rec = {"step": step, "loss": loss.item(), "lr": lr, "gnorm": gn, "tok_s": tps,
                   "tokens": tokens_seen, "elapsed_min": (time.time() - t0) / 60,
                   "rates": [round(r, 4) for r in rates.float().tolist()]}
            print(f"step {step:6d} loss {loss.item():.4f} lr {lr:.2e} gn {gn:.2f} {tps:,.0f} tok/s "
                  f"rate {rates.float().mean().item():.3f} [{' '.join(f'{r:.2f}' for r in rates.float().tolist())}]",
                  flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % a.eval_every == 0 or step == a.steps:
            vl, vr = evaluate(model, val, a)
            rec = {"step": step, "val_loss": vl, "val_rates": vr, "tokens": tokens_seen,
                   "elapsed_min": (time.time() - t0) / 60}
            print(f"== eval step {step} val_loss {vl:.4f}", flush=True)
            if a.sample_every == 0 or step % a.sample_every == 0 or step == a.steps:
                rec["samples"] = sample(model, tok, prompts, seed=step)
                for s_ in rec["samples"]:
                    print("   >> " + s_.replace("\n", " / "))
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "cfg": asdict(cfg), "arch": a.arch}, ckpt_path)
            tl = time.time()


if __name__ == "__main__":
    main()

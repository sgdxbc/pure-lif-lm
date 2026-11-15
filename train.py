"""Pretrain the all-LIF spiking LM on TinyStories (CPU or NVIDIA GPU, bf16 matmuls via autocast).

Optional training methods (all off by default; the architecture is unchanged by any of them):
  --teacher PATH --distill_weight a --distill_temp T
        knowledge distillation: loss = (1-a) * CE + a * T^2 * KL(teacher_T || student_T), where the teacher is a
        trained checkpoint (e.g. the Transformer baseline) run on the same batch. Validation loss is plain CE.
  --ema d
        keep an exponential moving average of the weights (decay d, e.g. 0.999); evaluations report both.
  --carry_state
        each batch row reads a continuous stretch of the training text, segment after segment, and the membrane
        potentials at the end of one segment are the starting state of the next (gradients stop at the segment
        boundary). Without it every segment starts from rest.
"""
import argparse, json, math, os, time
from dataclasses import asdict, fields

import numpy as np
import torch
import torch.nn.functional as F

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
    p.add_argument("--neuron_lr_mult", type=float, default=1.0,
                   help="learning-rate multiplier for the per-neuron dynamics parameters (beta, theta, gamma)")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample_every", type=int, default=0, help="0: sample at every eval")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:1, ... or auto")
    p.add_argument("--teacher", default="", help="checkpoint of a trained model to distill from")
    p.add_argument("--distill_weight", type=float, default=0.0)
    p.add_argument("--distill_temp", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights; 0 = off")
    p.add_argument("--carry_state", action="store_true",
                   help="continue each batch row through the text and carry the membrane potentials over")
    p.add_argument("--data", default="data", help="directory with train.bin / valid.bin (e.g. data/pg19)")
    p.add_argument("--tokenizer", default="tok4096.json", help="tokenizer file in --data")
    p.add_argument("--rope", action="store_true", help="tf: rotary position embedding instead of learned positions")
    a = p.parse_args()
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    return a


class Data:
    def __init__(self, split, seq, seed, data="data"):
        self.arr = np.memmap(os.path.join(HERE, data, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.seq = seq
        self.rng = np.random.default_rng(seed)
        self.pos = None   # stream positions (one per batch row) for --carry_state

    def batch(self, B, rng=None):
        rng = rng or self.rng
        ix = rng.integers(0, len(self.arr) - self.seq - 1, size=B)
        x = np.stack([self.arr[i:i + self.seq + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def stream_batch(self, B):
        """Next segment of B continuous streams. Returns x, y and a (B,) bool tensor: True where the stream
        continues from the previous segment (so its state should be kept), False where it (re)starts."""
        n = len(self.arr) - self.seq - 1
        if self.pos is None:
            self.pos = self.rng.integers(0, n, size=B)
            keep = np.zeros(B, dtype=bool)
        else:
            self.pos = self.pos + self.seq
            keep = self.pos < n
            self.pos[~keep] = self.rng.integers(0, n, size=int((~keep).sum()))
        x = torch.from_numpy(np.stack([self.arr[i:i + self.seq + 1] for i in self.pos]).astype(np.int64))
        return x[:, :-1].contiguous(), x[:, 1:].contiguous(), torch.from_numpy(keep)


def mask_state(state, keep):
    """Zero the carried state of the rows that start a new stream."""
    out = []
    for layer in state:
        out.append(tuple(t * keep.view(-1, *([1] * (t.dim() - 1))).to(t.dtype) for t in layer))
    return out


class EMA:
    """Exponential moving average of all floating-point weights (kept in float32)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def swap(self, model):
        """Exchange the model's weights with the averaged ones (call twice to restore)."""
        sd = model.state_dict()
        for k in self.shadow:
            cur = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k].to(sd[k].dtype))
            self.shadow[k] = cur.float()


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
        x, y = x.to(a.device), y.to(a.device)
        with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16):
            _, loss, _, r = model(x, y)
        tot += loss.item()
        rates = rates + r.float()
    model.train()
    return tot / a.eval_batches, (rates / a.eval_batches).tolist()


def sample(model, tok, prompts, n=120, seed=0, device="cpu"):
    eos = tok.token_to_id("<|endoftext|>")
    g = torch.Generator(device=device).manual_seed(seed)
    outs = []
    model.eval()
    for pr in prompts:
        ids = torch.tensor([tok.encode(pr).ids], device=device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            y = model.generate(ids, n, temperature=0.8, top_k=50, eos_id=eos, generator=g)
        outs.append(pr + "|" + tok.decode(y[0].tolist()))
    model.train()
    return outs


def main():
    a = get_args()
    dev = torch.device(a.device)
    if dev.type == "cpu":
        torch.set_num_threads(a.threads)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.arch == "tf":
        from model_tf import GPT, TFConfig
        cfg = TFConfig(vocab=a.vocab, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
                       max_len=max(1024, a.seq), rope=a.rope)
        model = GPT(cfg)
    else:
        cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
        model = SpikingLM(cfg)
    model.to(dev)
    neuron_keys = ("beta_logit", "log_theta", "gamma_logit")
    is_neuron = lambda n: a.neuron_lr_mult != 1.0 and n.endswith(neuron_keys)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad and not is_neuron(n)]
    groups = [{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if a.neuron_lr_mult != 1.0:   # separate group only when used, so older checkpoints still resume
        neuron = [p for n, p in model.named_parameters() if p.requires_grad and is_neuron(n)]
        groups.append({"params": neuron, "weight_decay": 0.0, "lr_mult": a.neuron_lr_mult})
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
    step = 0
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed from step {step}")
    ema = None
    if a.ema > 0:
        ema = EMA(model, a.ema)
        if a.resume and os.path.exists(ckpt_path) and "ema" in ck:
            ema.shadow = {k: v.to(dev) for k, v in ck["ema"].items()}
    teacher = None
    if a.teacher and a.distill_weight > 0:
        from sample import load
        teacher, _ = load(a.teacher)
        teacher.to(dev).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print(f"distilling from {a.teacher} (weight {a.distill_weight}, temperature {a.distill_temp})")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, a.data, a.tokenizer))
    assert tok.get_vocab_size() <= a.vocab, f"--vocab {a.vocab} is smaller than the tokenizer's {tok.get_vocab_size()}"
    train, val = Data("train", a.seq, a.seed + step, a.data), Data("valid", a.seq, 0, a.data)
    print(f"params {model.num_params()/1e6:.2f}M, LIF neurons {model.num_neurons():,}, "
          f"train tokens {len(train.arr)/1e6:.0f}M, tokens/step {a.batch*a.seq:,}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    prompts = ["Once upon a time", "Lily wanted to go to the park, but", "The little dog was sad because"]
    if a.data != "data":   # general text (PG-19, FineWeb-Edu)
        prompts = ["The history of the city", "In mathematics, a prime number is", "Plants need sunlight because"]

    t0 = time.time(); tl = time.time(); tokens_seen = step * a.batch * a.seq
    amp = lambda: torch.autocast(dev.type, dtype=torch.bfloat16)
    state = None   # carried membrane potentials (--carry_state)
    while step < a.steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        if a.carry_state:
            x, y, keep = train.stream_batch(a.batch)
            x, y, keep = x.to(dev), y.to(dev), keep.to(dev)
            if state is not None:
                state = mask_state(state, keep)
        else:
            x, y = train.batch(a.batch)
            x, y = x.to(dev), y.to(dev)
        with amp():
            logits, ce, new_state, rates = model(x, y, state=state)
        if a.carry_state:
            state = new_state
        loss, kd = ce, None
        if teacher is not None:
            with torch.no_grad(), amp():
                t_logits = teacher(x)[0]
            T = a.distill_temp
            V = logits.size(-1)
            kd = F.kl_div(F.log_softmax(logits.float().view(-1, V) / T, -1),
                          F.log_softmax(t_logits.float().view(-1, V) / T, -1),
                          log_target=True, reduction="batchmean") * T * T
            loss = (1 - a.distill_weight) * ce + a.distill_weight * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step()
        if ema is not None:
            ema.update(model, step)
        step += 1
        tokens_seen += a.batch * a.seq
        if step % a.log_every == 0 or step == 1:
            dt = time.time() - tl; tl = time.time()
            tps = a.batch * a.seq * (a.log_every if step > 1 else 1) / dt
            rec = {"step": step, "loss": ce.item(), "lr": lr, "gnorm": gn, "tok_s": tps,
                   "tokens": tokens_seen, "elapsed_min": (time.time() - t0) / 60,
                   "rates": [round(r, 4) for r in rates.float().tolist()]}
            if kd is not None:
                rec["distill_kl"] = kd.item()
            mem = ""
            if dev.type == "cuda":
                rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
                mem = f" mem {rec['gpu_mem_gb']:.1f}GB"
            print(f"step {step:6d} loss {ce.item():.4f}" + (f" kl {kd.item():.4f}" if kd is not None else "") +
                  f" lr {lr:.2e} gn {gn:.2f} {tps:,.0f} tok/s{mem} "
                  f"rate {rates.float().mean().item():.3f} [{' '.join(f'{r:.2f}' for r in rates.float().tolist())}]",
                  flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % a.eval_every == 0 or step == a.steps:
            vl, vr = evaluate(model, val, a)
            rec = {"step": step, "val_loss": vl, "val_rates": vr, "tokens": tokens_seen,
                   "elapsed_min": (time.time() - t0) / 60}
            msg = f"== eval step {step} val_loss {vl:.4f}"
            if ema is not None:
                ema.swap(model); rec["val_loss_ema"] = evaluate(model, val, a)[0]; ema.swap(model)
                msg += f" (EMA weights {rec['val_loss_ema']:.4f})"
            print(msg, flush=True)
            if a.sample_every == 0 or step % a.sample_every == 0 or step == a.steps:
                rec["samples"] = sample(model, tok, prompts, seed=step, device=a.device)
                for s_ in rec["samples"]:
                    print("   >> " + s_.replace("\n", " / "))
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                  "cfg": asdict(cfg), "arch": a.arch}
            if ema is not None:
                ck["ema"] = ema.shadow
            torch.save(ck, ckpt_path)
            tl = time.time()


if __name__ == "__main__":
    main()

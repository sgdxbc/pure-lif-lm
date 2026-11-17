"""Pretrain the spiking LM (or, with --arch tf, the Transformer baseline) on TinyStories, PG-19 or FineWeb-Edu
(--data, see docs/running.md), on CPU or an NVIDIA GPU, bf16 matmuls via autocast.

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
  --optim muon
        Muon (Jordan et al. 2024: momentum, then the update matrix is replaced by the nearest semi-orthogonal matrix,
        computed by a Newton-Schulz iteration) for the 2-D weights inside the layers (W_in, W_out, the leak and delay
        synapses; in the Transformer the attention and MLP matrices); AdamW, as before, for the input and readout
        synapses (embedding, head, positions), biases and the per-neuron parameters. The Muon update is scaled by
        0.2 * sqrt(max(rows, cols)) (Liu et al. 2025, "Muon is scalable"), so the AdamW learning rate carries over.
        --muon_skip leak_ keeps the matrices whose names contain "leak_" (the leak-modulating synapses) on AdamW.
  --teacher hf:NAME
        a Hugging Face model as the teacher (needs transformers), e.g. hf:openai-community/gpt2-large on data
        tokenized with GPT-2's tokenizer; its logits are padded to the student's vocabulary.
  --hint_weight w [--hint_layers K]
        hidden-state distillation (with a teacher; spiking student): K pairs of layers at matching relative depth
        (default: one per teacher layer). For each pair, a linear map, used only in training, predicts the teacher's
        residual stream after its layer (each channel standardised over the batch) from the student's bus after its
        layer (scaled to unit RMS per token); loss += w * mean squared error, averaged over the pairs. The aim is a
        denser training signal about what the teacher has gathered from the context at every position. The maps are
        not part of the model and cost nothing at inference.
"""
import argparse, json, math, os, time
from dataclasses import asdict, fields

import numpy as np
import torch
import torch.nn.functional as F

from model import Config, SpikingLM, round_weights_, restore_weights_

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
    p.add_argument("--accum", type=int, default=1, help="split each batch into this many micro-batches (less memory, "
                   "same result)")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                   help="cosine: from lr down to min_lr_frac * lr over all steps after the warm-up. wsd (warm-up, "
                        "stable, decay): lr until the last --decay_steps steps, then down to min_lr_frac * lr as "
                        "1 - sqrt(progress); a run with more --steps is the same run until its decay starts")
    p.add_argument("--decay_steps", type=int, default=0, help="wsd: the length of the final decay")
    p.add_argument("--keep_ckpt", default="", help="comma-separated steps (multiples of --eval_every) at which to "
                   "keep a copy of the checkpoint, ckpt_step<N>.pt (to branch decays off a wsd run)")
    p.add_argument("--class_file", default="", help="with --out_classes: .npy of the class of every word "
                   "(experiments/word_classes.py); default: by word number")
    p.add_argument("--in_class_file", default="", help="with --in_groups 2: .npy of a class for every word; a word "
                   "then fires its class's neuron and the neuron of its position in the class")
    p.add_argument("--init_ckpt", default="", help="start from this checkpoint (weights, optimizer, step, EMA) when "
                   "the run directory has none yet: a decay branch of a wsd run")
    p.add_argument("--wbits_from", type=int, default=0, help="with --wbits: train with the weight matrices rounded "
                   "from this step on (full precision before it); evaluations always use the rounded network")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--neuron_lr_mult", type=float, default=1.0,
                   help="learning-rate multiplier for the per-neuron dynamics parameters (beta, theta, gamma)")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--spike_log", type=float, default=5.0,
                   help="log the largest parameter gradients (and the batch offsets) of any step whose gradient norm "
                        "is above this many times the median of the last 200 steps; 0 = off. Only logs.")
    p.add_argument("--spike_skip", type=float, default=0.0,
                   help="skip the update (optimizer and EMA) of any step whose gradient norm is above this many times "
                        "the median of the last 200 steps (after 50 steps); 0 = off")
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
    p.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_skip", default="", help="with --optim muon: comma-separated name parts of 2-D weights "
                   "kept on AdamW, e.g. leak_ (the leak-modulating synapses)")
    p.add_argument("--hint_weight", type=float, default=0.0, help="hidden-state distillation weight (needs --teacher)")
    p.add_argument("--hint_layers", type=int, default=0, help="layer pairs for --hint_weight (0: one per teacher layer)")
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
        self.last_ix = ix   # for train.py's --spike_log records
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


def newton_schulz(G, steps=5):
    """Approximately the nearest semi-orthogonal matrix to G (U V^T of its SVD), by the quintic Newton-Schulz
    iteration of Muon; computed in bfloat16 on the GPU."""
    a_, b_, c_ = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a_ * X + (b_ * A + c_ * A @ A) @ X
    return (X.T if tall else X).float()


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weights: Nesterov momentum, orthogonalised update, decoupled weight decay."""

    def __init__(self, params, lr, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p, dtype=torch.float32)
                grad = p.grad.float()
                buf = st["buf"].mul_(g["momentum"]).add_(grad)
                u = newton_schulz(grad.add(buf, alpha=g["momentum"]), g["ns_steps"])
                u.mul_(0.2 * math.sqrt(max(p.size(0), p.size(1))))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-g["lr"])


class Optimizers:
    """Several optimizers stepped together (Muon for some weights, AdamW for the rest), seen as one."""

    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s_ in zip(self.opts, sd["optimizers"]):
            o.load_state_dict(s_)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if a.schedule == "wsd":
        start = a.steps - a.decay_steps
        if step < start:
            return a.lr
        prog = min(1.0, (step - start) / max(1, a.decay_steps))
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * (1 - math.sqrt(prog)))
    prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, val, a, rounded=True):
    """Validation loss and firing rates; with --wbits, of the network with its weight matrices rounded unless
    rounded=False."""
    saved = round_weights_(model, a.wbits, a.wgroup) if a.wbits and rounded else None
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
    if saved is not None:
        restore_weights_(model, saved)
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
    assert a.batch % a.accum == 0 and (a.accum == 1 or not a.carry_state), "--accum must divide --batch; not with --carry_state"
    assert not a.wbits or a.arch == "snn", "--wbits is for the spiking network"
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
        if a.class_file:   # (a checkpoint loaded below brings back its own classes)
            assert cfg.out_classes > 0, "--class_file needs --out_classes"
            model.head.set_classes(torch.from_numpy(np.load(a.class_file)).long())
        if a.in_class_file:
            assert cfg.in_groups == 2, "--in_class_file needs --in_groups 2"
            model.embed.set_classes(torch.from_numpy(np.load(a.in_class_file)).long())
    model.to(dev)
    teacher = None
    if a.teacher and a.distill_weight > 0:
        if a.teacher.startswith("hf:"):   # a Hugging Face model, e.g. hf:openai-community/gpt2-large
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(a.teacher[3:], torch_dtype=torch.bfloat16)

            class HFTeacher(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x, hidden=False):   # (logits, None, residual streams after each layer)
                    o = self.m(x, output_hidden_states=hidden)
                    return o.logits, None, list(o.hidden_states[1:]) if hidden else None
            teacher = HFTeacher(hf)
        else:
            from sample import load
            teacher, _ = load(a.teacher)
        teacher.to(dev).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print(f"distilling from {a.teacher} (weight {a.distill_weight}, temperature {a.distill_temp})")
    hint, hint_pairs = None, []
    if a.hint_weight > 0:
        assert teacher is not None and a.arch == "snn", "--hint_weight needs a teacher and a spiking student"
        with torch.no_grad():
            hs = teacher(torch.zeros(1, 2, dtype=torch.long, device=dev), hidden=True)[2]
        K, Ls = a.hint_layers or len(hs), a.n_layers
        hint_pairs = [(round(k * Ls / K) - 1, round(k * len(hs) / K) - 1) for k in range(1, K + 1)]
        hint = torch.nn.ModuleList([torch.nn.Linear(a.d_model, hs[0].size(-1), bias=False) for _ in hint_pairs]).to(dev)
        for m_ in hint:
            torch.nn.init.zeros_(m_.weight)
        print(f"hidden-state distillation (weight {a.hint_weight}): (student layer, teacher layer) {hint_pairs}")
    neuron_keys = ("beta_logit", "log_theta", "gamma_logit")
    is_neuron = lambda n: a.neuron_lr_mult != 1.0 and n.endswith(neuron_keys)
    # Muon: the 2-D weights inside the layers; the input/readout synapses (embedding, head, positions) stay on AdamW
    skip = [k for k in a.muon_skip.split(",") if k]
    is_muon = lambda n, p: (a.optim == "muon" and p.dim() == 2 and not n.split(".")[0] in ("embed", "head", "pos")
                            and not any(k in n for k in skip))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for n, p in named if p.dim() >= 2 and not is_muon(n, p)]
    no_decay = [p for n, p in named if p.dim() < 2 and not is_neuron(n)]
    groups = [{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if a.neuron_lr_mult != 1.0:   # separate group only when used, so older checkpoints still resume
        neuron = [p for n, p in named if is_neuron(n)]
        groups.append({"params": neuron, "weight_decay": 0.0, "lr_mult": a.neuron_lr_mult})
    if hint is not None:   # the training-only maps of --hint_weight
        groups.append({"params": list(hint.parameters()), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
    if a.optim == "muon":
        muon = [p for n, p in named if is_muon(n, p)]
        opt = Optimizers([opt, Muon(muon, lr=a.lr, momentum=a.muon_momentum, weight_decay=a.wd)])
        print(f"Muon for {len(muon)} matrices ({sum(p.numel() for p in muon) / 1e6:.1f}M weights), AdamW for the rest")
    step = 0
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    keep_steps = {int(x) for x in a.keep_ckpt.split(",") if x}
    assert all(k % a.eval_every == 0 for k in keep_steps), "--keep_ckpt steps must be multiples of --eval_every"
    assert a.schedule != "wsd" or 0 < a.decay_steps <= a.steps - a.warmup, "wsd needs 0 < --decay_steps <= steps - warmup"
    init_path = ckpt_path if a.resume and os.path.exists(ckpt_path) else a.init_ckpt
    if init_path:
        ck = torch.load(init_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        if hint is not None:
            hint.load_state_dict(ck["hint"])
        print(f"resumed from step {step}" + (f" of {init_path}" if init_path != ckpt_path else ""))
    ema = None
    if a.ema > 0:
        ema = EMA(model, a.ema)
        if init_path and "ema" in ck:
            ema.shadow = {k: v.to(dev) for k, v in ck["ema"].items()}
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
    gn_hist = []   # gradient norms of the last 200 steps (--spike_log)
    amp = lambda: torch.autocast(dev.type, dtype=torch.bfloat16)
    state = None   # carried membrane potentials (--carry_state)
    if teacher is not None and step == 0:   # the teacher's own loss on the validation batches, for reference
        rng, tot = np.random.default_rng(1234), 0.0
        for _ in range(a.eval_batches):
            xv, yv = val.batch(a.batch, rng)
            with torch.no_grad(), amp():
                tv = teacher(xv.to(dev))[0]
            tot += F.cross_entropy(tv.float().flatten(0, 1), yv.to(dev).flatten()).item()
        print(f"teacher val_loss {tot / a.eval_batches:.4f}", flush=True)
        logf.write(json.dumps({"step": 0, "teacher_val_loss": tot / a.eval_batches}) + "\n"); logf.flush()
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
        opt.zero_grad(set_to_none=True)
        logged = []
        # --wbits: forward and backward with the weight matrices rounded; the update goes to the full-precision values
        saved = round_weights_(model, a.wbits, a.wgroup) if a.wbits and step >= a.wbits_from else None
        for xm, ym in zip(x.chunk(a.accum), y.chunk(a.accum)):   # --accum: micro-batches, gradients summed
            model.keep_bus = hint is not None
            with amp():
                logits, ce, new_state, rates = model(xm, ym, state=state)
            model.keep_bus = False
            if a.carry_state:
                state = new_state
            loss, kd, hl = ce, None, None
            if teacher is not None:
                with torch.no_grad(), amp():
                    t_out = teacher(xm, hidden=True) if hint is not None else teacher(xm)
                    t_logits = t_out[0]
                T = a.distill_temp
                V = logits.size(-1)
                if t_logits.size(-1) < V:   # e.g. GPT-2's 50257 against the padded 50304: the rest get no probability
                    t_logits = F.pad(t_logits.float(), (0, V - t_logits.size(-1)), value=-1e4)
                kd = F.kl_div(F.log_softmax(logits.float().view(-1, V) / T, -1),
                              F.log_softmax(t_logits.float().view(-1, V) / T, -1),
                              log_target=True, reduction="batchmean") * T * T
                loss = (1 - a.distill_weight) * ce + a.distill_weight * kd
            if hint is not None:   # hidden-state distillation (see the top of this file)
                hl = 0.0
                for (si, ti), proj in zip(hint_pairs, hint):
                    h = t_out[2][ti].float()
                    z = (h - h.mean((0, 1))) / (h.std((0, 1)) + 1e-4)
                    b = model.last_bus[si].float()
                    b = b * torch.rsqrt(b.pow(2).mean(-1, keepdim=True) + 1e-6)
                    hl = hl + F.mse_loss(proj(b), z)
                hl = hl / len(hint_pairs)
                model.last_bus = None
                loss = loss + a.hint_weight * hl
            (loss / a.accum).backward()
            logged.append((ce.detach(), None if kd is None else kd.detach(), hl, rates))
            logits = loss = t_out = t_logits = None   # free this micro-batch's outputs before the next one
        if saved is not None:
            restore_weights_(model, saved); saved = None
        ce = torch.stack([v[0] for v in logged]).mean()
        kd = None if kd is None else torch.stack([v[1] for v in logged]).mean()
        rates = torch.stack([v[3] for v in logged]).mean(0)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        med = float(np.median(gn_hist)) if len(gn_hist) >= 50 else None
        spike = med is not None and a.spike_log > 0 and gn > a.spike_log * med
        skip_step = med is not None and a.spike_skip > 0 and gn > a.spike_skip * med   # --spike_skip
        if spike:
            # a gradient spike (docs/experiments.md 5.34): which parameters, which batch. The gradients are already
            # clipped, so scale back to their size before clipping. Changes nothing in training.
            named_g = [(n, q.grad) for n, q in model.named_parameters() if q.grad is not None]
            norms = (torch.stack([g_.float().norm() for _, g_ in named_g]) * max(1.0, gn / a.clip)).tolist()
            top = sorted(zip([n for n, _ in named_g], norms), key=lambda t: -t[1])[:8]
            ix = getattr(train, "last_ix", None)
            srec = {"step": step + 1, "spike_gn": gn, "median_gn": float(np.median(gn_hist)), "spike_loss": ce.item(),
                    "top": [[n, round(v, 4)] for n, v in top],
                    "ix": None if (ix is None or a.carry_state) else [int(i) for i in ix], "skipped": skip_step}
            print(f"gradient spike at step {step + 1}: gn {gn:.2f} (median {srec['median_gn']:.2f}), loss {ce.item():.4f}; "
                  + ", ".join(f"{n} {v:.2f}" for n, v in top[:4]) + (" -- update skipped" if skip_step else ""), flush=True)
            logf.write(json.dumps(srec) + "\n"); logf.flush()
        gn_hist = (gn_hist + [gn])[-200:]
        if skip_step:   # --spike_skip: no update at all from this batch (the step still counts for the schedule and data)
            if not spike:
                print(f"update skipped at step {step + 1}: gn {gn:.2f} (median {med:.2f})", flush=True)
        else:
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
            if hl is not None:
                rec["hint_mse"] = hl.item()
            mem = ""
            if dev.type == "cuda":
                rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
                mem = f" mem {rec['gpu_mem_gb']:.1f}GB"
            print(f"step {step:6d} loss {ce.item():.4f}" + (f" kl {kd.item():.4f}" if kd is not None else "") +
                  (f" hint {hl.item():.4f}" if hl is not None else "") +
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
                ema.swap(model); rec["val_loss_ema"] = evaluate(model, val, a)[0]
                if a.wbits:   # the same weights unrounded, to see what the rounding costs
                    rec["val_loss_ema_full"] = evaluate(model, val, a, rounded=False)[0]
                ema.swap(model)
                msg += f" (EMA weights {rec['val_loss_ema']:.4f})"
                if a.wbits:
                    msg += f" [{a.wbits}-bit weights; unrounded EMA {rec['val_loss_ema_full']:.4f}]"
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
            if hint is not None:
                ck["hint"] = hint.state_dict()
            torch.save(ck, ckpt_path)
            if step in keep_steps:
                torch.save(ck, os.path.join(a.out, f"ckpt_step{step}.pt"))
            tl = time.time()


if __name__ == "__main__":
    main()

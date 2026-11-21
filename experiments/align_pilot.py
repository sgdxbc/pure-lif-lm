"""Three-arm 5.63 pilot for source-address distillation into g44_all3."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fpga_est
from evaluate import full_val_loss
from experiments.head_screen import lambada_ids
from experiments.memory_maps import (
    Case, LAGS, case_batches, measure_cases, natural_cases, natural_induction,
    paired_bootstrap, synthetic_cases, traced_forward,
)
from hw_cost import firing, snn_cost
from sample import load

STUDENT = ((9, 3), (10, 5), (11, 2), (11, 3))
TEACHER = ((10, 5), (11, 1), (11, 7), (11, 0))
ARMS = ("plain", "sham", "aligned")
SEED = 56321
STEPS, BATCH, MICRO, SEQ = 1000, 32, 8, 1024
LR, WD = 6e-5, .1
PSEUDOCOUNT = .001
SOURCE_RUN = ROOT / "runs/g44_all3"


class SelectedRows(nn.Module):
    def __init__(self, weight, ids):
        super().__init__()
        self.register_buffer("ids", torch.tensor(ids, device=weight.device))
        self.rows = nn.Parameter(weight.detach().index_select(0, self.ids))

    def forward(self, frozen):
        return frozen.index_copy(0, self.ids, self.rows)


def train_rows(model, heads):
    model.requires_grad_(False)
    projections = []
    for layer in sorted({layer for layer, _ in heads}):
        m = model.mems[str(layer)]
        width, nheads = m.cfg.mem_keys, m.cfg.mem_heads
        selected = sorted(h for l, h in heads if l == layer)
        ids = [k for qk in range(2) for head in selected
               for k in range((qk * nheads + head) * width, (qk * nheads + head + 1) * width)]
        for name in ("weight", "bias"):
            parametrize.register_parametrization(m.kq_in, name, SelectedRows(getattr(m.kq_in, name), ids))
        projections.append(m.kq_in)
    return [p for p in model.parameters() if p.requires_grad], projections


def eligible_queries(windows, eot, seed, cap=8):
    """Return (stream, query, unique earlier source) within the same document."""
    rng = np.random.default_rng(seed)
    rows = []
    for stream, tokens in enumerate(windows):
        last = {}
        choices = []
        for q, (current, target) in enumerate(zip(tokens[:-1], tokens[1:])):
            if current == eot:
                last.clear()
            if target != eot and target != current and last.get(target, -1) >= 0:
                choices.append((stream, q, last[target]))
            if current != eot:
                last[current] = q if current not in last else -1
        if len(choices) > cap:
            choices = [choices[i] for i in sorted(rng.choice(len(choices), cap, replace=False))]
        rows.extend(choices)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def make_offsets(token_count, seq=SEQ, steps=STEPS, batch=BATCH):
    return np.random.default_rng(SEED).integers(0, token_count - seq, size=(steps + 1, batch)).tolist()


def microbatches(data, offsets, seq=SEQ, micro=MICRO):
    for start in range(0, len(offsets), micro):
        yield np.stack([data[o:o + seq + 1] for o in offsets[start:start + micro]]).astype(np.int64)


def capture(model, x, heads, logit_slice=None):
    out, _, trace = traced_forward(model, x, query_slice=logit_slice,
                                  trace_layers={layer for layer, _ in heads})
    return out, trace


def group_distribution(traces, heads, stream_query, teacher, return_silence=False):
    streams, queries = stream_query.unbind(-1)
    width = next(iter(traces.values())).k.size(2)
    causal = torch.arange(width, device=queries.device)[None] <= queries[:, None]
    total = 0.
    silent = 0.
    for layer, head in heads:
        tr = traces[layer]
        q = tr.q[streams, head, queries].unsqueeze(1)
        k = tr.k[streams, head]
        match = (q @ k.transpose(-1, -2)).squeeze(1)
        if teacher:
            p = match.masked_fill(~causal, -torch.inf).softmax(-1)
        else:
            decay = tr.log_decay.expand(*tr.q.shape[:3])[streams, head].float().cumsum(-1)
            survival = (decay.gather(1, queries[:, None]) - decay).masked_fill(~causal, -torch.inf).exp()
            coeff = match * survival
            if return_silence:
                silent = silent + coeff.sum(-1).eq(0).float().mean()
            smooth = coeff + causal * (PSEUDOCOUNT / (queries[:, None] + 1))
            p = smooth / smooth.sum(-1, keepdim=True)
        total = total + p
    p = total / len(heads)
    return (p, silent / len(heads)) if return_silence else p


def address_kl(student, target):
    """Mean row KL(target || student), including zero-mass future positions."""
    term = torch.special.xlogy(target, target) - target * student.clamp_min(1e-30).log()
    return term.sum(-1).mean()


def sham_targets(target, queries, seed):
    """Independently shuffle probability values within each causal lag band."""
    rng = np.random.default_rng(seed)
    order = np.tile(np.arange(target.size(1)), (len(target), 1))
    for row, q in enumerate(queries.detach().cpu().tolist()):
        for lo, hi in zip(LAGS[:-1], LAGS[1:]):
            start, stop = max(0, q - hi + 1), q - lo + 1
            if stop > start:
                order[row, start:stop] = rng.permutation(np.arange(start, stop))
    return target.gather(1, torch.as_tensor(order, device=target.device))


def losses(student, teacher, windows, queries, dev, auxiliary, seed):
    t = torch.as_tensor(windows, device=dev)
    x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
    align = auxiliary and len(queries) > 0
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        pred, trace = capture(student, x, STUDENT) if align else (student(x)[0], None)
        with torch.no_grad():
            logits, ref = capture(teacher, x, TEACHER) if align else (teacher(x)[0], None)
    student_lp = pred.float().log_softmax(-1)
    target_lp = logits.float().log_softmax(-1)
    kd = F.kl_div(student_lp.flatten(0, 1), target_lp.flatten(0, 1), reduction="batchmean", log_target=True)
    ce = F.nll_loss(student_lp.flatten(0, 1), y.flatten())
    task = .1 * ce + .9 * kd
    if not align:
        return task, None, None, None
    q = torch.as_tensor(queries, device=dev)
    p = group_distribution(trace, STUDENT, q[:, :2], False)
    target = group_distribution(ref, TEACHER, q[:, :2], True)
    sham = sham_targets(target, q[:, 1], seed)
    mass = [z.gather(1, q[:, 2:]).mean().detach() for z in (target, sham)]
    return task, address_kl(p, target), address_kl(p, sham), mass


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_args():
    return json.loads((SOURCE_RUN / "args.json").read_text())


def setup_data(args):
    data = ROOT / args["data"]
    tok = Tokenizer.from_file(str(data / args["tokenizer"]))
    arr = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    return data, tok, arr, tok.token_to_id("<|endoftext|>")


def prepare(path):
    args = model_args()
    data, tok, arr, eot = setup_data(args)
    diagnostic = json.loads((ROOT / "runs/memory_maps/summary.json").read_text())
    exclude = set(diagnostic["dev_passages"] + diagnostic["held_out_passages"])
    natural = [c for c in natural_cases(tok, SEQ, 0) if c.label not in exclude]
    rng = np.random.default_rng(SEED + 1)
    rng.shuffle(natural)
    if not natural:
        raise ValueError("no eligible fresh LAMBADA passages")
    valid = np.memmap(data / "valid.bin", dtype=np.uint16, mode="r")
    n = (len(valid) - 1) // SEQ
    old = np.random.default_rng(diagnostic["seed"]).permutation(n) * SEQ
    ranges = diagnostic["causal_pairs_per_lag_bin"][0] // SEQ
    old_natural = set(c.label for c in itertools.islice(natural_induction(valid, old[ranges:], SEQ, eot), 64))
    candidates = natural_induction(valid, rng.permutation(n) * SEQ, SEQ, eot)
    fresh_natural = list(itertools.islice((c for c in candidates if c.label not in old_natural), 64))
    repeated = list(synthetic_cases(tok.get_vocab_size(), SEQ, 64, rng, eot))
    if len(fresh_natural) != 64:
        raise ValueError("fewer than 64 fresh natural copy cases")
    write_json(path, {"lambada_eligible": len(natural), "lambada": [asdict(c) for c in natural[:256]],
                      "natural": [asdict(c) for c in fresh_natural], "repeated": [asdict(c) for c in repeated],
                      "train_offsets": make_offsets(len(arr))})
    print(f"Saved {min(len(natural), 256)} of {len(natural)} eligible fresh LAMBADA passages to {path}")


def kq_rates(model, arr, dev):
    starts = np.linspace(0, len(arr) - SEQ - 1, 32).astype(np.int64)
    count = torch.zeros(len(STUDENT), 2, model.cfg.mem_keys, device=dev)
    active = torch.zeros_like(count)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for windows in microbatches(arr, starts):
            t = torch.as_tensor(windows[:, :-1], device=dev)
            _, traces = capture(model, t, STUDENT, logit_slice=[-1])
            for i, (layer, h) in enumerate(STUDENT):
                kq = torch.stack([traces[layer].k[:, h], traces[layer].q[:, h]])
                count[i] += kq.sum((1, 2))
                active[i] += kq.ne(0).sum((1, 2))
    return {"heads": STUDENT, "order": ["key", "query"], "positions": len(starts) * SEQ,
            "mean_firing_per_threshold_neuron_by_row": (count / (len(starts) * SEQ * model.cfg.mem_kq_levels)).tolist(),
            "nonzero_count_fraction_by_row": (active / (len(starts) * SEQ)).tolist()}


def training_step(student, teacher, params, arr, plan, step, arm, dev, opt, weight, eot):
    # The first effective batch is calibration only. The other offsets and
    # query permutations are chosen by step and microbatch, independent of arm.
    aux = arm != "plain"
    grad_sums = [[torch.zeros_like(p) for p in params] for _ in range(3)] if opt is None else None
    acc = np.zeros(5)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    for i, windows in enumerate(microbatches(arr, plan["train_offsets"][step])):
        seed = SEED + step * (BATCH // MICRO) + i
        q = eligible_queries(windows, eot, seed)
        task, aligned, sham, mass = losses(student, teacher, windows, q, dev, aux, seed + 2 ** 20)
        if grad_sums is not None:
            for j, loss in enumerate((task, aligned, sham)):
                if loss is not None:
                    grads = torch.autograd.grad(loss, params, retain_graph=j < 2 and aligned is not None)
                    for total, grad in zip(grad_sums[j], grads):
                        total.add_(grad.detach(), alpha=MICRO / BATCH)
        else:
            extra = aligned if arm == "aligned" else sham if arm == "sham" else None
            total = task if extra is None else task + weight * extra
            (total * (MICRO / BATCH)).backward()
        acc[0] += float(task.detach())
        if mass is not None:
            acc[1:] += [float(aligned.detach()), float(sham.detach()), *map(float, mass)]
    if opt is not None:
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
    row = dict(zip(("task", "aligned_kl", "sham_kl", "teacher_source_mass", "sham_source_mass"),
                   (acc / (BATCH // MICRO)).tolist()))
    if grad_sums is not None:
        row["grad_norms"] = [float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
                             for grads in grad_sums]
    return row


@torch.no_grad()
def manipulation(student, teacher, cases, dev, eot):
    mass, target_mass, divergences, silent = [], [], [], []
    for _, block in case_batches(cases, MICRO):
        width = max(len(c.ids) for c in block)
        x = torch.tensor([c.ids + [eot] * (width - len(c.ids)) for c in block], device=dev)
        q = torch.tensor([(i, c.query) for i, c in enumerate(block)], device=dev)
        source = torch.tensor([[c.source] for c in block], device=dev)
        _, a = capture(student, x, STUDENT, logit_slice=[-1])
        _, b = capture(teacher, x, TEACHER, logit_slice=[-1])
        p, fraction = group_distribution(a, STUDENT, q, False, return_silence=True)
        target = group_distribution(b, TEACHER, q, True)
        mass.extend(p.gather(1, source).flatten().tolist())
        target_mass.extend(target.gather(1, source).flatten().tolist())
        divergences.append((len(block), float(address_kl(p, target))))
        silent.append(len(block) * float(fraction))
    return {"source_mass": float(np.mean(mass)), "teacher_source_mass": float(np.mean(target_mass)),
            "kl_to_teacher": sum(n * kl for n, kl in divergences) / len(cases),
            "silent_head_row_fraction": sum(silent) / len(cases)}


@torch.no_grad()
def lambada_accuracy(student, args, dev, eot):
    from probe_lambada import score

    examples = list(lambada_ids(args, 0))
    values = score(lambda x: student(x)[0], [tokens for tokens, _ in examples], dev, eot, batch_tokens=4096)
    return sum(bool(correct[-n:].all()) for (_, n), (_, correct) in zip(examples, values)) / len(examples)


def assess(student, teacher, plan, args, data, dev, eot):
    def score_cases(cases, **kw):
        return measure_cases(student, cases, MICRO, dev, eot, **kw)
    fresh = [Case(**c) for c in plan["lambada"]]
    manip = manipulation(student, teacher, fresh, dev, eot)
    clean, _ = score_cases(fresh)
    redirect, _ = score_cases(fresh, selected=STUDENT, arm="source")
    sham = np.mean([score_cases(fresh, selected=STUDENT, arm=arm)[0] for arm in ("sham0", "sham1")], axis=0)
    copies = {}
    for name in ("natural", "repeated"):
        cases = [Case(**c) for c in plan[name]]
        intact, _ = score_cases(cases)
        replaced, _ = score_cases(cases, arm="replace")
        copies[name] = {"intact": float(intact.mean()), "replaced": float(replaced.mean()),
                        "replacement_drop": paired_bootstrap(intact - replaced, seed=SEED)}
    val, tokens = full_val_loss(student, SEQ, MICRO, str(data))
    activity = firing(student, 64, dev=dev, data=str(data))
    rows = fpga_est.estimate(snn_cost(student, activity), args, 4,
                            dict(fpga_est.DEVICES["u55c"], cards=1, eff=.6))
    return {"manipulation": manip, "lambada_ids": [c.label for c in fresh],
            "unassisted_logp": clean.tolist(), "redirect_logp_mean": float(redirect.mean()),
            "redirect_headroom": paired_bootstrap(redirect - clean, seed=SEED),
            "redirect_vs_sham": paired_bootstrap(redirect - sham, seed=SEED), "copy": copies,
            "validation_nll": val, "validation_tokens": tokens,
            "full_lambada_accuracy": lambada_accuracy(student, args, dev, eot),
            "u55c_mJ_per_token": {str(r["streams"]): r["mJ_per_token"] for r in rows if r["streams"] in (1, 16)}}


def run(stage, folder, dev, arm):
    args = model_args()
    data, _, arr, eot = setup_data(args)
    plan_path = folder / "plan.json"
    plan = json.loads(plan_path.read_text())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    student, checkpoint = load(SOURCE_RUN / "ckpt.pt", ema=True, device=dev)
    teacher, _ = load(ROOT / "runs/g13_tf163/ckpt.pt", ema=False, device=dev)
    teacher.requires_grad_(False)
    params, layers = train_rows(student, STUDENT)
    path = folder / "calibration.json"
    rate_data = np.memmap(data / "valid.bin", dtype=np.uint16, mode="r")
    if stage == "calibrate":
        row = training_step(student, teacher, params, arr, plan, 0, "aligned", dev, None, 0., eot)
        task, real, sham = row["grad_norms"]
        if not (task > 0 and real > 0 and np.isfinite(row["grad_norms"]).all()):
            raise ValueError("the calibration batch has a zero or nonfinite task or alignment gradient")
        weight = .1 * task / real
        row.update(weight=weight, calibrated_grad_ratio={"aligned": weight * real / task,
                                                         "sham": weight * sham / task, "plain": 0.},
                   student_step=checkpoint, plan_sha256=digest,
                   aligned_target_has_more_source_mass=row["teacher_source_mass"] > row["sham_source_mass"],
                   rates_before=kq_rates(student, rate_data, dev))
        write_json(path, row)
        print({k: v for k, v in row.items() if k != "rates_before"}, flush=True)
        return
    calibration = json.loads(path.read_text())
    if calibration["plan_sha256"] != digest or checkpoint != calibration["student_step"]:
        raise ValueError("batch plan or checkpoint differs from the calibration")
    weight = 0. if arm == "plain" else calibration["weight"]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    out = folder / arm
    out.mkdir()
    write_json(out / "args.json", args)
    with (out / "training.jsonl").open("w") as log:
        for step in range(1, STEPS + 1):
            row = training_step(student, teacher, params, arr, plan, step, arm, dev, opt, weight, eot)
            log.write(json.dumps(dict(step=step, **row), allow_nan=False) + "\n")
            if step % 50 == 0:
                log.flush()
                print(arm, step, row, flush=True)
    for projection in layers:
        for name in ("weight", "bias"):
            parametrize.remove_parametrizations(projection, name, leave_parametrized=True)
    torch.save({"model": student.state_dict(), "cfg": asdict(student.cfg), "arch": "snn",
                "step": checkpoint + STEPS}, out / "ckpt.pt")
    rates = kq_rates(student, rate_data, dev)
    # Match the original diagnostic's FP32 attention/retrieval passes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = assess(student, teacher, plan, args, data, dev, eot)
    report.update(arm=arm, aux_weight=weight, calibration_grad_ratio=calibration["calibrated_grad_ratio"][arm],
                  rates_before=calibration["rates_before"], rates_after=rates, plan_sha256=digest,
                  peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None)
    write_json(out / "metrics.json", report)
    print(arm, "finished", flush=True)


def summarize(folder):
    rows = {arm: json.loads((folder / arm / "metrics.json").read_text()) for arm in ARMS}
    plain, aligned = (rows[n] for n in ("plain", "aligned"))
    if any(r["lambada_ids"] != plain["lambada_ids"] or r["plan_sha256"] != plain["plan_sha256"] for r in rows.values()):
        raise ValueError("arms do not share the same plan and LAMBADA passages")
    diagnostic = {arm: r["manipulation"] for arm, r in rows.items()}
    rose = diagnostic["aligned"]["source_mass"] > diagnostic["plain"]["source_mass"]
    print("Source mass rose over plain" if rose else "Source mass did not rise over plain", flush=True)
    print(json.dumps(diagnostic, indent=2), flush=True)
    gain = {n: paired_bootstrap(np.asarray(aligned["unassisted_logp"]) - rows[n]["unassisted_logp"], seed=SEED)
            for n in ("plain", "sham")}
    nll = aligned["validation_nll"] - plain["validation_nll"]
    acc = aligned["full_lambada_accuracy"] - plain["full_lambada_accuracy"]
    cost = {s: aligned["u55c_mJ_per_token"][s] / plain["u55c_mJ_per_token"][s] - 1 for s in ("1", "16")}
    advance = (all(g["mean"] >= .20 and g["ci95"][0] > 0 for g in gain.values())
               and nll <= .012 and acc >= -.02 and all(abs(c) <= .01 for c in cost.values()))
    result = {"source_mass_rose": rose, "manipulation": diagnostic, "fresh_lambada_n": len(plain["lambada_ids"]),
              "aligned_logprob_gain": gain, "validation_nll_delta": nll,
              "full_lambada_accuracy_delta": acc, "cost_fraction_delta": cost, "advance": advance}
    write_json(folder / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "calibrate", "train", "summarize"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--out", type=Path, default=Path("runs/align_pilot"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        if (args.out / "plan.json").exists():
            p.error("use a fresh output directory to prepare a plan")
        prepare(args.out / "plan.json")
    elif args.stage == "summarize":
        summarize(args.out)
    else:
        if args.stage == "train" and args.arm is None:
            p.error("train requires --arm")
        run(args.stage, args.out, torch.device(args.device), args.arm)


if __name__ == "__main__":
    main()

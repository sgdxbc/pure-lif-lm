"""Section 5.63 frozen-head conversion. Run from the repository root.

  python3 experiments/head_screen.py --device cuda --out runs/head_screen
  python3 experiments/head_screen.py --device cuda --out runs/head_screen --nonlinear

The default runs cache, fit and report, also available individually via --stage.
The optional nonlinear fit reuses the cache and starts from the primary linear
fit. It is allowed only after all four primary fits fail the validation NLL gate.

Validation activations are streamed through the EMA body and scored in FP32.
Fit, calibration and LAMBADA caches, including word rows and class log-marginals,
use about 1.4 GB across all four models. Shortlist logits are recomputed. Quality
uses the original EMA weights. Modeled cost uses W4 and exact unit-temperature
sampling, including a word block per draw, as in fpga_est.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fpga_est
from experiments.fpga_compare import rates_of
from model import ClassReadout
from probe_lambada import passages
from sample import load


RUNS = ("g45_kd25", "g46_kd25s1", "g48_gla31", "g48_gla50")
PARTITIONS = {"linear": "tf163_head225.npy", "shortlist": "tf163_head225_s8192.npy"}
STREAMS = (1, 16, 256)


class FrozenHead(nn.Module):
    def __init__(self, weight, partition, classes, nonlinear=False):
        super().__init__()
        vocab, width = weight.shape
        short = int((partition < 0).sum())
        layout = ClassReadout(vocab, width, classes, 0, short).to(weight.device)
        layout.set_classes(partition)
        self.short = short
        self.register_buffer("groups", layout.word_at_slot[short:].reshape(classes, -1))
        self.register_buffer("short_words", layout.word_at_slot[:short])
        group = layout.slot.clone()
        tail = group >= short
        group[tail] = (group[tail] - short) // self.groups.size(1) + short
        self.register_buffer("group", group)
        self.register_buffer("short_weights", weight[self.short_words])
        self.predictor = nn.Linear(width, classes, device=weight.device)
        nn.init.zeros_(self.predictor.bias)
        self.correction = (nn.Sequential(nn.Linear(width, 256, bias=False), nn.ReLU(),
                                         nn.Linear(256, classes, bias=False)).to(weight.device)
                           if nonlinear else None)
        if nonlinear:
            nn.init.zeros_(self.correction[-1].weight)

    def flat_top(self, flat):
        """Return exact top logits, with one logsumexp per class."""
        grouped = flat[:, self.groups.clamp_min(0)].masked_fill(self.groups < 0, -torch.inf)
        mass = grouped.logsumexp(-1)
        return torch.cat((flat[:, self.short_words], mass), -1)

    def top(self, features, short_logits=None):
        scores = self.predictor(features)
        if self.correction is not None:
            scores = scores + self.correction(features)
        if short_logits is None:
            short_logits = F.linear(features, self.short_weights)
        return torch.cat((short_logits, scores), -1)

    def log_probs(self, flat, flat_top, top):
        """Full exact distribution; each word keeps its original within-class score."""
        return flat + (top.log_softmax(-1) - flat_top)[:, self.group]

    def target_log_probs(self, flat, features, words):
        # The exact likelihood needs only the target's conditional, unlike global argmax.
        top = self.top(features, flat[:, self.short_words]).log_softmax(-1)
        group = self.group[words]
        members = self.groups[(group - self.short).clamp_min(0)]
        conditional = (flat.gather(1, words[:, None]).squeeze(1)
                       - flat.gather(1, members.clamp_min(0)).masked_fill(members < 0, -torch.inf).logsumexp(-1))
        return top.gather(1, group[:, None]).squeeze(1) + conditional.masked_fill(group < self.short, 0.)

    def kl(self, features, class_masses):
        short = F.linear(features, self.short_weights)
        target = torch.cat((short, class_masses), -1).log_softmax(-1)
        fitted = self.top(features, short).log_softmax(-1)
        return F.kl_div(fitted, target, log_target=True, reduction="batchmean")


def run_args(run):
    return json.loads((ROOT / "runs" / run / "args.json").read_text())


def make_head(weight, variant):
    partition = np.load(ROOT / "runs" / "classes" / PARTITIONS["shortlist" if variant == "shortlist" else "linear"])
    return FrozenHead(weight, torch.as_tensor(partition, device=weight.device), 225, variant == "nonlinear")


def window_batches(data, starts, seq, batch):
    for i in range(0, len(starts), batch):
        yield np.stack([data[j:j + seq + 1] for j in starts[i:i + batch]])


def lambada_ids(args, limit):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / args["data"] / args["tokenizer"]))
    eot = tok.token_to_id("<|endoftext|>")
    for context, answer in passages(limit):
        ctx = tok.encode(context).ids
        whole = tok.encode(context + answer).ids
        cont = whole[len(ctx):] if whole[:len(ctx)] == ctx else tok.encode(answer).ids
        yield ([eot] + ctx + cont)[-args["seq"] - 1:], len(cont)


@torch.no_grad()
def cache(run, args, config, folder, dev):
    (folder / "cache.json").unlink(missing_ok=True)
    model, step = load(ROOT / "runs" / run / "ckpt.pt", ema=True, device=dev)
    width = model.cfg.d_model
    weight = model.head.weight.detach()
    heads = [make_head(weight, variant) for variant in PARTITIONS]
    torch.save(weight.cpu(), folder / "word_rows.pt")
    data = ROOT / args["data"]
    seq = args["seq"]
    train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    valid = np.memmap(data / "valid.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(config.seed)
    # Use disjoint contiguous windows and a single random draw for both training splits.
    n = config.fit_windows + config.cal_windows
    starts = rng.choice((len(train) - 1) // seq, n, replace=False) * seq
    valid_starts = np.arange((len(valid) - 1) // seq) * seq
    if config.valid_windows:
        valid_starts = valid_starts[:config.valid_windows]
    manifest = {"step": step, "seq": seq, "seed": config.seed, "fit_starts": starts[:config.fit_windows].tolist(),
                "cal_starts": starts[config.fit_windows:].tolist(), "valid_windows": len(valid_starts),
                "lambada_limit": config.lambada_limit, "body_batch": config.body_batch}
    for split, offsets in (("fit", starts[:config.fit_windows]), ("cal", starts[config.fit_windows:])):
        count = len(offsets) * seq
        x_cache = np.lib.format.open_memmap(folder / f"{split}_x.npy", "w+", np.float32, (count, width))
        y_cache = np.lib.format.open_memmap(folder / f"{split}_y.npy", "w+", np.uint16, (count,))
        mass_cache = np.lib.format.open_memmap(folder / f"{split}_masses.npy", "w+", np.float32, (count, 2, 225))
        # Retain the events requested by 5.63 for the later direct-event rejection test.
        capture = run == RUNS[0]
        records = []
        handle = (model.blocks[-1].fc2.register_forward_pre_hook(
            lambda _, inp: records.append(np.packbits(inp[0].flatten(0, 1).detach().ne(0).cpu().numpy(), axis=-1)))
                  if capture else nullcontext())
        if capture:
            hidden = model.blocks[-1].fc2.in_features
            event_cache = np.lib.format.open_memmap(folder / f"{split}_events.npy", "w+", np.uint8,
                                                   (count, (hidden + 7) // 8))
            manifest["event_channels"] = hidden
        position = 0
        with handle:
            for ids in window_batches(train, offsets, seq, config.body_batch):
                x = torch.as_tensor(ids[:, :-1].astype(np.int64), device=dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    features = model._features(x)[0].flatten(0, 1)
                end = position + len(features)
                x_cache[position:end] = features.float().cpu().numpy()
                y_cache[position:end] = ids[:, 1:].reshape(-1)
                if capture:
                    event_cache[position:end] = records.pop()
                for start in range(0, len(features), config.projection_batch):
                    z = F.linear(features[start:start + config.projection_batch].float(), weight)
                    for i, head in enumerate(heads):
                        mass_cache[position + start:position + start + len(z), i] = head.flat_top(z)[:, -225:].cpu()
                position = end

    xs, ys, lengths = [], [], []
    for ids, size in lambada_ids(args, config.lambada_limit):
        tokens = torch.tensor([ids[:-1]], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            features = model._features(tokens)[0][0, -size:]
        xs.append(features.float().cpu().numpy())
        ys.extend(ids[-size:])
        lengths.append(size)
    np.save(folder / "lambada_x.npy", np.concatenate(xs))
    np.save(folder / "lambada_y.npy", np.array(ys, dtype=np.uint16))
    np.save(folder / "lambada_lengths.npy", lengths)
    write_json(folder / "cache.json", manifest)


def arrays(folder, split, masses=False):
    keys = ("x", "masses") if masses else ("x", "y")
    return tuple(np.load(folder / f"{split}_{key}.npy", mmap_mode="r") for key in keys)


def tensor(array, dev):
    return torch.as_tensor(np.array(array), device=dev)


def fit(head, folder, variant, config, dev):
    train_x, train_mass = arrays(folder, "fit", masses=True)
    cal_x, cal_mass = arrays(folder, "cal", masses=True)
    column = int(variant == "shortlist")
    rng = np.random.default_rng(config.seed)
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr)
    best = math.inf
    calibration_history = []
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(train_x))
        for ix in np.array_split(order, math.ceil(len(order) / config.fit_batch)):
            opt.zero_grad(set_to_none=True)
            head.kl(tensor(train_x[ix], dev), tensor(train_mass[ix, column], dev)).backward()
            opt.step()
        with torch.no_grad():
            total = 0.
            for i in range(0, len(cal_x), config.fit_batch):
                ix = slice(i, i + config.fit_batch)
                x = tensor(cal_x[ix], dev)
                total += head.kl(x, tensor(cal_mass[ix, column], dev)).item() * len(x)
            mean = total / len(cal_x)
        calibration_history.append(mean)
        if mean < best:
            best = mean
            result = {"epoch": epoch, "calibration_kl": best,
                      "budget": {"epochs": config.epochs, "lr": config.lr, "batch": config.fit_batch, "seed": config.seed},
                      "predictor": {k: t.cpu().clone() for k, t in head.predictor.state_dict().items()}}
            if head.correction is not None:
                result["correction"] = {k: t.cpu().clone() for k, t in head.correction.state_dict().items()}
    result.update(calibration_kl_by_epoch=calibration_history, best_epoch_is_last=result["epoch"] == config.epochs)
    torch.save(result, folder / f"{variant}.pt")


def fitted_head(weight, folder, variant):
    head = make_head(weight, variant)
    state = torch.load(folder / f"{variant}.pt", map_location="cpu", weights_only=True)
    head.predictor.load_state_dict(state.pop("predictor"))
    if "correction" in state:
        head.correction.load_state_dict(state.pop("correction"))
    return head, state


@torch.no_grad()
def score_validation(model, weight, heads, data_path, manifest, projection_batch, dev):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    seq = manifest["seq"]
    offsets = np.arange(manifest["valid_windows"]) * seq
    nll = {"flat": 0., **dict.fromkeys(heads, 0.)}
    for ids in window_batches(data, offsets, seq, manifest["body_batch"]):
        tokens = tensor(ids[:, :-1].astype(np.int64), dev)
        targets = tensor(ids[:, 1:].reshape(-1).astype(np.int64), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            features = model._features(tokens)[0].flatten(0, 1)
        for x, y in zip(features.split(projection_batch), targets.split(projection_batch)):
            x = x.float()
            flat = F.linear(x, weight)
            exact = flat.gather(1, y[:, None]).squeeze(1) - flat.logsumexp(-1)
            nll["flat"] -= exact.sum(dtype=torch.float64).item()
            for name, head in heads.items():
                nll[name] -= head.target_log_probs(flat, x, y).sum(dtype=torch.float64).item()
    count = len(offsets) * seq
    if not count:
        raise ValueError("validation requires at least one complete window")
    return {name: loss / count for name, loss in nll.items()}, count


@torch.no_grad()
def score_lambada(weight, heads, folder, batch, dev):
    xs, ys = arrays(folder, "lambada")
    correct = {name: [] for name in ("flat", *heads)}
    for start in range(0, len(xs), batch):
        x = tensor(xs[start:start + batch], dev)
        y = tensor(ys[start:start + batch].astype(np.int64), dev)
        flat = F.linear(x, weight)
        correct["flat"].append((flat.argmax(-1) == y).cpu().numpy())
        for name, head in heads.items():
            top = head.flat_top(flat)
            logp = head.log_probs(flat, top, head.top(x, top[:, :head.short]))
            correct[name].append((logp.argmax(-1) == y).cpu().numpy())
    counts = np.load(folder / "lambada_lengths.npy")
    starts = np.r_[0, counts.cumsum()[:-1]]
    return {name: float(np.logical_and.reduceat(np.concatenate(values), starts).mean())
            for name, values in correct.items()}


def cost(args, rates, width, variant):
    head = None if variant == "flat" else (8192 if variant == "shortlist" else 0, 225)
    hw = fpga_est.tf_hw(args, rates, readout=head, bits=4)
    if variant == "nonlinear":
        weights = 256 * (width + 225)
        for key in ("weight_bytes_onchip", "dense_bytes"):
            hw[key] += weights / 2
        hw["dense_macs"] += weights + 256 + 225  # ReLU and residual add each charged one MAC-equivalent
    dev = dict(fpga_est.DEVICES["u55c"], cards=1, eff=0.6, add_cost=1)
    return {str(row["streams"]): row for row in fpga_est.estimate_tf(hw, dev) if row["streams"] in STREAMS}


def within(value, threshold):
    return value <= threshold + 1e-12


def report(run, args, folder, variants, config, dev):
    weight = torch.load(folder / "word_rows.pt", map_location=dev, weights_only=True)
    loaded = {name: fitted_head(weight, folder, name) for name in variants}
    heads = {name: entry[0] for name, entry in loaded.items()}
    manifest = json.loads((folder / "cache.json").read_text())
    run_dir = ROOT / "runs" / run
    model, step = load(run_dir / "ckpt.pt", ema=True, device=dev)
    if step != manifest["step"] or not torch.equal(model.head.weight, weight):
        raise ValueError(f"{run} checkpoint changed since the fit cache was generated")
    valid, count = score_validation(model, weight, heads, ROOT / args["data"] / "valid.bin",
                                    manifest, config.projection_batch, dev)
    del model
    lambada = score_lambada(weight, heads, folder, config.projection_batch, dev)
    logged = json.loads((run_dir / "final_eval_ema.json").read_text())
    rates = rates_of(run)
    if args.get("relay") == "quant" and rates is None:
        raise ValueError(f"{run} is missing measured firing rates")
    path = folder.parent / f"{run}.json"
    result = json.loads(path.read_text()) if path.exists() and variants == ("nonlinear",) else {}
    result.update(cache=manifest, class_partitions=PARTITIONS, validation_tokens=count,
                  lambada_passages=len(np.load(folder / "lambada_lengths.npy")))
    for name in valid:
        result[name] = dict(valid_nll=valid[name], lambada=lambada[name],
                            cost=cost(args, rates, weight.size(1), name))
        if name in loaded:
            result[name]["fit"] = loaded[name][1]
            if result[name]["fit"]["best_epoch_is_last"]:
                print(f"{run} {name}: lowest calibration KL was at the last epoch", flush=True)
    flat = result["flat"]
    flat.update(logged_ema_val_loss=logged["val_loss"], logged_ema_eval_tokens=logged.get("eval_tokens"),
                pipeline_minus_logged_nll=flat["valid_nll"] - logged["val_loss"])
    for name in variants:
        row = result[name]
        nll = row["valid_nll"] - flat["valid_nll"]
        acc = flat["lambada"] - row["lambada"]
        savings = {str(s): 1 - energy(row, s) / energy(flat, s) for s in STREAMS}
        row.update(excess_nll=nll, lambada_drop_pp=100 * acc, modeled_savings=savings,
                   passes_nll=within(nll, .012),
                   passes=within(nll, .012) and within(acc, .02)
                   and all(within(.15, savings[str(s)]) for s in (1, 16)))
    write_json(path, result)


def energy(row, stream):
    return row["cost"][str(stream)]["mJ_per_token"]


def frontier_at(seeds, dense, stream):
    low, high = sorted(dense, key=lambda row: energy(row, stream))
    x0, x1 = energy(low, stream), energy(high, stream)
    differences = []
    for seed in seeds:
        cost = energy(seed, stream)
        if x0 <= cost <= x1 and x0 < x1:
            pos = math.log(cost / x0) / math.log(x1 / x0)
            estimate = low["valid_nll"] + pos * (high["valid_nll"] - low["valid_nll"])
            differences.append(estimate - seed["valid_nll"])
        else:
            differences.append(None)
    dominance = [[i for i, point in enumerate(dense) if energy(point, stream) < energy(seed, stream)
                  and point["valid_nll"] <= seed["valid_nll"] and point["lambada"] >= seed["lambada"]] for seed in seeds]
    result = {"D_per_seed": differences, "D_mean": None, "band": "outside_dense_bracket", "credit": False,
              "dominating_dense_indices_per_seed": dominance}
    if None not in differences:
        mean = sum(differences) / len(differences)
        band = ("favors_relay" if not within(mean, .020) else "favors_dense" if not within(-mean, .020)
                else "parity" if abs(mean) < .012 - 1e-12 else "undecided")
        result.update(D_mean=mean, band=band, credit=band == "favors_relay"
                      and all(d > 0 and seed["passes"] for d, seed in zip(differences, seeds)))
    return result


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=ROOT / "runs/head_screen")
    p.add_argument("--stage", choices=("all", "cache", "fit", "report"), default="all")
    p.add_argument("--nonlinear", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit-windows", type=int, default=64)
    p.add_argument("--cal-windows", type=int, default=8)
    p.add_argument("--body-batch", type=int, default=4)
    p.add_argument("--projection-batch", type=int, default=512)
    p.add_argument("--fit-batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--valid-windows", type=int, default=0, help="positive limit for smoke tests")
    p.add_argument("--lambada-limit", type=int, default=0, help="positive limit for smoke tests")
    config = p.parse_args()
    if min(config.fit_windows, config.cal_windows, config.body_batch, config.projection_batch,
           config.fit_batch, config.epochs, config.lr) <= 0:
        p.error("fit, calibration and batch budgets must be positive")
    if min(config.valid_windows, config.lambada_limit) < 0:
        p.error("evaluation limits must be nonnegative")
    info = {run: run_args(run) for run in RUNS}
    if len({(args["data"], args["tokenizer"], args["seq"]) for args in info.values()}) != 1:
        p.error("the four runs must share data, tokenizer and sequence length")
    torch.set_num_threads(4)
    dev = torch.device(config.device)
    variants = tuple(PARTITIONS)
    if config.nonlinear:
        if config.stage == "cache":
            p.error("the nonlinear fit reuses the linear cache")
        for name in RUNS:
            path = config.out / f"{name}.json"
            if not path.exists():
                p.error("report all four linear fits before using --nonlinear")
            row = json.loads(path.read_text())
            if row["linear"]["passes_nll"] or row["cache"]["lambada_limit"] or config.valid_windows:
                p.error("--nonlinear requires all four primary linear fits to fail full-validation NLL")
            args = info[name]
            data = ROOT / args["data"] / "valid.bin"
            current = json.loads((config.out / name / "cache.json").read_text())
            if row["cache"] != current or current["valid_windows"] != (data.stat().st_size // 2 - 1) // args["seq"]:
                p.error("--nonlinear requires full validation")
        variants = ("nonlinear",)
    for run in RUNS:
        print(f"{config.stage}: {run} ({', '.join(variants)})", flush=True)
        args = info[run]
        folder = config.out / run
        folder.mkdir(parents=True, exist_ok=True)
        if not config.nonlinear and config.stage in ("all", "cache"):
            cache(run, args, config, folder, dev)
        if config.stage in ("all", "fit"):
            weight = torch.load(folder / "word_rows.pt", map_location=dev, weights_only=True)
            for variant in variants:
                torch.manual_seed(config.seed)
                head = make_head(weight, variant)
                if variant == "nonlinear":
                    linear, _ = fitted_head(weight, folder, "linear")
                    head.predictor.load_state_dict(linear.predictor.state_dict())
                fit(head, folder, variant, config, dev)
        if config.stage in ("all", "report"):
            report(run, args, folder, variants, config, dev)
    if config.stage in ("all", "report"):
        rows = {run: json.loads((config.out / f"{run}.json").read_text()) for run in RUNS}
        summary_path = config.out / "summary.json"
        summary = json.loads(summary_path.read_text()) if config.nonlinear else {"runs": list(RUNS)}
        for variant in variants:
            seeds, dense = ([rows[run][variant] for run in group] for group in (RUNS[:2], RUNS[2:]))
            by_stream = {str(s): frontier_at(seeds, dense, s) for s in STREAMS}
            summary[variant] = dict(by_stream=by_stream, modeled_advantage=all(by_stream[str(s)]["credit"] for s in (1, 16)))
            summary[variant]["full_evaluation"] = all(
                row["cache"]["valid_windows"] == ((ROOT / info[run]["data"] / "valid.bin").stat().st_size // 2 - 1)
                // row["cache"]["seq"] and not row["cache"]["lambada_limit"] for run, row in rows.items()
            )
            summary[variant]["modeled_advantage"] &= summary[variant]["full_evaluation"]
        write_json(summary_path, summary)


if __name__ == "__main__":
    main()

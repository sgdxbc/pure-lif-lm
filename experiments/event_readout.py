"""Fit a direct kd25 event readout and screen it on the 5.63 calibration cache."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sample import load

FIT_WINDOWS = (512, 2048, 8192)
FANOUTS = (32, 128, 512)
SEED = 563
KL_MARGIN = .012
JOB_SECONDS = 3500


def nested_windows(token_count, seq, excluded, sizes, seed=SEED):
    """Draw one ordered pool of disjoint windows and take nested prefixes."""
    windows = (token_count - 1) // seq
    blocked = set(excluded)
    if len(blocked) + max(sizes) > windows or any(p % seq or not 0 <= p < windows * seq for p in blocked):
        raise ValueError("insufficient training windows or invalid excluded offsets")
    draw = np.random.default_rng(seed).choice(windows, size=max(sizes) + len(blocked), replace=False) * seq
    ordered = np.asarray([p for p in draw if p not in blocked])
    return tuple(ordered[:n] for n in sizes)


@torch.no_grad()
def capture_training(model, data, starts, seq, channels, batch, directory, dev, budget):
    tokens = len(starts) * seq
    packed = np.lib.format.open_memmap(directory / "events.npy", mode="w+", dtype=np.uint8,
                                       shape=(tokens, (channels + 7) // 8))
    targets = np.lib.format.open_memmap(directory / "targets.npy", mode="w+", dtype=np.uint16, shape=(tokens,))
    records = []
    layer = model.blocks[-1].fc2
    def record(_, args):
        active = args[0].flatten(0, 1).detach().ne(0)
        records.append(np.packbits(active.cpu().numpy(), axis=1))
    with layer.register_forward_pre_hook(record):
        for i in range(0, len(starts), batch):
            budget()
            ids = np.stack([data[s:s + seq + 1] for s in starts[i:i + batch]])
            x = torch.as_tensor(ids[:, :-1].astype(np.int64), device=dev)
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                model._features(x)
            pos = slice(i * seq, (i + len(ids)) * seq)
            packed[pos] = records.pop()
            targets[pos] = ids[:, 1:].reshape(-1)
    packed.flush()
    targets.flush()
    return packed, targets


def unpack(packed, channels, dev):
    x = torch.as_tensor(np.array(packed), device=dev)
    bits = torch.arange(7, -1, -1, dtype=torch.uint8, device=dev)
    return ((x.unsqueeze(-1) >> bits) & 1).flatten(1)[:, :channels]


def event_edges(events, words, weights):
    row, channel = events.nonzero(as_tuple=True)
    return row[:, None].expand(-1, words.size(1)), words[channel], weights[channel]


def sparse_scores(events, words, weights, bias):
    """Evaluate additive event scores and track which words received any edge."""
    row, word, delta = event_edges(events, words, weights)
    logits = bias.expand(len(events), -1).clone()
    logits.index_put_((row, word), delta, accumulate=True)
    touched = torch.zeros_like(logits, dtype=torch.bool)
    touched[row, word] = True
    return logits, touched


def sparse_likelihood(events, words, weights, bias, targets):
    """Compute exact target log-probs from touched words and a fixed baseline sum."""
    row, word, delta = event_edges(events, words, weights)
    vocab = len(bias)
    hits, inverse = (row * vocab + word).flatten().unique(return_inverse=True)
    delta_sum = bias.new_zeros(len(hits), dtype=torch.float64).index_add_(0, inverse, delta.flatten().double())
    row, word = hits // vocab, hits % vocab
    base = bias.double().exp()
    change = base[word] * delta_sum.expm1()
    partition = base.sum().expand(len(events)).clone().index_add_(0, row, change)
    targets_edit = torch.where(word == targets[row], delta_sum, 0.)
    logits = bias[targets].double().index_add_(0, row, targets_edit)
    return logits - partition.log()


def untouched_bounds(flat, touched, bias, targets):
    """Best possible KL and target NLL when only the untouched conditional is fixed."""
    logp = flat.double().log_softmax(-1)
    logp_rest = logp.masked_fill(touched, -torch.inf)
    log_mass = logp_rest.logsumexp(-1, keepdim=True)
    b = bias.double().expand_as(logp).masked_fill(touched, -torch.inf)
    log_z = b.logsumexp(-1, keepdim=True)
    # Empty complements produce -inf - -inf. Set touched terms to zero.
    diff = (logp_rest - log_mass - b + log_z).masked_fill(touched, 0)
    kl = (logp_rest.exp() * diff).sum(-1).clamp_min(0)
    nll = (log_z[:, 0] - bias[targets]).masked_fill(touched.gather(1, targets[:, None])[:, 0], 0)
    return kl, nll, log_z[:, 0]


def count_range(packed, targets, counts, activity, freq, first, end, budget):
    """Accumulate only the new suffix at each learning-curve size."""
    channels = counts.size(1)
    for start in range(first, end, 8192):
        budget()
        ix = slice(start, min(end, start + 8192))
        active = unpack(packed[ix], channels, counts.device)
        target = torch.as_tensor(targets[ix].astype(np.int64), device=counts.device)
        row, event = active.nonzero(as_tuple=True)
        counts.view(-1).index_add_(0, target[row] * channels + event,
                                   torch.ones_like(event, dtype=counts.dtype))
        activity.add_(active.sum(0, dtype=activity.dtype))
        freq.add_(target.bincount(minlength=len(freq)))


def assignments(counts, activity, freq, limit):
    """Rank by smoothed excess co-occurrence. Work in channel blocks."""
    rows = []
    for start in range(0, len(activity), 64):
        ix = slice(start, start + 64)
        expected = activity[ix, None] * freq
        score = (counts[:, ix].T - expected) * (1 + expected).rsqrt()
        rows.append(score.topk(limit, dim=1).indices)
    return torch.cat(rows)


def fit(packed, targets, n, words, bias, args, dev, budget):
    """Fit only the selected edge deltas with exact full-vocabulary CE."""
    weights = torch.nn.Parameter(torch.zeros_like(words, dtype=torch.float32))
    opt = torch.optim.AdamW([weights], lr=.03)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(args.epochs):
        opt.param_groups[0]["lr"] = .03 if epoch == 0 else .01
        total = torch.zeros((), device=dev)
        # Keep reads contiguous within each batch of packed rows.
        for start in rng.permutation(np.arange(0, n, args.fit_batch)):
            budget()
            ix = slice(start, min(n, start + args.fit_batch))
            opt.zero_grad(set_to_none=True)
            active = unpack(packed[ix], len(words), dev).to(dtype)
            # A masked dense projection uses tensor cores rather than scattering
            # all events x fan-out edges for each token. Only weights is trainable.
            matrix = torch.zeros((len(words), len(bias)), device=dev, dtype=dtype)
            matrix.scatter_(1, words, weights.to(dtype))
            logits = (active @ matrix).float() + bias
            loss = F.cross_entropy(logits, torch.as_tensor(targets[ix].astype(np.int64), device=dev))
            loss.backward()
            opt.step()
            total += loss.detach() * len(active)
        history.append(float(total / n))
        print(f"fit tokens={n} fanout={words.size(1)} epoch={epoch + 1} nll={history[-1]:.5f}", flush=True)
    return weights.detach(), history


def summary(values):
    return dict(zip(("mean", "p50", "p90", "p99", "max"),
                    [float(np.mean(values)), *np.quantile(values, [.5, .9, .99, 1]).tolist()]))


def traffic(events, touched, bias_rest, bias, fanout, streams):
    """Logical storage traffic with a shared fan-out fetch per simultaneous tick."""
    # Transpose contiguous, equal-length cache shards so each shard is a stream.
    packed = events.reshape(streams, -1, events.shape[-1])
    fetch = np.logical_or.reduce(packed, axis=0).sum()
    hits = events.sum() * fanout
    n = len(events)
    # Fixed global alias table, sampled by rejection only when drawing from
    # the untouched distribution. Conditional expected trials = Z_bias / Z_rest.
    # Caller passes P(untouched) / bias_mass(untouched) per token for the total.
    alias = bias_rest.sum()
    return {
        "streams": streams,
        "reads_bytes_per_token": {
            "fanout_lists": float(fetch * (8 + fanout * 4) / n),
            "stream_accumulators": float(hits * 8 / n),
            "bias_and_normalize_sample": float(touched.sum() * 16 / n + 8),
            "fixed_alias_table": float(alias * 8 / n),
        },
        "writes_bytes_per_token": {
            "stream_accumulators": float(hits * 8 / n),
            "touched_list_and_cdf": float(touched.sum() * 8 / n),
        },
        "static_bytes": int(events.shape[-1] * (8 + fanout * 4) + len(bias) * 12 + 8),
        "scratch_bytes": int(streams * (len(bias) * 8 + int(touched.max()) * 8)),
    }


@torch.no_grad()
def assess(events, features, targets, rows, words, weights, bias, args, dev, budget):
    totals = np.zeros(5)
    counts, alias = [], []
    for start in range(0, len(targets), args.batch):
        budget()
        ix = slice(start, start + args.batch)
        x = torch.as_tensor(np.array(features[ix]), device=dev)
        target = torch.as_tensor(targets[ix], device=dev)
        active = torch.as_tensor(events[ix], device=dev)
        flat = F.linear(x, rows)
        logits, touched = sparse_scores(active, words, weights.double(), bias.double())
        kl, floor, log_rest = untouched_bounds(flat, touched, bias, target)
        lp = sparse_likelihood(active, words, weights, bias, target)
        lp_explicit = logits.log_softmax(-1)
        torch.testing.assert_close(lp, lp_explicit.gather(1, target[:, None])[:, 0], atol=1e-9, rtol=1e-9)
        nll = -lp.sum()
        flat_nll = F.cross_entropy(flat, target, reduction="sum")
        full_kl = F.kl_div(lp_explicit, flat.double().log_softmax(-1),
                           log_target=True, reduction="sum")
        totals += [float(v) for v in (nll, flat_nll, full_kl, kl.sum(), floor.sum())]
        counts.extend(touched.sum(-1).cpu().numpy())
        # P_model(rest) * expected alias trials conditional on choosing rest.
        factor = (bias.double().logsumexp(-1) - logits.double().logsumexp(-1)).exp()
        alias.extend(factor.masked_fill(~log_rest.isfinite(), 0).cpu().numpy())
    keys = ("fitted_nll", "flat_nll", "fitted_kl", "untouched_kl_floor", "untouched_nll_floor")
    return {**dict(zip(keys, (totals / len(targets)).tolist())),
            "touched_words": summary(counts),
            "traffic": [traffic(events, np.asarray(counts), np.asarray(alias), bias, words.size(1), n)
                        for n in (1, 16)]}


def rejection(fanouts, conventional_kl):
    limit = conventional_kl + KL_MARGIN
    if any(row["fitted_kl"] <= limit for row in fanouts.values()):
        return "parity_candidate"
    if any(row["untouched_kl_floor"] <= limit for row in fanouts.values()):
        return "not retired, fitting-limited"
    return "retire"


def learning_curve(packed, targets, sizes, cal, rows, channels, args, dev, budget):
    counts = torch.zeros((len(rows), channels), dtype=torch.int32, device=dev)
    activity = counts.new_zeros(channels)
    freq = counts.new_zeros(len(rows))
    points = []
    first = 0
    for n in sizes:
        count_range(packed, targets, counts, activity, freq, first, n, budget)
        marginal = (freq.float() + 1) / (n + len(freq))
        bias = marginal.log()
        mapping = assignments(counts, activity, marginal, max(args.fanout))
        point = {"fit_tokens": n, "fanouts": {}}
        for size in args.fanout:
            words = mapping[:, :size].contiguous()
            weights, history = fit(packed, targets, n, words, bias, args, dev, budget)
            point["fanouts"][size] = assess(*cal, rows, words, weights, bias, args, dev, budget)
            point["fanouts"][size]["train_nll_by_epoch"] = history
            path = args.out.with_name(f"{args.out.stem}_{n}_{size}.pt")
            torch.save({"words": words.cpu().to(torch.int32), "weights": weights.cpu(), "bias": bias.cpu()}, path)
        points.append(point)
        first = n
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "runs/g45_kd25")
    p.add_argument("--cache", type=Path, default=Path("runs/head_screen/g45_kd25"))
    p.add_argument("--head-report", type=Path, default=Path("runs/head_screen/g45_kd25.json"))
    p.add_argument("--out", type=Path, default=Path("runs/event_readout/kd25.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--fanout", nargs="+", type=int, default=list(FANOUTS))
    p.add_argument("--fit-windows", nargs=3, type=int, default=list(FIT_WINDOWS), help="reduce for a smoke test")
    p.add_argument("--batch", type=int, default=64, help="held-out score batch")
    p.add_argument("--fit-batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    started = time.monotonic()

    def budget():
        if time.monotonic() - started > JOB_SECONDS:
            raise TimeoutError("event readout reached its time cap before completing the curve")

    torch.set_num_threads(4)
    dev = torch.device(args.device)
    manifest = json.loads((args.cache / "cache.json").read_text())
    conventional = json.loads(args.head_report.read_text())
    if conventional["cache"] != manifest:
        p.error("conventional head report does not use this exact cache")
    baseline = {name: conventional[name]["fit"]["calibration_kl"]
                for name in ("linear", "nonlinear", "shortlist") if name in conventional}
    best = min(baseline, key=baseline.get)
    seq, channels = manifest["seq"], manifest["event_channels"]

    def cached(kind):
        return np.load(args.cache / f"cal_{kind}.npy", mmap_mode="r")
    events = np.unpackbits(cached("events"), axis=1, count=channels).astype(bool)
    targets = cached("y").astype(np.int64)
    features = cached("x")
    rows = torch.load(args.cache / "word_rows.pt", map_location=dev, weights_only=True).float()
    sizes = [n * seq for n in args.fit_windows]
    if (min(args.fanout) < 1 or max(args.fanout) > len(rows) or min(args.batch, args.fit_batch, args.epochs) < 1
            or len(features) % 16 or not len(features) == len(events) == len(targets) == len(manifest["cal_starts"]) * seq
            or sizes != sorted(set(sizes)) or min(sizes) <= 0):
        p.error("invalid fit budget, support size, or cache dimensions")
    capture_bytes = sizes[-1] * ((channels + 7) // 8 + 2)
    output_bytes = len(sizes) * (channels * sum(args.fanout) * 8 + len(args.fanout) * len(rows) * 4)
    if capture_bytes + output_bytes > 4_900_000_000:
        p.error("packed capture and fitted outputs would exceed the 5 GB disk cap")

    settings = json.loads((args.run / "args.json").read_text())
    data = np.memmap(ROOT / settings["data"] / "train.bin", dtype=np.uint16, mode="r")
    pools = nested_windows(len(data), seq, manifest["fit_starts"] + manifest["cal_starts"], args.fit_windows)
    model, step = load(args.run / "ckpt.pt", ema=True, device=dev)
    if (step != manifest["step"] or model.cfg.mlp != "spike"
            or model.blocks[-1].fc2.in_features != channels
            or not torch.equal(model.head.weight.float(), rows)):
        p.error("EMA checkpoint does not agree with the head_screen cache")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cal = events, features, targets
    with tempfile.TemporaryDirectory(prefix=".event-fit-", dir=args.out.parent) as tmp:
        packed, labels = capture_training(model, data, pools[-1], seq, channels, manifest["body_batch"],
                                          Path(tmp), dev, budget)
        del model
        print(f"captured {len(labels)} new training tokens", flush=True)
        curve = learning_curve(packed, labels, sizes, cal, rows, channels, args, dev, budget)
        del packed, labels
    budget()
    registered = (tuple(args.fit_windows) == FIT_WINDOWS and tuple(args.fanout) == FANOUTS and sizes[-1] >= 8_000_000
                  and args.run.resolve() == ROOT / "runs/g45_kd25" and args.epochs == 2 and args.fit_batch == 2048)
    report = {
        "scope": "5.63 direct-event rejection screen on the conventional heads' calibration windows",
        "cache_step": manifest["step"],
        "fit_window_order": pools[-1].tolist(),
        "heldout_tokens": len(features),
        "conventional_calibration_kl": baseline,
        "best_conventional": best,
        "kl_limit": baseline[best] + KL_MARGIN,
        "conventional_selection": "best epochs were chosen on cal, which favors those heads on this comparison",
        "decision": rejection(curve[-1]["fanouts"], baseline[best]) if registered else "diagnostic_only",
        "events_per_token": summary(events.sum(-1)),
        "curve": curve,
        "elapsed_seconds": time.monotonic() - started,
        "capture_bytes": capture_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
        "layout": {
            "fanout_entry": "16-bit word plus 16-bit delta",
            "fit_precision": "BF16 projection on CUDA, FP32 loss and learned deltas; FP64 sparse vs explicit check",
            "stream_scratch": "32-bit accumulator plus 32-bit epoch tag per word, reused every token",
            "simultaneous_rows": "16 equal contiguous cache shards, broadcast each active fan-out list once per tick",
            "sampling": "unit temperature, touched CDF plus fixed alias table with rejection of touched words",
            "omissions": "initial scratch clear, bank conflicts, body traffic and latency, arithmetic and fitted-weight quantization error",
        },
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "fit_window_order"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

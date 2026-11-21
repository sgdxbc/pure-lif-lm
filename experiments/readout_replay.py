"""Replay the full vocabulary head of an EMA model_tf checkpoint without training.

  python3 experiments/readout_replay.py runs/g45_kd25 --device cuda
  python3 experiments/readout_replay.py runs/g48_gla50 --device cuda --thresholds 1 --lambada

The first token of each evaluation sequence and every --refresh tokens thereafter use an exact projection
and reset both caches. Between refreshes, signed events move each reconstructed channel by one calibrated step
when its residual crosses +/- one step. The INT4 control rounds to the same step and clips to [-7, 7].
Its channel error is at most max(step/2, abs(residual) - 7*step), up to floating point roundoff.
Both update the cached logits with the projection of their delta and keep the same per-stream FP32 caches.
The per-channel step is a sweep multiplier times RMS feature change on a small training-text calibration set.

NLL and KL use the same EMA weights in both arms and FP32 vocabulary projection. Four-bit weights apply to the
U55C estimate. It budgets a read and a write of both caches per token, including refreshes, and reports the
bandwidth the cache requires. Resident traffic uses the existing estimator's on-chip assumption.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fpga_est
from model_tf import GPT
from sample import load


def encode_updates(features, step, codec):
    """Encode independent refresh blocks of shape (blocks, tokens, channels)."""
    updates = np.empty_like(features)
    active = np.zeros(features.shape[:2], dtype=np.int64)
    state = features[:, 0].copy()
    updates[:, 0] = state
    for pos in range(1, features.shape[1]):
        residual = features[:, pos] - state
        if codec == "int4":
            code = np.clip(np.rint(residual / step), -7, 7)
        else:
            code = np.sign(residual) * (np.abs(residual) >= step)
        active[:, pos] = np.count_nonzero(code, axis=-1)
        updates[:, pos] = step * code
        state += updates[:, pos]
    return updates, active


def replay_logits(weight, updates):
    """Batch the signed row sums in software, then accumulate them in token order."""
    return F.linear(updates, weight).cumsum(1)


def replay_hw(hw, d, vocab, active_rate, refresh_rate, codec):
    """Whole-model counts with one weight layout shared by refresh and incremental projection."""
    head_ops = d * vocab
    head_bytes = head_ops * hw["bits"] / 8
    update_rate = (1 - refresh_rate) * active_rate
    row_ops = head_ops * update_rate
    cache_bytes = 4 * (d + vocab)
    return dict(
        hw, dense_macs=hw["dense_macs"] - head_ops + refresh_rate * head_ops
        + (row_ops if codec == "int4" else 0),
        # One residual subtraction per channel and a reconstruction add per active channel.
        sparse_adds=hw["sparse_adds"] + d * (1 - refresh_rate + update_rate)
        + (row_ops if codec == "events" else 0),
        dense_bytes=hw["dense_bytes"] - head_bytes,
        state_bytes=hw["state_bytes"] + cache_bytes,
        readout_replay=dict(weight_bytes=head_bytes, refresh_rate=refresh_rate, active_rate=active_rate),
    )


def board_cost(hw, cache_bytes=0):
    dev = dict(fpga_est.DEVICES["u55c"], cards=1, eff=0.6)
    return {str(r["streams"]): dict(r, cache_required_GB_s=2 * cache_bytes * r["tok_s"] / 1e9)
            for r in fpga_est.estimate_tf(hw, dev) if r["streams"] in (1, 16)}


def batches(path, seq, batch, limit):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(len(data) - 1, limit or len(data)) // seq * seq
    if n < seq:
        raise ValueError(f"{path} needs at least one {seq}-token evaluation sequence")
    for start in range(0, n, batch * seq):
        stop = min(start + batch * seq, n)
        ids = torch.from_numpy(data[start:stop + 1].astype(np.int64))
        yield ids[:-1].reshape(-1, seq), ids[1:].reshape(-1, seq)


def normalized(model, ids, dev):
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        features, _ = model._features(ids.to(dev))
    return features.float()


def calibration(model, data, seq, batch, limit, dev):
    total, count, tokens = np.zeros(model.cfg.d_model, dtype=np.float64), 0, 0
    for ids, _ in batches(data / "train.bin", seq, batch, limit):
        delta = normalized(model, ids, dev).diff(dim=1)
        total += delta.square().sum((0, 1)).cpu().numpy()
        count += delta.size(0) * delta.size(1)
        tokens += ids.numel()
    return np.sqrt(total / count).astype(np.float32).clip(min=np.finfo(np.float32).eps), tokens


def refresh_blocks(features, targets, refresh):
    """Group full refresh blocks into one batch and retain the short tail of each sequence."""
    length, width = features.shape[1:]
    end = length // refresh * refresh
    if end:
        yield features[:, :end].reshape(-1, refresh, width), targets[:, :end].reshape(-1, refresh)
    if end < length:
        yield features[:, end:], targets[:, end:]


def burst_stats(hist, width):
    n = hist.sum()
    cumulative = hist.cumsum()
    channels = {"mean": float(hist @ np.arange(hist.size)) / n if n else 0.0}
    for percentile in (50, 90, 99):
        channels[f"p{percentile}"] = int(np.searchsorted(cumulative, np.ceil(n * percentile / 100)))
    return {"channels": channels, "fraction": {k: float(v / width) for k, v in channels.items()}}


def nll_sum(logp, y):
    return -logp.gather(-1, y[..., None]).sum(dtype=torch.float64).item()


def evaluate(model, data, args, scale, dev):
    weight = model.head.weight.float()
    vocab, width = weight.shape
    steps = {ratio: scale * ratio for ratio in args.thresholds}
    stats = {(ratio, codec): dict(nll=0., kl=0., hist=np.zeros(width + 1, dtype=np.int64))
             for ratio in steps for codec in ("events", "int4")}
    exact_nll, tokens, refreshes, rate_sum = 0., 0, 0, 0
    for ids, targets in batches(data / "valid.bin", args.seq, args.batch, args.max_tokens):
        features = normalized(model, ids, dev)
        rate_sum = rate_sum + ids.numel() * model.rates(dev)
        for vectors, labels in refresh_blocks(features, targets, args.refresh):
            # Score with a bounded number of vocabulary rows in memory while encoding all blocks together on CPU.
            cpu = vectors.cpu().numpy()
            pieces = max(1, 1024 // vectors.size(1))
            encoded = {}
            for (ratio, codec), stat in stats.items():
                delta, active = encode_updates(cpu, steps[ratio], codec)
                stat["hist"] += np.bincount(active[:, 1:].ravel(), minlength=width + 1)
                encoded[ratio, codec] = torch.from_numpy(delta)
            for start in range(0, vectors.size(0), pieces):
                sl = slice(start, start + pieces)
                y = labels[sl].to(dev)
                exact = F.linear(vectors[sl], weight).log_softmax(-1)
                exact_nll += nll_sum(exact, y)
                p = exact.exp()
                for key, stat in stats.items():
                    logp = replay_logits(weight, encoded[key][sl].to(dev)).log_softmax(-1)
                    stat["nll"] += nll_sum(logp, y)
                    stat["kl"] += (p * (exact - logp)).sum(dtype=torch.float64).item()
            tokens += labels.numel()
            refreshes += labels.size(0)
    return exact_nll / tokens, stats, tokens, refreshes, (rate_sum / tokens).tolist()


def lambada(model, data, tokenizer, seq, refresh, step, dev):
    """Reuse the repository's LAMBADA harness with the chosen readout variants."""
    import lm_eval
    from tokenizers import Tokenizer
    from eval_lm import make_lm, summarise

    tok = Tokenizer.from_file(str(data / tokenizer))

    class Readout(torch.nn.Module):
        def __init__(self, codec):
            super().__init__()
            self.codec = codec

        def forward(self, ids):
            vectors = normalized(model, ids, dev)
            weight = model.head.weight.float()
            with torch.autocast(dev.type, enabled=False):
                if self.codec == "exact":
                    return (F.linear(vectors, weight),)
                logits = []
                for block in vectors.split(refresh, dim=1):
                    delta, _ = encode_updates(block.cpu().numpy(), step, self.codec)
                    logits.append(replay_logits(weight, torch.from_numpy(delta).to(dev)))
                return (torch.cat(logits, dim=1),)

    result = {}
    for codec in ("exact", "events", "int4"):
        lm = make_lm(Readout(codec), tok, dev, seq, batch_tokens=2048)
        r = lm_eval.simple_evaluate(model=lm, tasks=["lambada_openai"], num_fewshot=0)
        result[codec] = summarise(r)["lambada_openai"]
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq", type=int, help="validation sequence length, defaults to training length")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--refresh", type=int, default=128, help="exact projection period within each sequence")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1., 2., 4.])
    p.add_argument("--calibration-tokens", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=0, help="validation token cap, 0 for the full split")
    p.add_argument("--lambada", action="store_true", help="also score one selected threshold and the exact baseline")
    p.add_argument("--output", type=Path, help="defaults to RUN/readout_replay.json")
    args = p.parse_args()
    run_args = json.loads((args.run / "args.json").read_text())
    args.seq = run_args.get("seq", 1024) if args.seq is None else args.seq
    if (args.seq < 2 or min(args.refresh, args.batch) < 1 or args.max_tokens < 0
            or args.calibration_tokens < args.seq):
        p.error("need seq >= 2, positive refresh and batch, and at least one calibration sequence")
    if any(not np.isfinite(x) or x < 0 or (x == 0 and args.refresh != 1) for x in args.thresholds):
        p.error("thresholds must be finite and positive, or zero with --refresh 1")
    if args.lambada and len(args.thresholds) != 1:
        p.error("--lambada requires one value in --thresholds")
    dev = torch.device(args.device)
    model, ckpt_step = load(args.run / "ckpt.pt", ema=True, device=dev)
    if not isinstance(model, GPT) or not isinstance(model.head, torch.nn.Linear):
        p.error("readout replay requires a model_tf.py checkpoint with a full vocabulary head")
    if model.pos is not None and args.seq > model.cfg.max_len:
        p.error("--seq exceeds the learned position table")
    data = ROOT / run_args.get("data", "data")
    scale, calibration_tokens = calibration(model, data, args.seq, args.batch, args.calibration_tokens, dev)
    exact, stats, tokens, refreshes, rates = evaluate(model, data, args, scale, dev)
    vocab, width = model.head.weight.shape
    hw = fpga_est.tf_hw({**run_args, **vars(model.cfg)}, rates=rates, ctx=args.seq, bits=4)
    base_cost = board_cost(hw)
    cache_bytes = 4 * (width + vocab)
    report = dict(step=ckpt_step, ema=True, split="valid", sequence_tokens=args.seq, validation_tokens=tokens,
                  refresh_tokens=args.refresh, refresh_count=refreshes, refresh_rate=refreshes / tokens,
                  refresh_policy="sequence start and every refresh_tokens thereafter",
                  fpga_device="u55c", weight_bits=4, add_cost=1.0,
                  cache_bytes_per_stream=cache_bytes, cache_traffic_bytes_per_token=2 * cache_bytes,
                  calibration_tokens=calibration_tokens, rms_delta=scale.tolist(),
                  exact=dict(nll=exact, cost=base_cost), replays={})
    for (ratio, codec), stat in stats.items():
        burst = burst_stats(stat["hist"], width)
        row_hist = stat["hist"].copy()
        row_hist[width] += refreshes
        cost = board_cost(replay_hw(hw, width, vocab, burst["fraction"]["mean"], refreshes / tokens, codec),
                          cache_bytes)
        savings = {b: 1 - cost[b]["mJ_per_token"] / base_cost[b]["mJ_per_token"] for b in cost}
        row = dict(threshold_multiplier=ratio, codec=codec, nonrefresh_burst=burst,
                   projection_row_burst=burst_stats(row_hist, width),
                   all_token_crossing_fraction=burst["fraction"]["mean"] * (1 - refreshes / tokens),
                   kl_exact_to_replay=stat["kl"] / tokens, nll=stat["nll"] / tokens,
                   nll_delta=stat["nll"] / tokens - exact, cost=cost, energy_savings=savings)
        report["replays"].setdefault(str(ratio), {})[codec] = row
        print(f"{ratio:g} {codec:6s}  NLL {row['nll']:.5f}  KL {row['kl_exact_to_replay']:.5f}  "
              f"active {burst['fraction']['mean']:.2%}  energy savings {savings['1']:.1%}/{savings['16']:.1%}",
              flush=True)
    if args.lambada:
        report["lambada_openai"] = lambada(model, data, run_args.get("tokenizer", "tok4096.json"), args.seq,
                                           args.refresh, scale * args.thresholds[0], dev)
    output = args.output or args.run / "readout_replay.json"
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"exact NLL {exact:.5f} on {tokens} tokens. Wrote {output}", flush=True)


if __name__ == "__main__":
    main()

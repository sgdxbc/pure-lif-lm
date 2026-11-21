"""Probe quantized relay insertion without training, from a recorded --init_weights run.

From the repo root:
  python3 experiments/relay_calib.py runs/g45_ft25 --device cuda --out runs/g45_ft25/relay_calib.json

Reads the insertion run's args and first-step log and its source checkpoint's EMA.
Calibration and scoring use disjoint windows of the training split. Separately,
replay the original first training batch to compare directly with the logged NLL.
The matched arm targets the default thresholds' rate on the calibration windows.
Only the arm targeting the registered budget enters the pass rule.
The top-level fraction includes values rounded to the cap, and the clipped fraction
counts those that would have rounded above it. Silent channels emit no events across
the entire held-out sample.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_tf import GPT, TFConfig, _Tern
from relay_calib import calibrate_relays, measure_relay_rate, observe_relay_inputs
from sample import load
from train import Data


def disjoint_offsets(data, seed, ncal, nheld):
    """Draw whole nonoverlapping input-plus-target windows from train.bin."""
    length = data.seq + 1
    starts = np.random.default_rng(seed).choice(len(data.arr) // length, ncal + nheld, replace=False) * length
    return starts[:ncal], starts[ncal:]


def read_windows(data, offsets, device):
    return torch.from_numpy(np.stack([data.arr[i:i + data.seq + 1] for i in offsets]).astype(np.int64)).to(device)


def inserted_models(source_path, args):
    source, step = load(source_path, ema=True)
    if not isinstance(source, GPT) or source.relays():
        raise ValueError("--init_weights must refer to a no-relay model_tf checkpoint")
    cfg = TFConfig(**{f.name: args.get(f.name, f.default) for f in fields(TFConfig) if f.name != "max_len"},
                   max_len=max(1024, args["seq"]))
    if args["arch"] != "tf" or cfg.attn != "linear" or cfg.relay != "quant" or args["carry_state"]:
        raise ValueError("expected an independent-window quantized-relay GLA insertion run")
    torch.manual_seed(args["seed"])
    student = GPT(cfg)
    missing, unexpected = student.load_state_dict(source.state_dict(), strict=False)
    thresholds = {name + ".log_theta" for name, m in student.named_modules() if m in student.relays()}
    if set(missing) != thresholds or unexpected:
        raise ValueError(f"source must supply every parameter except relay thresholds: missing {missing}, "
                         f"unexpected {unexpected}")
    return source, student.eval(), step


@torch.no_grad()
def score(model, batches, relay_stats=False):
    relays = model.relays()
    counts, active = {}, {}

    def collect(relay, inputs):
        x = inputs[0]
        scaled = x / relay.log_theta.exp().to(x.dtype)
        magnitude = _Tern.apply(scaled, relay.levels).abs()
        cap = (magnitude == relay.levels).sum()
        clipped = ((scaled.abs() + 0.5).floor() > relay.levels).sum()
        increments = torch.stack((cap, clipped))
        fired = (magnitude != 0).any(dim=(0, 1))
        counts[relay] = counts[relay] + increments if relay in counts else increments
        active[relay] = active[relay] | fired if relay in active else fired

    nll, tokens = 0., 0
    rates = []
    with observe_relay_inputs(relays if relay_stats else [], collect):
        for batch in batches:
            ids, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            # Use the same head, CE, and microbatch shape as train.py.
            output = model(ids, targets)
            nll += output[1].item() * targets.numel()
            tokens += targets.numel()
            if relays:
                rates.append(torch.stack([r.rate for r in relays]) * targets.numel())
            del output
    result = {"nll": nll / tokens}
    if relays:
        rates = (torch.stack(rates).sum(0) / tokens).tolist()
        result["weighted_rate"] = sum(rate * r.fan for rate, r in zip(rates, relays)) / sum(r.fan for r in relays)
    if relay_stats:
        names = {m: name for name, m in model.named_modules()}
        result["relays"] = {}
        for rate, relay in zip(rates, relays):
            cap, clipped = counts[relay].tolist()
            result["relays"][names[relay]] = {
                "rate": rate, "at_top_fraction": cap / (tokens * relay.D),
                "clipped_fraction": clipped / (tokens * relay.D),
                "silent_channel_fraction": 1 - active[relay].sum().item() / relay.D,
            }
    return result


def run_probe(run, device, seed, calibration_batches, heldout_batches):
    args = json.loads((run / "args.json").read_text())
    source_path = ROOT / args["init_weights"]
    source, student, step = inserted_models(source_path, args)
    source.to(device)
    student.to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.set_num_threads(args.get("threads", 4))

    data = Data("train", args["seq"], args["seed"], args["data"])
    data.batch(args["batch"])
    replay_offsets = data.last_ix
    microbatch = args["batch"] // args["accum"]
    cal_offsets, held_offsets = disjoint_offsets(data, seed, microbatch * calibration_batches,
                                                microbatch * heldout_batches)
    replay = read_windows(data, replay_offsets, device).split(microbatch)
    heldout = read_windows(data, held_offsets, device).split(microbatch)
    cal_ids = read_windows(data, cal_offsets, device)[:, :-1].contiguous()

    log_path = run / "log.jsonl"
    recorded = {}
    if log_path.exists():
        with log_path.open() as f:
            recorded = next((row for row in map(json.loads, f) if row.get("step") == 1 and "loss" in row), {})
    with torch.autocast(device.type, dtype=torch.bfloat16):
        first = score(student, replay)
        source_nll = score(source, heldout)["nll"]
        default = score(student, heldout, relay_stats=True)
        default_calibration_rate = measure_relay_rate(student, cal_ids)
    heldout_arms = {"source_nll": source_nll, "default": default}
    fits = {}
    for arm, target in (("calibrated", args["relay_target"]), ("calibrated_matched", default_calibration_rate)):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            fits[arm] = calibrate_relays(student, cal_ids, target)
        # Refresh autocast's parameter cache after changing the thresholds.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            heldout_arms[arm] = score(student, heldout, relay_stats=True)

    if recorded:
        first["logged_nll"] = recorded["loss"]
        first["nll_minus_logged"] = first["nll"] - recorded["loss"]
        if "relay_spikes" in recorded:
            first["logged_weighted_rate"] = recorded["relay_spikes"]
    for name in ("default", *fits):
        result = heldout_arms[name]
        result["excess_nll"] = result["nll"] - source_nll
    max_excess = default["excess_nll"] / 2
    return {
        "source_checkpoint": str(source_path), "source_ema_step": step,
        "relay": args["relay"], "relay_levels": args["relay_levels"], "seq": args["seq"], "microbatch": microbatch,
        "sample_split": "train", "sample_seed": seed, "precision": "bf16 autocast", "device": str(device),
        "first_step": first, "first_step_offsets": replay_offsets.tolist(),
        "calibration": fits["calibrated"], "matched_calibration": fits["calibrated_matched"],
        "default_calibration_rate": default_calibration_rate,
        "calibration_offsets": cal_offsets.tolist(), "heldout_offsets": held_offsets.tolist(),
        "heldout": heldout_arms,
        "pass_max_excess_nll": max_excess, "pass": heldout_arms["calibrated"]["excess_nll"] <= max_excess,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="insertion run, e.g. runs/g45_ft25")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0, help="sampling seed for the calibration and holdout windows")
    p.add_argument("--calibration-batches", type=int, default=4, help="number of insertion-size microbatches to pool")
    p.add_argument("--heldout-batches", type=int, default=16, help="number of insertion-size microbatches to score")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.seed < 0 or min(a.calibration_batches, a.heldout_batches) < 1:
        p.error("need a nonnegative seed and at least one microbatch for each split")
    result = run_probe(a.run, torch.device(a.device), a.seed, a.calibration_batches, a.heldout_batches)
    text = json.dumps(result, indent=2, allow_nan=False)
    out = a.out or a.run / "relay_calib.json"
    out.write_text(text + "\n")
    first, heldout = result["first_step"], result["heldout"]
    logged = f" (logged {first['logged_nll']:.4f})" if "logged_nll" in first else ""
    print(f"First batch NLL {first['nll']:.4f}{logged}")
    print(f"Held-out NLL source {heldout['source_nll']:.4f}, default {heldout['default']['nll']:.4f}, "
          f"calibrated {heldout['calibrated']['nll']:.4f}, matched {heldout['calibrated_matched']['nll']:.4f}")
    matched = result["matched_calibration"]
    print(f"Matched multiplier {matched['multiplier']:.4f}, rate {matched['weighted_rate']:.4f} "
          f"(default calibration rate {result['default_calibration_rate']:.4f})")
    print(f"Calibrated multiplier {result['calibration']['multiplier']:.4f}, "
          f"rate {result['calibration']['weighted_rate']:.4f}, pass {result['pass']} ({out})")


if __name__ == "__main__":
    main()

"""Data-initialized thresholds for model_tf's stateless quantized relays."""
from contextlib import contextmanager, ExitStack
import math

import torch


@contextmanager
def observe_relay_inputs(relays, hook):
    with ExitStack() as stack:
        for relay in relays:
            stack.callback(relay.register_forward_pre_hook(hook).remove)
        yield


@torch.no_grad()
def measure_relay_rate(model, ids):
    """Measure the penalty's fan-weighted hard count without the vocabulary head."""
    model._features(ids)
    relays = model.relays()
    return (sum(r.fan * r.rate for r in relays) / sum(r.fan for r in relays)).item()


@torch.no_grad()
def calibrate_relays(model, ids, target=0.25, tolerance=0.0025):
    """Set theta = multiplier * input RMS (RMS floor 1e-6) in forward order.

    Pool the calibration rows in ids [batch, time] on the model's device. Use
    the same autocast policy as training. Recompute the RMS inputs for every
    candidate so each relay sees its predecessors at that candidate's thresholds.
    Returns the multiplier and achieved hard-count rate, or None without relays.
    """
    relays = model.relays()
    if not relays:
        return None
    if any(r.kind != "quant" for r in relays):
        raise ValueError("RMS calibration needs stateless quantized relays")
    fan = sum(r.fan for r in relays)
    max_rate = sum(r.fan * r.levels for r in relays) / fan
    if not 0 < target < max_rate or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid relay budget or tolerance")

    def measure(multiplier):
        def install(relay, inputs):
            rms = inputs[0].float().square().mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
            relay.log_theta.copy_(rms.log() + math.log(multiplier))

        with observe_relay_inputs(relays, install):
            rate = measure_relay_rate(model, ids)
        if not math.isfinite(rate):
            raise ValueError("non-finite weighted relay rate")
        return rate

    original = [r.log_theta.clone() for r in relays]
    success = False
    try:
        multiplier = 1.0
        rate = measure(multiplier)
        best = (abs(rate - target), multiplier, rate)
        if best[0] > tolerance:
            # Bracket opposite signs, then bisect in log space. Do not assume
            # the whole-model rate is monotone as activations change.
            direction = 2.0 if rate > target else 0.5
            start = (multiplier, rate)
            for _ in range(12):
                multiplier *= direction
                rate = measure(multiplier)
                best = min(best, (abs(rate - target), multiplier, rate))
                if abs(rate - target) <= tolerance:
                    break
                if (rate > target) != (start[1] > target):
                    above, below = (start, (multiplier, rate)) if start[1] > target else ((multiplier, rate), start)
                    for _ in range(16):
                        multiplier = math.sqrt(above[0] * below[0])
                        rate = measure(multiplier)
                        best = min(best, (abs(rate - target), multiplier, rate))
                        if abs(rate - target) <= tolerance:
                            break
                        if rate > target:
                            above = (multiplier, rate)
                        else:
                            below = (multiplier, rate)
                    break
                start = (multiplier, rate)
        if abs(rate - target) > tolerance:
            raise ValueError(f"relay budget {target:g} ± {tolerance:g} was not reached "
                             f"(closest rate {best[2]:.6g} at multiplier {best[1]:.6g})")
        success = True
        return {"multiplier": multiplier, "weighted_rate": rate, "target": target, "tolerance": tolerance}
    finally:
        if not success:
            for relay, theta in zip(relays, original):
                relay.log_theta.copy_(theta)

"""CPU development screen for sparse regime switches. See docs/probe_regime.md."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


CELLS = 32
TABLE_SEED, TRAIN_SEED, DEV_SEED = 201, 204, 202
HELD_OUT_SEEDS = (2001, 2002, 2003)
CLEAN = np.array([np.arange(8), np.r_[4:8, 4:8]])
NOISE = .05
LOG_ODDS = np.log((1 - NOISE) / (NOISE / 7))
CONFIDENCE = .99


@dataclass
class Traffic:
    read: int = 0
    written: int = 0
    input: int = 0


def generate(rng, batch, length):
    state = np.empty((length, batch), dtype=np.int8)
    for stream in range(batch):
        regime = int(rng.integers(2))
        start = 0
        while start < length:
            end = start + int(rng.integers(64, 257))
            state[start:end, stream] = regime
            start = end
            regime ^= 1
    cue = rng.integers(8, size=state.shape, dtype=np.uint8)
    corrupt = rng.random(state.shape) < NOISE
    observed = (CLEAN[state, cue] + corrupt * rng.integers(1, 8, size=state.shape)) % 8
    return cue, observed.astype(np.uint8), state, corrupt


def learn_tables(rng):
    counts = np.ones((2, 8, 8))
    for state in range(2):
        for cue in range(8):
            corrupt = rng.random(512) < NOISE
            sample = (CLEAN[state, cue] + corrupt * rng.integers(1, 8, size=512)) % 8
            counts[state, cue] += np.bincount(sample, minlength=8)
    modes = counts.argmax(-1)
    shared = modes[0] == modes[1]
    counts[:, shared] = counts[:, shared].sum(0)
    return (counts / counts.sum(-1, keepdims=True)).astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def transition(voltage, votes, evidence, beta, threshold):
    charged = beta * voltage + evidence[:, None]
    up, down = charged >= threshold, charged <= -threshold
    return charged, np.where(up, 1., np.where(down, 0., votes)).astype(np.float32)


def vote_mass(votes):
    mean = votes.mean(-1)
    return np.where(mean == .5, .5, np.where(mean > .5, CONFIDENCE, 1 - CONFIDENCE))


def bank_traffic(traffic, cells, observations, cell_events, refreshes, analog=False):
    # Read both FP32 probability rows, one FP32 mixture or one byte for the event-gated table.
    traffic.read += observations * (64 + (4 if analog else 1))
    # Per-cell FP32 membrane and leak. Thresholds and votes apply to the event detector.
    traffic.read += observations * cells * (8 if analog else 13)
    traffic.written += observations * cells * 4
    traffic.written += refreshes * (4 if analog else 1) + (0 if analog else cell_events)
    traffic.input += observations * 2


def signals(tables, cue, seen):
    p0, p1 = tables[0, cue, seen], tables[1, cue, seen]
    evidence = np.clip(np.log(p1 / p0) / LOG_ODDS, -1, 1)
    return p0, p1, evidence


def initial_params():
    return np.stack((np.linspace(.25, .80, CELLS), np.linspace(.65, 1.60, CELLS))).astype(np.float32)


def gradient(params, tables, cue, seen, method):
    beta, threshold = params
    voltage = np.zeros((cue.shape[1], len(beta)), dtype=np.float32)
    votes = np.full_like(voltage, .5)
    accumulated = np.zeros_like(params)
    if method == "eligibility":
        beta_trace, vote_beta, vote_theta = (np.zeros_like(voltage) for _ in range(3))
        learning_bytes = beta_trace.nbytes + vote_beta.nbytes + vote_theta.nbytes
    else:
        tape = np.empty((len(cue), 5, *voltage.shape), dtype=np.float32)
        adj_v, adj_vote = np.zeros_like(voltage), np.zeros_like(votes)
        learning_bytes = tape.nbytes + adj_v.nbytes + adj_vote.nbytes
    # The recurrent buffers and gradients count in both methods. Batch inputs are shared separately.
    learning_bytes += voltage.nbytes + votes.nbytes + accumulated.nbytes
    for t, (x, y) in enumerate(zip(cue, seen)):
        p0, p1, evidence = signals(tables, x, y)
        p = p0 + (p1 - p0) * vote_mass(votes)
        signal = (-((p1 - p0) / p) * (2 * CONFIDENCE - 1) / len(beta))[:, None]
        newer, next_votes = transition(voltage, votes, evidence, beta, threshold)
        up = sigmoid(3 * (newer - threshold))
        down = sigmoid(3 * (-newer - threshold))
        positive = 3 * up * (1 - up) * (1 - votes)
        negative = 3 * down * (1 - down) * votes
        slope_v, slope_theta = positive + negative, -positive + negative
        keep = (newer < threshold) & (newer > -threshold)
        if method == "eligibility":
            accumulated[0] += (signal * vote_beta).sum(0)
            accumulated[1] += (signal * vote_theta).sum(0)
            beta_trace = beta * beta_trace + voltage
            vote_beta = keep * vote_beta + slope_v * beta_trace
            vote_theta = keep * vote_theta + slope_theta
        else:
            tape[t] = (voltage, slope_v, slope_theta, keep, np.broadcast_to(signal, voltage.shape))
        voltage, votes = newer, next_votes
    if method == "bptt":
        for old_v, slope_v, slope_theta, keep, signal in tape[::-1]:
            accumulated[1] += (adj_vote * slope_theta).sum(0)
            adj_v += adj_vote * slope_v
            accumulated[0] += (adj_v * old_v).sum(0)
            adj_v *= beta
            adj_vote = keep * adj_vote + signal
    return accumulated / cue.size, learning_bytes


def train(tables, batches, batch, length, method):
    params = initial_params()
    rng = np.random.default_rng(TRAIN_SEED)
    for _ in range(batches):
        cue, seen, _, _ = generate(rng, batch, length)
        grad, _ = gradient(params, tables, cue, seen, method)
        params -= .5 * grad
        params[0] = params[0].clip(.05, .98)
        params[1] = params[1].clip(.2, 5.)
    return params


def infer(params, tables, cue, observed, method):
    beta, threshold = params
    voltage = np.zeros((cue.shape[1], len(beta)), dtype=np.float32)
    votes = np.full_like(voltage, .5)
    mass = np.full(cue.shape[1], .5, dtype=np.float32)
    probabilities = np.empty(cue.shape)
    selection = np.empty(cue.shape, dtype=np.int8)
    switches = np.zeros(cue.shape, dtype=bool)
    traffic = Traffic()
    cell_events = refreshes = 0
    for t, (x, y) in enumerate(zip(cue, observed)):
        p0, p1, evidence = signals(tables, x, y)
        selection[t] = np.where(mass == .5, -1, mass > .5)
        probabilities[t] = p0 + (p1 - p0) * mass
        if method == "likelihood_ratio":
            posterior = p1 * mass / probabilities[t]
            next_mass = posterior * (1 - 2 / 160) + 1 / 160
            traffic.read += len(x) * (64 + 4)
            traffic.written += len(x) * 4
            traffic.input += len(x) * 2
        elif method == "events":
            voltage, newer = transition(voltage, votes, evidence, beta, threshold)
            events = int(np.count_nonzero(newer != votes))
            cell_events += events
            next_mass = vote_mass(newer)
            refreshed = int(np.count_nonzero(next_mass != mass))
            bank_traffic(traffic, len(beta), len(x), events, refreshed)
            votes = newer
        else:
            voltage = beta * voltage + evidence[:, None]
            next_mass = sigmoid(2 * voltage.mean(-1))
            bank_traffic(traffic, len(beta), len(x), 0, len(x), analog=True)
        next_selection = np.where(next_mass == .5, -1, next_mass > .5)
        switches[t] = next_selection != selection[t]
        refreshes += int(np.count_nonzero(next_mass != mass)) if method == "events" else len(x)
        mass = next_mass
    return probabilities, selection, switches, traffic, cell_events, refreshes


def evaluate(params, tables, streams, method):
    cue, seen, state, corrupt = streams
    probabilities, choices, switches, traffic, events, refreshes = infer(params, tables, cue, seen, method)
    recovery, false, isolated = [], 0, 0
    boundaries = np.flatnonzero((state[1:] != state[:-1]).any(1))
    for i in range(cue.shape[1]):
        changed = boundaries[state[boundaries + 1, i] != state[boundaries, i]] + 1
        errors = np.flatnonzero(corrupt[:, i])
        for start, end in zip(changed, np.r_[changed[1:], len(cue)]):
            correct = np.flatnonzero(choices[start:end, i] == state[start, i])
            stop = start + correct[0] if len(correct) else end
            informative = np.count_nonzero(cue[start:stop, i] < 4)
            recovery.append(informative if len(correct) else informative + 1)
        for t in errors:
            if t + 1 == len(cue) or np.any(np.abs(changed - t) <= 4) or np.sum(np.abs(errors - t) <= 4) != 1:
                continue
            isolated += 1
            false += int(switches[t, i] and choices[t + 1, i] != state[t, i])
    return {"nll": float(-np.log(probabilities).mean()),
            "downstream_updates_per_observation": refreshes / cue.size,
            "cell_events_per_observation": events / cue.size,
            "median_informative_observations_to_recover": float(np.median(recovery)) if recovery else None,
            "switches": len(recovery), "isolated_corruptions": isolated,
            "false_switch_rate": false / isolated if isolated else None,
            "bytes_per_observation": {k: value / cue.size for k, value in asdict(traffic).items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--eval-streams", type=int, default=128)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--held-out", action="store_true", help="use after the spec is recorded")
    args = parser.parse_args()
    if min(args.batches, args.eval_streams) < 1:
        parser.error("development budgets must be positive")
    tables = learn_tables(np.random.default_rng(TABLE_SEED))
    if args.held_out:
        if args.batches != 12 or args.eval_streams != 128:
            parser.error("held-out uses the registered training and evaluation budgets")
        streams = [generate(np.random.default_rng(seed), 256, 1024) for seed in HELD_OUT_SEEDS]
        switch_counts = [int(np.count_nonzero(np.diff(values[2], axis=0))) for values in streams]
        if min(switch_counts) < 1000:
            raise ValueError("held-out seed generated fewer than 1,000 switches")
        evaluation = tuple(np.concatenate(field, axis=1) for field in zip(*streams))
    else:
        evaluation = generate(np.random.default_rng(DEV_SEED), args.eval_streams, 1024)
    trained = {method: train(tables, args.batches, 32, 512, method)
               for method in ("bptt", "eligibility")}
    results = {method: evaluate(params, tables, evaluation, "events") for method, params in trained.items()}
    results["before_training"] = evaluate(initial_params(), tables, evaluation, "events")
    results.update({method: evaluate(trained["bptt"], tables, evaluation, method)
                    for method in ("always_updated", "likelihood_ratio")})
    storage = {}
    for length in (64, 256):
        cue, seen, _, _ = generate(np.random.default_rng(TRAIN_SEED), 32, length)
        storage[length] = {method: gradient(params, tables, cue, seen, method)[1]
                           for method, params in trained.items()}
        storage[length]["shared_inputs"] = cue.nbytes + seen.nbytes
        storage[length]["parameters"] = trained["bptt"].nbytes + tables.nbytes
    result = {"split": "held-out" if args.held_out else "development",
              "seeds": {"tables": TABLE_SEED, "training": TRAIN_SEED,
                        "evaluation": HELD_OUT_SEEDS if args.held_out else DEV_SEED},
              "training_batches": args.batches, "evaluation_streams": evaluation[0].shape[1],
              "learning_state_bytes": storage, "conditions": results}
    if args.held_out:
        event, analog, local = (results[name] for name in ("bptt", "always_updated", "eligibility"))
        result.update(
            switches_per_seed=switch_counts,
            advance_detector=(event["nll"] - analog["nll"] <= .02
                              and event["downstream_updates_per_observation"]
                              <= .25 * analog["downstream_updates_per_observation"]
                              and event["median_informative_observations_to_recover"] <= 4
                              and event["false_switch_rate"] < .05),
            advance_local_rule=(abs(local["nll"] - event["nll"]) <= .02
                                and storage[64]["eligibility"] == storage[256]["eligibility"]),
        )
    data = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()

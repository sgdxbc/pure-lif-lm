"""CPU development screen for ordered binding. See docs/probe_binding.md."""
import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np


CAP = 32 * 1024
SCORES = 32 * 4
TAUS = (16, 64, 256, 2048, math.inf)
DEV_SEED = 101
HELD_OUT_SEEDS = (1001, 1002, 1003)


@dataclass
class Traffic:
    read: int = 0
    written: int = 0

    def bytes(self):
        return self.read + self.written


def contacts(first, second):
    return [int(a) * 128 + int(b) for a in first for b in second]


def choose(scores, tie, traffic):
    traffic.read += SCORES
    best = scores.max()
    winners = np.flatnonzero(scores == best)
    return int(winners[tie % len(winners)])


class Coincidence:
    slots = 959
    state_bytes = slots * 34 + SCORES + 4

    def __init__(self):
        self.tags = np.full(self.slots, 65535, dtype=np.uint16)
        self.values = np.zeros((self.slots, 32), dtype=np.uint8)

    def locate(self, contact, traffic):
        slot = (contact * 2654435761 % (2 ** 32)) % self.slots
        while True:
            traffic.read += 2
            tag = int(self.tags[slot])
            if tag == contact or tag == 65535:
                return slot, tag == contact
            slot = (slot + 1) % self.slots

    def write(self, first, second, value, tick, traffic):
        traffic.written += len(first)  # A leaves a trace until B arrives
        traffic.read += len(first)
        for tag in contacts(first, second):
            slot, present = self.locate(tag, traffic)
            if not present:
                self.tags[slot] = tag
                traffic.written += 2
            traffic.read += 1
            self.values[slot, value] += 1
            traffic.written += 1

    def query(self, first, second, tick, tie, traffic):
        scores = np.zeros(32, dtype=np.int32)
        traffic.written += SCORES + len(first)
        traffic.read += len(first)
        for tag in contacts(first, second):
            slot, present = self.locate(tag, traffic)
            if present:
                scores += self.values[slot]
                traffic.read += 32 + SCORES
                traffic.written += SCORES
        return choose(scores, tie, traffic)


class OuterProduct:
    rows = 246
    def __init__(self, tau):
        self.tau = tau
        self.matrix = np.zeros((self.rows, 32), dtype=np.float32)
        self.time = None if math.isinf(tau) else np.zeros(self.rows, dtype=np.int32)
        self.state_bytes = self.matrix.nbytes + (self.time.nbytes if self.time is not None else 0) + SCORES

    def features(self, first, second):
        return Counter((tag * 2654435761 % (2 ** 32)) % self.rows for tag in contacts(first, second))

    def write(self, first, second, value, tick, traffic):
        for row, mult in self.features(first, second).items():
            state = self.matrix[row]
            if self.time is None:
                traffic.read += 4
                traffic.written += 4
            else:
                state *= math.exp(-(tick - self.time[row]) / self.tau)
                self.time[row] = tick
                traffic.read += 132
                traffic.written += 132
            state[value] += mult

    def query(self, first, second, tick, tie, traffic):
        scores = np.zeros(32, dtype=np.float32)
        traffic.written += SCORES
        for row, mult in self.features(first, second).items():
            decay = 1 if self.time is None else math.exp(-(tick - self.time[row]) / self.tau)
            scores += self.matrix[row] * (mult * decay)
            traffic.read += 128 + SCORES + (4 if self.time is not None else 0)
            traffic.written += SCORES
        return choose(scores, tie, traffic)


class DigitalTable:
    slots = 128
    state_bytes = slots * 10

    def __init__(self, overlap):
        self.overlap = overlap
        self.state_bytes += self.slots if overlap else 0
        self.keys = np.zeros((self.slots, 8), dtype=np.uint8)
        self.values = np.full(self.slots, 255, dtype=np.uint8)
        self.live = np.zeros(self.slots, dtype=np.uint8)

    def locate(self, key, traffic):
        address = sum((i + 1) * a for i, a in enumerate(key)) % self.slots
        while True:
            traffic.read += 1
            if not self.live[address]:
                return address
            traffic.read += 8
            if tuple(self.keys[address]) == key:
                return address
            address = (address + 1) % self.slots

    def write(self, first, second, value, tick, traffic):
        key = (*first, *second)
        row = self.locate(key, traffic)
        self.keys[row], self.values[row], self.live[row] = key, value, 1
        traffic.written += 10

    def query(self, first, second, tick, tie, traffic):
        if self.overlap:
            # Scan occupancy and the stored addresses, then fetch the winning value.
            keys = self.keys[self.live != 0]
            traffic.read += self.slots + keys.size
            a = np.isin(keys[:, :4], first).sum(1)
            b = np.isin(keys[:, 4:], second).sum(1)
            scores = a * b
            # Each overlap score fits in a byte. Charge the scratch scores through max and tie selection.
            traffic.written += len(scores)
            traffic.read += len(scores)
            winners = np.flatnonzero(scores == scores.max())
            row = np.flatnonzero(self.live)[winners[tie % len(winners)]]
        else:
            if len(first) != 4 or len(second) != 4:
                return tie % 32
            row = self.locate((*first, *second), traffic)
        traffic.read += 1
        value = int(self.values[row])
        return value if value < 32 else tie % 32


def episode(rng, count, gap):
    code = [tuple(sorted(int(x) for x in rng.choice(128, 4, replace=False))) for _ in range(8)]
    while len(set(code)) != 8:
        code = [tuple(sorted(int(x) for x in rng.choice(128, 4, replace=False))) for _ in range(8)]
    pairs = list(combinations(range(8), 2))
    chosen = rng.permutation(len(pairs))
    original = []
    for i in chosen[:count // 2]:
        a, b = pairs[i]
        v = int(rng.integers(32))
        original.extend([(a, b, v), (b, a, (v + int(rng.integers(1, 32))) % 32)])
    rng.shuffle(original)
    decoys = []
    for i in chosen[count // 2:count // 2 + 4]:
        a, b = pairs[i]
        decoys.extend([(a, b, int(rng.integers(32))), (b, a, int(rng.integers(32)))])
    rng.shuffle(decoys)
    tick = count + gap
    bindings = [(code[a], code[b], value, i) for i, (a, b, value) in enumerate(original)]
    bindings += [(code[a], code[b], value, count + (i + 1) * gap // 9)
                 for i, (a, b, value) in enumerate(decoys)]
    a, b, expected = original[int(rng.integers(count))]
    clean = (code[a], code[b])
    damaged = list(clean)
    which, position = (int(rng.integers(n)) for n in (2, 4))
    damaged[which] = damaged[which][:position] + damaged[which][position + 1:]
    return bindings, clean, damaged, expected, tick, int(rng.integers(2 ** 30))


def evaluate(count, gap, episodes, seed=DEV_SEED, taus=TAUS):
    models = [("coincidence", Coincidence), ("timing_shuffled", Coincidence),
              ("exact_table", lambda: DigitalTable(False)), ("overlap_lookup", lambda: DigitalTable(True))]
    models.extend((f"outer_tau_{tau:g}", lambda tau=tau: OuterProduct(tau)) for tau in taus)
    totals = {name: Counter() for name, _ in models}
    states = {}
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        bindings, clean, damaged, expected, tick, tie = episode(rng, count, gap)
        for name, build in models:
            memory = build()
            states[name] = memory.state_bytes
            if memory.state_bytes > CAP:
                raise ValueError(f"{name} exceeds the state cap")
            write = Traffic()
            for i, (first, second, value, time) in enumerate(bindings):
                if name == "timing_shuffled":
                    first = bindings[(i + 1) % len(bindings)][0]
                memory.write(first, second, value, time, write)
            totals[name].update(write_read=write.read, write_written=write.written,
                                input_bytes=len(bindings) * 11 + (gap - 8) * 5)
            for condition, query in (("clean", clean), ("dropped", damaged)):
                traffic = Traffic()
                prediction = memory.query(*query, tick, tie, traffic)
                totals[name].update({condition: prediction == expected,
                                     condition + "_read": traffic.read, condition + "_written": traffic.written,
                                     condition + "_input_bytes": 2 + sum(len(part) for part in query)})
    return {name: {**{k: value / episodes for k, value in row.items()},
                   "state_bytes": states[name], "reset_bytes_per_episode": states[name]}
            for name, row in totals.items()}


def held_out(development):
    tau = float(development["selected_decay"])
    outer = f"outer_tau_{tau:g}"
    rows = {gap: [evaluate(32, gap, 1000, seed, (tau,)) for seed in HELD_OUT_SEEDS] for gap in (256, 2048)}
    pooled = {gap: {name: {metric: float(np.mean([seed[name][metric] for seed in seeds]))
                           for metric in model} for name, model in seeds[0].items()}
              for gap, seeds in rows.items()}
    relevant = ("clean", "dropped", "write", "clean_query", "dropped_query")

    def traffic(row, op):
        prefix = op.removesuffix("_query")
        return row[prefix + "_read"] + row[prefix + "_written"]

    advance = all(
        row["coincidence"]["clean"] >= .90
        and (gap != 2048 or row["coincidence"]["dropped"] >= .75)
        and row["coincidence"]["clean"] - row["timing_shuffled"]["clean"] >= .20
        and row["coincidence"]["clean"] > row[outer]["clean"]
        for gap, row in pooled.items()
    )
    overlap_matches = all(
        all(row["overlap_lookup"][metric] >= row["coincidence"][metric] for metric in relevant[:2])
        and all(traffic(row["overlap_lookup"], metric) <= traffic(row["coincidence"], metric)
                for metric in relevant[2:])
        for row in pooled.values()
    )
    return {"split": "held-out", "seeds": HELD_OUT_SEEDS, "episodes_per_seed_and_gap": 1000,
            "selected_decay": development["selected_decay"], "conditions": pooled, "by_seed": rows,
            "advance_capability": advance, "hardware_claim_closed_by_overlap": overlap_matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="development episodes per condition")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--held-out", type=Path, metavar="DEVELOPMENT_JSON", help="use after the spec is recorded")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.held_out:
        development = json.loads(args.held_out.read_text())
        if (development["split"], development["seed"], development["episodes_per_condition"]) != ("development", DEV_SEED, 100):
            parser.error("held-out selection requires the registered development run")
        result = held_out(development)
    else:
        cases = {f"{count}x{gap}": evaluate(count, gap, args.episodes) for count in (8, 16) for gap in (32, 64)}
        best = max(TAUS, key=lambda tau: (np.mean([row[f"outer_tau_{tau:g}"][condition] for row in cases.values()
                                                for condition in ("clean", "dropped")]), tau))
        result = {"split": "development", "seed": DEV_SEED, "episodes_per_condition": args.episodes,
                  "selected_decay": f"{best:g}", "conditions": cases}
    data = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()

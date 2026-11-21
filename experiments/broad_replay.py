"""Estimate whole-text NLL headroom from isolated fixed-key query witnesses."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_long import paired_bootstrap
from experiments.align_pilot import TEACHER as TEACHER_HEADS
from experiments.memory_maps import traced_forward
from experiments.name_competition import BASE, EOT, TEACHER
from experiments.query_oracle import oracle
from experiments.witness_replay import HEADS, MAX_COUNT, answer_logprobs, query_counts, query_editor, validate_counts
from experiments.write_read import Candidate, passed_candidates, window_candidates
from sample import load

GROUPS = ("eligible_names", "other_competing_repeats", "everything_else")
WINDOWS, TIMING_WINDOWS, BATCH = 1024, 32, 4
WIN_QUOTA, LOSS_QUOTA, TIMING_SOLVES = 128, 512, 128
WINDOW_SEED, SAMPLE_SEED, BOOTSTRAP_SEED = 5665, 5666, 5667
INTERNAL_SECONDS, RESERVE_SECONDS = 3450, 120


@dataclass(frozen=True)
class Position:
    window: int
    candidate: Candidate
    group: int
    win: bool


@dataclass
class Geometry:
    keys: np.ndarray
    queries: np.ndarray
    cumulative_decay: np.ndarray

    def survival(self, queries):
        return np.exp(np.minimum(0, self.cumulative_decay[:, queries, None]
                                 - self.cumulative_decay[:, None, :]))

    def wins(self, candidates):
        if not candidates:
            return []
        queries = [c.query for c in candidates]
        coefficients = (self.queries[:, queries].astype(np.float32)
                        @ self.keys.astype(np.float32).transpose(0, 2, 1))
        coefficients *= self.survival(queries)
        coefficients *= np.arange(self.keys.shape[1]) <= np.array(queries)[:, None]
        coefficients = coefficients.astype(np.float64)
        total = coefficients.sum(axis=-1, keepdims=True)
        shares = np.divide(coefficients, total, out=np.zeros_like(coefficients), where=total > 0).mean(0)
        return [bool(row[c.source] > row[c.wrong].max()) for row, c in zip(shares, candidates)]


def assign_groups(seq, candidates, teacher_passes):
    groups = np.full(seq, 2, dtype=np.int8)
    for candidate in candidates:
        groups[candidate.query] = 0 if candidate.name_source and candidate.query in teacher_passes else 1
    return groups


def draw(pool, count, rng):
    count = min(count, len(pool))
    if count == 0:
        return []
    probability = count / len(pool)
    return [(pool[i], probability) for i in rng.choice(len(pool), size=count, replace=False)]


def loss_quotas(desired, capacity):
    """Reduce both quotas proportionally, then assign any rounding remainder."""
    desired = np.asarray(desired, dtype=int)
    if capacity >= desired.sum():
        return desired.tolist()
    active = desired > 0
    if capacity < active.sum():
        raise TimeoutError("insufficient time to sample both nonempty loss strata")
    remaining = desired - active
    fractional = remaining * ((capacity - active.sum()) / remaining.sum())
    extra = fractional.astype(int)
    for index in np.argsort(-(fractional - extra), kind="stable")[:capacity - int(active.sum()) - extra.sum()]:
        extra[index] += 1
    return (active + extra).tolist()


def weighted_windows(records, num_windows):
    sums = np.zeros((num_windows, 3), dtype=np.float64)
    for position, probability, score in records:
        sums[position.window, position.group] += score["gain"] / probability
    return sums


def summarize(counts, original_nll, gains, seq):
    tokens = len(counts) * seq
    overall = paired_bootstrap(gains.sum(axis=1) / seq, seed=BOOTSTRAP_SEED)
    groups = {
        name: dict(tokens=int(counts[:, group].sum()), prevalence=float(counts[:, group].sum() / tokens),
                   original_nll_ceiling=float(original_nll[:, group].sum() / tokens),
                   gain_per_all_tokens=float(gains[:, group].sum() / tokens))
        for group, name in enumerate(GROUPS)
    }
    return dict(tokens=tokens, baseline_nll=float(original_nll.sum() / tokens), groups=groups, gain=overall,
                reopen_training=overall["mean"] >= .012 and overall["ci95"][0] > 0)


def geometries(traces, heads):
    keys = torch.stack([traces[layer].k[:, head] for layer, head in heads], dim=1)
    queries = torch.stack([traces[layer].q[:, head] for layer, head in heads], dim=1)
    decay = torch.stack([traces[layer].log_decay[0, head, 0] for layer, head in heads])
    cumulative = decay[:, None].expand(-1, keys.size(2)).cumsum(-1).cpu().numpy()
    return [Geometry(k, q, cumulative) for k, q in
            zip(keys.to(torch.uint8).cpu().numpy(), queries.to(torch.uint8).cpu().numpy())]


@torch.no_grad()
def baseline(model, teacher, windows, pieces, eot, device, deadline):
    counts = np.zeros((len(windows), 3), dtype=np.int64)
    original_nll = np.zeros_like(counts, dtype=np.float64)
    pools = {(group, win): [] for group in range(2) for win in (True, False)}
    cached = []
    layers = {layer for layer, _ in HEADS}
    for start in range(0, len(windows), BATCH):
        if time.monotonic() >= deadline:
            raise TimeoutError("baseline window scan exceeded the time available")
        batch = windows[start:start + BATCH]
        candidates = [window_candidates(ids, [pieces[t] for t in ids], eot)[1] for ids in batch.tolist()]
        names = [(stream, c) for stream, row in enumerate(candidates) for c in row if c.name_source]
        passes = [set() for _ in batch]
        for stream, candidate in passed_candidates(teacher, batch, names, TEACHER_HEADS, device):
            passes[stream].add(candidate.query)
        x = torch.as_tensor(batch, device=device)
        logits, _, traces = traced_forward(model, x[:, :-1], trace_layers=layers)
        targets = x[:, 1:].reshape(-1)
        nll = F.cross_entropy(logits.float().flatten(0, 1), targets, reduction="none")
        nll = nll.reshape(len(batch), -1).cpu().numpy()
        del logits
        geometry = geometries(traces, HEADS)
        cached.extend(geometry)
        for stream, (row, passed, geom) in enumerate(zip(candidates, passes, geometry)):
            window = start + stream
            groups = assign_groups(batch.shape[1] - 1, row, passed)
            counts[window] = np.bincount(groups, minlength=3)
            original_nll[window] = np.bincount(groups, weights=nll[stream], minlength=3)
            for candidate, win in zip(row, geom.wins(row)):
                group = int(groups[candidate.query])
                pools[group, win].append(Position(window, candidate, group, win))
    return counts, original_nll, pools, cached


def solve(position, cached):
    candidate = position.candidate
    geom = cached[position.window]
    stop = candidate.query + 1
    return oracle(geom.keys[:, :stop], geom.survival([candidate.query])[:, 0, :stop],
                  geom.queries[:, candidate.query], candidate.source, candidate.wrong)


@torch.no_grad()
def replay_batch(model, windows, positions, replacements, heads, device):
    """Each batch row is a fresh copy with exactly one query position to edit."""
    window_ids = [p.window for p in positions]
    # Keep the forward shape the same as the full-window baseline, including the last batch.
    x = torch.as_tensor(windows[window_ids + window_ids[:1] * (BATCH - len(positions))], device=device)
    streams = torch.arange(len(positions), device=device)
    queries = torch.tensor([p.candidate.query for p in positions], device=device)
    values = torch.as_tensor(np.asarray(replacements), device=device)
    answers = x[streams, queries + 1]
    logit_slice, indices = torch.unique(queries, sorted=True, return_inverse=True)
    options = dict(query_slice=logit_slice, trace_layers={layer for layer, _ in heads})
    original, _, traces = traced_forward(model, x[:, :-1], **options)
    validate_counts(query_counts(traces, heads, streams, queries).cpu().numpy(), values.cpu().numpy())
    edit = query_editor(heads, streams, queries, values)
    replayed, _, after = traced_forward(model, x[:, :-1], plastic_query=edit, **options)
    gains = answer_logprobs(replayed, streams, indices, answers) - answer_logprobs(original, streams, indices, answers)
    wins = [g.wins([p.candidate])[0] for g, p in zip(geometries(after, heads), positions)]
    return [dict(gain=gain, replay_win=win) for gain, win in zip(gains.tolist(), wins)]


def evaluate(model, windows, cached, selected, device, deadline):
    records, pending = [], []
    solve_wall = solve_cpu = replay_wall = 0.
    solves = 0
    for position, probability in selected:
        if time.monotonic() >= deadline:
            raise TimeoutError("the planned sample did not finish within the internal deadline")
        if position.win:
            records.append((position, probability, dict(gain=0., status="baseline_win")))
            continue
        wall_start, cpu_start = time.monotonic(), time.process_time()
        witness = solve(position, cached)
        solve_wall += time.monotonic() - wall_start
        solve_cpu += time.process_time() - cpu_start
        solves += 1
        if witness["status"] == "win":
            pending.append((position, probability, witness))
        else:
            records.append((position, probability, dict(gain=0., status=witness["status"])))
    for start in range(0, len(pending), BATCH):
        if time.monotonic() >= deadline:
            raise TimeoutError("the planned witness replays did not finish within the internal deadline")
        batch = pending[start:start + BATCH]
        positions, probabilities, witnesses = zip(*batch)
        wall_start = time.monotonic()
        scores = replay_batch(model, windows, positions, [w["query"] for w in witnesses], HEADS, device)
        replay_wall += time.monotonic() - wall_start
        records.extend((p, prob, dict(score, status="witness", fixed_key_margin=w["margin"]))
                       for p, prob, w, score in zip(positions, probabilities, witnesses, scores))
    if time.monotonic() >= deadline:
        raise TimeoutError("the planned sample did not finish within the internal deadline")
    return records, dict(solves=solves, oracle_wall_seconds=solve_wall, oracle_cpu_seconds=solve_cpu,
                         replay_wall_seconds=replay_wall)


def timing_sample(pools):
    rng = np.random.default_rng(WINDOW_SEED)
    sizes = [min(TIMING_SOLVES // 2, len(pools[g, False])) for g in range(2)]
    for g in range(2):
        sizes[g] += min(TIMING_SOLVES - sum(sizes), len(pools[g, False]) - sizes[g])
    return [entry for group, size in enumerate(sizes) for entry in draw(pools[group, False], size, rng)]


def run(model, teacher, windows, timing_windows, pieces, eot, device, deadline):
    _, _, pilot_pools, pilot_cache = baseline(model, teacher, timing_windows, pieces, eot, device, deadline)
    pilot, timing = evaluate(model, timing_windows, pilot_cache, timing_sample(pilot_pools), device, deadline)
    replays = sum(r["status"] == "witness" for _, _, r in pilot)
    if pilot and not replays:
        position = pilot[0][0]
        original = pilot_cache[position.window].queries[:, position.candidate.query]
        started = time.monotonic()
        replay_batch(model, timing_windows, [position], [original], HEADS, device)
        timing["identity_replay_wall_seconds"] = time.monotonic() - started
    del pilot_cache, pilot_pools
    print(json.dumps(dict(timing_pilot=timing)), flush=True)
    counts, original_nll, pools, cached = baseline(model, teacher, windows, pieces, eot, device, deadline)
    desired = [min(LOSS_QUOTA, len(pools[g, False])) for g in range(2)]
    if sum(desired) and not pilot:
        raise RuntimeError("no timing-pilot losses available to budget the main oracle sample")
    # Budget for a replay on every solve, including when few timing queries produced witnesses.
    cost = 2 * (timing["oracle_wall_seconds"] / max(1, timing["solves"])
                + timing["replay_wall_seconds"] / max(1, replays)
                + timing.get("identity_replay_wall_seconds", 0.))
    available = max(0., deadline - time.monotonic() - RESERVE_SECONDS)
    capacity = int(available / cost) if cost else sum(desired)
    quotas = loss_quotas(desired, capacity)
    rng = np.random.default_rng(SAMPLE_SEED)
    selected, strata = [], []
    for (group, win), pool in pools.items():
        sample = draw(pool, WIN_QUOTA if win else quotas[group], rng)
        selected.extend(sample)
        strata.append(dict(group=GROUPS[group], baseline_win=win, population=len(pool), sample=len(sample),
                           inclusion_probability=len(sample) / len(pool) if pool else None))
    budget = dict(full_loss_sample=desired, final_loss_sample=quotas, available_seconds=available,
                  seconds_per_loss_with_margin=cost, full_estimate_seconds=sum(desired) * cost,
                  selected_estimate_seconds=sum(quotas) * cost)
    print(json.dumps(dict(budget=budget, strata=strata)), flush=True)
    records, main_timing = evaluate(model, windows, cached, selected, device, deadline)
    gains = weighted_windows(records, len(windows))
    report = summarize(counts, original_nll, gains, windows.shape[1] - 1)
    report.update(timing_pilot=timing, main_timing=main_timing, budget=budget, strata=strata,
                  outcomes=dict(Counter(score["status"] for _, _, score in records)),
                  replay_wins=sum(score.get("replay_win", False) for _, _, score in records),
                  windows=[dict(counts=c.tolist(), original_nll=n.tolist(), weighted_gain=g.tolist())
                           for c, n, g in zip(counts, original_nll, gains)],
                  rows=[dict(window=p.window, query=p.candidate.query, source=p.candidate.source,
                             group=GROUPS[p.group], baseline_win=p.win, inclusion_probability=prob, **score)
                        for p, prob, score in records])
    return report


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    folder = ROOT / "runs" / BASE
    cfg = json.loads((folder / "args.json").read_text())
    data = ROOT / cfg["data"]
    validation = np.memmap(data / "valid.bin", dtype=np.uint16, mode="r")
    width = cfg["seq"] + 1
    offsets = width * np.random.default_rng(WINDOW_SEED).choice(
        len(validation) // width, size=WINDOWS + TIMING_WINDOWS, replace=False)
    windows = np.stack([validation[o:o + width] for o in offsets]).astype(np.int64)
    tok = Tokenizer.from_file(str(data / cfg["tokenizer"]))
    pieces = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    model, step = load(folder / "ckpt.pt", ema=True, device=device)
    if step != 20_000 or model.cfg.mem_kq_levels != MAX_COUNT or model.cfg.mem_kind != "plastic":
        raise ValueError("expected the step-20,000 four-level plastic memory checkpoint")
    # g13_tf163 was trained without EMA (args ema 0.0); every earlier screen used its raw weights.
    teacher_args = json.loads((ROOT / "runs" / TEACHER / "args.json").read_text())
    teacher, teacher_step = load(ROOT / "runs" / TEACHER / "ckpt.pt", ema=bool(teacher_args.get("ema")), device=device)
    report = run(model, teacher, windows[:WINDOWS], windows[WINDOWS:], pieces, tok.token_to_id(EOT),
                 device, start + INTERNAL_SECONDS)
    report.update(step=step, teacher_step=teacher_step, heads=HEADS, teacher_heads=TEACHER_HEADS,
                  window_offsets=offsets[:WINDOWS].tolist(), timing_window_offsets=offsets[WINDOWS:].tolist(),
                  elapsed_seconds=time.monotonic() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({k: report[k] for k in ("tokens", "groups", "gain", "reopen_training", "elapsed_seconds")},
                     indent=2))


if __name__ == "__main__":
    main()

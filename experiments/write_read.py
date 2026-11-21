"""Fixed-key retrieval diagnosis and source-supervision coverage, inference only."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.align_pilot import group_distribution
from experiments.memory_maps import LAGS, RUNS, case_batches, traced_forward
from experiments.name_competition import (
    BASE, EOT, FREQUENCY_TOKENS, SUMMARY, TEACHER, diagnostic_cases, group_shares,
    interval, lexical_words,
)
from experiments.query_oracle import oracle
from sample import load

SCREEN_WINDOWS = 4096
SCREEN_SEED = 5661
SOURCE_MIN = .20
EXPECTED_WINS = {TEACHER: 41, BASE: 33, "g39_snngla": 31, "g38_la": 38}
BASELINE = ROOT / "runs/name_competition.json"
ALIGN_PLAN = ROOT / "runs/align_pilot/plan.json"


def similarity(source, other, count_keys=False):
    a, b = np.asarray(source, dtype=np.float64), np.asarray(other, dtype=np.float64)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    result = dict(source_norm=norm_a, other_norm=norm_b, source_silent=norm_a == 0,
                  other_silent=norm_b == 0, exact_equal=bool(np.array_equal(a, b)),
                  cosine=float(a @ b / (norm_a * norm_b)) if norm_a and norm_b else None)
    if count_keys:
        union = np.count_nonzero((a > 0) | (b > 0))
        result["active_jaccard"] = float(np.count_nonzero((a > 0) & (b > 0)) / union) if union else None
    return result


def rerank(coeff, match, source, wrong, recurrent=True):
    def rank(row):
        shares = group_shares(row)
        rival = wrong[int(np.argmax(shares[wrong]))]
        margin = float(shares[source] - shares[rival])
        return dict(win=margin > 0, margin=margin, best_wrong=rival)
    return rank(coeff), rank(match if recurrent else coeff)


def decision(original, no_decay, observed_without_decay):
    read = original["status"] == "win"
    retention = observed_without_decay or (
        original["status"] == "impossible" and no_decay["status"] == "win")
    write = no_decay["status"] == "impossible"
    paths = [label for label, flag in (("read_side_headroom", read), ("retention", retention),
                                      ("write_code_limit", write)) if flag]
    return dict(label=paths[0] if len(paths) == 1 else "mixed" if paths else "inconclusive",
                indications=paths)


def describe_passage(meta, keys, query, coeff, match, survival, count_keys):
    source, wrong = meta["source"], meta["wrong"]
    actual, neutral = rerank(coeff, match, source, wrong, recurrent=survival is not None)
    rival = actual["best_wrong"]
    key_pairs = {p: [similarity(k[source], k[p], count_keys) for k in keys]
                 for p in sorted(set(wrong) | {ctrl for _, ctrl in meta["pairs"]})}
    row = dict(label=meta["label"], actual=actual, unit_survival=neutral, key_pairs=key_pairs,
               raw_margin_per_head=(match[:, source] - match[:, wrong].max(axis=1)).tolist(),
               raw_margin_vs_actual_rival=(match[:, source] - match[:, rival]).tolist())
    if survival is not None:
        row["survival_source_vs_actual_rival"] = survival[:, [source, rival]].tolist()
        pre_decay = group_shares(match)
        row["share_margin_change_from_decay_vs_actual_rival"] = (
            actual["margin"] - float(pre_decay[source] - pre_decay[rival]))
    if count_keys and not actual["win"]:
        original = oracle(keys, survival, query, source, wrong)
        no_decay = oracle(keys, np.ones_like(survival), query, source, wrong)
        row["oracle"] = dict(actual=original, unit_survival=no_decay,
                             decision=decision(original, no_decay, neutral["win"]))
    return row


@torch.no_grad()
def diagnose(model, run, cases, meta, heads, device, eot):
    rows = {}
    layers = {layer for layer, _ in heads}
    # Preserve the diagnostic's original padding and batch composition.
    for indices, block in case_batches(cases, 4):
        if all(meta[i] is None for i in indices):
            continue
        width = len(block[0].ids)
        x = torch.tensor([c.ids + [eot] * (width - len(c.ids)) for c in block], device=device)
        qpos = torch.tensor([[c.query] for c in block], device=device)
        _, _, traces = traced_forward(model, x, query_slice=slice(0, 0), trace_layers=layers)
        parts = {layer: trace.parts(qpos) for layer, trace in traces.items()}
        decay = None
        coeff = torch.stack([parts[layer][0][:, h, 0] for layer, h in heads], 1).cpu().numpy()
        match = torch.stack([parts[layer][2][:, h, 0] for layer, h in heads], 1).cpu().numpy()
        if run != TEACHER:
            decay = torch.stack([parts[layer][1][:, h, 0].exp() for layer, h in heads], 1).cpu().numpy()
        keys = torch.stack([traces[layer].k[:, h] for layer, h in heads], 1).cpu().numpy()
        query = torch.stack([traces[layer].q[torch.arange(len(block), device=device), h, qpos[:, 0]]
                             for layer, h in heads], 1).cpu().numpy()
        for j, i in enumerate(indices):
            if meta[i] is not None:
                n = len(block[j].ids)
                rows[i] = describe_passage(meta[i], keys[j, :, :n], query[j], coeff[j, :, :n], match[j, :, :n],
                                           None if decay is None else decay[j, :, :n], run == BASE)
    return [rows[i] for i, item in enumerate(meta) if item is not None]


def summarize(rows, meta, expected, recorded):
    wins = sum(r["actual"]["win"] for r in rows)
    if wins != expected:
        raise ValueError(f"recorded {expected} wins but reproduced {wins}")
    delta = [int(r["unit_survival"]["win"]) - int(r["actual"]["win"]) for r in rows]
    similarity_excess = {}
    for metric in ("cosine", "active_jaccard"):
        passage_means = {}
        for row, item in zip(rows, meta):
            pairs = [(a.get(metric), b.get(metric)) for w, c in item["pairs"]
                     for a, b in zip(row["key_pairs"][w], row["key_pairs"][c])]
            excess = [a - b for a, b in pairs if a is not None and b is not None]
            if excess:
                passage_means[row["label"]] = float(np.mean(excess))
        if passage_means:
            similarity_excess[metric] = dict(by_passage=passage_means, interval=interval(list(passage_means.values())))
    wrong_pairs = [p for row, item in zip(rows, meta) for w in item["wrong"] for p in row["key_pairs"][w]]
    geometry = {m: sum(bool(p[m]) for p in wrong_pairs)
                for m in ("source_silent", "other_silent", "exact_equal")}
    geometry["head_write_pairs"] = len(wrong_pairs)
    geometry["undefined_cosine"] = sum(p["cosine"] is None for p in wrong_pairs)
    if wrong_pairs and "active_jaccard" in wrong_pairs[0]:
        geometry["undefined_active_jaccard"] = sum(p["active_jaccard"] is None for p in wrong_pairs)
        geometry["nonzero_exact_count_collisions"] = sum(p["exact_equal"] and not p["source_silent"]
                                                         for p in wrong_pairs)
    for m in ("source_norm", "other_norm", "cosine", "active_jaccard"):
        available = [p[m] for p in wrong_pairs if p.get(m) is not None]
        geometry[m + "_mean"] = float(np.mean(available)) if available else None
    oracles = [r["oracle"] for r in rows if "oracle" in r]
    outcomes = {mode: dict(Counter(r[mode]["status"] for r in oracles))
                for mode in ("actual", "unit_survival")} if oracles else {}
    return dict(wins=wins, unit_survival_wins=wins + sum(delta), loss_to_win=delta.count(1),
                win_to_loss=delta.count(-1), survival_removed_minus_actual=interval(delta),
                recorded_disagreements=[r["label"] for r, x in zip(rows, recorded) if r["actual"]["win"] != x],
                key_geometry=geometry, similarity_excess=similarity_excess,
                oracle_outcomes=outcomes, oracle_decisions=dict(Counter(r["decision"]["label"] for r in oracles)))


@dataclass
class Candidate:
    query: int
    source: int
    wrong: list[int]
    name_source: bool


def window_candidates(ids, pieces, eot):
    """Match visible within-document sources and names before applying teacher scores."""
    words, initial = lexical_words(pieces)
    last, title_count, lower_count, title_positions = {}, Counter(), Counter(), []
    candidates, unique = [], 0
    for q, (current, target) in enumerate(zip(ids[:-1], ids[1:])):
        if current == eot:
            last.clear()
            title_count.clear()
            lower_count.clear()
            title_positions.clear()
            continue
        pos = q - 1
        if pos >= 0 and ids[pos] != eot:
            last[ids[pos]] = pos if ids[pos] not in last else -1
            word = words.get(pos, "")
            if word.istitle():
                title_count[word] += 1
                title_positions.append(pos)
            elif word.islower():
                lower_count[word] += 1
        source = last.get(target, -1)
        if target == current or target == eot or source < 0:
            continue
        unique += 1
        answer = words.get(source, "").casefold()
        names = [p for p in title_positions if p not in initial or
                 (title_count[words[p]] > 1 and lower_count[words[p].lower()] == 0)]
        wrong = [p for p in names if ids[p] != target and words[p].casefold() != answer]
        if wrong:
            candidates.append(Candidate(q, source, wrong, source in names))
    return unique, candidates


def teacher_hits(probabilities, candidates):
    if not candidates:
        return []
    index = torch.arange(len(candidates), device=probabilities.device)
    source = probabilities[index, [c.source for c in candidates]]
    width = max(len(c.wrong) for c in candidates)
    wrong = [c.wrong + [c.wrong[0]] * (width - len(c.wrong)) for c in candidates]
    best = probabilities.gather(1, torch.as_tensor(wrong, device=probabilities.device)).amax(1)
    return ((source >= SOURCE_MIN) & (source > best)).cpu().tolist()


def screen_offsets(n_tokens, seq, old_offsets, count=SCREEN_WINDOWS):
    size = seq + 1
    blocks = np.arange(n_tokens // size) * size
    old = np.sort(np.ravel(old_offsets))
    nearby = np.searchsorted(old, blocks - seq)
    overlap = np.r_[old, n_tokens + size][nearby] <= blocks + seq
    choices = blocks[(blocks >= FREQUENCY_TOKENS) & ~overlap]
    return np.random.default_rng(SCREEN_SEED).choice(choices, count, replace=False)


def lag_summary(lags, tokens):
    edges = [*LAGS, LAGS[-1] + 1]
    return dict(count=len(lags), per_million=len(lags) / tokens * 1e6, edges=edges,
                histogram=np.histogram(lags, bins=edges)[0].tolist(),
                quantiles=dict(zip(("p10", "p50", "p90"), np.quantile(lags, [.1, .5, .9]).tolist())) if lags else None)


@torch.no_grad()
def screen(teacher, train, tok, heads, device, seq, eot):
    plan = json.loads(ALIGN_PLAN.read_text())
    offsets = screen_offsets(len(train), seq, plan["train_offsets"])
    lags = {key: [] for key in ("unique_with_wrong_name", "name_source_with_wrong_name",
                                "teacher_pass", "name_source_teacher_pass")}
    windows = np.zeros(len(offsets), dtype=int)
    unique = 0
    pieces = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    layers = {layer for layer, _ in heads}
    for start in range(0, len(offsets), 8):
        batch = np.stack([train[o:o + seq + 1] for o in offsets[start:start + 8]]).astype(np.int64)
        indexed = []
        for stream, ids in enumerate(batch.tolist()):
            n, candidates = window_candidates(ids, [pieces[t] for t in ids], eot)
            unique += n
            indexed.extend((stream, c) for c in candidates)
            for c in candidates:
                lags["unique_with_wrong_name"].append(c.query - c.source)
                if c.name_source:
                    lags["name_source_with_wrong_name"].append(c.query - c.source)
        if not indexed:
            continue
        x = torch.as_tensor(batch[:, :-1], device=device)
        _, _, traces = traced_forward(teacher, x, query_slice=slice(0, 0), trace_layers=layers)
        for i in range(0, len(indexed), 64):
            chunk = indexed[i:i + 64]
            q = torch.tensor([[b, c.query] for b, c in chunk], device=device)
            group = group_distribution(traces, heads, q, teacher=True)
            for (stream, c), passed in zip(chunk, teacher_hits(group, [c for _, c in chunk])):
                if passed:
                    lags["teacher_pass"].append(c.query - c.source)
                    if c.name_source:
                        lags["name_source_teacher_pass"].append(c.query - c.source)
                        windows[start + stream] += 1
    tokens = len(offsets) * seq
    per_window = interval(windows.tolist())
    forecast = [x * 32_000 for x in per_window["ci95"]]
    nonempty = float((windows.reshape(-1, 32).sum(axis=1) > 0).mean())
    return dict(windows=len(offsets), tokens=tokens, seed=SCREEN_SEED, min_teacher_source_share=SOURCE_MIN,
                offsets=offsets.tolist(), unique_source=unique,
                unique_source_per_million=unique / tokens * 1e6,
                stages={key: lag_summary(values, tokens) for key, values in lags.items()},
                forecast_name_source_at_32000=dict(mean=per_window["mean"] * 32_000, ci95=forecast),
                name_source_nonempty_batch_fraction=nonempty, adequate=forecast[0] >= 4000 and nonempty >= .90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    diag = json.loads(SUMMARY.read_text())
    baseline = json.loads(BASELINE.read_text())
    info = {run: json.loads((ROOT / "runs" / run / "args.json").read_text()) for run in RUNS}
    settings = {(a["data"], a["tokenizer"], a["seq"]) for a in info.values()}
    (data, tokenizer, seq), = settings
    tok = Tokenizer.from_file(str(ROOT / data / tokenizer))
    eot = tok.token_to_id(EOT)
    cases = diagnostic_cases(tok, seq, diag)
    records = baseline["passages"]
    if [case.label for case in cases] != [r["label"] for r in records]:
        raise ValueError("recorded passages differ from the 5.63 holdout")
    meta = [r if r["name_answer"] and r["wrong"] else None for r in records]
    selected = [r for r in meta if r is not None]
    if len(selected) != 57 or any(case.source != r["source"] for case, r in zip(cases, records)):
        raise ValueError("expected the recorded 57 passages and source positions")

    report = dict(models={}, passages=selected)
    for run in (*[r for r in RUNS if r != TEACHER], TEACHER):
        print(f"diagnosing {run}", flush=True)
        model, step = load(ROOT / "runs" / run / "ckpt.pt", ema=bool(info[run].get("ema")), device=device)
        heads = diag["models"][run]["intervention"]["heads"]
        if step != baseline["models"][run]["step"]:
            raise ValueError(f"checkpoint step differs from the recorded diagnostic for {run}")
        if heads != baseline["models"][run]["heads"]:
            raise ValueError(f"recorded heads differ for {run}")
        rows = diagnose(model, run, cases, meta, heads, device, eot)
        old = [row["correct_beats_best_wrong"] for row in baseline["models"][run]["passage_scores"]
               if row["correct_beats_best_wrong"] is not None]
        report["models"][run] = dict(step=step, heads=heads, rows=rows,
                                     **summarize(rows, selected, EXPECTED_WINS[run], old))
        if run == TEACHER:
            print("screening fresh FineWeb windows", flush=True)
            train = np.memmap(ROOT / data / "train.bin", dtype=np.uint16, mode="r")
            report["screen"] = screen(model, train, tok, heads, device, seq, eot)
        del model
    teacher = report["models"][TEACHER]["similarity_excess"].get("cosine", {}).get("by_passage", {})
    for run, scores in report["models"].items():
        if run != TEACHER:
            excess = scores["similarity_excess"].get("cosine", {}).get("by_passage", {})
            common = [label for label in excess if label in teacher]
            scores["cosine_excess_minus_teacher"] = dict(
                passages=len(common), interval=interval([excess[label] - teacher[label] for label in common]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

"""Measure name competition on the recorded 5.63 LAMBADA holdout."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from tokenizers import Tokenizer, decoders

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_long import paired_bootstrap
from experiments.memory_maps import RUNS, case_batches, natural_cases, traced_forward
from sample import load

BASE = "g44_all3"
TEACHER = "g13_tf163"
SUMMARY = ROOT / "runs/memory_maps/summary.json"
FREQUENCY_TOKENS = 10_000_000
MAX_LAG_GAP = 16
EOT = "<|endoftext|>"
WORD = re.compile(r" ?([A-Za-z]{2,})")


def lexical_words(pieces):
    """Full single-token words and positions at sentence starts."""
    decoder = decoders.ByteLevel()
    words, initial = {}, set()
    decoded = {raw: decoder.decode([raw]) for raw in set(pieces)}
    for pos, raw in enumerate(pieces):
        piece = decoded[raw]
        match = WORD.fullmatch(piece)
        if not match:
            continue
        following = decoded[pieces[pos + 1]] if pos + 1 < len(pieces) else ""
        if following and following[0].isalnum():
            continue
        preceding = decoder.decode(list(pieces[:pos])) if not piece.startswith(" ") or match[1][0].isupper() else ""
        if not piece.startswith(" ") and preceding and preceding[-1].isalnum():
            continue
        words[pos] = match[1]
        if not match[1][0].isupper():
            continue
        previous = preceding.rstrip(" \t\r\"'“”‘’([{")
        if not previous or previous.endswith((EOT, ".", "?", "!", "\n")):
            initial.add(pos)
    return words, initial


def lexical_positions(pieces):
    """Full single-token words and name-like positions in a visible GPT-2 prefix."""
    words, initial = lexical_words(pieces)
    title = {pos: word for pos, word in words.items() if word[0].isupper() and word[1:].islower()}
    repeated = Counter(title.values())
    lowercase = {word for word in words.values() if word.islower()}
    names = {pos for pos, word in title.items()
             if pos not in initial or (repeated[word] > 1 and word.lower() not in lowercase)}
    return words, names


def match_controls(ids, query, wrong, controls, counts, max_gap=MAX_LAG_GAP):
    """Match writes using the lag distance, count bin, count distance and position."""
    def freq(pos):
        return int(counts[ids[pos]])
    bins = {}
    for pos in controls:
        if pos < query:
            bins.setdefault((freq(pos) + 1).bit_length() - 1, []).append(pos)
    pairs = []
    for pos in wrong:
        count = freq(pos)
        eligible = [c for c in bins.get((count + 1).bit_length() - 1, ())
                    if abs(pos - c) <= max_gap]
        if eligible:
            control = min(eligible, key=lambda c: (abs(pos - c), abs(count - freq(c)), c))
            pairs.append((pos, control))
    return pairs


@dataclass
class Passage:
    label: str
    source: int
    name_answer: bool
    wrong: list[int]
    pairs: list[tuple[int, int]]


def describe(case, pieces, counts):
    words, names = lexical_positions(pieces)
    answer = words.get(case.source, "").casefold()
    # The diagnostic identifies the earlier answer by token ID. The lexical
    # comparison also excludes occurrences with a different case or space.
    name_answer = case.source in names
    wrong = [p for p in sorted(names) if p < case.query and words[p].casefold() != answer
             and case.ids[p] != case.answer]
    spellings = {words[p].casefold() for p in names}
    controls = [p for p, word in words.items() if p < case.query and word.islower()
                and word != answer and word not in spellings and case.ids[p] != case.answer]
    pairs = match_controls(case.ids, case.query, wrong, controls, counts) if name_answer else []
    return Passage(case.label, case.source, name_answer, wrong, pairs)


def group_shares(coefficients):
    """Normalize each head's causal row and average head shares."""
    mass = np.abs(np.asarray(coefficients, dtype=np.float64))
    totals = mass.sum(axis=-1, keepdims=True)
    return np.divide(mass, totals, out=np.zeros_like(mass), where=totals != 0).mean(axis=-2)


def passage_metrics(passage, shares):
    wrong = shares[passage.wrong]
    correct = float(shares[passage.source])
    rows = [[float(shares[a]), float(shares[b])] for a, b in passage.pairs]
    return dict(correct_share=correct, wrong_shares=wrong.tolist(), pair_shares=rows,
                correct_beats_best_wrong=(float(correct > wrong.max())
                                          if passage.name_answer and len(wrong) else None),
                competition_effect=float(np.mean([name - control for name, control in rows])) if rows else None)


def interval(values):
    return paired_bootstrap(values, seed=0, draws=10_000) if values else None


METRICS = ("correct_beats_best_wrong", "competition_effect")


def summarize(rows, teacher_rows=None):
    result = {}
    for metric in METRICS:
        eligible = [i for i, row in enumerate(rows) if row[metric] is not None]
        stat = interval([rows[i][metric] for i in eligible])
        result[metric] = dict(passages=len(eligible), interval=stat)
        if teacher_rows is not None:
            result[metric]["minus_teacher"] = interval([rows[i][metric] - teacher_rows[i][metric]
                                                        for i in eligible])
    competition = result["competition_effect"]["interval"]
    result["general_name_competition"] = bool(competition and competition["ci95"][0] > 0)
    return result


@torch.no_grad()
def measure(model, cases, passages, heads, device, eot, batch_size):
    rows = [None] * len(cases)
    layers = {layer for layer, _ in heads}
    for indices, block in case_batches(cases, batch_size):
        width = max(len(case.ids) for case in block)
        x = torch.tensor([case.ids + [eot] * (width - len(case.ids)) for case in block], device=device)
        query = torch.tensor([[case.query] for case in block], device=device)
        _, _, traces = traced_forward(model, x, query_slice=slice(0, 0), trace_layers=layers)
        coeff = {layer: trace.parts(query)[0][:, :, 0] for layer, trace in traces.items()}
        shares = group_shares(torch.stack([coeff[layer][:, head] for layer, head in heads], dim=1).cpu().numpy())
        for j, idx in enumerate(indices):
            rows[idx] = passage_metrics(passages[idx], shares[j, :len(block[j].ids)])
    return rows


def diagnostic_cases(tok, seq, diagnostic):
    labels = diagnostic["held_out_passages"]
    if (diagnostic["intervention_set"] != "lambada" or len(labels) != 192 or len(set(labels)) != 192
            or any(not label.startswith("lambada:") for label in labels)
            or set(labels).intersection(diagnostic["dev_passages"])):
        raise ValueError("expected the 192 distinct 5.63 held-out LAMBADA IDs")
    recovered = {case.label: case for case in natural_cases(tok, seq, 0, labels=set(labels))}
    if set(recovered) != set(labels):
        raise ValueError(f"could not reconstruct held-out IDs: {sorted(set(labels) - recovered.keys())}")
    return [recovered[label] for label in labels]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, action="append", default=[],
                        help="also score an alignment-pilot checkpoint with the all3 heads; repeat for each arm")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if len({path.resolve() for path in args.ckpt}) != len(args.ckpt):
        parser.error("pass each optional checkpoint once")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    diagnostic = json.loads(SUMMARY.read_text())
    info = {run: json.loads((ROOT / "runs" / run / "args.json").read_text()) for run in RUNS}
    settings = {(a["data"], a["tokenizer"], a["seq"]) for a in info.values()}
    if len(settings) != 1:
        raise ValueError("expected all 5.63 models to share data, tokenizer and sequence length")
    (data, tokenizer, seq), = settings
    tok = Tokenizer.from_file(str(ROOT / data / tokenizer))
    train = np.memmap(ROOT / data / "train.bin", dtype=np.uint16, mode="r")
    if len(train) < FREQUENCY_TOKENS:
        raise ValueError("FineWeb train.bin is shorter than the registered unigram window")
    counts = np.bincount(train[:FREQUENCY_TOKENS], minlength=tok.get_vocab_size())
    cases = diagnostic_cases(tok, seq, diagnostic)
    passages = [describe(case, [tok.id_to_token(t) for t in case.ids], counts) for case in cases]
    coverage = dict(passages=len(passages), name_answers=sum(p.name_answer for p in passages),
                    no_wrong_name=sum(not p.wrong for p in passages),
                    name_answers_no_wrong_name=sum(p.name_answer and not p.wrong for p in passages),
                    eligible_pairs=sum(len(p.pairs) for p in passages),
                    unmatched_wrong_writes=sum(len(p.wrong) - len(p.pairs) for p in passages if p.name_answer))
    report = dict(coverage=coverage, frequency_tokens=FREQUENCY_TOKENS, max_lag_gap=MAX_LAG_GAP,
                  passages=[asdict(p) for p in passages], models={})
    jobs = [(run, ROOT / "runs" / run / "ckpt.pt", run, bool(info[run].get("ema"))) for run in RUNS]
    jobs.extend((f"{BASE}:{path}", path, BASE, False) for path in args.ckpt)
    base_cfg = None
    for label, path, head_run, ema in jobs:
        heads = diagnostic["models"][head_run]["intervention"]["heads"]
        if len(heads) != 4 or len(set(map(tuple, heads))) != 4:
            raise ValueError(f"expected four distinct recorded heads for {head_run}")
        print(f"scoring {label}", flush=True)
        model, step = load(path, ema=ema, device=device)
        if label == BASE:
            base_cfg = asdict(model.cfg)
        elif head_run == BASE and asdict(model.cfg) != base_cfg:
            raise ValueError(f"{path} does not have the all3 model configuration")
        rows = measure(model, cases, passages, heads, device, tok.token_to_id(EOT), args.batch)
        del model
        reference = report["models"][TEACHER]["passage_scores"] if label != TEACHER else None
        report["models"][label] = dict(step=step, heads=heads, ckpt=str(path), ema=ema, passage_scores=rows,
                                       **summarize(rows, reference))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

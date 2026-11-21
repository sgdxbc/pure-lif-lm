"""Replay the saved legal queries in the g44_all3 forward pass."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_long import paired_bootstrap
from experiments.memory_maps import case_batches, traced_forward
from experiments.name_competition import BASE, EOT, SUMMARY, diagnostic_cases, group_shares
from sample import load

WITNESSES = ROOT / "runs/write_read.json"
HEADS = ((9, 3), (10, 5), (11, 2), (11, 3))
ARMS = ("original", "witness", "sham")
MAX_COUNT = 4
SEED = 5664


def validate_counts(original, replacement):
    if original.shape != replacement.shape:
        raise ValueError("query count vectors differ in shape")
    for counts in (original, replacement):
        if not ((counts >= 0) & (counts <= MAX_COUNT) & (counts == np.floor(counts))).all():
            raise ValueError("expected integer query counts from 0 through 4")
    if not np.array_equal(original.sum(-1), replacement.sum(-1)):
        raise ValueError("query replacement changes a head's count budget")


def draw_sham(original, witness, rng):
    """Sample sequentially from feasible counts at the same budget and distance."""
    validate_counts(original, witness)
    original = original.astype(np.int64)
    moves = int(np.abs(witness - original).sum()) // 2
    if not moves:
        return original.copy()
    order = rng.permutation(len(original))
    width = moves + 1
    feasible = np.zeros((len(order) + 1, width, width), dtype=bool)
    feasible[-1, 0, 0] = True
    for j in range(len(order) - 1, -1, -1):
        for value in range(MAX_COUNT + 1):
            delta = value - original[order[j]]
            up, down = max(0, delta), max(0, -delta)
            if up <= moves and down <= moves:
                feasible[j, up:, down:] |= feasible[j + 1, :width - up, :width - down]
    sham = original.copy()
    up, down = moves, moves
    for j, i in enumerate(order):
        options = []
        for value in range(MAX_COUNT + 1):
            delta = value - original[i]
            u, d = up - max(0, delta), down - max(0, -delta)
            if min(u, d) >= 0 and feasible[j + 1, u, d]:
                options.append((value, u, d))
        value, up, down = options[int(rng.integers(len(options)))]
        sham[i] = value
    validate_counts(original, sham)
    if np.abs(sham - original).sum() != 2 * moves:
        raise AssertionError("sham does not preserve query edit size")
    return sham


def query_editor(heads, streams, queries, counts):
    """Replace the selected plastic scan inputs at the selected query positions."""
    streams, queries = streams[:, None], queries[:, None]
    by_layer = {}
    for layer in sorted({layer for layer, _ in heads}):
        selected = [(i, head) for i, (l, head) in enumerate(heads) if l == layer]
        head_ids = torch.tensor([h for _, h in selected], device=counts.device)[None, :]
        by_layer[layer] = (head_ids, counts[:, [i for i, _ in selected]])

    def edit(layer, q):
        if layer not in by_layer:
            return q
        head_ids, values = by_layer[layer]
        values = values.to(q.dtype)
        if torch.equal(q[streams, head_ids, queries], values):
            return q
        # Preserve the scan input layout and leave every other query intact.
        edited = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device).copy_(q)
        edited[streams, head_ids, queries] = values
        return edited

    return edit


def query_counts(traces, heads, streams, queries):
    return torch.stack([traces[layer].q[streams, head, queries] for layer, head in heads], dim=1)


def answer_logprobs(logits, streams, positions, answers):
    return logits[streams, positions].float().log_softmax(-1).gather(1, answers[:, None]).squeeze(1)


def rank_rows(traces, heads, all_queries, streams, cases, metadata):
    parts = {layer: trace.parts(all_queries[:, None])[0] for layer, trace in traces.items()}
    coeff = torch.stack([parts[layer][streams, head, 0] for layer, head in heads], dim=1).cpu().numpy()
    ranks = []
    for case, meta, row in zip(cases, metadata, coeff):
        shares = group_shares(row[:, :len(case.ids)])
        best = max(meta["wrong"], key=lambda pos: shares[pos])
        margin = float(shares[case.source] - shares[best])
        ranks.append(dict(win=margin > 0, margin=margin, best_wrong=best))
    return ranks


@torch.no_grad()
def replay(model, cases, loss_rows, heads, device, eot):
    layers = {layer for layer, _ in heads}
    outputs = {}
    for indices, block in case_batches(cases, 4):
        active = [(j, i, case, loss_rows[case.label]) for j, (i, case) in enumerate(zip(indices, block))
                  if case.label in loss_rows]
        if not active:
            continue
        width = len(block[0].ids)
        x = torch.tensor([c.ids + [eot] * (width - len(c.ids)) for c in block], device=device)
        all_queries = torch.tensor([c.query for c in block], device=device)
        streams = torch.tensor([j for j, _, _, _ in active], device=device)
        queries = all_queries[streams]
        selected = [c for _, _, c, _ in active]
        metadata = [meta for _, _, _, meta in active]
        answers = torch.tensor([c.answer for c in selected], device=device)
        logit_slice, positions = torch.unique(queries, sorted=True, return_inverse=True)
        forward_opts = dict(query_slice=logit_slice, trace_layers=layers)
        baseline, _, traces = traced_forward(model, x, **forward_opts)
        original = query_counts(traces, heads, streams, queries).cpu().numpy()
        witness = np.asarray([meta["query"] for meta in metadata])
        validate_counts(original, witness)
        fixed_edit = query_editor(heads, streams, queries, torch.as_tensor(witness, device=device))
        fixed_traces = {layer: replace(trace, q=fixed_edit(layer, trace.q)) for layer, trace in traces.items()}
        fixed = rank_rows(fixed_traces, heads, all_queries, streams, selected, metadata)
        for case, rank in zip(selected, fixed):
            if not rank["win"]:
                raise ValueError(f"saved witness did not reproduce a fixed-key win for {case.label}")
        sham = np.stack([
            [draw_sham(a, b, np.random.default_rng(np.random.SeedSequence((SEED, i, layer, h))))
             for a, b, (layer, h) in zip(o, w, heads)]
            for (_, i, _, _), o, w in zip(active, original, witness)
        ])
        arms = {}
        for arm, counts in zip(ARMS, (original, witness, sham)):
            values = torch.as_tensor(counts, device=device)
            edit = query_editor(heads, streams, queries, values)
            logits, _, traces = traced_forward(model, x, plastic_query=edit, **forward_opts)
            if arm == "original" and not torch.equal(baseline[streams, positions], logits[streams, positions]):
                raise ValueError("replaying the original queries changed the answer-position logits")
            if not torch.equal(query_counts(traces, heads, streams, queries), values):
                raise ValueError(f"{arm} did not replay the specified query counts")
            ranks = rank_rows(traces, heads, all_queries, streams, selected, metadata)
            score = answer_logprobs(logits, streams, positions, answers).tolist()
            arms[arm] = [dict(logprob=s, **r) for s, r in zip(score, ranks)]
        for i, case in enumerate(selected):
            if arms["original"][i]["win"]:
                raise ValueError(f"original rank win on recorded loss {case.label}")
            outputs[case.label] = dict(
                source=case.source, wrong=metadata[i]["wrong"], query_position=case.query,
                oracle_margin=metadata[i]["oracle_margin"], fixed_key_margin=fixed[i]["margin"],
                budget=original[i].sum(-1).astype(int).tolist(),
                l1_edit=np.abs(witness[i] - original[i]).sum(-1).astype(int).tolist(),
                sham_equals_witness=(sham[i] == witness[i]).all(-1).tolist(),
                **{arm: arms[arm][i] for arm in ARMS})
    return outputs


def saved_losses(saved, cases):
    model = saved["models"][BASE]
    if tuple(map(tuple, model["heads"])) != HEADS or model["step"] != 20_000:
        raise ValueError("expected the recorded all3 checkpoint and four heads")
    meta = {row["label"]: row for row in saved["passages"]}
    labels = {c.label: c for c in cases}
    losses = {}
    for row in model["rows"]:
        if row["actual"]["win"]:
            continue
        label = row["label"]
        source, wrong = meta[label]["source"], meta[label]["wrong"]
        c = labels[label]
        oracle = row["oracle"]["actual"]
        if (oracle["status"] != "win" or source != c.source or not wrong or source in wrong
                or any(not 0 <= pos < c.query for pos in wrong)):
            raise ValueError(f"expected a saved original-survival witness and ranking positions for {label}")
        losses[label] = dict(wrong=wrong, query=oracle["query"], oracle_margin=oracle["margin"])
    if len(losses) != 24:
        raise ValueError("expected exactly 24 recorded original losses")
    return losses


def summarize(rows):
    gains = {arm: paired_bootstrap([r["witness"]["logprob"] - r[arm]["logprob"] for r in rows.values()], seed=SEED)
             for arm in ("original", "sham")}
    return dict(passages=len(rows), seed=SEED,
                ranking_wins={arm: sum(r[arm]["win"] for r in rows.values()) for arm in ARMS},
                sham_witness_matching_heads=sum(sum(r["sham_equals_witness"]) for r in rows.values()),
                sham_witness_matching_nonzero_edits=sum(
                    sum(equal and size > 0 for equal, size in zip(r["sham_equals_witness"], r["l1_edit"]))
                    for r in rows.values()),
                witness_logprob_gain=gains,
                reopen_discussion=all(g["mean"] >= .20 and g["ci95"][0] > 0 for g in gains.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    run = ROOT / "runs" / BASE
    cfg = json.loads((run / "args.json").read_text())
    tokenizer = Tokenizer.from_file(str(ROOT / cfg["data"] / cfg["tokenizer"]))
    cases = diagnostic_cases(tokenizer, cfg["seq"], json.loads(SUMMARY.read_text()))
    losses = saved_losses(json.loads(WITNESSES.read_text()), cases)
    model, step = load(run / "ckpt.pt", ema=True, device=device)
    if step != 20_000 or model.cfg.mem_kq_levels != MAX_COUNT or model.cfg.mem_kind != "plastic":
        raise ValueError("expected the step-20,000 four-level plastic memory checkpoint")
    scores = replay(model, cases, losses, HEADS, device, tokenizer.token_to_id(EOT))
    rows = {label: scores[label] for label in losses}
    report = summarize(rows)
    print(json.dumps(report, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(step=step, heads=HEADS, **report, rows=rows), indent=2,
                                   allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

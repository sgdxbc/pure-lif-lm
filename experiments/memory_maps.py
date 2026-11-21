"""Section 5.63 inference-only memory maps and paired interventions.

  python3 experiments/memory_maps.py --device cuda --out runs/memory_maps --intervention-set lambada

Run from the repository root. Uses the existing checkpoints and LAMBADA dataset
(probe_lambada.py). Coefficients describe writes, conditional on the realized
queries, keys and decays. A write can already incorporate an earlier token.
The forward and reconstruction use FP32, including the vocabulary score.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model as snn
import model_tf as tf
from eval_long import paired_bootstrap
from probe_lambada import passages
from sample import load


RUNS = ("g13_tf163", "g44_all3", "g39_snngla", "g38_la")
LAGS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@dataclass
class Read:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    log_decay: torch.Tensor | None
    actual: torch.Tensor
    incoming: torch.Tensor | None = None

    def positions(self, queries=None):
        if queries is None:
            queries = torch.arange(self.q.size(2), device=self.q.device)[None, :].expand(self.q.size(0), -1)
        return queries

    def parts(self, queries=None):
        """Coefficients, log survival and raw dot products for selected query rows."""
        queries = self.positions(queries)
        index = queries[:, None, :, None].expand(-1, self.q.size(1), -1, self.q.size(3))
        match = self.q.gather(2, index) @ self.k.transpose(-1, -2)
        keys = torch.arange(self.k.size(2), device=self.k.device)
        mask = keys <= queries[..., None]
        if self.log_decay is None:
            coeff = match.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            return coeff, None, match.masked_fill(~mask[:, None], 0)
        decays = self.log_decay.expand(*self.q.shape[:3]).cumsum(-1)
        at_query = decays.gather(2, queries[:, None].expand(-1, self.q.size(1), -1))
        log_survival = (at_query[..., None] - decays.unsqueeze(-2)).masked_fill(~mask[:, None], -torch.inf)
        match = match.masked_fill(~mask[:, None], 0)
        return match * log_survival.exp(), log_survival, match

    def reconstructed(self, absolute=False):
        """The read from the exact map. absolute=True sums the terms' absolute values instead: the scale of the
        floating-point rounding in any order of summation (multi-level spike keys make single terms large)."""
        f = torch.abs if absolute else (lambda t: t)
        out = f(self.parts()[0]) @ f(self.v)
        if self.incoming is not None:
            prefix = self.log_decay.expand(*self.q.shape[:3]).cumsum(-1).exp()
            out = out + f(self.q * prefix[..., None]) @ f(self.incoming)
        return out


def layer_ids(model):
    return model.mem_layers if isinstance(model, snn.SpikingLM) else list(range(len(model.blocks)))


def traced_forward(model, x, state=None, intervention=None, return_reads=True, query_slice=None):
    """Capture the real scan arguments or intervene on a single raw read before downstream processing."""
    traces = {}
    layers = iter(layer_ids(model))
    scan_gla, scan_plastic, attention = tf.gla_scan, snn.plastic_scan, F.scaled_dot_product_attention

    def observe(q, k, v, decay, actual, incoming=None):
        layer = next(layers)
        trace = Read(q.float(), k.float(), v.float(), decay, actual.float(), incoming)
        if return_reads:
            traces[layer] = trace
        return actual if intervention is None else intervention(layer, trace).to(actual.dtype)

    def gla(q, k, v, log_a, chunk=64, S0=None):
        out, after = scan_gla(q, k, v, log_a, chunk, S0)
        return observe(q, k, v, log_a, out, S0), after

    def plastic(q, k, v, log_lam, S0=None, chunk=128):
        out, after = scan_plastic(q, k, v, log_lam, S0, chunk)
        decay = log_lam.float().view(1, -1, 1)
        return observe(q, k, v, decay, out, S0), after

    def softmax(q, k, v, *, is_causal):
        assert is_causal
        out = attention(q, k, v, is_causal=True)
        return observe(q * q.size(-1) ** -.5, k, v, None, out)

    carry = {"return_state": True} if isinstance(model, tf.GPT) and model.cfg.attn == "linear" else {}
    with patch.object(tf, "gla_scan", gla), patch.object(snn, "plastic_scan", plastic), \
            patch.object(F, "scaled_dot_product_attention", softmax):
        logits, _, after, _ = model(x, state=state, logit_slice=query_slice, **carry)
    return logits, after, traces


def check_reconstruction(model, x, split=137, atol=3e-5, rel_to_terms=1e-4):
    """Check against the actual scans and, for recurrent models, against carried and explicit prefix writes.
    An element passes if |error| <= atol + rel_to_terms * (sum of the absolute terms in its read). FP32 rounding is
    at most about n * 1.2e-7 of that sum (n <= 257 terms); a wrong map (lag, decay or state) misses by a sizeable
    fraction of it. The split pass and the full pass can feed a layer different inputs when rounding flips a spike
    upstream (a membrane within FP32 noise of threshold; spiking layers amplify it). The split-vs-full comparison is
    then recorded, not asserted: every map check still runs on each pass's own inputs, the analysis itself uses
    full passes only, and the CPU tests check the carried state exactly. Returns the largest absolute error, the
    largest share of the tolerance used, and the layers whose inputs diverged."""
    if not 1 <= split < x.size(1):
        raise ValueError("reconstruction needs a split inside the sequence")
    _, _, full = traced_forward(model, x, query_slice=slice(0, 0))
    checks = [(f"layer {layer} map vs scan", read.reconstructed(), read.actual, read.reconstructed(True))
              for layer, read in full.items()]
    diverged = []
    if isinstance(model, snn.SpikingLM) or model.cfg.attn == "linear":
        _, state, left = traced_forward(model, x[:, :split], query_slice=slice(0, 0))
        _, _, right = traced_forward(model, x[:, split:], state=state, query_slice=slice(0, 0))
        for layer, read in right.items():
            first = left[layer]
            first_decay = first.log_decay.expand(*first.q.shape[:3])
            second_decay = read.log_decay.expand(*read.q.shape[:3])
            joined = Read(torch.cat((first.q, read.q), 2), torch.cat((first.k, read.k), 2),
                          torch.cat((first.v, read.v), 2), torch.cat((first_decay, second_decay), 2),
                          full[layer].actual)
            scale = full[layer].reconstructed(True)
            checks.extend(((f"layer {layer} carried-state map vs scan", read.reconstructed(), read.actual,
                            read.reconstructed(True)),
                           (f"layer {layer} explicit prefix map vs carried scan", joined.reconstructed()[:, :, split:],
                            read.actual, scale[:, :, split:])))
            whole = full[layer]
            differ = {}
            for name, a, b in (("q", joined.q, whole.q), ("k", joined.k, whole.k), ("v", joined.v, whole.v),
                               ("log_decay", joined.log_decay, whole.log_decay.expand(*whole.q.shape[:3]))):
                off = (a - b).abs() > atol + rel_to_terms * b.abs().amax()
                if off.any():
                    differ[name] = dict(entries=off.sum().item(), of=off.numel(),
                                        first_position=off.nonzero()[:, 2].min().item())
            joined_actual = torch.cat((first.actual, read.actual), 2)
            if differ:
                gap = (joined_actual - whole.actual).abs()
                diverged.append(dict(layer=layer, inputs=differ, max_abs_output_difference=gap.max().item(),
                                     output_elements_outside=(gap > atol + rel_to_terms * scale).sum().item()))
            else:
                checks.append((f"layer {layer} split scan vs full scan", joined_actual, whole.actual, scale))
    worst_abs, worst_share = 0., 0.
    for label, actual, expected, terms in checks:
        error = (actual - expected).abs()
        share = error / (atol + rel_to_terms * terms)
        if not torch.isfinite(share).all() or share.max() > 1:
            bad = share.max().item()
            raise AssertionError(f"{label}: {(share > 1).sum().item()} of {share.numel()} elements outside "
                                 f"tolerance; max |error| {error.max().item():.3g}, max error/tolerance {bad:.3g}, "
                                 f"max term sum {terms.max().item():.3g}")
        worst_abs, worst_share = max(worst_abs, error.max().item()), max(worst_share, share.max().item())
    return dict(max_abs_error=worst_abs, max_share_of_tolerance=worst_share,
                tolerance=f"{atol:g} + {rel_to_terms:g} x sum of absolute terms", split=split,
                split_inputs_diverged=diverged)


@dataclass
class Case:
    ids: list
    answer: int
    source: int
    shams: tuple[int, int]
    label: str

    @property
    def query(self):
        return len(self.ids) - 1


def induction_source(ids, answer):
    """Unique earlier A B write when ids ends in A. B is the token to predict."""
    matches = [s for s in range(1, len(ids) - 1) if ids[s - 1] == ids[-1] and ids[s] == answer]
    return matches[0] if len(matches) == 1 else None


def matched_shams(ids, source, answer):
    """Two distractors at symmetric lags, with mean lag equal to the source lag."""
    for gap in range(1, 9):
        a, b = source - gap, source + gap
        if a >= 1 and b < len(ids) - 1 and ids[a] != answer and ids[b] != answer:
            return a, b
    return None


def make_case(ids, target, source, label):
    shams = matched_shams(ids, source, target)
    return Case(list(ids), target, source, shams, label) if shams is not None else None


def natural_cases(tok, seq, limit):
    eot = tok.token_to_id("<|endoftext|>")
    for i, (context, word) in enumerate(passages(limit)):
        ctx = tok.encode(context).ids
        whole = tok.encode(context + word).ids
        suffix = whole[len(ctx):] if whole[:len(ctx)] == ctx else tok.encode(word).ids
        if len(suffix) != 1:
            continue
        ids = ([eot] + ctx)[-seq:]
        target = suffix[0]
        locations = [s for s, token in enumerate(ids) if token == target]
        if len(locations) == 1:
            case = make_case(ids, target, locations[0], f"lambada:{i}")
            if case:
                yield case


def synthetic_cases(vocab, seq, count, rng, eot):
    for i in range(count):
        width = int(rng.choice([16, 32, 64, 128]))
        width = min(width, (seq - 4) // 2)
        seq_ids = np.delete(np.arange(vocab), eot)
        base = rng.choice(seq_ids, width, replace=False).tolist()
        pos = int(rng.integers(2, width - 2))
        ids = [eot] + base + base[:pos + 1]
        target = base[pos + 1]
        case = make_case(ids, target, induction_source(ids, target), f"repeated:{i}")
        yield case


def natural_induction(data, offsets, seq, eot):
    for start in offsets:
        window = data[start:start + seq + 1].astype(int).tolist()
        ids, target = window[:-1], window[-1]
        source = induction_source(ids, target)
        if source is not None and target != eot:
            case = make_case(ids, target, source, f"valid:{start}")
            if case:
                yield case


def intervene(layer, read, selected, queries, positions, strength, teacher):
    heads = [head for candidate_layer, head in selected if candidate_layer == layer]
    if not heads or strength == 0:
        return read.actual
    stream = torch.arange(len(queries), device=queries.device)[:, None]
    queries = queries[:, None]
    positions = positions[:, None]
    head = torch.as_tensor(heads, device=queries.device)[None, :]
    old = read.actual[stream, head, queries]
    if teacher:
        # Remove the write and renormalize each selected causal softmax row.
        q = read.q[stream, head, queries]
        raw = (read.k[:, heads] * q.unsqueeze(-2)).sum(-1)
        raw = raw.masked_fill(torch.arange(raw.size(-1), device=raw.device) > queries.unsqueeze(-1), -torch.inf)
        raw.scatter_(-1, positions[:, None].expand_as(raw[..., :1]), -torch.inf)
        new = (raw.softmax(-1).unsqueeze(-1) * read.v[:, heads]).sum(-2)
    else:
        # Use the student's own source value and retain the realized L1 coefficient mass.
        mass = read.parts(queries)[0][:, heads, 0].abs().sum(-1)
        new = mass[..., None] * read.v[stream, head, positions]
    out = read.actual.clone()
    out[stream, head, queries] = torch.lerp(old, new, strength)
    return out


def case_batches(cases, batch):
    ordered = sorted(enumerate(cases), key=lambda e: -len(e[1].ids))
    for offset in range(0, len(ordered), batch):
        block = ordered[offset:offset + batch]
        yield [i for i, _ in block], [case for _, case in block]


def measure_cases(model, cases, batch, dev, eot, selected=None, strength=1., arm="source"):
    scores = np.empty(len(cases))
    maps = {}
    teacher = isinstance(model, tf.GPT) and model.cfg.attn != "linear"
    for indices, block in case_batches(cases, batch):
        width = max(len(case.ids) for case in block)
        ids = [case.ids + [eot] * (width - len(case.ids)) for case in block]
        x = torch.tensor(ids, device=dev)
        rows = torch.arange(len(block), device=dev)
        queries = torch.tensor([c.query for c in block], device=dev)
        src = torch.tensor([c.source for c in block], device=dev)
        answer = torch.tensor([c.answer for c in block], device=dev)
        callback = None
        if selected is not None:
            edit_at = src if arm == "source" else torch.tensor(
                [c.shams[int(arm[-1])] for c in block], device=dev)
            # Bind edit_at now: `positions` below is reused for the logit rows.
            callback = lambda layer, read, edit_at=edit_at: intervene(layer, read, selected, queries, edit_at,
                                                                      strength, teacher)
        if arm == "replace":
            x[rows, src] = x[rows, torch.tensor([c.shams[0] for c in block], device=dev)]
        scored = sorted(set(queries.tolist()))
        lookup = {q: pos for pos, q in enumerate(scored)}
        positions = torch.tensor([lookup[c.query] for c in block], device=dev)
        logits, _, traces = traced_forward(model, x, intervention=callback, return_reads=selected is None and arm == "source",
                                            query_slice=scored)
        scores[indices] = logits[rows, positions].float().log_softmax(-1)[rows, answer].cpu().numpy()
        for layer, trace in traces.items():
            coeff = trace.parts(queries[:, None])[0].squeeze(2)
            values = torch.stack([coeff[rows, :, pos] for pos in (src, queries - 1)]).float()
            total = coeff.abs().sum(-1).clamp_min(1e-30)
            normalized = (values / total).permute(1, 0, 2).cpu().numpy()
            if layer not in maps:
                maps[layer] = np.empty((len(cases), *normalized.shape[1:]), dtype=np.float32)
            maps[layer][indices] = normalized
    return scores, maps


def measure_range(traces, bins):
    out = {}
    for layer, trace in traces.items():
        coeff, log_age, match = trace.parts()
        lag = torch.arange(coeff.size(-1), device=coeff.device)
        lag = lag[:, None] - lag[None, :]
        weighted = coeff.abs() * trace.v.norm(dim=-1).unsqueeze(-2)
        mass = coeff.abs().sum(-1)
        energy = weighted.sum(-1)
        signed_balance = 1 - coeff.sum(-1).abs() / mass.clamp_min(1e-30)
        vector_balance = 1 - trace.actual.norm(dim=-1) / energy.clamp_min(1e-30)
        ratios = [mass.eq(0).float().mean((0, 2)),
                  (signed_balance * mass.gt(0)).sum((0, 2)) / mass.gt(0).sum((0, 2)).clamp_min(1),
                  (vector_balance.clamp(0, 1) * energy.gt(0)).sum((0, 2)) / energy.gt(0).sum((0, 2)).clamp_min(1),
                  (coeff.diagonal(-1, -2, -1) / mass[..., 1:].clamp_min(1e-30)).mean((0, 2))]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            take = (lag >= lo) & (lag < hi)
            def reduce(x):
                return x[..., take].sum((0, 2))
            rows.append(torch.stack((reduce(coeff.abs()), reduce(weighted), reduce(match.abs()),
                                     reduce(log_age.exp()) if log_age is not None else torch.zeros_like(ratios[0]))))
        out[layer] = (torch.stack(rows).cpu().numpy(), torch.stack(ratios).cpu().numpy())
    return out


def reference_inputs(model, data, offsets, seq, dev):
    values = []
    for offset in offsets:
        tokens = torch.tensor(data[offset:offset + seq].astype(np.int64)[None, :], device=dev)
        traces = traced_forward(model, tokens, query_slice=slice(0, 0))[2]
        values.append(measure_range(traces, LAGS))
    return {layer: {"lag_sums": np.sum([row[layer][0] for row in values], axis=0).tolist(),
                    "row_means": np.mean([row[layer][1] for row in values], axis=0).tolist()}
            for layer in values[0]}


def save_maps(model, cases, path, dev, tok, chosen):
    arrays = {}
    for i, case in enumerate(cases):
        x = torch.tensor([case.ids], device=dev)
        logits, _, traces = traced_forward(model, x, query_slice=[case.query])
        arrays[f"case_{i}_tokens"] = x[0].cpu().numpy()
        arrays[f"case_{i}_text"] = np.asarray([tok.id_to_token(t) or "" for t in case.ids])
        arrays[f"case_{i}_positions"] = np.array([case.query, case.source, *case.shams, case.answer,
                                                 logits[0, 0].argmax().item()])
        arrays[f"case_{i}_label"] = np.array(case.label)
        first, last = min(traces), max(traces)
        examples = {(first, 0), (last, traces[last].q.size(1) - 1), *chosen}
        for layer, head in sorted(examples):
            read = traces[layer]
            select = slice(head, head + 1)
            decay = None if read.log_decay is None else read.log_decay[:, select]
            one = Read(read.q[:, select], read.k[:, select], read.v[:, select], decay, read.actual[:, select])
            arrays[f"case_{i}_layer_{layer}_head_{head}"] = one.parts()[0][0, 0].cpu().numpy()
    np.savez_compressed(path, **arrays)


def test_effect(model, cases, measure, head, strength, teacher):
    real = measure(model, cases, selected=head, strength=strength, arm="source")[0]
    shams = [measure(model, cases, selected=head, strength=strength, arm=arm)[0] for arm in ("sham0", "sham1")]
    sham = np.mean(shams, axis=0)
    return (sham - real if teacher else real - sham), float(real.mean()), float(sham.mean())


def diagnose(model, dev_cases, test_cases, measure, teacher, candidates, seed):
    _, dev_maps = measure(model, dev_cases)
    # Choose a small set with high correct-write mass on development passages.
    heads = sorted(((float(np.abs(values[:, 0, h]).mean()), (layer, h))
                    for layer, values in dev_maps.items() for h in range(values.shape[-1])), reverse=True)
    groups = [tuple(head for _, head in heads[:size]) for size in (1, 2, 4)
              if size <= min(candidates, len(heads))]
    strengths = (1.,) if teacher else (.5, 1.)
    options = [(float(test_effect(model, dev_cases, measure, group, w, teacher)[0].mean()), group, w)
               for group in groups for w in strengths]
    dev_effect, selected, strength = max(options, key=lambda option: option[0])
    effect, real, sham = test_effect(model, test_cases, measure, selected, strength, teacher)
    baseline, maps = measure(model, test_cases)
    return {"heads": selected, "strength": strength, "development_effect": dev_effect,
            "development_candidates": [{"heads": group, "strength": w, "mean": gain} for gain, group, w in options],
            "intervention_compute_per_passage": {
                "development_forwards": 3 * len(options),
                "development_head_edits": 3 * sum(len(group) for _, group, _ in options),
                "held_out_forwards": 3, "held_out_head_edits": 3 * len(selected)},
            "baseline_mean_answer_logprob": float(baseline.mean()),
            "source_mean_answer_logprob": real, "sham_mean_answer_logprob": sham,
            "baseline_mean_source_weights": [float(maps[layer][:, 0, head].mean()) for layer, head in selected],
            "held_out": paired_bootstrap(effect, seed=seed), "paired_effects": effect.tolist()}


def induction_report(model, cases, measure, seed):
    actual, maps = measure(model, cases)
    replaced, _ = measure(model, cases, arm="replace")
    return {"cases": len(cases), "mean_answer_logprob": float(actual.mean()),
            "answer_logprob_original_minus_replacement": paired_bootstrap(actual - replaced, seed=seed),
            "heads": {layer: {"source_weight": row[:, 0].mean(0).tolist(),
                              "previous_token_weight": row[:, 1].mean(0).tolist()}
                      for layer, row in maps.items()}}


@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/memory_maps")
    parser.add_argument("--intervention-set", choices=("lambada", "repeated"), required=True,
                        help="which passages supply the pre-registered intervention gate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-cases", type=int, default=48)
    parser.add_argument("--test-cases", type=int, default=192)
    parser.add_argument("--induction-cases", type=int, default=64)
    parser.add_argument("--range-windows", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--head-candidates", type=int, choices=(1, 2, 4), default=4,
                        help="largest group in the top-1, top-2, top-4 development search")
    config = parser.parse_args()
    if min(config.dev_cases, config.test_cases, config.induction_cases, config.range_windows,
           config.batch, config.head_candidates) < 1:
        parser.error("case, window, batch and candidate counts must be positive")
    torch.set_num_threads(4)
    dev = torch.device(config.device)
    config.out.mkdir(parents=True, exist_ok=True)
    info = {name: json.loads((ROOT / "runs" / name / "args.json").read_text()) for name in RUNS}
    refs = info[RUNS[0]]
    assert all((a["data"], a["tokenizer"], a["seq"]) == (refs["data"], refs["tokenizer"], refs["seq"])
               for a in info.values())
    seq = refs["seq"]
    tok = Tokenizer.from_file(str(ROOT / refs["data"] / refs["tokenizer"]))
    eot = tok.token_to_id("<|endoftext|>")
    data = np.memmap(ROOT / refs["data"] / "valid.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(config.seed)
    offsets = rng.permutation((len(data) - 1) // seq) * seq
    repeated = list(synthetic_cases(tok.get_vocab_size(), seq, config.induction_cases, rng, eot))
    natural = list(itertools.islice(natural_induction(data, offsets[config.range_windows:], seq, eot),
                                    config.induction_cases))
    size = config.dev_cases + config.test_cases
    if config.intervention_set == "lambada":
        intervention_cases = list(natural_cases(tok, seq, 0))
        rng.shuffle(intervention_cases)
    else:
        intervention_cases = list(synthetic_cases(tok.get_vocab_size(), seq, size, rng, eot))
        for i, case in enumerate(intervention_cases):
            case.label = f"intervention_repeated:{i}"
    if len(intervention_cases) < size or len(natural) < config.induction_cases or len(offsets) < config.range_windows:
        raise ValueError("insufficient matching passages for the requested evaluation budgets")
    development = intervention_cases[:config.dev_cases]
    held_out = intervention_cases[config.dev_cases:size]
    examples = held_out[:2] + natural[:1] + repeated[:1]
    report = dict(lag_edges=LAGS, causal_pairs_per_lag_bin=[
                      config.range_windows * sum(seq - lag for lag in range(lo, min(hi, seq)))
                      for lo, hi in zip(LAGS[:-1], LAGS[1:])],
                  sham="mean of symmetric distractor writes, each at most eight tokens from the source",
                  lag_channels=["abs_coeff", "abs_coeff_times_value_norm", "abs_qk",
                                                "decay_survival_sum_or_zero_for_softmax"],
                  range_row_metrics=["silent_fraction", "signed_cancellation", "value_vector_cancellation",
                                     "previous_token_signed_weight"],
                  map_position_columns=["query", "source", "sham0", "sham1", "answer", "argmax"],
                  dev_passages=[c.label for c in development], held_out_passages=[c.label for c in held_out],
                  intervention_set=config.intervention_set, seed=config.seed, models={})
    teacher_supported = False
    for name, args in info.items():
        print(f"analyzing {name}", flush=True)
        model, step = load(ROOT / "runs" / name / "ckpt.pt", ema=bool(args.get("ema")), device=dev)
        prefix = data[offsets[0]:offsets[0] + min(seq, 257)].astype(np.int64)
        verify = check_reconstruction(model, torch.tensor(prefix[None, :], device=dev))
        ranges = reference_inputs(model, data, offsets[:config.range_windows], seq, dev)
        def measure(net, cases, **kwargs):
            return measure_cases(net, cases, config.batch, dev, eot, **kwargs)
        teacher = name == RUNS[0]
        paired = diagnose(model, development, held_out, measure, teacher, config.head_candidates, config.seed)
        if teacher:
            teacher_supported = paired["held_out"]["mean"] > 0 and paired["held_out"]["ci95"][0] > 0
            report["teacher_supports_alignment"] = teacher_supported
        gain = paired["held_out"]
        save_maps(model, examples, config.out / f"{name}.npz", dev, tok, paired["heads"])
        report["models"][name] = dict(step=step, reconstruction=verify, range=ranges,
                                       repeated=induction_report(model, repeated, measure, config.seed),
                                       natural_induction=induction_report(model, natural, measure, config.seed),
                                       intervention=paired,
                                       intervention_kind="suppress" if teacher else "redirect",
                                       qualifies_for_alignment=(teacher_supported and gain["mean"] >= .20
                                                                and gain["ci95"][0] > 0)
                                       if isinstance(model, snn.SpikingLM) else None)
        (config.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del model


if __name__ == "__main__":
    main()

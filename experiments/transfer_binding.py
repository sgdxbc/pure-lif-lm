"""Train and score small persistent memory adapters on frozen, reset GLA features."""

import argparse
from collections import Counter, defaultdict
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

from experiments.transfer_binding_data import BLOCK, Passages
from experiments.transfer_binding_memory import BindingMemory, reset_features, select_positions
from model_tf import GPT
from sample import load

FIT_DOCS = 2048
EVAL_DOCS = 128
INIT_SEED = 3102
BOOT_SEED = 3105
SECONDS = 3600
TRAIN_CELLS = ((128, 2), (256, 2), (256, 8))
METHODS = ("body", "spike", "dense", "overlap")


def name_log_odds(features, rows, correct, swapped):
    """The softmax normaliser cancels when the two names share a prefix."""
    return ((rows[correct].float() - rows[swapped].float()) * features.float()).sum(-1)


def pairs_batch(body, examples, device):
    original = [row.tokens for row in examples]
    other = [row.other for row in examples]
    tokens = torch.tensor(np.stack(original + other), dtype=torch.long, device=device)
    positions = torch.tensor([row.position for row in examples] * 2, device=device)[:, None]
    correct = [row.correct for row in examples]
    swapped = [row.swapped for row in examples]
    truth = torch.tensor(correct + swapped, device=device)
    foil = torch.tensor(swapped + correct, device=device)
    features = reset_features(body, tokens, BLOCK).float()
    return features, positions, truth, foil


def ordinary_batch(body, tokens, count, device):
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    length = tokens.size(1) - 1
    pos = torch.linspace(length // 3, length - 1, count, device=device).long()
    pos = pos[None].expand(tokens.size(0), -1)
    return reset_features(body, tokens[:, :-1], BLOCK).float(), pos, tokens.gather(1, pos + 1)


def exact_nll(body, features, targets):
    with torch.autocast(features.device.type, torch.bfloat16, enabled=features.is_cuda):
        logits = body.head(features)
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")


def fit(body, models, generator, args, budget):
    device = next(body.parameters()).device
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=.001) for name, model in models.items()}
    schedule = []
    for i, (distance, competitors) in enumerate(TRAIN_CELLS):
        n = args.train_docs // len(TRAIN_CELLS) + (i < args.train_docs % len(TRAIN_CELLS))
        examples = generator.pairs(distance, competitors, n)
        schedule.extend(examples[i:i + args.batch] for i in range(0, n, args.batch))
    results = {name: Counter() for name in models}
    for index in generator.rng.permutation(len(schedule)):
        budget()
        batch = schedule[index]
        x, pos, truth, foil = pairs_batch(body, batch, device)
        plain, plain_pos, targets = ordinary_batch(body, np.stack([row.original for row in batch]), 4, device)
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(x, pos)
            odds = name_log_odds(pred[:, 0], body.head.weight, truth, foil)
            name_nll = F.softplus(-odds).mean()
            pred_plain, _ = model(plain, plain_pos)
            nll = exact_nll(body, pred_plain, targets).mean()
            (name_nll + .25 * nll).backward()
            optimizer.step()
            results[name].update(documents=len(batch), name_nll_sum=len(batch) * float(name_nll.detach()),
                                 ordinary_nll_sum=len(batch) * float(nll.detach()))
    return {name: {key.removesuffix("_sum"): value / row["documents"]
                   for key, value in row.items() if key.endswith("_sum")} for name, row in results.items()}


def score_features(models, x, pos):
    spike, codes = models["spike"](x, pos)
    dense, _ = models["dense"](x, pos)
    return {"body": select_positions(x, pos), "spike": spike, "dense": dense,
            "overlap": models["spike"].digital(x, pos, codes)}


@torch.no_grad()
def evaluate(body, models, generator, cells, args, budget):
    device = next(body.parameters()).device
    metrics, preferences = {}, {}
    for distance, competitors in cells:
        budget()
        examples = generator.pairs(distance, competitors, args.eval_docs)
        odds = defaultdict(list)
        for j in range(0, len(examples), args.batch):
            budget()
            x, pos, truth, foil = pairs_batch(body, examples[j:j + args.batch], device)
            for name, features in score_features(models, x, pos).items():
                delta = name_log_odds(features[:, 0], body.head.weight, truth, foil)
                odds[name].append(delta.cpu().numpy().reshape(2, -1))
        cell = f"{distance}x{competitors}"
        metrics[cell], preferences[cell] = {}, {}
        for name in METHODS:
            pred = np.concatenate(odds[name], axis=1)
            preferences[cell][name] = pred.mean(axis=0)
            metrics[cell][name] = {"name_log_odds": float(pred.mean()),
                                   "two_choice_accuracy": float(((pred > 0) + .5 * (pred == 0)).mean())}
    nll = Counter()
    work = defaultdict(Counter)
    ordinary = generator.ordinary(args.eval_docs)
    for j in range(0, len(ordinary), args.batch):
        budget()
        x, pos, target = ordinary_batch(body, ordinary[j:j + args.batch], 16, device)
        preds = {"body": select_positions(x, pos)}
        for name, model in models.items():
            preds[name], codes = model(x, pos)
            work[name].update(model.work(codes))
            if name == "spike":
                preds["overlap"] = model.digital(x, pos, codes)
                work["overlap"].update(model.work(codes, digital=True))
        for name, pred in preds.items():
            nll[name] += float(exact_nll(body, pred, target).sum())
    for name, counters in work.items():
        work[name] = {"state_bytes_per_stream": (models["spike"].overlap_bytes if name == "overlap"
                                                 else models[name].state_bytes),
                      **{key + "_per_token": value / counters["tokens"] for key, value in counters.items()
                         if key not in ("tokens", "allocated_state_bytes")}}
    return {"name_probe": metrics, "ordinary_nll": {k: nll[k] / (len(ordinary) * 16) for k in METHODS},
            "ordinary_adapter_work": dict(work)}, preferences


def within_work_budget(report):
    spike, dense = (report["ordinary_adapter_work"][name] for name in ("spike", "dense"))
    def work(row):
        return (sum(row[key] for key in ("state_bytes_read_per_token", "state_bytes_written_per_token",
                                        "reset_bytes_per_token")), row["adds_per_token"] + row["macs_per_token"])
    return all(a <= b for a, b in zip(work(spike), work(dense)))


def binding_verdict(report, preferences, dev_work_ok):
    row = preferences["2048x8"]
    deltas = np.stack([row["spike"] - row[control] for control in ("dense", "overlap")])
    samples = np.random.default_rng(BOOT_SEED).integers(len(row["spike"]), size=(10_000, len(row["spike"])))
    # Taking the worse comparison in each draw also covers a control switch.
    interval = np.quantile(deltas[:, samples].mean(-1).min(axis=0), (.025, .975))
    stronger = max(("dense", "overlap"), key=lambda name: row[name].mean())
    gap = float(deltas.mean(axis=1).min())
    nll = report["ordinary_nll"]
    excess = nll["spike"] - nll["dense"]
    return {"stronger_control": stronger, "name_log_odds_gain": gap, "paired_ci95": interval.tolist(),
            "ordinary_excess_nll": excess, "development_work_gate": dev_work_ok,
            "advance": bool(dev_work_ok and gap >= .20 and interval[0] > 0 and excess <= .02)}


def main(development_revision=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/g48_gla31")
    parser.add_argument("--data", type=Path, help="override the checkpoint run's FineWeb directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train-docs", type=int, default=FIT_DOCS, help="reduce for development smoke tests")
    parser.add_argument("--eval-docs", type=int, default=EVAL_DOCS, help="reduce for development smoke tests")
    parser.add_argument("--held-out", action="store_true", help="use once, after the spec is recorded")
    if development_revision:
        parser.add_argument("--smoke-steps", type=int, help="shorten each stage for an unregistered CPU smoke run")
    args = parser.parse_args()
    if not (3 <= args.train_docs <= FIT_DOCS and 1 <= args.eval_docs <= EVAL_DOCS and args.batch > 0):
        parser.error("invalid development sample or batch size")
    if args.held_out and (args.train_docs, args.eval_docs, args.batch) != (FIT_DOCS, EVAL_DOCS, 8):
        parser.error("held-out uses the registered sample counts and batch")
    started = time.monotonic()
    prior_seconds = 0
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "adapters.pt"
    result_path = args.out / ("heldout.json" if args.held_out else "development.json")
    spec_run = ROOT / "runs/g48_gla31"
    registered = args.run.resolve() == spec_run.resolve() and args.data is None
    if development_revision:
        if args.smoke_steps is not None:
            if args.device != "cpu" or args.smoke_steps < 1 or args.held_out:
                parser.error("--smoke-steps requires CPU development and a positive step cap")
            registered = False
        if not args.held_out:
            if any(args.out.iterdir()):
                parser.error("the development revision requires a fresh output directory")
            original = ROOT / "runs/transfer_binding/development.json"
            prior_seconds = json.loads(original.read_text())["elapsed_seconds"]
    if args.held_out:
        if not registered or result_path.exists():
            parser.error("held-out evaluation requires the registered source and may only run once")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not saved["registered"] or not saved["allow_heldout"]:
            parser.error("the registered development run did not clear its gate")
        prior_seconds = saved["elapsed_seconds"]

    def budget():
        if time.monotonic() - started + prior_seconds > SECONDS:
            raise TimeoutError("binding transfer exceeded its total one-hour budget")

    settings = json.loads((args.run / "args.json").read_text())
    data = args.data or ROOT / settings["data"]
    tok = Tokenizer.from_file(str(data / settings["tokenizer"]))
    if tok.token_to_id("<|endoftext|>") != 50256:
        parser.error("the transfer requires GPT-2 BPE and FineWeb data")
    body, step = load(args.run / "ckpt.pt", ema=True, device=args.device)
    if not isinstance(body, GPT) or body.cfg.attn != "linear" or body.cfg.out_classes:
        parser.error("expected a flat-head model_tf GLA checkpoint")
    body.requires_grad_(False)
    models = {}
    for name in ("spike", "dense"):
        torch.manual_seed(INIT_SEED)
        models[name] = BindingMemory(body.cfg.d_model, sparse=name == "spike").to(args.device)
    if args.held_out:
        if step != saved["source_step"]:
            parser.error("source checkpoint step changed since development")
        for name, model in models.items():
            model.load_state_dict(saved["models"][name])
        generator = Passages(tok, data / "valid.bin", "held")
        cells = ((2048, 2), (2048, 8), (256, 2), (256, 8))
        report, preferences = evaluate(body, models, generator, cells, args, budget)
        report["verdict"] = binding_verdict(report, preferences, saved["work_gate"])
    else:
        train = Passages(tok, data / "train.bin", "train")
        generator = Passages(tok, data / "valid.bin", "dev")
        if development_revision:
            report = development_revision(body, models, train, generator, args, budget)
        else:
            training = fit(body, models, train, args, budget)
            report, _ = evaluate(body, models, generator, ((256, 2), (256, 8)), args, budget)
            chance = max(report["name_probe"]["256x8"][arm]["two_choice_accuracy"] for arm in models)
            report.update(training=training, allow_heldout=chance > .5, development_work_gate=within_work_budget(report))
        budget()
        torch.save({"registered": registered and (args.train_docs, args.eval_docs, args.batch) == (FIT_DOCS, EVAL_DOCS, 8),
                    "allow_heldout": report["allow_heldout"], "work_gate": report["development_work_gate"],
                    "source_step": step, "elapsed_seconds": prior_seconds + time.monotonic() - started,
                    "models": {name: model.state_dict() for name, model in models.items()}}, checkpoint)
    budget()
    report.update(split="held-out" if args.held_out else "development", source_step=step,
                  train_documents=args.train_docs, eval_documents_per_cell=args.eval_docs,
                  elapsed_seconds=time.monotonic() - started)
    result = json.dumps(report, indent=2) + "\n"
    result_path.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()

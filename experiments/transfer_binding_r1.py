"""One development revision of the learned binding transfer, starting at gap 128."""
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.transfer_binding import (TRAIN_CELLS, evaluate, exact_nll, main, name_log_odds,
                                          ordinary_batch, pairs_batch, within_work_budget)
from experiments.transfer_binding_data import Passages

TARGET = .04
MIXED_DEV = ((256, 2), (256, 8))


def rates(gate, key, positions=None):
    wrote = key.detach().ne(0).any(-1).to(gate.dtype)
    if positions is None:
        return gate.mean(), wrote.mean()
    mask = torch.arange(gate.size(1), device=gate.device) <= positions
    return (gate * mask).sum() / mask.sum(), (wrote * mask).sum() / mask.sum()


def stage_lr(progress):
    if progress <= .05:
        return .001 * progress / .05
    return .0001 + .00045 * (1 + math.cos(math.pi * (progress - .05) / .95))


def training_batches(generator, cells, docs, batch_size):
    """Revisit the same disjoint donors with fresh paired stories on later epochs."""
    batches = []
    for i, cell in enumerate(cells):
        n = docs // len(cells) + (i < docs % len(cells))
        rows = generator.pairs(*cell, n)
        batches.extend((cell, rows[j:j + batch_size]) for j in range(0, n, batch_size))
    first = True
    while True:
        for index in generator.rng.permutation(len(batches)):
            cell, rows = batches[index]
            if first:
                yield rows
                continue
            fresh = []
            for row in rows:
                for _ in range(1000):
                    new = generator.make_example(row.original, *cell)
                    if new is not None:
                        fresh.append(new)
                        break
                else:
                    raise ValueError("could not redraw a valid assignment on a previously accepted donor")
            yield fresh
        first = False


def stage(body, models, generator, args, budget, log, label, cells, steps, seconds):
    device = body.head.weight.device
    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=.001) for name, m in models.items()}
    batches = training_batches(generator, cells, args.train_docs, args.batch)
    limit = min(steps, args.smoke_steps or steps)
    metrics = {}
    start = time.monotonic()
    docs = 0
    for step in range(1, limit + 1):
        budget()
        paired = next(batches)
        x, pos, truth, foil = pairs_batch(body, paired, device)
        plain, targets_pos, targets = ordinary_batch(body, np.stack([ex.original for ex in paired]), 4, device)
        idx = torch.arange(x.size(0), device=device)
        body_loss = F.softplus(-name_log_odds(x[idx, pos[:, 0]], body.head.weight, truth, foil)).mean()
        progress = min(1., max(step / limit, (time.monotonic() - start) / seconds))
        lr = stage_lr(progress)
        count = len(paired)
        for name, model in models.items():
            opt = optimizers[name]
            opt.param_groups[0]["lr"] = lr
            opt.zero_grad(set_to_none=True)
            pred, (_, key, _), gate = model(x, pos, return_gate=True)
            odds = name_log_odds(pred[:, 0], body.head.weight, truth, foil)
            name_loss = F.softplus(-odds).mean()
            paired_rate, writes = rates(gate, key, pos)
            pred, (_, key, _), gate = model(plain, targets_pos, return_gate=True)
            ordinary_loss = exact_nll(body, pred, targets)
            plain_rate, plain_writes = rates(gate, key)
            penalty = 5 * ((paired_rate - TARGET).square() + (plain_rate - TARGET).square())
            (name_loss + .25 * ordinary_loss.mean() + penalty).backward()
            opt.step()
            values = torch.stack((name_loss, body_loss, odds.mean(), ordinary_loss.mean(), paired_rate,
                                  writes, plain_rate, plain_writes)).detach().double() * count
            if name in metrics:
                metrics[name] += values
            else:
                metrics[name] = values
        docs += count
        elapsed = time.monotonic() - start
        done = step == limit or elapsed >= seconds
        if step % 100 == 0 or done:
            keys = ("name_nll", "body_name_nll", "paired_name_log_odds", "ordinary_nll", "gate_rate",
                    "write_rate", "ordinary_gate_rate", "ordinary_write_rate")
            last = {name: dict(zip(keys, (v / docs).tolist())) for name, v in metrics.items()}
            record = dict(stage=label, step=step, seconds=elapsed, interval_documents=docs, lr=lr, arms=last)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            metrics.clear()
            docs = 0
        if done:
            return record


def revise(body, models, train, dev, args, budget):
    # Reproduce the registered 256×2, 256×8 and ordinary dev samples in the
    # final evaluation. Use other even-indexed documents for the easy gate.
    easy_dev = Passages(dev.tok, dev.data.filename, "dev")
    for cell in MIXED_DEV:
        easy_dev.pairs(*cell, args.eval_docs)
    easy_dev.ordinary(args.eval_docs)
    easy = ((128, 2),)
    with (args.out / "training.jsonl").open("x") as log:
        fit_easy = stage(body, models, train, args, budget, log, "easy", easy, 4096, 240)
        screen, _ = evaluate(body, models, easy_dev, easy, args, budget)
        passed = max(screen["name_probe"]["128x2"][arm]["two_choice_accuracy"] for arm in models) > .60
        screen.update(passed=passed, last_training_interval=fit_easy)
        (args.out / "learnability.json").write_text(json.dumps(screen, indent=2) + "\n")
        if not passed:
            return dict(learnability=screen, stopped_at="128x2", allow_heldout=False, development_work_gate=None)
        fit_mixed = stage(body, models, train, args, budget, log, "mixed", TRAIN_CELLS, 65536, 2100)
    report, _ = evaluate(body, models, dev, MIXED_DEV, args, budget)
    passed = max(report["name_probe"]["256x8"][arm]["two_choice_accuracy"] for arm in models) > .50
    report.update(learnability=screen, last_training_interval=fit_mixed, allow_heldout=passed,
                  development_work_gate=within_work_budget(report))
    return report


if __name__ == "__main__":
    main(development_revision=revise)

"""Query-only source supervision using the recorded FineWeb eligibility screen."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import align_pilot as align
from experiments.memory_maps import (
    Case, case_batches, natural_cases, natural_induction, paired_bootstrap, synthetic_cases,
)
from experiments.name_competition import Passage, describe, diagnostic_cases, measure
from experiments.write_read import passed_candidates, screen_offsets, window_candidates
from sample import load

ARMS = ("plain", "sham", "true")
SEED = 5662
START_STEP = 20_000
ALIGN_PLAN = ROOT / "runs/align_pilot/plan.json"
WRITE_READ = ROOT / "runs/write_read.json"
DIAGNOSTIC = ROOT / "runs/memory_maps/summary.json"
NAME_COMPETITION = ROOT / "runs/name_competition.json"
TEACHER_CKPT = ROOT / "runs/g13_tf163/ckpt.pt"
STEPS, BATCH, MICRO, SEQ = align.STEPS, align.BATCH, align.MICRO, align.SEQ


def target_rows(teacher, windows, pieces, eot, device):
    indexed = []
    for stream, ids in enumerate(windows.tolist()):
        _, candidates = window_candidates(ids, [pieces[i] for i in ids], eot)
        indexed.extend((stream, c) for c in candidates if c.name_source)
    return [(b, c.query, c.source, min(c.wrong, key=lambda p: (abs(p - c.source), p)))
            for b, c in passed_candidates(teacher, windows, indexed, align.TEACHER, device)]


def evaluation_plan(tok, eot, data, diagnostic, earlier):
    exclude = set(diagnostic["dev_passages"] + diagnostic["held_out_passages"])
    exclude.update(row["label"] for row in earlier["lambada"])
    rng = np.random.default_rng(SEED + 1)
    available = [c for c in natural_cases(tok, SEQ, 0) if c.label not in exclude]
    rng.shuffle(available)
    fresh = available[:512]
    if not fresh:
        raise ValueError("no fresh passages under the 5.63 eligibility rules")
    wrong = {}
    for case in fresh:
        meta = describe(case, [tok.id_to_token(t) for t in case.ids])
        if meta.name_answer and meta.wrong:
            wrong[case.label] = meta.wrong

    valid = np.memmap(data / "valid.bin", dtype=np.uint16, mode="r")
    n = (len(valid) - 1) // SEQ
    old = np.random.default_rng(diagnostic["seed"]).permutation(n) * SEQ
    range_windows = diagnostic["causal_pairs_per_lag_bin"][0] // SEQ
    old_natural = {c.label for c in itertools.islice(natural_induction(valid, old[range_windows:], SEQ, eot), 64)}
    old_natural.update(c["label"] for c in earlier["natural"])
    candidates = natural_induction(valid, rng.permutation(n) * SEQ, SEQ, eot)
    natural = list(itertools.islice((c for c in candidates if c.label not in old_natural), 64))
    if len(natural) != 64:
        raise ValueError("fewer than 64 fresh natural copy cases")
    repeated = list(synthetic_cases(tok.get_vocab_size(), SEQ, 64, rng, eot))
    return dict(lambada_eligible=len(available), lambada=[asdict(c) for c in fresh], wrong_names=wrong,
                natural=[asdict(c) for c in natural], repeated=[asdict(c) for c in repeated])


def prepare(folder, device):
    data, tok, arr, eot = align.setup_data(align.model_args())
    earlier = json.loads(ALIGN_PLAN.read_text())
    screen = json.loads(WRITE_READ.read_text())["screen"]
    diagnosis = json.loads(DIAGNOSTIC.read_text())
    report = evaluation_plan(tok, eot, data, diagnosis, earlier)
    excluded = np.concatenate((np.ravel(earlier["train_offsets"]), screen["offsets"]))
    offsets = screen_offsets(len(arr), SEQ, excluded, count=(STEPS + 1) * BATCH, seed=SEED)
    report["train_offsets"] = offsets.reshape(-1, BATCH).tolist()
    print(f"Prepared {len(report['lambada'])} decision passages, {len(report['wrong_names'])} for the margin",
          flush=True)
    teacher, teacher_step = load(TEACHER_CKPT, device=device)
    if teacher_step != diagnosis["models"]["g13_tf163"]["step"]:
        raise ValueError("teacher checkpoint step differs from the write/read screen")
    pieces = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    report.update(teacher_step=teacher_step, targets=[], seed=SEED)
    for step, batch_offsets in enumerate(report["train_offsets"]):
        rows = []
        for micro, windows in enumerate(align.microbatches(arr, batch_offsets)):
            rows.extend((b + micro * MICRO, q, source, sham)
                        for b, q, source, sham in target_rows(teacher, windows, pieces, eot, device))
        report["targets"].append(rows)
        if step % 100 == 0:
            print(f"Screened step {step}/{STEPS}, {len(rows)} eligible positions in this batch", flush=True)
    counts = [len(rows) for rows in report["targets"]]
    gaps = [abs(source - sham) for rows in report["targets"][1:] for _, _, source, sham in rows]
    report["eligible_positions"] = dict(calibration=counts[0], training=sum(counts[1:]),
                                        empty_updates=counts[1:].count(0),
                                        per_million=sum(counts[1:]) * 1e6 / (STEPS * BATCH * SEQ),
                                        sham_lag_gap_quantiles=dict(zip(("p50", "p90", "p99"),
                                                                        np.quantile(gaps, [.5, .9, .99]).tolist()))
                                        if gaps else None)
    align.write_json(folder / "plan.json", report)
    print(report["eligible_positions"], flush=True)


def step_batches(data, plan, step):
    rows = plan["targets"][step]
    for i, windows in enumerate(align.microbatches(data, plan["train_offsets"][step])):
        first, end = i * MICRO, i * MICRO + len(windows)
        positions = [(b - first, q, source, sham) for b, q, source, sham in rows if first <= b < end]
        yield windows, positions


def address_loss(probs, targets):
    return -probs.gather(1, targets[:, None]).log().mean()


def losses(student, teacher, windows, positions, device):
    tokens = torch.as_tensor(windows, device=device)
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, traces = align.capture(student, x, align.STUDENT, logit_slice=None) if positions else (
            student(x)[0], None)
        with torch.no_grad():
            ref = teacher(x)[0]
    task = align.task_loss(logits, ref, y)
    if not positions:
        return task, None, None
    targets = torch.as_tensor(positions, device=device)
    probs = align.group_distribution(traces, align.STUDENT, targets[:, :2], teacher=False)
    return task, address_loss(probs, targets[:, 2]), address_loss(probs, targets[:, 3])


def training_step(student, teacher, params, arr, plan, step, arm, device, opt=None, weight=0.):
    positions = len(plan["targets"][step])
    calibration = opt is None
    sums = [[torch.zeros_like(p) for p in params] for _ in range(3)] if calibration else None
    totals = np.zeros(3)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    for windows, targets in step_batches(arr, plan, step):
        task, real, sham = losses(student, teacher, windows, targets, device)
        weights = (len(windows) / BATCH, len(targets) / max(1, positions), len(targets) / max(1, positions))
        objectives = (task, real, sham)
        for i, loss in enumerate(objectives):
            if loss is not None:
                totals[i] += weights[i] * float(loss.detach())
                if calibration:
                    grads = torch.autograd.grad(loss, params, retain_graph=real is not None and i < 2)
                    for total, grad in zip(sums[i], grads):
                        total.add_(grad.detach(), alpha=weights[i])
        if opt is not None:
            aux = real if arm == "true" else sham if arm == "sham" else None
            loss = weights[0] * task
            if aux is not None:
                loss = loss + weights[1] * weight * aux
            loss.backward()
    result = dict(task=totals[0], true_nll=totals[1] if positions else None,
                  sham_nll=totals[2] if positions else None, eligible_positions=positions)
    if calibration:
        result["grad_norms"] = dict(zip(("task", "true", "sham"),
                                        [float(torch.stack([p.square().sum() for p in grads]).sum().sqrt())
                                         for grads in sums]))
    else:
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
    return result


@torch.no_grad()
def snapshot(model, cases, wrong, device, eot):
    source_shares = np.empty(len(cases))
    margins, keys = {}, {}
    for indices, block in case_batches(cases, MICRO):
        width = max(len(c.ids) for c in block)
        x = torch.tensor([c.ids + [eot] * (width - len(c.ids)) for c in block], device=device)
        q = torch.tensor([(i, c.query) for i, c in enumerate(block)], device=device)
        _, traces = align.capture(model, x, align.STUDENT, logit_slice=slice(0, 0))
        probs = align.group_distribution(traces, align.STUDENT, q, teacher=False).cpu().numpy()
        counts = torch.stack([traces[layer].k[:, h] for layer, h in align.STUDENT], dim=1)
        counts = counts.to(torch.uint8).cpu().numpy()
        for i, (idx, case) in enumerate(zip(indices, block)):
            source_shares[idx] = probs[i, case.source]
            if case.label in wrong:
                margins[case.label] = float(probs[i, case.source] - probs[i, wrong[case.label]].max())
            keys[case.label] = counts[i, :, :len(case.ids)].copy()
    ids = [c.label for c in cases if c.label in margins]
    margins = [margins[label] for label in ids]
    result = dict(source_mass=float(source_shares.mean()), passages=len(cases), margin_ids=ids, margins=margins,
                  source_mass_name_subset=float(np.mean([source_shares[i] for i, c in enumerate(cases)
                                                        if c.label in wrong])) if ids else None,
                  mean_margin=float(np.mean(margins)) if margins else None,
                  win_rate=float(np.mean(np.asarray(margins) > 0)) if margins else None)
    return result, keys


def key_drift(cases, wrong, before, after):
    categories = ("all_causal", "source", "wrong_names")
    totals = {key: np.zeros((len(align.STUDENT), 2)) for key in categories}
    counts = dict.fromkeys(categories, 0)
    for c in cases:
        a, b = before[c.label], after[c.label]
        delta = np.abs(a.astype(np.int16) - b)
        for category, positions in zip(categories, (slice(None), [c.source], wrong.get(c.label, []))):
            d = delta[:, positions]
            counts[category] += d.shape[1]
            totals[category][:, 0] += d.any(-1).sum(1)
            totals[category][:, 1] += d.mean(-1).sum(1)
    return dict(heads=align.STUDENT, by_category={
        k: dict(writes=n, changed_vector_fraction=(totals[k][:, 0] / n).tolist() if n else None,
                mean_abs_count_change=(totals[k][:, 1] / n).tolist() if n else None)
        for k, n in counts.items()})


def old_name_competition(student, tok, device, eot):
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    cases = diagnostic_cases(tok, SEQ, diagnostic)
    rows = json.loads(NAME_COMPETITION.read_text())["passages"]
    if [c.label for c in cases] != [p["label"] for p in rows]:
        raise ValueError("name-competition passage order differs from the original diagnostic")
    rows = measure(student, cases, [Passage(**row) for row in rows], align.STUDENT, device, eot, 4)
    wins = [r["correct_beats_best_wrong"] for r in rows if r["correct_beats_best_wrong"] is not None]
    if len(wins) != 57:
        raise ValueError(f"expected 57 recorded name-competition passages, found {len(wins)}")
    return dict(wins=int(sum(wins)), passages=len(wins), win_rate=float(np.mean(wins)))


def run(stage, folder, device, arm):
    args = align.model_args()
    data, tok, arr, eot = align.setup_data(args)
    raw_plan = (folder / "plan.json").read_bytes()
    plan = json.loads(raw_plan)
    digest = hashlib.sha256(raw_plan).hexdigest()
    student, start = load(align.SOURCE_RUN / "ckpt.pt", ema=True, device=device)
    teacher, teacher_step = load(TEACHER_CKPT, device=device)
    if start != START_STEP or teacher_step != plan["teacher_step"]:
        raise ValueError("expected the recorded student initialization and teacher from preparation")
    teacher.requires_grad_(False)
    params, layers = align.train_rows(student, align.STUDENT, query_only=True)
    rate_data = np.memmap(data / "valid.bin", dtype=np.uint16, mode="r")
    calibration_path = folder / "calibration.json"
    if stage == "calibrate":
        res = training_step(student, teacher, params, arr, plan, 0, "true", device)
        norms = res["grad_norms"]
        if not all(np.isfinite(n) and n > 0 for n in norms.values()):
            raise ValueError("calibration requires positive finite task, true and sham gradient norms")
        weights = {"plain": 0., **{arm: .1 * norms["task"] / norms[arm] for arm in ("true", "sham")}}
        ratios = {"plain": 0., **{arm: weights[arm] * norms[arm] / norms["task"] for arm in ("true", "sham")}}
        res.update(weights=weights, achieved_ratios=ratios, plan_sha256=digest, student_step=start,
                   rates_before=align.kq_rates(student, rate_data, device))
        align.write_json(calibration_path, res)
        print({k: v for k, v in res.items() if k != "rates_before"}, flush=True)
        return

    calibration = json.loads(calibration_path.read_text())
    if calibration["plan_sha256"] != digest or calibration["student_step"] != start:
        raise ValueError("the calibrated plan or student step differs")
    weight, ratio = calibration["weights"][arm], calibration["achieved_ratios"][arm]
    cases = [Case(**c) for c in plan["lambada"]]
    _, keys_before = snapshot(student, cases, plan["wrong_names"], device, eot)
    if stage == "evaluate":
        # Re-score an arm whose training finished and saved ckpt.pt but whose evaluation did not.
        out = folder / arm
        if not (out / "ckpt.pt").exists() or not (out / "training.jsonl").exists():
            raise ValueError(f"{out} has no finished training to evaluate")
        if (out / "metrics.json").exists():
            raise ValueError(f"{out / 'metrics.json'} already exists")
        steps = sum(1 for _ in (out / "training.jsonl").open())
        if steps != STEPS or torch.load(out / "ckpt.pt", map_location="cpu")["step"] != start + STEPS:
            raise ValueError(f"{out} training is incomplete ({steps} logged updates)")
        del teacher, params, layers
        evaluate_arm(out, arm, plan, digest, calibration, cases, keys_before, args, data, tok, rate_data, device, eot)
        return
    opt = torch.optim.AdamW(params, lr=align.LR, weight_decay=align.WD)
    out = folder / arm
    out.mkdir()
    align.write_json(out / "args.json", args)
    print(dict(arm=arm, weight=weight, initial_gradient_ratio=ratio), flush=True)
    with (out / "training.jsonl").open("w") as log:
        for step in range(1, STEPS + 1):
            row = training_step(student, teacher, params, arr, plan, step, arm, device, opt, weight)
            log.write(json.dumps(dict(step=step, **row), allow_nan=False) + "\n")
            if step % 50 == 0:
                log.flush()
                print(arm, step, row, flush=True)
    for layer in layers:
        for attribute in ("weight", "bias"):
            parametrize.remove_parametrizations(layer, attribute, leave_parametrized=True)
    checkpoint = out / "ckpt.pt"
    torch.save(dict(model=student.state_dict(), cfg=asdict(student.cfg), arch="snn", step=start + STEPS), checkpoint)
    del student, teacher, opt, params, layers
    evaluate_arm(out, arm, plan, digest, calibration, cases, keys_before, args, data, tok, rate_data, device, eot)


@torch.no_grad()
def evaluate_arm(out, arm, plan, digest, calibration, cases, keys_before, args, data, tok, rate_data, device, eot):
    """Score a saved final checkpoint. Reloaded weights require grad by default, so switch it off."""
    student, _ = load(out / "ckpt.pt", device=device)
    student.requires_grad_(False)
    weight, ratio = calibration["weights"][arm], calibration["achieved_ratios"][arm]
    manip, keys_after = snapshot(student, cases, plan["wrong_names"], device, eot)
    print(f"{arm} manipulation", {k: v for k, v in manip.items() if k not in ("margins", "margin_ids")}, flush=True)
    res = align.assess(student, None, plan, args, data, device, eot, manipulation_results=manip, seed=SEED)
    res.update(arm=arm, aux_weight=weight, initial_gradient_ratio=ratio, plan_sha256=digest,
               rates_before=calibration["rates_before"], rates_after=align.kq_rates(student, rate_data, device),
               key_drift=key_drift(cases, plan["wrong_names"], keys_before, keys_after),
               original_name_competition=old_name_competition(student, tok, device, eot),
               eligible_positions=plan["eligible_positions"],
               peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None)
    align.write_json(out / "metrics.json", res)
    print(arm, "finished", flush=True)


def summarize(folder):
    rows = {arm: json.loads((folder / arm / "metrics.json").read_text()) for arm in ARMS}
    plain, true = rows["plain"], rows["true"]
    diagnostic = {arm: r["manipulation"] for arm, r in rows.items()}
    if any(r["lambada_ids"] != plain["lambada_ids"] or r["plan_sha256"] != plain["plan_sha256"]
           or r["manipulation"]["margin_ids"] != plain["manipulation"]["margin_ids"] for r in rows.values()):
        raise ValueError("arms must share the same plan and decision passages")
    print("Manipulation", json.dumps({arm: {k: v for k, v in r.items() if k not in ("margins", "margin_ids")}
                                     for arm, r in diagnostic.items()}, indent=2), flush=True)
    diff = np.asarray(diagnostic["true"]["margins"]) - diagnostic["plain"]["margins"]
    margin = paired_bootstrap(diff, seed=SEED) if len(diff) else None
    stage1 = margin is not None and margin["ci95"][0] > 0
    gains = {arm: paired_bootstrap(np.asarray(true["unassisted_logp"]) - rows[arm]["unassisted_logp"], seed=SEED)
             for arm in ("plain", "sham")}
    nll = true["validation_nll"] - plain["validation_nll"]
    accuracy = true["full_lambada_accuracy"] - plain["full_lambada_accuracy"]
    cost = {s: true["u55c_mJ_per_token"][s] / plain["u55c_mJ_per_token"][s] - 1 for s in ("1", "16")}
    stage2 = (all(g["mean"] >= .20 and g["ci95"][0] > 0 for g in gains.values())
              and nll <= .012 and accuracy >= -.02 and all(abs(c) <= .01 for c in cost.values()))
    result = dict(manipulation=diagnostic, fresh_lambada_n=len(plain["lambada_ids"]), margin_n=len(diff),
                  margin_gain_vs_plain=margin, stage1_passed=stage1, true_logprob_gain=gains,
                  validation_nll_delta=nll, full_lambada_accuracy_delta=accuracy, cost_fraction_delta=cost,
                  stage2_passed=stage2 if stage1 else None, advance=bool(stage1 and stage2),
                  original_name_competition={arm: row["original_name_competition"] for arm, row in rows.items()})
    align.write_json(folder / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "manipulation"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "calibrate", "train", "evaluate", "summarize"),
                        help="evaluate: score an arm's saved final ckpt.pt after a failed evaluation")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out", type=Path, default=Path("runs/query_pilot"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.stage == "prepare":
        if (args.out / "plan.json").exists():
            parser.error("use a fresh output directory to prepare a plan")
        prepare(args.out, device)
    elif args.stage == "summarize":
        summarize(args.out)
    else:
        if args.stage in ("train", "evaluate") and args.arm is None:
            parser.error(f"{args.stage} requires --arm")
        run(args.stage, args.out, device, args.arm)


if __name__ == "__main__":
    main()

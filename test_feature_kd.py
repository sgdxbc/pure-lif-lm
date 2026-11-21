"""CPU correctness checks for feature distillation and the continuation path."""
from dataclasses import asdict
import json
from math import isclose
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import torch

from feature_kd import calibrated_weight, capture_grams, cosine_gram, feature_mse, sample_positions
from model_tf import GPT, TFConfig
import train


def test_projection_loss():
    torch.manual_seed(10)
    cfg = TFConfig(vocab=16, d_model=16, n_layers=2, n_heads=2, attn="linear", mlp="spike")
    student, teacher = GPT(cfg), GPT(cfg).eval().requires_grad_(False)
    teacher.load_state_dict(student.state_dict())
    ids = torch.randint(cfg.vocab, (2, 12))
    positions = sample_positions(2, 12, 8, 1, 15000, "cpu")
    assert torch.equal(positions, sample_positions(2, 12, 8, 1, 15000, "cpu"))
    assert not torch.equal(positions, sample_positions(2, 12, 8, 1, 15001, "cpu"))
    with capture_grams(student, positions) as student_grams, torch.autocast("cpu", dtype=torch.bfloat16):
        student(ids)
    with torch.no_grad(), capture_grams(teacher, positions) as teacher_grams, torch.autocast("cpu", dtype=torch.bfloat16):
        teacher._features(ids)
    assert len(student_grams) == len(teacher_grams) == 4
    loss = feature_mse(student_grams, teacher_grams)
    assert loss.item() == 0 and loss.requires_grad
    assert all(not x.requires_grad for x in teacher_grams)
    assert all(not module._forward_hooks for module in student.modules())

    x = torch.randn(2, 8, 16, requires_grad=True)
    target = torch.randn_like(x)
    permuted = target[..., torch.randperm(target.size(-1))]
    loss = feature_mse([cosine_gram(x)], [cosine_gram(target)])
    perm_loss = feature_mse([cosine_gram(x)], [cosine_gram(permuted)])
    torch.testing.assert_close(loss, perm_loss)
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], torch.autograd.grad(perm_loss, x)[0])

    silent = torch.tensor([[[0., 0.], [1., 1.]]], requires_grad=True)
    gram = cosine_gram(silent)
    assert torch.count_nonzero(gram[:, 0]) == 0
    assert torch.count_nonzero(torch.autograd.grad(gram.sum(), silent)[0][:, 0]) == 0


def test_calibration():
    p = torch.nn.Parameter(torch.tensor([2., 3.]))
    task_grads = torch.tensor([[3., 1.], [-1., 1.]])
    feat_grads = torch.tensor([[4., 6.], [0., 2.]])
    pairs = ((p @ task, p @ feat) for task, feat in zip(task_grads, feat_grads))
    w = calibrated_weight([pairs], (p,), accum=2, fraction=0.1)
    assert isclose(w * feat_grads.mean(0).norm().item() / task_grads.mean(0).norm().item(), 0.1, rel_tol=1e-6)
    assert p.grad is None


def test_continuation_and_resume():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tok = Tokenizer(WordLevel({str(i): i for i in range(16)}, unk_token="0"))
        tok.save(str(root / "toy.json"))
        ids = np.random.default_rng(9).integers(0, 16, size=256, dtype=np.uint16)
        for split in ("train", "valid"):
            ids.tofile(root / f"{split}.bin")
        cfg = TFConfig(vocab=16, d_model=16, n_layers=2, n_heads=2, attn="linear", mlp="spike")
        # Different raw/EMA weights make the expected EMA selection testable.
        torch.manual_seed(99)
        teacher, raw = GPT(cfg), GPT(cfg)
        twin_path = root / "twin.pt"
        torch.save(dict(model=raw.state_dict(), ema=teacher.state_dict(), cfg=asdict(cfg), arch="tf"), twin_path)
        flags = ("--arch tf --attn linear --mlp spike --relay quant --relay_levels 4 --relay_target 0.25 "
                 "--n_layers 2 --n_heads 2 --vocab 16 --d_model 16 --ema 0.999 --batch 4 --seq 8 --accum 2 "
                 "--warmup 1 --eval_every 1 --eval_batches 1 --log_every 1 --threads 1 --tokenizer toy.json").split()

        def run(out, steps, extra=()):
            argv = ["train.py", "--data", str(root), "--out", str(out), "--steps", str(steps), *flags, *extra]
            with patch.object(sys, "argv", argv), patch.object(train, "sample", return_value=[]):
                train.main()
            records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
            return torch.load(out / "ckpt.pt"), records

        source, continuation = root / "source", root / "continuation"
        src, records = run(source, 1)
        assert "feat_weight" not in src and all("feat_mse" not in r for r in records)
        extra = ["--init_ckpt", str(source / "ckpt.pt"), "--feat_teacher", str(twin_path), "--feat_positions", "4",
                 "--teacher", str(twin_path), "--distill_weight", "0.9"]
        from sample import load

        def check_load(*args, **kwargs):
            model, step = load(*args, **kwargs)
            if kwargs["ema"]:
                for k, v in model.state_dict().items():
                    assert torch.equal(v, teacher.state_dict()[k])
            return model, step

        with patch("sample.load", side_effect=check_load) as loader:
            ck, records = run(continuation, 2, [*extra, "--resume"])
        loader.assert_called_with(str(twin_path), ema=True, device=torch.device("cpu"))
        w = ck["feat_weight"]
        assert ck["step"] == 2 and w > 0
        calibrated = ck
        assert any(r.get("feat_weight_frac") == 0.1 and r["step"] == 1 for r in records)
        assert any(r.get("feat_mse", 0) > 0 and r["step"] == 2 for r in records)
        with patch.object(train, "calibrated_weight", side_effect=AssertionError("resume recalibrated")):
            ck, records = run(continuation, 3, [*extra, "--resume"])
            fixed, _ = run(root / "fixed", 2, [*extra, "--feat_weight", str(w)])
        assert ck["step"] == 3 and ck["feat_weight"] == fixed["feat_weight"] == w
        assert any(r.get("feat_mse", 0) > 0 and r["step"] == 3 for r in records)
        for k in calibrated["model"]:
            assert torch.equal(calibrated["model"][k], fixed["model"][k])


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_projection_loss()
    test_calibration()
    test_continuation_and_resume()
    print("feature KD loss, calibration, continuation and resume: OK")

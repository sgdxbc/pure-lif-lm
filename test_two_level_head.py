"""CPU checks for the optional GPT readout."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from fpga_est import tf_hw
from model_tf import GPT, TFConfig
from sample import load


def check_flat_head_matches_legacy(attn):
    cfg = TFConfig(vocab=17, d_model=16, n_layers=2, n_heads=2, max_len=12, attn=attn)
    torch.manual_seed(8)
    model = GPT(cfg).eval()
    ids = torch.arange(8).unsqueeze(0)
    targets = ids + 1
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        # Original GPT forward with its single, biasless head.
        x = model.embed(ids)
        if model.pos is not None:
            x = x + model.pos(torch.arange(ids.size(1)))
        for block in model.blocks:
            x = block(x)
        expected = F.linear(model.norm(x), model.head.weight)
        legacy_loss = F.cross_entropy(expected.float().view(-1, cfg.vocab), targets.view(-1))
        logits, loss, _, _ = model(ids, targets)
        assert torch.equal(logits, expected)
        assert torch.equal(loss, legacy_loss)

    old_cfg = {k: v for k, v in asdict(cfg).items() if k not in {"out_classes", "out_short"}}
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"cfg": old_cfg, "arch": "tf", "model": model.state_dict()}, path)
        restored, _ = load(path)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(restored(ids)[0], expected)


def test_flat_head_matches_legacy():
    for attn in ("softmax", "linear"):
        check_flat_head_matches_legacy(attn)


def check_likelihood_and_causality(attn, short):
    torch.manual_seed(0)
    cfg = TFConfig(vocab=11, d_model=16, n_layers=2, n_heads=2, max_len=12,
                   attn=attn, out_classes=5, out_short=short)
    model = GPT(cfg).eval()
    if short:
        # Interleave the shortlist and tail, with one class empty.
        model.head.set_classes(torch.tensor([0, -1, 1, 0, -1, 1, 2, 2, 3, -1, 3]))
    ids = torch.arange(8).unsqueeze(0)
    targets = (ids + 1) % cfg.vocab
    with torch.no_grad():
        lp, loss, _, _ = model(ids, targets)
        x, _ = model._features(ids)
        head = model.head
        groups = [(head.word_cls == c).nonzero().flatten() for c in range(head.C)]
        top = F.linear(x, head.w_cls)
        for c, words in enumerate(groups):
            if not words.numel():
                top[..., short + c] = -torch.inf
        top = top.softmax(-1)
        ref = torch.empty_like(lp)
        ref[..., head.word_cls == -1] = top[..., :short]
        for c, words in enumerate(groups):
            if words.numel():
                weights = head.w_word[c * head.m:c * head.m + words.numel()]
                ref[..., words] = top[..., short + c, None] * F.linear(x, weights).softmax(-1)
        torch.testing.assert_close(lp.exp(), ref)
        torch.testing.assert_close(lp.exp().sum(-1), torch.ones_like(ids, dtype=torch.float32), rtol=0, atol=3e-7)
        torch.testing.assert_close(loss, -ref.gather(-1, targets[..., None]).log().mean())
        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % cfg.vocab
        assert torch.equal(model(changed)[0][:, :4], lp[:, :4])


def test_likelihood_and_causality():
    for attn in ("softmax", "linear"):
        for short in (0, 3):
            check_likelihood_and_causality(attn, short)


def test_tiny_distilled_training():
    torch.manual_seed(2)
    model = GPT(TFConfig(vocab=11, d_model=16, n_layers=2, n_heads=2, max_len=12,
                         attn="linear", mlp="spike", relay="quant", relay_levels=4, out_classes=5, out_short=3))
    model.head.set_classes(torch.tensor([0, -1, 1, 0, -1, 1, 2, 2, 3, -1, 3]))
    ids = (torch.arange(11)[:, None] + torch.arange(8)) % 11
    targets = (ids + 1) % 11
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    teacher = F.one_hot(targets, 11).float() * 8
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lp, nll, _, _ = model(ids, targets)
        kd = F.kl_div(lp.flatten(0, 1).log_softmax(-1), teacher.flatten(0, 1).log_softmax(-1),
                      log_target=True, reduction="batchmean")
        (0.1 * nll + 0.9 * kd).backward()
        for weight in (model.head.w_cls, model.head.w_word):
            assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
        opt.step()
        if step == 0:
            initial = nll.item()
    final = nll.item()
    assert final < 0.1 * initial
    print(f"Tiny distilled training NLL: {initial:.4f} -> {final:.4f}")


def test_sampling_uses_one_word_block_and_restores_mapping():
    cfg = TFConfig(vocab=11, d_model=16, n_layers=2, n_heads=2, max_len=12, out_classes=3, out_short=3)
    model = GPT(cfg)
    model.head.set_classes(torch.tensor([0, -1, 1, 0, -1, 1, 1, 2, 0, -1, 2]))
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"cfg": asdict(cfg), "arch": "tf", "model": model.state_dict()}, path)
        restored, _ = load(path)
    assert torch.equal(model.head.word_at_slot, restored.head.word_at_slot)
    head = restored.head
    ids = torch.tensor([[0, 1], [2, 3]])
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        prob = restored(ids)[0][:, -1].exp()
    # Force a shortlist draw (word 4) and a class-2 draw (second word is 10).
    with patch.object(head, "forward", side_effect=AssertionError("full head used")), \
         patch("torch.multinomial", side_effect=[torch.tensor([[1], [5]]), torch.tensor([[1]])]) as draw, \
         patch("torch.bmm", wraps=torch.bmm) as project:
        samples = restored.generate(ids, 1, temperature=1, top_k=0, generator=g)
    assert samples.tolist() == [[4], [10]]
    tail = prob[1, [7, 10]]
    top = torch.stack([prob[:, head.word_cls == c].sum(-1) for c in range(head.C)], -1)
    expected_top = torch.cat([prob[:, head.word_cls == -1], top], -1)
    torch.testing.assert_close(draw.call_args_list[0].args[0], expected_top)
    torch.testing.assert_close(draw.call_args_list[1].args[0], F.pad(tail[None] / tail.sum(), (0, 1)))
    assert all(call.kwargs["generator"] is g for call in draw.call_args_list)
    weights, _ = project.call_args.args
    assert project.call_count == 1 and weights.shape == (1, 3, cfg.d_model)
    assert torch.equal(weights[0], head.w_word[6:9])


def test_fineweb_map_and_fpga_cost():
    cfg = TFConfig(vocab=50304, d_model=16, n_layers=2, n_heads=2, out_classes=225, out_short=8192)
    model = GPT(cfg)
    path = Path(__file__).with_name("runs") / "classes/tf163_head225_s8192.npy"
    model.head.set_classes(torch.from_numpy(np.load(path)))
    head = model.head
    args = asdict(cfg)
    cost = tf_hw(args)
    flat = tf_hw(dict(args, out_classes=0, out_short=0))
    top, block = head.w_cls.numel(), head.m * cfg.d_model
    assert cost["readout"] == (8192, 225) and head.m == 188
    assert cost["params"] == model.num_params()
    assert cost["table_weights"] - flat["table_weights"] == head.w_word.numel()
    assert cost["dense_macs"] - flat["dense_macs"] == top + block - cfg.vocab * cfg.d_model
    assert cost["dense_bytes"] - flat["dense_bytes"] == (top - cfg.vocab * cfg.d_model) / 2
    assert cost["per_stream_bytes"] - flat["per_stream_bytes"] == block / 2


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_flat_head_matches_legacy()
    test_likelihood_and_causality()
    test_tiny_distilled_training()
    test_sampling_uses_one_word_block_and_restores_mapping()
    test_fineweb_map_and_fpga_cost()
    print("Flat compatibility, likelihood, causality, training, sampling and FPGA accounting: OK")

"""CPU checks for the optional relay penalty gradient and the distillation taper."""
from math import isclose
from types import SimpleNamespace

import torch

from model_tf import GPT, Relay, TFConfig
from train import distill_weight_at


def test_relay_count_gradient():
    relay = Relay(1, "quant", levels=4)
    relay.keep_count = True
    x = torch.tensor([[-5.0], [-0.25], [0.0], [0.25], [5.0]], requires_grad=True)
    ste_out = relay(x)
    ste_count = relay.count.detach()
    ste_xgrad, = torch.autograd.grad(0.75 * relay.count, x)
    assert torch.count_nonzero(ste_xgrad) == 0

    relay.count_grad_width = 0.5
    smooth_out = relay(x)
    assert torch.equal(ste_out, smooth_out)
    assert torch.equal(ste_count, relay.count) and torch.equal(ste_count, relay.rate)
    xgrad, thgrad = torch.autograd.grad(0.75 * relay.count, (x, relay.log_theta))
    assert xgrad[0] < 0 and xgrad[1] < 0 and xgrad[2] == 0 and xgrad[3] > 0 and xgrad[4] > 0
    assert thgrad < 0

    thresholds = torch.arange(relay.levels) + 0.5
    z = x / relay.log_theta.exp()
    surrogate = (torch.sigmoid((z - thresholds) / 0.5) + torch.sigmoid((-z - thresholds) / 0.5)).sum(-1).mean()
    ref_x, ref_th = torch.autograd.grad(0.75 * surrogate, (x, relay.log_theta))
    torch.testing.assert_close(xgrad, ref_x)
    torch.testing.assert_close(thgrad, ref_th)


def test_causal_relay_forward():
    torch.manual_seed(0)
    model = GPT(TFConfig(vocab=32, d_model=16, n_layers=2, n_heads=2, attn="linear", mlp="spike",
                         relay="quant", relay_levels=4))
    ids = torch.arange(12).unsqueeze(0)
    changed_suffix = torch.cat((ids[:, :6], ids[:, 6:] + 12), 1)
    for relay in model.relays():
        relay.keep_count = True

    def forward(tokens):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _, _, rates = model(tokens)
        return logits, rates, model.relay_spikes()

    ste_logits, ste_rates, ste_count = forward(ids)
    for relay in model.relays():
        relay.count_grad_width = 0.5
    smooth_logits, smooth_rates, smooth_count = forward(ids)
    assert torch.equal(ste_logits, smooth_logits)
    assert torch.equal(ste_rates, smooth_rates) and torch.equal(ste_count, smooth_count)
    changed_logits, _, _ = forward(changed_suffix)
    assert torch.equal(smooth_logits[:, :6], changed_logits[:, :6])


def test_distill_weight_at():
    args = SimpleNamespace(steps=20000, distill_taper_steps=5000, distill_weight=0.9, distill_weight_final=0.0)
    assert distill_weight_at(14999, args) == 0.9
    assert isclose(distill_weight_at(15000, args), 0.9 * 4999 / 5000)
    assert isclose(distill_weight_at(17499, args), 0.45)
    assert distill_weight_at(19999, args) == 0.0
    args.distill_weight_final = 0.2
    assert isclose(distill_weight_at(17499, args), 0.55)
    assert distill_weight_at(19999, args) == 0.2
    args.distill_taper_steps = 1
    assert distill_weight_at(19998, args) == 0.9 and distill_weight_at(19999, args) == 0.2
    args.distill_taper_steps = 0
    assert distill_weight_at(19999, args) == 0.9


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_relay_count_gradient()
    test_causal_relay_forward()
    test_distill_weight_at()
    print("relay penalty gradients, hard counts, causality and distillation taper: OK")

"""CPU calibration and insertion-probe checks on tiny GLA models."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiments.relay_calib import inserted_models, read_windows, run_probe
from model_tf import GPT, TFConfig
from relay_calib import calibrate_relays, measure_relay_rate, observe_relay_inputs
from train import Data


def test_calibration():
    torch.manual_seed(4)
    cfg = TFConfig(vocab=32, d_model=16, n_layers=2, n_heads=2, attn="linear", mlp="spike",
                   relay="quant", relay_levels=4)
    model = GPT(cfg).eval()
    before = {name: t.clone() for name, t in model.state_dict().items()}
    ids = torch.randint(cfg.vocab, (8, 32))
    relays = model.relays()
    names = {name + ".log_theta" for name, module in model.named_modules() if module in relays}

    with torch.autocast("cpu", dtype=torch.bfloat16):
        calibration = calibrate_relays(model, ids)
        thetas = [r.log_theta.clone() for r in relays]
        assert calibration == calibrate_relays(model, ids)
        assert all(torch.equal(r.log_theta, t) for r, t in zip(relays, thetas))

        def confirm_rms(relay, inputs):
            rms = inputs[0].float().square().mean((0, 1)).sqrt().clamp_min(1e-6)
            torch.testing.assert_close(relay.log_theta.exp(), rms * calibration["multiplier"])

        model.train()
        for relay in relays:
            relay.keep_count = True
        with torch.no_grad(), observe_relay_inputs(relays, confirm_rms):
            model(ids)
        # Check against the actual hard-count penalty, including its fan weights.
        rate = model.relay_spikes().item()
        assert abs(rate - 0.25) <= 0.0025
        assert rate == calibration["weighted_rate"]
    assert all(torch.equal(t, before[name]) for name, t in model.state_dict().items() if name not in names)
    assert all(not r._forward_pre_hooks for r in relays)
    assert all(p.grad is None for p in model.parameters())


def test_no_relays_untouched():
    model = GPT(TFConfig(vocab=16, d_model=16, n_layers=1, n_heads=2, attn="linear"))
    before = {name: t.clone() for name, t in model.state_dict().items()}
    assert calibrate_relays(model, torch.zeros(1, 2, dtype=torch.long)) is None
    assert model.training
    assert all(torch.equal(t, before[name]) for name, t in model.state_dict().items())


def test_insertion_probe():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        cfg = TFConfig(vocab=16, d_model=16, n_layers=1, n_heads=2, attn="linear", mlp="spike")
        torch.manual_seed(0)
        raw, ema = GPT(cfg), GPT(cfg)
        source = run / "checkpoint.pt"
        torch.save(dict(model=raw.state_dict(), ema=ema.state_dict(), cfg=asdict(cfg), arch="tf", step=10), source)
        np.random.default_rng(0).integers(0, 16, 4096, dtype=np.uint16).tofile(run / "train.bin")
        args = dict(asdict(cfg), arch="tf", relay="quant", relay_levels=4, relay_target=0.25, seq=16, batch=4,
                    accum=2, data=str(run), init_weights=str(source), carry_state=False, seed=1, threads=1)
        (run / "args.json").write_text(json.dumps(args))

        no_relay, inserted, _ = inserted_models(source, args)
        for name, param in ema.state_dict().items():
            assert torch.equal(no_relay.state_dict()[name], param)
            assert torch.equal(inserted.state_dict()[name], param)
        data = Data("train", args["seq"], args["seed"], args["data"])
        x, y = data.batch(args["batch"])
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            nll = torch.stack([inserted(xm, ym)[1] for xm, ym in zip(x.chunk(2), y.chunk(2))]).mean().item()
        (run / "log.jsonl").write_text(json.dumps({"step": 1, "loss": nll}) + "\n")

        result = run_probe(run, torch.device("cpu"), 42, 3, 4)
        starts = sorted(result["calibration_offsets"] + result["heldout_offsets"])
        assert min(np.diff(starts)) >= args["seq"] + 1
        assert result["first_step"]["nll_minus_logged"] == 0
        assert result == run_probe(run, torch.device("cpu"), 42, 3, 4)
        default, calibrated, matched = (result["heldout"][key] for key in ("default", "calibrated", "calibrated_matched"))
        cal_ids = read_windows(data, result["calibration_offsets"], "cpu")[:, :-1].contiguous()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            default_rate = measure_relay_rate(inserted, cal_ids)
        fit = result["matched_calibration"]
        assert fit["target"] == result["default_calibration_rate"] == default_rate
        assert abs(fit["weighted_rate"] - default_rate) <= fit["tolerance"]
        assert result["pass"] == (calibrated["excess_nll"] <= default["excess_nll"] / 2)
        for arm in (default, calibrated, matched):
            assert arm["excess_nll"] == arm["nll"] - result["heldout"]["source_nll"]
            assert len(arm["relays"]) == 3
            for stats in arm["relays"].values():
                assert 0 <= stats["clipped_fraction"] <= stats["at_top_fraction"] <= 1
                assert 0 <= stats["silent_channel_fraction"] <= 1


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_calibration()
    test_no_relays_untouched()
    test_insertion_probe()
    print("relay calibration, hard budget, determinism and no-training insertion probe: OK")

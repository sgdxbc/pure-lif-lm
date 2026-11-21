"""CPU checks for signed vocabulary-head replay."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import fpga_est
from experiments import readout_replay as replay
from model_tf import GPT, TFConfig


class ReadoutReplayTest(unittest.TestCase):
    def test_refresh_each_token_equals_exact(self):
        torch.manual_seed(92)
        cfg = TFConfig(vocab=32, d_model=12, n_layers=2, n_heads=2, attn="linear", max_len=16)
        model = GPT(cfg).eval()
        ids = torch.randint(cfg.vocab, (2, 15))
        with torch.no_grad():
            exact = model(ids)[0]
            features = model._features(ids)[0]
            for codec in ("events", "int4"):
                updates, active = replay.encode_updates(features.reshape(-1, 1, 12).numpy(), np.zeros(12), codec)
                self.assertEqual(active.sum(), 0)
                logits = replay.replay_logits(model.head.weight, torch.from_numpy(updates)).reshape_as(exact)
                torch.testing.assert_close(logits, exact, rtol=0, atol=0)
            updates, active = replay.encode_updates(features.numpy(), np.full(12, 0.25), "events")
            self.assertGreater(active.sum(), 0)
            updates = torch.from_numpy(updates)
            torch.testing.assert_close(replay.replay_logits(model.head.weight, updates),
                                       F.linear(updates.cumsum(1), model.head.weight))

    def test_accounting_includes_refresh_row_reads_and_cache(self):
        cfg = TFConfig(vocab=8, d_model=4, n_layers=1, n_heads=2, attn="linear")
        base = fpga_est.tf_hw(vars(cfg), bits=4)
        dev = dict(fpga_est.DEVICES["u55c"], cards=1, eff=0.6, onchip_mb=0)
        baseline = fpga_est.estimate_tf(base, dev)
        for codec in ("events", "int4"):
            hw = replay.replay_hw(base, 4, 8, active_rate=0.5, refresh_rate=0.25, codec=codec)
            # 32 dense MACs become 8 refresh MACs plus 12 incremental row ops.
            self.assertEqual(hw["dense_macs"], base["dense_macs"] - 32 + 8 + (12 if codec == "int4" else 0))
            # 4.5 codec adds = 3 residual subtractions and 1.5 reconstruction adds per token.
            self.assertEqual(hw["sparse_adds"], 4.5 + (12 if codec == "events" else 0))
            for original, row in zip(baseline, fpga_est.estimate_tf(hw, dev)):
                streams = row["streams"]
                cache_traffic = row["placement"]["state"]["hbm_bytes_per_step"]
                self.assertEqual(cache_traffic - original["placement"]["state"]["hbm_bytes_per_step"],
                                 2 * (4 + 8) * 4 * streams)
                # Four 4-byte rows, read in full at refresh or once if any stream needs a row.
                head = row["placement"]["sparse_weights"]
                self.assertEqual(head["hbm_bytes"], 16)
                wanted_bytes = 16 * (0.25 + 0.75 * (1 - 0.5 ** streams))
                self.assertEqual(head["hbm_bytes_per_step"], wanted_bytes)
                expected_ops = hw["dense_macs"] + hw["sparse_adds"]
                self.assertEqual(row["ops_per_step"], streams * expected_ops)

    def test_int4_error_bound_with_and_without_saturation(self):
        x = np.array([[[0.25, 1.], [0.8, -0.55], [9., -30.], [7., -9.], [5., 25.]]], np.float32)
        step = np.array([0.5, 1.25], np.float32)
        updates, active = replay.encode_updates(x, step, "int4")
        state = updates.cumsum(1)
        residual = x[:, 1:] - state[:, :-1]
        bound = np.maximum(step / 2, np.abs(residual) - 7 * step)
        self.assertTrue(np.any(bound > step / 2))
        self.assertTrue(np.any(bound == step / 2))
        self.assertTrue(np.all(np.abs(x[:, 1:] - state[:, 1:]) <= bound + 1e-6))
        np.testing.assert_array_equal(active[:, 1:], np.count_nonzero(updates[:, 1:], axis=-1))
        np.testing.assert_array_equal(updates[:, 1:] / step, (updates[:, 1:] / step).round())

    def test_cli_reads_ema_and_writes_paired_metrics(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            cfg = TFConfig(vocab=32, d_model=8, n_layers=1, n_heads=2, attn="linear", max_len=16)
            model = GPT(cfg)
            weights = model.state_dict()
            raw = {key: torch.zeros_like(t) for key, t in weights.items()}
            torch.save(dict(arch="tf", cfg=vars(cfg), model=raw, ema=weights, step=42), path / "ckpt.pt")
            (path / "args.json").write_text(json.dumps(dict(data=temp, seq=8)))
            (np.arange(50) % cfg.vocab).astype(np.uint16).tofile(path / "train.bin")
            (np.arange(50) % cfg.vocab).astype(np.uint16).tofile(path / "valid.bin")
            cli = ["readout_replay.py", temp, "--thresholds", "0.5", "--refresh", "3",
                   "--batch", "3", "--calibration-tokens", "16"]
            with patch("sys.argv", cli):
                replay.main()
            report = json.loads((path / "readout_replay.json").read_text())
            self.assertEqual(report["step"], 42)
            self.assertTrue(report["ema"])
            self.assertEqual(report["validation_tokens"], 48)
            self.assertEqual(report["refresh_count"], 18)
            self.assertEqual(report["cache_traffic_bytes_per_token"], 2 * (8 + 32) * 4)
            self.assertNotAlmostEqual(report["exact"]["nll"], float(np.log(32)), places=5)
            for result in report["replays"]["0.5"].values():
                self.assertGreaterEqual(result["kl_exact_to_replay"], 0)
                self.assertEqual(set(result["cost"]), {"1", "16"})
                self.assertEqual(set(result["nonrefresh_burst"]["channels"]), {"mean", "p50", "p90", "p99"})
                self.assertEqual(result["projection_row_burst"]["channels"]["p90"], cfg.d_model)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

"""Verify partial placement without checkpoints or accelerator access."""
import unittest

from fpga_est import DEVICES, STREAMS, estimate, estimate_tf, place, tf_hw


def device(budget):
    return dict(DEVICES["u55c"], onchip_mb=budget / 900_000, cards=1, eff=0.6)


class PlacementTest(unittest.TestCase):
    def setUp(self):
        self.args = dict(vocab=16, d_model=2, d_hidden=4, n_layers=2)
        self.snn = dict(params=128, table_weights=16, dense_weights=96, per_stream_weights=2,
                        state_bytes=10, dense_macs=12, sparse_adds=2,
                        firing=dict(token=[0.2, 0.5], token_any=[0.3, 0.8]))
        self.tf = tf_hw(dict(vocab=48, d_model=12, n_layers=2, n_heads=3, seq=16, mlp="lif"),
                        rates=[0.25, 0.75], ctx=16, bits=4)

    def test_priority_by_traffic_per_byte(self):
        regions = [("sparse_weights", 20, 5), ("state", 10, 20), ("dense_weights", 16, 16),
                   ("sparse_weights", 40, 30), ("kv", 14, 14)]
        read, _, split = place(regions, 50, 100, 8)
        self.assertEqual(split["state"]["onchip_bytes"], 10)
        self.assertEqual(split["dense_weights"]["onchip_bytes"], 16)
        self.assertEqual(split["kv"]["onchip_bytes"], 14)
        self.assertEqual(split["sparse_weights"]["onchip_bytes"], 10)
        self.assertEqual(read, 5 + 22.5 + 8)

    def test_nothing_fits(self):
        for s, t, streams in zip(estimate(self.snn, self.args, 4, device(0)),
                                 estimate_tf(self.tf, device(0)), STREAMS):
            shares = ([0.2, 0.5] if streams == 1 else
                      [1 - (1 - p) ** (streams / 16) for p in (0.3, 0.8)])
            expected_snn = 48 + 4 * sum(shares) + streams * (1 + 20)
            self.assertAlmostEqual(s["bytes_per_step"], expected_snn)
            expected_tf = self.tf["dense_bytes"] + sum(
                (1 - (1 - rate) ** streams) * rows * width for rate, rows, width in self.tf["sparse_layers"])
            expected_tf += streams * (self.tf["per_stream_bytes"] + 2 * self.tf["state_bytes"] + self.tf["kv_bytes"])
            self.assertAlmostEqual(t["bytes_per_step"], expected_tf)
            for row in (s, t):
                self.assertTrue(all(p["onchip_bytes"] == 0 for p in row["placement"].values()))
                self.assertGreater(row["placement"]["state"]["hbm_bytes"], 0)

    def test_all_fits_preserves_estimate(self):
        dev = device(1_000_000)
        bw = dev["bw_gbs"] * 1e9 * dev["eff"]
        rate = dev["dsp"] * dev["macs_per_dsp"] * dev["clock_mhz"] * 1e6 * dev["eff"]
        tf_ops = self.tf["dense_macs"] + self.tf["sparse_adds"]
        for rows, own, ops in ((estimate(self.snn, self.args, 4, dev), 1, 14),
                               (estimate_tf(self.tf, dev), self.tf["per_stream_bytes"], tf_ops)):
            for row in rows:
                streams = row["streams"]
                old_time = max(streams * own / bw, streams * ops / rate)
                self.assertEqual(row["where"], "weights on chip, word tables in HBM")
                self.assertEqual(row["bytes_per_step"], streams * own)
                self.assertEqual(row["ops_per_step"], streams * ops)
                self.assertAlmostEqual(row["ms_per_token"], old_time * 1000)
                self.assertAlmostEqual(row["mJ_per_token"], dev["watts"] * old_time * 1000 / streams)
                self.assertTrue(all(p["hbm_bytes"] == 0 for k, p in row["placement"].items() if k != "word_tables"))
        # Old SNN files sometimes omit the separate embedding/word table size.
        legacy = dict(self.snn, table_weights=0)
        self.assertTrue(all(row["bytes_per_step"] == 0 for row in estimate(legacy, self.args, 4, dev)))

    def test_budget_and_monotonicity(self):
        for compute in (lambda dev: estimate(self.snn, self.args, 4, dev), lambda dev: estimate_tf(self.tf, dev)):
            budgets = (0, 5, 20, 53, 56, 90, 100, 250, 1000, 5000, 20000, 1_000_000)
            rows = [compute(device(b)) for b in budgets]
            for budget, batch in zip(budgets, rows):
                for row in batch:
                    placement = row["placement"]
                    self.assertAlmostEqual(sum(p["onchip_bytes"] for p in placement.values()),
                                           min(budget, row["required_onchip_bytes"]))
                    self.assertAlmostEqual(row["bytes_per_step"],
                                           sum(p["hbm_bytes_per_step"] for p in placement.values()))
            for smaller, larger in zip(rows, rows[1:]):
                for a, b in zip(smaller, larger):
                    for key in ("bytes_per_step", "ms_per_token", "mJ_per_token"):
                        self.assertLessEqual(b[key], a[key] + 1e-9)

    def test_trained_sparse_row_bits_and_scales(self):
        args = dict(self.args, wgroup=2)
        snn = dict(self.snn, wbits=4, weights_bytes_trained=140, table_bytes_trained=12, wbits8="layers.0.w_out")
        row = estimate(snn, args, 4, device(0))[0]
        sparse = row["placement"]["sparse_weights"]
        self.assertEqual(sparse["hbm_bytes"], 16 + 12)  # Eight weights per layer, four 16-bit scales per layer.
        self.assertAlmostEqual(sparse["hbm_bytes_per_step"], 0.2 * 16 + 0.5 * 12)
        self.assertEqual(row["placement"]["dense_weights"]["hbm_bytes"], 128 - 28)


if __name__ == "__main__":
    unittest.main()

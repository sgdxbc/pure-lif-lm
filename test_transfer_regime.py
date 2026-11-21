import unittest

from experiments.transfer_regime_bound import upper_bound


class TrafficBoundTest(unittest.TestCase):
    def test_shared_weights_and_per_stream_state(self):
        one = upper_bound(400, 100, 80, 40, 1, 10)
        four = upper_bound(400, 100, 80, 40, 4, 10)
        self.assertEqual(one["always_refresh_bytes_per_token"], 620)
        self.assertEqual(one["best_case_bytes_per_token"], 490)
        self.assertEqual(one["maximum_saving_fraction"], 130 / 620)
        self.assertTrue(one["clears_bound"])
        self.assertEqual(four["always_refresh_bytes_per_token"], 245)
        self.assertEqual(four["best_case_bytes_per_token"], 190)
        self.assertEqual(four["maximum_saving_fraction"], 55 / 245)

    def test_detector_cost_can_close_gate(self):
        row = upper_bound(0, 0, 78, 22, 1, 3)
        self.assertEqual(row["required_suppressible_bytes_at_this_baseline"], 23)
        self.assertFalse(row["clears_bound"])


if __name__ == "__main__":
    unittest.main()

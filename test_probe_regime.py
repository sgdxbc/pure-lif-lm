"""Known-answer, gradient and traffic checks for the CPU regime probe."""
import unittest

import numpy as np

from experiments.probe_regime import CLEAN, Traffic, bank_traffic, gradient, infer, transition


class RegimeTest(unittest.TestCase):
    def setUp(self):
        self.tables = np.full((2, 8, 8), .05 / 7, dtype=np.float32)
        for r in range(2):
            self.tables[r, np.arange(8), CLEAN[r]] = .95

    def test_crossing_and_hold_after_single_mismatch(self):
        config = np.array([[.5], [.8]], dtype=np.float32)
        cue = np.zeros((7, 1), dtype=np.uint8)
        seen = np.array([[0], [0], [4], [0], [4], [4], [4]], dtype=np.uint8)
        _, selected, _, traffic, events, refreshes = infer(config, self.tables, cue, seen, "events")
        self.assertEqual(selected[:, 0].tolist(), [-1, 0, 0, 0, 0, 0, 1])
        self.assertEqual((events, refreshes), (2, 2))
        self.assertEqual((traffic.read, traffic.written, traffic.input), (546, 32, 14))
        v, votes = transition(np.zeros((1, 1)), np.ones((1, 1)), np.array([-.3]), *config)
        self.assertAlmostEqual(v.item(), -.3)
        self.assertEqual(votes.item(), 1.)

    def test_hand_count_one_observation(self):
        traffic = Traffic()
        bank_traffic(traffic, cells=2, observations=1, cell_events=1, refreshes=1)
        self.assertEqual(traffic, Traffic(read=91, written=10, input=2))
        analog = Traffic()
        bank_traffic(analog, cells=2, observations=1, cell_events=0, refreshes=1, analog=True)
        self.assertEqual(analog, Traffic(read=84, written=12, input=2))

    def test_forward_eligibility_matches_bptt(self):
        rng = np.random.default_rng(203)
        cue = rng.integers(8, size=(11, 3), dtype=np.uint8)
        observed = rng.integers(8, size=cue.shape, dtype=np.uint8)
        params = np.array([[.2, .6], [.7, 1.2]], dtype=np.float32)
        a, _ = gradient(params, self.tables, cue, observed, "bptt")
        b, storage = gradient(params, self.tables, cue, observed, "eligibility")
        np.testing.assert_allclose(a, b, atol=1e-6)
        longer = gradient(params, self.tables, np.tile(cue, (4, 1)),
                          np.tile(observed, (4, 1)), "eligibility")[1]
        self.assertEqual(longer, storage)


if __name__ == "__main__":
    unittest.main()

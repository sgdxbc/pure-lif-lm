import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from experiments.event_readout import nested_windows, rejection, sparse_likelihood, sparse_scores, untouched_bounds


class EventReadoutTests(unittest.TestCase):
    def test_sparse_scores_match_dense_sum_with_duplicates_and_silence(self):
        words = torch.tensor([[0, 1], [1, 3], [2, 3]])
        weights = torch.tensor([[.2, .3], [-.5, .7], [1., 0.]])
        events = torch.tensor([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.bool)
        bias = torch.tensor([.1, .5, -.3, .1])
        scores, touched = sparse_scores(events, words, weights, bias)
        dense = torch.zeros(3, 4).scatter_(1, words, weights)
        torch.testing.assert_close(scores, events.float() @ dense + bias)
        targets = torch.tensor([1, 0, 3])
        logp = (events.double() @ dense.double() + bias.double()).log_softmax(-1)
        torch.testing.assert_close(sparse_likelihood(events, words, weights, bias, targets),
                                   logp.gather(1, targets[:, None])[:, 0])
        self.assertEqual(touched.tolist(), [[True, True, False, True], [False] * 4, [False, False, True, True]])

    def test_untouched_conditional_fixes_kl(self):
        p = torch.tensor([[.5, .4, .1]] * 3, dtype=torch.float64)
        touched = torch.tensor([[True, False, False], [False] * 3, [True] * 3])
        target = torch.tensor([1, 1, 1])
        bias = torch.zeros(3, dtype=torch.float64)
        kl, floor, log_z = untouched_bounds(p.log(), touched, bias, target)
        oracle = torch.tensor([.5, .25, .25], dtype=torch.float64)
        self.assertAlmostEqual(kl[0].item(), F.kl_div(oracle.log(), p[0], reduction="sum").item())
        self.assertAlmostEqual(kl[1].item(), (p[1] * (p[1].log() + math.log(3))).sum().item())
        self.assertEqual(kl[2].item(), 0.)
        self.assertEqual(floor.tolist(), [math.log(2), math.log(3), 0.])
        self.assertEqual(log_z[2].item(), -math.inf)

    def test_fit_sizes_share_one_disjoint_window_order(self):
        excluded = {8, 16, 40}
        sizes = (2, 5, 9)
        fit = nested_windows(97, 4, excluded, sizes)
        again = nested_windows(97, 4, excluded, sizes)
        for expected, windows, repeat in zip(sizes, fit, again):
            self.assertEqual(len(windows), expected)
            np.testing.assert_array_equal(windows, fit[-1][:expected])
            np.testing.assert_array_equal(windows, repeat)
        self.assertEqual(len(set(fit[-1])), 9)
        self.assertFalse(set(fit[-1]) & excluded)

    def test_rejection_requires_both_gaps_at_every_fanout(self):
        baseline = .25
        points = {32: {"fitted_kl": .8, "untouched_kl_floor": .5},
                  128: {"fitted_kl": .4, "untouched_kl_floor": .3}}
        self.assertEqual(rejection(points, baseline), "retire")
        points[128]["untouched_kl_floor"] = baseline + .012
        self.assertEqual(rejection(points, baseline), "not retired, fitting-limited")
        points[128]["fitted_kl"] = baseline + .012
        self.assertEqual(rejection(points, baseline), "parity_candidate")


if __name__ == "__main__":
    unittest.main()

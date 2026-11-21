"""CPU checks for the 5.63 frozen-head conversion."""
import unittest

import torch
import torch.nn.functional as F

from experiments.head_screen import FrozenHead, frontier_at


class HeadScreenTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(63)
        self.x = torch.randn(13, 7)
        self.weight = torch.randn(11, 7)
        self.flat = F.linear(self.x, self.weight)
        self.targets = torch.arange(13) % 11

    def test_marginal_kl_is_full_vocab_kl(self):
        partition = (torch.arange(11) - 2).clamp_min(0) % 3
        partition[:2] = -1
        head = FrozenHead(self.weight, partition, 3, nonlinear=True)
        flat_top = head.flat_top(self.flat)
        top = head.top(self.x, flat_top[:, :2])
        fitted = head.log_probs(self.flat, flat_top, top)
        full = F.kl_div(fitted, self.flat.log_softmax(-1), log_target=True, reduction="batchmean")
        torch.testing.assert_close(head.kl(self.x, flat_top[:, 2:]), full, atol=1e-6, rtol=1e-6)
        sizes = [parameter.numel() for parameter in head.parameters()]
        self.assertEqual(sizes, [7 * 3, 3, 7 * 256, 256 * 3])

    def test_conditionals_and_exact_target_likelihood(self):
        for short in (0, 2):
            with self.subTest(short=short):
                partition = (torch.arange(11) - short).clamp_min(0) % 3
                partition[:short] = -1
                head = FrozenHead(self.weight, partition, 3)
                exact_top = head.flat_top(self.flat)
                top = head.top(self.x, exact_top[:, :short])
                actual = head.log_probs(self.flat, exact_top, top)
                explicit = actual.clone()
                for group in range(-1, 3):
                    words = (partition == group).nonzero().flatten()
                    if group == -1:
                        explicit[:, words] = top.log_softmax(-1)[:, :short]
                        torch.testing.assert_close(top[:, :short], self.flat[:, words], atol=0., rtol=0.)
                    else:
                        raw = F.linear(self.x, self.weight[words])
                        conditional = raw.log_softmax(-1)
                        result = actual[:, words].log_softmax(-1)
                        torch.testing.assert_close(result, conditional, atol=1e-6, rtol=1e-6)
                        explicit[:, words] = conditional + top.log_softmax(-1)[:, short + group, None]
                target = explicit.gather(-1, self.targets[:, None]).squeeze(1)
                torch.testing.assert_close(actual, explicit, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(head.target_log_probs(self.flat, self.x, self.targets),
                                           target, atol=1e-6, rtol=1e-6)
                perfect = head.log_probs(self.flat, exact_top, exact_top)
                torch.testing.assert_close(perfect, self.flat.log_softmax(-1), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(F.nll_loss(perfect, self.targets),
                                           F.cross_entropy(self.flat, self.targets), atol=1e-6, rtol=0.)


def point(loss, cost, passes=True, lambada=.5):
    return dict(valid_nll=loss, cost={"1": {"mJ_per_token": cost}}, passes=passes, lambada=lambada)


class VerdictTest(unittest.TestCase):
    def test_band_edges_and_guards(self):
        dense = [point(4., 1.), point(2., 4.)]
        cases = ((-.021, "favors_dense"), (-.020, "undecided"), (-.012, "undecided"),
                 (-.011, "parity"), (0., "parity"), (.011, "parity"),
                 (.012, "undecided"), (.020, "undecided"), (.021, "favors_relay"))
        for gap, band in cases:
            result = frontier_at([point(3. - gap, 2.)] * 2, dense, 1)
            self.assertEqual(result["band"], band)
            self.assertAlmostEqual(result["D_mean"], gap)
            self.assertEqual(result["credit"], band == "favors_relay")
        self.assertFalse(frontier_at([point(2.9, 2.), point(2.9, 2., False)], dense, 1)["credit"])
        self.assertFalse(frontier_at([point(2.9, 2.), point(3.01, 2.)], dense, 1)["credit"])
        result = frontier_at([point(3., 5.), point(2.9, 2.)], dense, 1)
        self.assertEqual(result["band"], "outside_dense_bracket")
        self.assertEqual(result["dominating_dense_indices_per_seed"], [[1], []])

    def test_interpolate_at_each_seeds_cost(self):
        dense = [point(4., 1.), point(2., 4.)]
        result = frontier_at([point(3.49, 2 ** .5), point(2.48, 2 ** 1.5)], dense, 1)
        self.assertAlmostEqual(result["D_per_seed"][0], .01)
        self.assertAlmostEqual(result["D_per_seed"][1], .02)
        self.assertAlmostEqual(result["D_mean"], .015)


if __name__ == "__main__":
    unittest.main()

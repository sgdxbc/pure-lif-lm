import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parametrize

from experiments import align_pilot as pilot
from experiments.memory_maps import Read
from model import Config, SpikingLM
from model_tf import GPT, TFConfig


class AlignmentPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_selected_projection_rows_can_change_under_adamw(self):
        torch.manual_seed(3)
        net = SpikingLM(Config(vocab=24, d_model=8, d_hidden=16, n_layers=2,
                                mem_heads=2, mem_keys=3, mem_dv=4, mem_every=1))
        before = {name: t.clone() for name, t in net.state_dict().items()}
        params, projections = pilot.train_rows(net, ((1, 1),))
        self.assertEqual(len(params), 2)
        self.assertEqual(sum(p.numel() for p in params), 6 * 9)
        opt = torch.optim.AdamW(params, lr=.02, weight_decay=.1)
        # An explicit zero gradient still triggers decay on parameters in AdamW.
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
        projections[0](torch.ones(2, 8)).sum().backward()
        opt.step()
        for attribute in ("weight", "bias"):
            parametrize.remove_parametrizations(projections[0], attribute, leave_parametrized=True)
        picked = [3, 4, 5, 9, 10, 11]
        fixed = [0, 1, 2, 6, 7, 8]
        for name, actual in net.state_dict().items():
            if name in ("mems.1.kq_in.weight", "mems.1.kq_in.bias"):
                self.assertTrue(torch.equal(before[name][fixed], actual[fixed]), name)
                self.assertTrue((before[name][picked] != actual[picked]).all(), name)
            else:
                self.assertTrue(torch.equal(before[name], actual), name)

    def test_sham_preserves_probability_multisets_by_lag_and_entropy(self):
        positions = torch.tensor([13, 10])
        cols = torch.arange(17)[None, :]
        raw = (cols + 1.).expand(2, -1).masked_fill(cols > positions[:, None], 0.)
        target = raw / raw.sum(-1, keepdim=True)
        sham = pilot.sham_targets(target, positions, 43)
        self.assertFalse(torch.equal(target, sham))
        for q, a, b in zip(positions, target, sham):
            lags = q - cols[0]
            for lo, hi in zip(pilot.LAGS[:-1], pilot.LAGS[1:]):
                band = (lo <= lags) & (lags < hi)
                torch.testing.assert_close(a[band].sort().values, b[band].sort().values, atol=0, rtol=0)
            self.assertTrue(torch.equal(a[lags < 0], b[lags < 0]))
        torch.testing.assert_close(torch.special.xlogy(target, target).sum(-1),
                                   torch.special.xlogy(sham, sham).sum(-1))

    def test_group_target_and_kl_match_manual_causal_decay(self):
        q = torch.ones(1, 2, 3, 1)
        k = torch.tensor([[[[2.], [1.], [8.]], [[0.], [3.], [7.]]]])
        trace = Read(q, k, k, torch.tensor([[[math.log(.5)], [0.]]]), k)
        rows = torch.tensor([[0, 1]])
        heads = ((0, 0), (0, 1))
        teacher = pilot.group_distribution({0: trace}, heads, rows, True)
        hi = math.exp(2) / (math.exp(2) + math.exp(1))
        lo = 1 / (1 + math.exp(3))
        p = (hi + lo) / 2
        expected_teacher = torch.tensor([[p, 1 - p, 0.]])
        torch.testing.assert_close(teacher, expected_teacher)
        student = pilot.group_distribution({0: trace}, heads, rows, False)
        s = (.5 + .0005 / 3.001) / 2
        torch.testing.assert_close(student, torch.tensor([[s, 1 - s, 0.]]))
        expected_kl = p * math.log(p / s) + (1 - p) * math.log((1 - p) / (1 - s))
        self.assertAlmostEqual(float(pilot.address_kl(student, teacher)), expected_kl, places=6)

    def test_arms_use_identical_windows_queries_and_seeds(self):
        data = np.tile([23, 2, 3, 2, 3, 4, 2], 32)
        plan = {"train_offsets": pilot.make_offsets(len(data), seq=6, steps=1)}
        seen = {}
        for arm in pilot.ARMS:
            p = torch.nn.Parameter(torch.tensor(1.))
            traces = []
            def inspect(student, teacher, windows, queries, dev, auxiliary, seed):
                traces.append((windows.tolist(), queries.tolist(), seed))
                return (p - 2).square(), None, None, None
            with patch.object(pilot, "losses", side_effect=inspect), patch.object(
                    pilot, "microbatches", side_effect=lambda arr, ix: pilot_windows(arr, ix)):
                pilot.training_step(None, None, [p], data, plan, 1, arm, torch.device("cpu"),
                                    torch.optim.AdamW([p]), .1, 23)
            seen[arm] = traces
        self.assertEqual(len(seen["plain"]), 4)
        self.assertTrue(any(batch[1] for batch in seen["plain"]))
        self.assertEqual(seen["plain"], seen["sham"])
        self.assertEqual(seen["plain"], seen["aligned"])

    def test_tiny_models_trace_and_train_with_real_losses(self):
        torch.manual_seed(7)
        student = SpikingLM(Config(vocab=24, d_model=8, d_hidden=16, n_layers=1, mem_heads=2,
                                    mem_keys=3, mem_dv=4, mem_theta=.05, mem_every=1))
        teacher = GPT(TFConfig(vocab=24, d_model=8, n_layers=1, n_heads=2))
        teacher.requires_grad_(False)
        heads = ((0, 1),)
        params, _ = pilot.train_rows(student, heads)
        windows = np.array([[23, 2, 3, 2, 4, 3, 2]])
        q = pilot.eligible_queries(windows, 23, 4)
        with patch.object(pilot, "STUDENT", heads), patch.object(pilot, "TEACHER", heads):
            task, aux, sham, _ = pilot.losses(student, teacher, windows, q, torch.device("cpu"), True, 43)
        self.assertTrue(torch.isfinite(torch.stack((task, aux, sham))).all())
        (task + aux).backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in params))


def pilot_windows(data, offsets):
    for i in range(0, len(offsets), pilot.MICRO):
        yield np.stack([data[j:j + 7] for j in offsets[i:i + pilot.MICRO]]).astype(np.int64)


if __name__ == "__main__":
    unittest.main()

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parametrize

from experiments import align_pilot as align
from experiments import query_pilot as pilot
from experiments import write_read
from model import Config, SpikingLM
from model_tf import GPT, TFConfig


class QueryPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_q_weight_and_bias_rows_change(self):
        torch.manual_seed(7)
        model = SpikingLM(Config(vocab=16, d_model=8, d_hidden=16, n_layers=2,
                                 mem_heads=2, mem_keys=3, mem_dv=4, mem_every=1))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        params, layers = align.train_rows(model, ((1, 1),), query_only=True)
        self.assertEqual(sum(p.numel() for p in params), 3 * 9)
        opt = torch.optim.AdamW(params, lr=.01, weight_decay=.1)
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad()
        layers[0](torch.ones(2, 8)).sum().backward()
        opt.step()
        for name in ("weight", "bias"):
            parametrize.remove_parametrizations(layers[0], name, leave_parametrized=True)
        for key, value in model.state_dict().items():
            if key in ("mems.1.kq_in.weight", "mems.1.kq_in.bias"):
                self.assertTrue(torch.equal(before[key][:9], value[:9]), key)
                self.assertTrue((before[key][9:] != value[9:]).all(), key)
            else:
                self.assertTrue(torch.equal(before[key], value), key)

    def test_screen_eligibility_and_closest_wrong_target(self):
        pieces = ["<|endoftext|>", "He", "Ġsaw", "ĠBob", "Ġand", "ĠAlice",
                  "Ġwith", "ĠDan", "Ġthen", "ĠCarol", "Ġasked"]
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 0, 10, 5]
        windows = np.array([ids, ids])
        dev = torch.device("cpu")
        _, known = write_read.window_candidates(ids, [pieces[i] for i in ids], 0)
        self.assertEqual(known, [write_read.Candidate(query=10, source=5, wrong=[3, 7, 9], name_source=True)])

        def screen(teacher, batch, indexed, heads, device):
            self.assertEqual(indexed, [(0, known[0]), (1, known[0])])
            scores = torch.zeros((2, len(ids) - 1), dtype=torch.float64)
            scores[:, 5] = torch.tensor([.20, .19], dtype=torch.float64)
            scores[:, [3, 7, 9]] = .18
            hits = write_read.teacher_hits(scores, [candidate for _, candidate in indexed])
            return [row for row, hit in zip(indexed, hits) if hit]

        with patch.object(pilot, "passed_candidates", side_effect=screen):
            self.assertEqual(pilot.target_rows(None, windows, pieces, 0, dev), [(0, 10, 5, 3)])

    def test_address_loss_on_known_probabilities(self):
        p = torch.tensor([[.25, .5, .25, 0.], [.10, .20, .25, .45]], dtype=torch.float64)
        true = pilot.address_loss(p, torch.tensor([0, 3]))
        sham = pilot.address_loss(p, torch.tensor([1, 2]))
        self.assertAlmostEqual(true.item(), -(math.log(.25) + math.log(.45)) / 2)
        self.assertAlmostEqual(sham.item(), -(math.log(.50) + math.log(.25)) / 2)

    def test_arms_see_identical_batches_and_calibration_uses_query_counts(self):
        data = np.arange(30)
        plan = dict(train_offsets=[[0, 6, 12, 18]], targets=[[(1, 3, 0, 1), (1, 4, 0, 1), (3, 3, 1, 2)]])
        seen = {}
        dev = torch.device("cpu")
        with patch.object(pilot, "MICRO", 2), patch.object(pilot, "BATCH", 4), patch.object(
                align, "microbatches", side_effect=lambda data, offsets: (
                    np.stack([data[i:i + 6] for i in offsets[j:j + 2]]) for j in (0, 2))):
            for arm in (*pilot.ARMS, "calibration"):
                p = torch.nn.Parameter(torch.tensor(1.))
                batches = []
                def inspect(student, teacher, windows, positions, device):
                    batches.append((windows.tolist(), positions))
                    return p * 2, p * len(positions), p * (2 * len(positions))
                opt = None if arm == "calibration" else torch.optim.AdamW([p])
                with patch.object(pilot, "losses", side_effect=inspect):
                    row = pilot.training_step(None, None, [p], data, plan, 0, arm, dev, opt=opt, weight=.1)
                self.assertEqual(row["eligible_positions"], 3)
                if opt is None:
                    self.assertAlmostEqual(row["grad_norms"]["task"], 2.)
                    self.assertAlmostEqual(row["grad_norms"]["true"], 5 / 3, places=6)
                    self.assertAlmostEqual(row["grad_norms"]["sham"], 10 / 3, places=6)
                seen[arm] = batches
        self.assertTrue(all(batches == seen["plain"] for batches in seen.values()))
        self.assertEqual(seen["plain"][1][1], [(1, 3, 1, 2)])

    def test_tiny_cpu_forward_calibration_and_update(self):
        torch.manual_seed(19)
        student = SpikingLM(Config(vocab=16, d_model=8, d_hidden=16, n_layers=1,
                                   mem_heads=2, mem_keys=3, mem_dv=4, mem_theta=.05, mem_every=1))
        teacher = GPT(TFConfig(vocab=16, d_model=8, n_layers=1, n_heads=2)).requires_grad_(False)
        heads = ((0, 1),)
        params, _ = align.train_rows(student, heads, query_only=True)
        windows = np.array([[7, 2, 3, 4, 6, 2]])
        plan = dict(train_offsets=[[0]], targets=[[(0, 4, 1, 2)]])
        with patch.object(align, "STUDENT", heads), patch.object(pilot, "BATCH", 1), patch.object(
                align, "microbatches", return_value=[windows]):
            row = pilot.training_step(student, teacher, params, windows[0], plan, 0, "true", torch.device("cpu"))
            for value in row["grad_norms"].values():
                self.assertTrue(np.isfinite(value))
        with patch.object(align, "STUDENT", heads), patch.object(pilot, "BATCH", 1), patch.object(
                align, "microbatches", return_value=[windows]):
            opt = torch.optim.AdamW(params)
            row = pilot.training_step(student, teacher, params, windows[0], plan, 0, "true", torch.device("cpu"),
                                      opt=opt, weight=.1)
            self.assertTrue(np.isfinite(row["task"]))


if __name__ == "__main__":
    unittest.main()

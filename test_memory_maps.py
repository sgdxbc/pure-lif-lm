"""Tiny CPU tests of write attribution, carried state and paired interventions."""
import unittest

import numpy as np
import torch

from experiments.memory_maps import (Read, check_reconstruction, induction_source, intervene,
                                     make_case, measure_cases, measure_range, synthetic_cases, traced_forward)
from model import Config, SpikingLM
from model_tf import GPT, TFConfig


class MemoryMapsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(63)
        self.tokens = torch.randint(32, (2, 157))

    def models(self):
        opts = dict(vocab=32, d_model=12, d_hidden=16, n_layers=2, mem_heads=2, mem_theta=.6)
        plastic = SpikingLM(Config(**opts, mem_keys=4, mem_dv=6, mem_delays=4, mem_kq_levels=4,
                                   mem_write_avg=True, mem_chunk=23, mem_gate="channel"))
        for mem in plastic.mems.values():
            mem.delay_dw.data.normal_(std=.15)
            mem.w_out.weight.data.normal_(std=.1)
        opts_tf = dict(vocab=32, d_model=12, n_layers=2, n_heads=2, max_len=256)
        return (plastic, SpikingLM(Config(**opts, mem_kind="gla")),
                GPT(TFConfig(**opts_tf, attn="linear")), GPT(TFConfig(**opts_tf, rope=True)))

    @torch.no_grad()
    def test_exact_reads_across_chunk_boundary_and_carried_state(self):
        for model in self.models():
            with self.subTest(config=model.cfg):
                model.eval()
                self.assertLess(check_reconstruction(model, self.tokens, split=77), 3e-5)

    @torch.no_grad()
    def test_zero_sham_leaves_full_model_unchanged(self):
        for model in self.models():
            with self.subTest(config=model.cfg):
                model.eval()
                query = torch.tensor([9, 11])
                source = torch.tensor([3, 4])
                teacher = isinstance(model, GPT) and model.cfg.attn != "linear"
                baseline = model(self.tokens)[0]
                actual = traced_forward(
                    model, self.tokens,
                    intervention=lambda layer, read: intervene(layer, read, ((0, 0),), query, source, 0., teacher))[0]
                torch.testing.assert_close(baseline, actual, rtol=0, atol=0)

    def test_known_induction_and_signed_read(self):
        ids = [0, 1, 2, 3, 4, 5, 1]
        self.assertEqual(induction_source(ids, 2), 2)
        self.assertIsNone(induction_source(ids, 4))
        case = make_case(ids, 2, 2, "known")
        self.assertEqual(case.shams, (1, 3))
        self.assertEqual(case.query, 6)
        q = torch.ones(1, 1, 7, 1)
        k = torch.zeros_like(q)
        k[..., 2, :] = 3.
        k[..., 5, :] = -1.
        v = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7, 1)
        decay = torch.zeros(1, 1, 7)
        trace = Read(q, k, v, decay, torch.zeros_like(v))
        weights = trace.parts(torch.tensor([[case.query]]))[0]
        torch.testing.assert_close(weights[0, 0, 0], torch.tensor([0., 0., 3., 0., 0., -1., 0.]))
        torch.testing.assert_close(trace.reconstructed()[0, 0, -1], torch.tensor([1.]))
        redirected = intervene(0, trace, ((0, 0),), torch.tensor([6]), torch.tensor([2]), 1., False)
        self.assertEqual(redirected[0, 0, 6, 0].item(), 8.)
        torch.testing.assert_close(redirected[..., :-1, :], trace.actual[..., :-1, :])
        row = measure_range({0: trace}, (0, 1, 2, 7))[0]
        self.assertAlmostEqual(row[1][0, 0], 2 / 7)
        self.assertAlmostEqual(row[1][3, 0], (1 - 1 / 4) / 6)

    def test_joint_teacher_and_student_edits_match_single_head_edits(self):
        q, k, v = (torch.randn(2, 3, 7, 4) for _ in range(3))
        queries, writes = torch.tensor([5, 6]), torch.tensor([1, 2])
        for teacher in (False, True):
            decay = None if teacher else -torch.rand(2, 3, 7)
            trace = Read(q, k, v, decay, torch.zeros_like(v))
            trace.actual = trace.reconstructed()
            both = intervene(0, trace, ((0, 0), (0, 2), (1, 1)), queries, writes, 1., teacher)
            for head in (0, 2):
                single = intervene(0, trace, ((0, head),), queries, writes, 1., teacher)
                torch.testing.assert_close(both[:, head], single[:, head])
            torch.testing.assert_close(both[:, 1], trace.actual[:, 1], rtol=0, atol=0)

    @torch.no_grad()
    def test_paired_batch_uses_query_and_per_case_source(self):
        model = GPT(TFConfig(vocab=512, d_model=12, n_layers=2, n_heads=2, max_len=256)).eval()
        cases = list(synthetic_cases(512, 128, 4, np.random.default_rng(7), 0))
        for case in cases:
            self.assertEqual(case.source, induction_source(case.ids, case.answer))
        batched, maps = measure_cases(model, cases, 4, torch.device("cpu"), 0)
        single, _ = measure_cases(model, cases, 1, torch.device("cpu"), 0)
        np.testing.assert_allclose(batched, single, atol=1e-5)
        self.assertEqual(maps[0].shape, (len(cases), 2, 2))


if __name__ == "__main__":
    unittest.main()

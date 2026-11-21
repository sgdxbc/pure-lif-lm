import math
import unittest

import numpy as np
import torch

from experiments.memory_maps import traced_forward
from experiments.witness_replay import (
    answer_logprobs, draw_sham, query_counts, query_editor, summarize,
)
from model import Config, SpikingLM, plastic_scan


class WitnessReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @torch.no_grad()
    def test_explicit_replay_of_original_counts_is_bitwise_identical(self):
        torch.manual_seed(17)
        cfg = Config(vocab=32, d_model=12, d_hidden=16, n_layers=3, mem_every=1,
                     mem_heads=2, mem_keys=4, mem_dv=6, mem_kq_levels=4,
                     mem_theta=.05, mem_delays=4, mem_chunk=7)
        model = SpikingLM(cfg).eval()
        x = torch.randint(32, (2, 17))
        streams, queries = torch.tensor([0, 1]), torch.tensor([10, 16])
        heads = ((0, 1), (1, 0), (2, 0), (2, 1))
        selected = {layer for layer, _ in heads}
        before, _, traces = traced_forward(model, x, trace_layers=selected)
        original = query_counts(traces, heads, streams, queries)
        edit = query_editor(heads, streams, queries, original)
        after, _, replayed = traced_forward(model, x, trace_layers=selected, plastic_query=edit)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(query_counts(replayed, heads, streams, queries), original))

    def test_scan_edit_changes_only_requested_reads_and_leaves_state(self):
        shape = (2, 3, 11, 4)
        q = torch.ones(2, 11, 3, 4).transpose(1, 2)
        k = torch.ones(shape)
        v = torch.ones(*shape[:-1], 2)
        decay = torch.full((3,), -.11)
        streams, queries = torch.tensor([0, 1]), torch.tensor([6, 9])
        heads = ((0, 1), (1, 0), (1, 2))
        counts = torch.tensor([[[2., 2., 0., 0.], [4., 1., 0., 0.], [3., 3., 0., 0.]],
                               [[0., 2., 2., 0.], [0., 2., 4., 0.], [4., 2., 0., 0.]]])
        edit = query_editor(heads, streams, queries, counts)
        unchanged_layer = edit(2, q)
        self.assertIs(unchanged_layer, q)
        edited = edit(1, q)
        self.assertEqual(edited.stride(), q.stride())
        np.testing.assert_array_equal(q.numpy(), np.ones(shape))
        torch.testing.assert_close(edited[streams[:, None], torch.tensor([0, 2]), queries[:, None]], counts[:, 1:])
        original, state = plastic_scan(q, k, v, decay, chunk=5)
        after, next_state = plastic_scan(edited, k, v, decay, chunk=5)
        expected = torch.zeros(shape[:-1], dtype=torch.bool)
        expected[streams[:, None], torch.tensor([0, 2]), queries[:, None]] = True
        self.assertTrue(torch.equal((after != original).any(-1), expected))
        self.assertTrue(torch.equal(state, next_state))

    def test_sham_preserves_budget_edit_size_caps_and_seed(self):
        original = np.array([4, 0, 3, 1, 0, 2, 2, 4])
        witness = np.array([0, 4, 0, 3, 3, 2, 3, 1])
        edit_size = np.abs(witness - original).sum()
        seen = set()
        for seed in range(12):
            sham = draw_sham(original, witness, np.random.default_rng(seed))
            self.assertEqual(sham.sum(), original.sum())
            self.assertEqual(np.abs(sham - original).sum(), edit_size)
            self.assertTrue(((0 <= sham) & (sham <= 4)).all())
            np.testing.assert_array_equal(sham, draw_sham(original, witness, np.random.default_rng(seed)))
            seen.add(tuple(sham))
        self.assertGreater(len(seen), 1)
        np.testing.assert_array_equal(draw_sham(original, original, np.random.default_rng(0)), original)
        forced = np.array([4, 0])
        np.testing.assert_array_equal(draw_sham(forced, forced[::-1], np.random.default_rng(0)), forced[::-1])

    def test_known_answer_logprob_and_paired_decision(self):
        logits = torch.log(torch.tensor([[[1., 1., 1.], [1., 3., 2.]], [[4., 3., 1.], [1., 1., 1.]]]))
        scores = answer_logprobs(logits, torch.tensor([1, 0]), torch.tensor([0, 1]), torch.tensor([2, 1]))
        torch.testing.assert_close(scores, torch.tensor([math.log(1 / 8), math.log(1 / 2)]))
        rows = {}
        for i, score in enumerate(scores.tolist()):
            rows[str(i)] = dict(original=dict(logprob=score, win=False),
                                witness=dict(logprob=score + .4, win=i == 0),
                                sham=dict(logprob=score + .1, win=False),
                                sham_equals_witness=[False], l1_edit=[2])
        result = summarize(rows)
        self.assertEqual(result["ranking_wins"]["witness"], 1)
        self.assertAlmostEqual(result["witness_logprob_gain"]["original"]["mean"], .4)
        self.assertAlmostEqual(result["witness_logprob_gain"]["sham"]["mean"], .3)
        self.assertTrue(result["reopen_discussion"])
        rows["1"]["witness"]["logprob"] -= .3
        self.assertFalse(summarize(rows)["reopen_discussion"])


if __name__ == "__main__":
    unittest.main()

import itertools
import unittest

import numpy as np
import torch

from experiments.query_oracle import exact_geometry, exact_margin, oracle
from experiments.write_read import Candidate, decision, rerank, screen_offsets, similarity, teacher_hits, window_candidates


class WriteReadTest(unittest.TestCase):
    def test_key_similarity_and_silence(self):
        row = similarity([4, 0, 3], [2, 1, 0], count_keys=True)
        self.assertEqual(row["source_norm"], 5)
        self.assertAlmostEqual(row["cosine"], 8 / (5 * np.sqrt(5)))
        self.assertAlmostEqual(row["active_jaccard"], 1 / 3)
        self.assertFalse(row["exact_equal"])
        silent = similarity([0, 0], [0, 0], count_keys=True)
        self.assertTrue(silent["source_silent"])
        self.assertTrue(silent["other_silent"])
        self.assertTrue(silent["exact_equal"])
        self.assertIsNone(silent["cosine"])
        self.assertIsNone(silent["active_jaccard"])

    def test_reranking_with_unit_survival(self):
        raw = np.array([[4., 2., 0.], [-4., -2., 0.]])
        coeff = raw * [.20, .75, 1.]
        actual, neutral = rerank(coeff, raw, 0, [1])
        self.assertFalse(actual["win"])
        self.assertTrue(neutral["win"])
        self.assertAlmostEqual(neutral["margin"], 1 / 3)
        self.assertEqual(rerank(coeff, raw, 0, [1], recurrent=False), (actual, actual))
        actual, neutral = rerank(coeff, raw, 1, [0])
        self.assertTrue(actual["win"])
        self.assertFalse(neutral["win"])

    def test_integer_oracle_finds_simultaneous_winner(self):
        keys = np.array([[[0, 2], [2, 0], [1, 1]], [[1, 0], [2, 0], [1, 1]]])
        survival = np.ones((2, 3))
        observed = np.ones((2, 2), dtype=int)
        result = oracle(keys, survival, observed, 0, [1, 2])
        self.assertEqual(result["status"], "win")
        query = np.asarray(result["query"])
        np.testing.assert_array_equal(query.sum(-1), [2, 2])
        self.assertTrue(((query >= 0) & (query <= 4)).all())
        self.assertGreater(exact_margin(query, exact_geometry(keys, survival, 0, [1, 2])), 0)

    def test_infeasible_against_both_rivals_even_if_each_can_be_beaten(self):
        keys = np.array([[[1, 1], [2, 0], [0, 2]]])
        survival = np.ones((1, 3))
        observed = np.ones((1, 2), dtype=int)
        for rival in (1, 2):
            self.assertEqual(oracle(keys, survival, observed, 0, [rival])["status"], "win")
        result = oracle(keys, survival, observed, 0, [1, 2])
        self.assertEqual(result["status"], "impossible")
        self.assertEqual(result["bound_exact"], "0")
        self.assertEqual(result["upper_bound"], 0)

    def test_bound_respects_silent_head_and_four_count_cap(self):
        keys = np.array([[[1, 0, 0], [2, 0, 0], [0, 0, 0]],
                         [[1, 2, 0], [2, 2, 0], [2, 0, 0]]])
        survival = np.array([[.25, .5, 1.], [.25, .5, 1.]])
        observed = np.array([[1, 0, 4], [2, 1, 1]])
        result = oracle(keys, survival, observed, 0, [1])
        self.assertEqual(result["status"], "impossible")
        geometry = exact_geometry(keys, survival, 0, [1])
        options = [[q for q in itertools.product(range(5), repeat=3) if sum(q) == n]
                   for n in observed.sum(-1)]
        self.assertLessEqual(max(exact_margin(np.array(q), geometry) for q in itertools.product(*options)),
                             result["upper_bound"])

    def test_retention_and_mixed_decisions(self):
        win, loss, unresolved = ({"status": x} for x in ("win", "impossible", "unresolved"))
        self.assertEqual(decision(loss, win, False)["label"], "retention")
        self.assertEqual(decision(unresolved, loss, False)["label"], "write_code_limit")
        self.assertEqual(decision(win, win, False)["label"], "read_side_headroom")
        self.assertEqual(decision(win, win, True)["label"], "mixed")
        self.assertEqual(decision(unresolved, unresolved, False)["label"], "inconclusive")

    def test_screen_counts_window_and_applies_strict_teacher_rule(self):
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 3, 0, 7, 3]
        pieces = ["<|endoftext|>", "He", "Ġsaw", "ĠAlice", "Ġand", "ĠBob", "Ġthen",
                  "Ġasked", "ĠAlice", "<|endoftext|>", "Ġasked", "ĠAlice"]
        unique, candidates = window_candidates(ids, pieces, eot=0)
        self.assertEqual(unique, 1)
        self.assertEqual(candidates, [Candidate(7, 3, [5], True)])
        p = torch.zeros((3, len(ids) - 1), dtype=torch.float64)
        p[:, 3] = torch.tensor([.20, .19, .30], dtype=torch.float64)
        p[:, 5] = torch.tensor([.19, .10, .30], dtype=torch.float64)
        self.assertEqual(teacher_hits(p, candidates * 3), [True, False, False])
        ids[2] = 3
        self.assertEqual(window_candidates(ids, pieces, eot=0), (0, []))

    def test_fresh_nonoverlapping_screen_windows(self):
        from experiments.write_read import FREQUENCY_TOKENS
        seq = 8
        old = np.array([[FREQUENCY_TOKENS + 6, FREQUENCY_TOKENS + 80]])
        offsets = screen_offsets(FREQUENCY_TOKENS + 120, seq, old, count=5)
        self.assertTrue(np.all(np.diff(np.sort(offsets)) >= seq + 1))
        self.assertTrue(np.all(np.abs(offsets[:, None] - old.ravel()) > seq))


if __name__ == "__main__":
    unittest.main()

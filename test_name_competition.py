import unittest

import numpy as np

from experiments.name_competition import (
    Passage, describe, group_shares, lexical_positions, match_controls, passage_metrics, summarize,
)
from experiments.memory_maps import Case


class NameCompetitionTest(unittest.TestCase):
    def test_name_like_words(self):
        pieces = ["<|endoftext|>", "Alice", "Ġsaw", "ĠBob", ".", "ĠThe", "Ġlight", "Ġwas",
                  "Ġon", ".", "Ċ", "Bob", "Ġmet", "ĠCar", "oline", "Ġand", "ĠUSA", "."]
        words, names = lexical_positions(pieces)
        self.assertEqual([(pos, words[pos]) for pos in sorted(names)], [(3, "Bob"), (11, "Bob")])
        self.assertNotIn(13, words)
        self.assertNotIn(14, words)
        self.assertNotIn(0, words)

    def test_repeated_capital_with_lowercase_use(self):
        pieces = ["Rose", "Ġand", "Ġrose", ".", "ĠRose", "Ġagain"]
        _, names = lexical_positions(pieces)
        self.assertEqual(names, set())

    def test_lag_and_frequency_matching(self):
        # Position 4 is just one token away from write 5 but is in a different bin.
        # Position 8 would match write 6 at a wider gap, but is the query itself.
        counts = [3, 9, 10, 7, 17, 11, 10, 13, 10]
        self.assertEqual(match_controls(list(range(9)), 8, [2, 6, 0], [1, 4, 5, 7, 8],
                                        counts, max_gap=1), [(2, 1), (6, 5)])
        self.assertEqual(match_controls(list(range(9)), 8, [6], [1, 4, 8],
                                        counts, max_gap=2), [])
        self.assertEqual(match_controls(list(range(9)), 8, [5], [4, 6, 7],
                                        counts, max_gap=2), [(5, 6)])
        # Equal lag and count distance resolve to the earlier position.
        self.assertEqual(match_controls(list(range(3)), 3, [1], [0, 2],
                                        [9, 10, 11], max_gap=1), [(1, 0)])

    def test_matching_excludes_answer_spelling_and_query(self):
        pieces = ["He", "Ġsaw", "ĠAlice", "Ġnear", "ĠBob", "Ġand", "Ġalice", "ĠCarol"]
        case = Case(list(range(len(pieces))), answer=2, source=2, shams=(1, 3), label="example")
        result = describe(case, pieces, [10] * len(pieces))
        self.assertTrue(result.name_answer)
        self.assertEqual(result.wrong, [4])
        self.assertEqual(result.pairs, [(4, 3)])

    def test_known_signed_map_and_silent_head(self):
        coeff = [[-2, 4, 2, 0], [9, -1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
        shares = group_shares(coeff)
        np.testing.assert_allclose(shares, [0.35, 0.2125, 0.125, 0.0625])
        passage = Passage("example", 0, True, [1], [(1, 2)])
        row = passage_metrics(passage, shares)
        self.assertEqual(row["correct_beats_best_wrong"], 1)
        self.assertAlmostEqual(row["competition_effect"], 0.0875)
        self.assertEqual(passage_metrics(passage, np.ones(4))["correct_beats_best_wrong"], 0)

    def test_interpretation_uses_paired_interval_and_eligibility(self):
        eligible = {"correct_beats_best_wrong": 1., "competition_effect": .1}
        teacher = {"correct_beats_best_wrong": 0., "competition_effect": .03}
        empty = {"correct_beats_best_wrong": None, "competition_effect": None}
        report = summarize([eligible, empty], [teacher, empty])
        self.assertTrue(report["general_name_competition"])
        self.assertEqual(report["competition_effect"]["passages"], 1)
        self.assertAlmostEqual(report["competition_effect"]["minus_teacher"]["mean"], .07)
        self.assertFalse(summarize([empty])["general_name_competition"])


if __name__ == "__main__":
    unittest.main()

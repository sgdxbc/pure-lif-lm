import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import broad_replay as broad
from experiments.memory_maps import traced_forward
from experiments.witness_replay import query_counts
from experiments.write_read import Candidate, teacher_hits, window_candidates
from model import Config, SpikingLM


class BroadReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_group_assignment_uses_item6_screen_on_hand_built_window(self):
        vocabulary = ["<|endoftext|>", "He", "Ġsaw", "ĠAlice", "Ġand", "ĠBob",
                      "Ġthen", "Ġasked", "Ġnear", "Ġwaited"]
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 3, 9, 8, 2, 0, 8, 2]
        _, candidates = window_candidates(ids, [vocabulary[i] for i in ids], eot=0)
        self.assertEqual(candidates, [Candidate(7, 3, [5], True), Candidate(10, 2, [3, 5, 8], False)])
        probabilities = torch.zeros(2, len(ids) - 1)
        probabilities[0, [3, 5]] = torch.tensor([.21, .20])
        probabilities[1, 2] = .50
        passed = {c.query for c, hit in zip(candidates, teacher_hits(probabilities, candidates)) if hit}
        expected = np.full(len(ids) - 1, 2)
        expected[[7, 10]] = [0, 1]
        np.testing.assert_array_equal(broad.assign_groups(len(ids) - 1, candidates, passed), expected)
        expected[7] = 1
        np.testing.assert_array_equal(broad.assign_groups(len(ids) - 1, candidates, set()), expected)

    def test_sampling_weights_recover_exact_total_in_expectation(self):
        pool = [broad.Position(i, Candidate(2, 0, [1], True), 0, False) for i in range(4)]
        gains = [2., -1., 4., 8.]
        _, probabilities = zip(*broad.draw(pool, 2, np.random.default_rng(17)))
        self.assertEqual(probabilities, (.5, .5))
        totals = []
        for sample in itertools.combinations(range(4), 2):
            records = [(pool[i], .5, dict(gain=gains[i])) for i in sample]
            totals.append(broad.weighted_windows(records, len(pool)).sum())
        self.assertEqual(np.mean(totals), sum(gains))
        self.assertEqual(broad.loss_quotas([500, 200], 140), [100, 40])
        self.assertEqual(broad.loss_quotas([500, 200], 2), [1, 1])
        self.assertEqual(broad.loss_quotas([0, 1], 1), [0, 1])
        with self.assertRaises(TimeoutError):
            broad.loss_quotas([1, 1], 1)

    def tiny_model_and_edits(self):
        torch.manual_seed(17)
        cfg = Config(vocab=32, d_model=12, d_hidden=16, n_layers=3, mem_every=1,
                     mem_heads=2, mem_keys=4, mem_dv=6, mem_kq_levels=4,
                     mem_theta=.05, mem_delays=4, mem_chunk=7)
        model = SpikingLM(cfg).eval()
        with torch.no_grad():
            for memory in model.mems.values():
                memory.w_out.weight.normal_(std=.1)
        windows = torch.randint(32, (2, 18)).numpy()
        heads = ((0, 1), (1, 0), (2, 0), (2, 1))
        positions = [broad.Position(window, Candidate(query, 0, [1], bool(group == 0)), group, False)
                     for window, query, group in [(0, 10, 0), (0, 15, 0), (1, 7, 1), (1, 14, 1)]]
        streams = torch.tensor([p.window for p in positions])
        queries = torch.tensor([p.candidate.query for p in positions])
        _, _, traces = traced_forward(model, torch.from_numpy(windows[:, :-1]))
        original = query_counts(traces, heads, streams, queries).cpu().numpy()
        edits = np.roll(original, 1, axis=-1)
        self.assertTrue((original != edits).any())
        return model, windows, heads, positions, edits

    @torch.no_grad()
    def test_one_edit_per_replay_even_for_two_queries_from_same_window(self):
        model, windows, heads, positions, edits = self.tiny_model_and_edits()
        with patch.object(broad, "traced_forward", wraps=traced_forward) as forwards:
            actual = broad.replay_batch(model, windows, positions, edits, heads, "cpu")
        x = forwards.call_args_list[1].args[1]
        self.assertTrue(torch.equal(x[0], x[1]))
        edit = forwards.call_args_list[1].kwargs["plastic_query"]
        q = torch.zeros((4, 2, x.size(1), 4))
        for layer in range(3):
            result = edit(layer, q)
            expected = q.clone()
            for stream, position in enumerate(positions):
                for index, (selected_layer, head) in enumerate(heads):
                    if selected_layer == layer:
                        expected[stream, head, position.candidate.query] = torch.from_numpy(edits[stream, index])
            self.assertTrue(torch.equal(result, expected))
        for i, position in enumerate(positions):
            separate = broad.replay_batch(model, windows, [position], [edits[i]], heads, "cpu")
            self.assertAlmostEqual(actual[i]["gain"], separate[0]["gain"], places=5)
            self.assertEqual(actual[i]["replay_win"], separate[0]["replay_win"])

    @torch.no_grad()
    def test_weighted_estimate_matches_brute_force_model_replays(self):
        model, windows, heads, positions, edits = self.tiny_model_and_edits()
        inputs = torch.from_numpy(windows[:, :-1])
        targets = torch.from_numpy(windows[:, 1:])
        original = model(inputs)[0].log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        brute = []
        for position, replacement in zip(positions, edits):
            stream, query = position.window, position.candidate.query

            def replace(layer, q):
                result = q.clone()
                for values, (chosen, head) in zip(replacement, heads):
                    if layer == chosen:
                        result[stream, head, query] = torch.from_numpy(values)
                return result

            replayed, _, _ = traced_forward(model, inputs, plastic_query=replace)
            value = replayed[stream, query].log_softmax(-1)[targets[stream, query]]
            brute.append((value - original[stream, query]).item())
        self.assertGreater(sum(map(abs, brute)), 1e-5)
        seq = inputs.shape[1]
        counts = np.zeros((len(windows), 3), dtype=int)
        nll = np.zeros_like(counts, dtype=float)
        for w in range(len(windows)):
            groups = np.full(seq, 2)
            for p in positions:
                if p.window == w:
                    groups[p.candidate.query] = p.group
            counts[w] = np.bincount(groups, minlength=3)
            nll[w] = np.bincount(groups, weights=-original[w].numpy(), minlength=3)
        summaries = []
        for sample in itertools.product((0, 1), (2, 3)):
            selected = [positions[i] for i in sample]
            scores = broad.replay_batch(model, windows, selected, edits[list(sample)], heads, "cpu")
            sums = broad.weighted_windows(list(zip(selected, [.5, .5], scores)), len(windows))
            summaries.append(broad.summarize(counts, nll, sums, seq))
        self.assertAlmostEqual(np.mean([s["gain"]["mean"] for s in summaries]), sum(brute) / inputs.numel(), places=6)
        for s in summaries:
            ceiling = sum(g["original_nll_ceiling"] for g in s["groups"].values())
            self.assertAlmostEqual(ceiling, -original.mean().item(), places=6)


if __name__ == "__main__":
    unittest.main()

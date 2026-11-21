from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.transfer_binding import binding_verdict, name_log_odds
from experiments.transfer_binding_data import NAMES, Passages
from experiments.transfer_binding_memory import BindingMemory, persistent_read, reset_features
from experiments import transfer_binding_r1 as revision
from model_tf import GPT, TFConfig


class TransferBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_body_reset_isolates_a_prefix(self):
        torch.manual_seed(11)
        body = GPT(TFConfig(vocab=32, d_model=16, n_heads=1, n_layers=1, attn="linear")).eval()
        ids = torch.tensor([[1, 2, 3, 4, 12, 13, 14, 15],
                            [5, 6, 7, 8, 12, 13, 14, 15]])
        features = reset_features(body, ids, block=4)
        full, _ = body._features(ids)
        self.assertFalse(features.requires_grad)
        self.assertFalse(torch.equal(features[0, :4], features[1, :4]))
        self.assertFalse(torch.equal(full[0, 4:], full[1, 4:]))
        torch.testing.assert_close(features[0, 4:], features[1, 4:], rtol=0, atol=0)

    def test_state_cap(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model = BindingMemory(16, sparse)
                self.assertLessEqual(model.state_bytes, 32768)
                if sparse:
                    self.assertLessEqual(model.overlap_bytes, 32768)
                with self.assertRaisesRegex(ValueError, "state cap"):
                    BindingMemory(16, sparse, cap=model.state_bytes - 1)

    @torch.no_grad()
    def test_known_write_and_read(self):
        x = torch.tensor([[[1., 0.], [0., 3.], [0., -1.]]])
        pos = torch.tensor([[0, 2]])
        for sparse, expected in ((True, 2.4), (False, 1.5)):
            with self.subTest(sparse=sparse):
                model = BindingMemory(2, sparse, addresses=4, values=1, winners=1)
                for param in model.parameters():
                    param.zero_()
                for project in (model.trace, model.key, model.query):
                    project.bias[0] = 1 if sparse else 0
                model.gate.weight[0, 1] = 1
                model.gate.bias.fill_(-.5)
                model.value.weight[0, 1] = 1
                model.decode.weight[0, 0] = 1
                actual, codes = model(x, pos)
                torch.testing.assert_close(actual, torch.tensor([[[1., 0.], [expected, -1.]]]))
                # The write at 1 must remain visible to the read in the next block.
                torch.testing.assert_close(persistent_read(*codes, block=2)[0, 2], torch.tensor([expected]))
                self.assertEqual(model.work(codes)["successful_writes"], 1)
                if sparse:
                    torch.testing.assert_close(model.digital(x, pos, codes), actual)

    def test_name_log_odds_matches_full_log_softmax(self):
        x = torch.tensor([[1., 2.], [1., 2.]])
        weights = torch.tensor([[1., 4.], [2., -1.], [3., 2.]])
        actual = name_log_odds(x, weights, torch.tensor([0, 1]), torch.tensor([1, 0]))
        logp = (x @ weights.T).log_softmax(-1)
        torch.testing.assert_close(actual, torch.tensor([9., -9.]))
        torch.testing.assert_close(actual, torch.stack((logp[0, 0] - logp[0, 1], logp[1, 1] - logp[1, 0])))

    def test_generator_paired_swap_and_single_token_names(self):
        tok_path = Path(__file__).parent / "data/fineweb/gpt2.json"
        tok = Tokenizer.from_file(str(tok_path))
        all_names = [set(names.split()) for names in NAMES.values()]
        self.assertEqual([len(names) for names in all_names], [32, 24, 32])
        self.assertEqual(len(set.union(*all_names)), 88)
        plain = np.array(tok.encode(" The residents gathered in the hall. " * 100).ids, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "train.bin"
            np.concatenate([np.array([50256], dtype=np.uint16), plain,
                            np.array([50256], dtype=np.uint16)]).tofile(corpus)
            passages = Passages(tok, corpus, "train")
            row, = passages.pairs(256, 8, 1)
        changed = np.flatnonzero(row.tokens != row.other)
        self.assertEqual(len(changed), 2)
        self.assertIn(row.position + 1 - 256, changed)
        np.testing.assert_equal(row.tokens[changed], row.other[changed][::-1])
        np.testing.assert_equal(row.tokens[-128:], row.other[-128:])

    def test_verdict_compares_both_controls(self):
        result = {"ordinary_nll": {"spike": 2.01, "dense": 2.}}
        prefs = {"2048x8": {"spike": np.array([.31, .32]), "dense": np.array([.02, .12]),
                           "overlap": np.array([.09, .10])}}
        decision = binding_verdict(result, prefs, True)
        self.assertTrue(decision["advance"])
        self.assertEqual(decision["stronger_control"], "overlap")
        self.assertFalse(binding_verdict(result, prefs, False)["advance"])
        prefs["2048x8"]["dense"] += .10
        self.assertFalse(binding_verdict(result, prefs, True)["advance"])

    def test_revision_penalty_uses_masked_hard_gate_and_has_gradient(self):
        logits = torch.tensor([[1., -1., 1.]], requires_grad=True)
        prob = logits.sigmoid()
        gate = (logits >= 0).float() + (prob - prob.detach())
        key = torch.tensor([[[0.], [0.], [1.]]])
        opening, written = revision.rates(gate, key, torch.tensor([[1]]))
        self.assertEqual(opening.item(), .5)
        self.assertEqual(written.item(), 0.)
        (opening - revision.TARGET).square().backward()
        self.assertTrue((logits.grad[0, :2] > 0).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)

    def test_revision_stages_stop_at_the_registered_edges(self):
        dev = SimpleNamespace(tok=None, data=SimpleNamespace(filename="valid.bin"))
        def result(cell, acc):
            row = {arm: {"two_choice_accuracy": val} for arm, val in (("dense", acc), ("spike", .5))}
            return {"name_probe": {cell: row}}, {}
        for easy, mixed in ((.60, None), (.61, .50), (.61, .51)):
            with self.subTest(easy=easy, mixed=mixed), tempfile.TemporaryDirectory() as out:
                args = SimpleNamespace(eval_docs=128, out=Path(out))
                results = [result("128x2", easy)]
                if mixed is not None:
                    results.append(result("256x8", mixed))
                with patch.object(revision, "Passages"), patch.object(revision, "stage", return_value={}) as fit, \
                     patch.object(revision, "evaluate", side_effect=results), \
                     patch.object(revision, "within_work_budget", return_value=False):
                    report = revision.revise(None, {"spike": None, "dense": None}, None, dev, args, None)
                self.assertEqual(fit.call_count, len(results))
                self.assertEqual(report["allow_heldout"], mixed == .51)


if __name__ == "__main__":
    unittest.main()

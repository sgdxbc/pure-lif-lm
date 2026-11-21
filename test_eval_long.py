"""CPU tests for paired long-context evaluation."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import eval_long
from model import Config, SpikingLM
from model_tf import GPT, TFConfig


CPU = torch.device("cpu")
TOKENIZER = Path(__file__).resolve().parent / "data" / "fineweb" / "gpt2.json"


class LongEvalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.x = torch.randint(32, (2, 83))

    def test_carried_models_match_whole_and_are_causal(self):
        snn = Config(vocab=32, d_model=12, d_hidden=16, n_layers=2, mem_heads=2,
                     mem_keys=4, mem_dv=6, mem_delays=4, mem_kq_levels=4, mem_gate="channel", bus_norm=True)
        tf = dict(vocab=32, d_model=12, n_layers=2, n_heads=2, attn="linear")
        models = [SpikingLM(snn), SpikingLM(Config(vocab=32, d_model=12, d_hidden=16, n_layers=2,
                                                  mem_heads=2, mem_kind="gla")),
                  GPT(TFConfig(**tf)), GPT(TFConfig(**tf, relay="quant", relay_levels=4, mlp="spike")),
                  GPT(TFConfig(**tf, relay="lif", mlp="lif", out_classes=4, out_short=4))]
        with torch.no_grad():
            for model in models:
                with self.subTest(model=model.cfg):
                    model.eval()
                    opts = {} if isinstance(model, SpikingLM) else {"return_state": True}
                    if isinstance(model, SpikingLM) and model.cfg.mem_kind == "plastic":
                        for mem in model.mems.values():
                            mem.w_out.weight.normal_(std=0.1)
                            mem.delay_dw.normal_(std=0.15)
                            mem.kq_log_theta.fill_(-0.5)
                    logits, _, final, _ = model(self.x, **opts)
                    torch.testing.assert_close(model(self.x)[0], logits, atol=0, rtol=0)
                    state, parts = None, []
                    for chunk in self.x.split(17, dim=1):
                        out, _, state, _ = model(chunk, state=state, **opts)
                        parts.append(out)
                    torch.testing.assert_close(torch.cat(parts, dim=1), logits, rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(state, final, rtol=2e-4, atol=3e-5)
                    altered = self.x.clone()
                    altered[:, 32:] = (altered[:, 32:] + 1) % model.cfg.vocab
                    torch.testing.assert_close(model(altered)[0][:, :32], logits[:, :32], rtol=1e-4, atol=1e-5)
                    # Prefill updates state even if no vocabulary positions are requested.
                    empty, _, state, _ = model(self.x[:, :-3], logit_slice=slice(0, 0), **opts)
                    self.assertEqual(empty.shape[1], 0)
                    tail = model(self.x[:, -3:], state=state, **opts)[0]
                    torch.testing.assert_close(tail, logits[:, -3:], rtol=1e-4, atol=1e-5)

    def test_same_targets_and_softmax_window_phase(self):
        opts = dict(vocab=32, d_model=12, n_layers=2, n_heads=2)
        recurrent = GPT(TFConfig(**opts, attn="linear")).eval()
        full, _ = eval_long.score(recurrent, self.x, 7, 0, CPU, restart=19)
        reset, _ = eval_long.score(recurrent, self.x, 7, 0, CPU, restart=19, reset=True)
        self.assertEqual(full.shape, (2, 82))
        self.assertTrue(torch.equal(full[:, :19], reset[:, :19]))
        reference = [eval_long.score(recurrent, self.x[:, lo:lo + 20], 7, 0, CPU)[0]
                     for lo in range(0, 82, 19)]
        torch.testing.assert_close(reset, torch.cat(reference, dim=1), rtol=0, atol=0)
        tail, _ = eval_long.score(recurrent, self.x, 7, 0, CPU, restart=19, reset=True, first=80)
        torch.testing.assert_close(tail, reset[:, 80:], rtol=0, atol=0.001)

        tf = GPT(TFConfig(**opts, max_len=12)).eval()
        full, _ = eval_long.score(tf, self.x, 7, 12, CPU, restart=17)
        self.assertTrue(torch.equal(full, eval_long.nll_tf(tf, self.x, 12, CPU)))
        reset, _ = eval_long.score(tf, self.x, 7, 12, CPU, restart=17, reset=True)
        expected = []
        for lo, hi, start, _ in eval_long.softmax_spans(82, 12, 0, 0):
            for i in range(start, hi):
                xi = self.x[:, max(lo, i // 17 * 17):i + 1]
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = tf(xi)[0][:, -1]
                expected.append(F.cross_entropy(logits.float(), self.x[:, i + 1], reduction="none"))
        torch.testing.assert_close(reset, torch.stack(expected, dim=1), rtol=0, atol=0.001)

    def test_recall_trials_and_metrics(self):
        tok = Tokenizer.from_file(str(TOKENIZER))
        background = np.full(900, tok.encode(" noise").ids[-1], dtype=np.uint16)
        starts, ends = np.array([0, 450]), np.array([450, 900])
        args = (tok, background, starts, ends, (32, 128, 192), 64, 5)
        trial_sets = list(eval_long.recall_trials(*args))
        for (distance, x, counts), (_, again, again_counts) in zip(trial_sets, eval_long.recall_trials(*args)):
            self.assertTrue(torch.equal(x, again))
            self.assertTrue(torch.equal(counts, again_counts))
            rng = np.random.default_rng(0)
            for i, row in enumerate(x):
                value = int(rng.integers(1000, 10000))
                fact = tok.encode(f"\nThe code for record {i:04d} is {value}.\n").ids
                tail = tok.encode(f"\nQuestion: What is the code for record {i:04d}?\nAnswer: {value}").ids
                start = len(row) - len(tail) - distance - len(fact)
                self.assertEqual(row[start:start + len(fact)].tolist(), fact)
                self.assertEqual(row[-len(tail):].tolist(), tail)
                self.assertEqual((len(row) - 1) % 64, 0)
                self.assertEqual(start >= len(row) - 1 - 64, distance == 32)
        scores = eval_long.recall_metrics(torch.tensor([[3., 2.], [8., 1.]]),
                                          torch.tensor([[True, False], [False, True]]), torch.tensor([2, 1]))
        self.assertEqual(scores, dict(mean_answer_logprob=-3., mean_token_logprob=-2.,
                                      token_top1=2 / 3, answer_top1=0.5))

    def test_warm_restarts_preserve_softmax_windows_and_prefill_recurrence(self):
        opts = dict(vocab=32, d_model=12, n_layers=2, n_heads=2)
        tf = GPT(TFConfig(**opts, max_len=12)).eval()
        full = eval_long.score(tf, self.x, 7, 12, CPU, restart=19, first=8, top1=True)
        warm = eval_long.score(tf, self.x, 7, 12, CPU, restart=19, first=8, top1=True, reset=True, warmup=12)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, warm)))

        gla = GPT(TFConfig(**opts, attn="linear")).eval()
        for model, window in ((gla, 0), (tf, 12)):
            args = (model, self.x, 7, window, CPU, 19, True)
            torch.testing.assert_close(eval_long.score(*args, warmup=0)[0], eval_long.score(*args)[0],
                                       atol=0, rtol=0)
        warm = eval_long.score(gla, self.x, 7, 0, CPU, restart=19, reset=True, warmup=19)[0]
        expected = []
        for lo in range(0, 82, 19):
            prefix = max(0, lo - 19)
            scores, _ = eval_long.score(gla, self.x[:, prefix:lo + 20], 7, 0, CPU,
                                        restart=19, first=lo - prefix)
            expected.append(scores)
        torch.testing.assert_close(warm, torch.cat(expected, 1), atol=0, rtol=0)

    def test_matched_recall_and_seeded_bootstrap(self):
        tok = Tokenizer.from_file(str(TOKENIZER))
        noise = tok.encode(" noise").ids[-1]
        arr = np.full(1400, noise, np.uint16)
        for distance, one_token in ((128, False), (512, True)):
            _, x, counts, swapped = next(eval_long.recall_trials(
                tok, arr, np.array([0]), np.array([1400]), (distance,), 64, 8, matched=True, one_token=one_token))
            self.assertEqual(x.shape, swapped.shape)
            for row, foil in zip(x.tolist(), swapped.tolist()):
                fact = [i for i, token in enumerate(row) if token != noise]
                end = next(i for i in fact if row[i + 1] == noise) + 1
                self.assertEqual(row[end:], foil[end:])
                self.assertEqual(row[:fact[0]], foil[:fact[0]])
                self.assertNotEqual(row[fact[0]:end], foil[fact[0]:end])
            if one_token:
                self.assertTrue(torch.equal(counts, torch.ones_like(counts)))
        delta = [0., 0.1, 0.4, 1., 2.]
        report = eval_long.paired_bootstrap(delta, seed=41)
        self.assertEqual(report, eval_long.paired_bootstrap(delta, seed=41))
        self.assertAlmostEqual(report["mean"], 0.7)
        self.assertEqual(eval_long.paired_bootstrap([0., 0.])["ci95"], [0., 0.])

    def test_legacy_and_paired_json_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cfg = TFConfig(vocab=50257, d_model=8, n_layers=1, n_heads=2, attn="linear")
            model = GPT(cfg)
            torch.save({"arch": "tf", "cfg": vars(cfg), "model": model.state_dict(), "ema": model.state_dict()},
                       path / "ckpt.pt")
            (path / "args.json").write_text(json.dumps({"data": directory, "seq": 128, "ema": 0.999,
                                                      "tokenizer": str(TOKENIZER)}))
            np.save(path / "valid_books.npy", np.array([0]))
            (np.arange(900) % 32).astype(np.uint16).tofile(path / "valid.bin")
            opts = ["eval_long.py", directory, "--length", "128", "--chunk", "17", "--ema"]
            with patch("sys.argv", opts):
                eval_long.main()
            legacy = json.loads((path / "long_valid_ema.json").read_text())
            self.assertEqual(set(legacy), {"split", "books", "length", "ema", "window", "trained_seq", "mean", "buckets"})
            with patch("sys.argv", opts + ["--restart", "64", "--recall", "--distances", "32", "96", "--max_trials", "2"]):
                eval_long.main()
            report = json.loads((path / "long_valid.json").read_text())
            pair = report["restart_control"]
            self.assertEqual(pair["tokens"], 64)
            self.assertEqual(report["buckets"], pair["full"]["buckets"])
            self.assertEqual(pair["full"]["buckets"].keys(), pair["restart"]["buckets"].keys())
            self.assertAlmostEqual(pair["gain"]["mean"], pair["restart"]["mean"] - pair["full"]["mean"], places=5)
            self.assertEqual(report["planted_recall"]["by_distance"]["96"]["trials"], 2)
            cold = (path / "long_valid.json").read_bytes()
            with patch("sys.argv", opts + ["--restart", "64", "--warmup", "64", "--recall",
                                           "--distances", "32", "96", "--max_trials", "2"]):
                eval_long.main()
            warm = json.loads((path / "long_valid_warm.json").read_text())
            self.assertEqual(warm["warm_restart_control"]["warmup_tokens"], 64)
            self.assertEqual(warm["planted_recall"]["positive_control"]["answer_tokens"], 2)
            self.assertIn("ci95", warm["planted_recall"]["by_distance"]["96"]["full_minus_warm_preference"])
            self.assertEqual((path / "long_valid.json").read_bytes(), cold)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

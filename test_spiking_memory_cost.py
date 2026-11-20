"""CPU checks for plastic-memory events, cost, simulation, and carried state."""
from dataclasses import asdict, replace
import itertools
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import fpga_est
import hw_cost
from model import Config, PlasticSynapses, SpikingLM
from quant_sim import calibrate, run, supported
from train import mask_state


class SpikingMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = Config(vocab=17, d_model=12, d_hidden=9, n_layers=2, leak_rank=4, mem_heads=2,
                          mem_keys=3, mem_dv=4, mem_every=2, mem_chunk=6, mem_theta=0.4,
                          mem_write_avg=True, reset_learn=True, hard_reset=True, bus_norm=True,
                          out_classes=3, out_short=2)

    def test_event_accounting_and_placement(self):
        spikes = torch.tensor([[[0, 1, 2, 3, 4, 0, 1, 0, 0, 4, 0, 2]]])
        measured = hw_cost.memory_firing(spikes)
        self.assertAlmostEqual(measured["mem_key_rate"], 4 / 6)
        self.assertAlmostEqual(measured["mem_query_rate"], 3 / 6)
        self.assertAlmostEqual(measured["mem_key_count"], 10 / 6)
        self.assertAlmostEqual(measured["mem_query_count"], 7 / 6)
        f = {k: np.zeros(2) for k in ("rate", "token", "token_any")}
        f.update({k: np.array([v]) for k, v in measured.items()})

        base = SpikingLM(self.cfg)
        cfg = replace(self.cfg, mem_kq_levels=4, mem_delays=4, mem_gate="channel")
        enhanced = SpikingLM(cfg)
        old = hw_cost.snn_cost(base, f)
        cost = hw_cost.snn_cost(enhanced, f)
        # The fixture has 17 pulses writing/reading four value channels each.
        self.assertAlmostEqual(cost["memory_sparse_adds"], 68, places=4)
        self.assertAlmostEqual(cost["sparse_adds"], 68, places=4)
        # 20 projected currents, four taps, 36 extra potentials, eight gates on a width-12 bus.
        self.assertEqual(cost["dense_macs"] - old["dense_macs"], 20 * 4 + 36 + 12 * 8 + 8)
        self.assertEqual(cost["dense_weights"] - old["dense_weights"], 20 * 4 + 12 * 8)
        self.assertEqual(cost["state_bytes"] - old["state_bytes"], 3 * (36 + 20 * 3))
        head = hw_cost.snn_cost(SpikingLM(replace(cfg, mem_gate="head")), f)
        self.assertEqual(cost["dense_macs"] - head["dense_macs"], 12 * (8 - 2))
        stateless = hw_cost.snn_cost(SpikingLM(replace(cfg, mem_kq_stateless=True)), f)
        self.assertEqual(cost["state_bytes"] - stateless["state_bytes"], 3 * 48)
        self.assertEqual(cost["dense_macs"] - stateless["dense_macs"], 48)
        binary_f = {k: v for k, v in f.items() if not k.endswith("_count")}
        self.assertAlmostEqual(hw_cost.snn_cost(base, binary_f)["memory_sparse_adds"], 28, places=4)
        with self.assertRaisesRegex(ValueError, "mean key and query counts"):
            hw_cost.snn_cost(enhanced, binary_f)

        budget = cost["state_bytes"] + (cost["params"] - cost["table_weights"]) / 2
        dev = dict(fpga_est.DEVICES["u55c"], cards=1, eff=0.6, onchip_mb=budget / 0.9e6 + 1e-6)
        first, _, sixteen, *_ = fpga_est.estimate(cost, asdict(cfg), 4, dev)
        self.assertEqual(first["where"], "weights on chip, word tables in HBM")
        self.assertEqual(first["required_onchip_bytes"], budget)
        self.assertGreater(sixteen["required_onchip_bytes"], sixteen["onchip_budget_bytes"])
        self.assertAlmostEqual(first["ops_per_step"], cost["dense_macs"] + 68, places=4)
        self.assertEqual(first["bytes_per_step"], cost["per_stream_weights"] / 2)

    @torch.no_grad()
    def test_stateless_thresholds_and_delay(self):
        bus = torch.zeros(1, 5, self.cfg.d_model)
        bus[..., 0] = torch.tensor([1.1, 0.05, 2.7, -0.8, 1.6])
        for levels, delays in itertools.product((1, 4), (0, 4)):
            with self.subTest(levels=levels, delays=delays):
                cfg = replace(self.cfg, mem_kq_levels=levels, mem_delays=delays, mem_kq_stateless=True)
                mem = PlasticSynapses(cfg)
                mem.kq_in.weight.zero_()
                mem.kq_in.weight[:, 0] = 1.0
                I = mem.kq_in(bus)
                if delays:
                    mem.delay_dw[:, 0, 0] = 0.5
                    I = I + 0.5 * F.pad(I[:, :-3], (0, 0, 3, 0))
                expected = (I.unsqueeze(-1) >= mem.kq_log_theta.exp().view(I.size(-1), levels)).sum(-1)
                _, counts, state = mem(bus)
                self.assertTrue(torch.equal(counts, expected))
                self.assertIsNone(state[1])
                _, poisoned, _ = mem(bus, (torch.zeros_like(state[0]), torch.full((1, I.size(-1) * levels), 1e5))
                                    + ((torch.zeros_like(state[2]),) if delays else ()))
                self.assertTrue(torch.equal(counts, poisoned))

    @torch.no_grad()
    def test_unrounded_parity_causality_and_chunks(self):
        x = torch.randint(self.cfg.vocab, (2, 17))
        other = x.clone()
        other[:, 9:] = (x[:, 9:] + 1) % self.cfg.vocab
        for levels, delays, stateless in itertools.product((1, 4), (0, 4), (False, True)):
            with self.subTest(levels=levels, delays=delays, stateless=stateless):
                gate = "channel" if levels == 4 else ("head" if delays else "")
                cfg = replace(self.cfg, mem_kq_levels=levels, mem_delays=delays, mem_kq_stateless=stateless, mem_gate=gate)
                model = SpikingLM(cfg).eval()
                for mem in model.mems.values():
                    mem.w_out.weight.normal_(std=0.2)
                    if delays:
                        mem.delay_dw.normal_(std=0.1)
                whole, _, state, rates = model(x, return_spikes=True)
                left, _, saved, l_rates = model(x[:, :9], return_spikes=True)
                right, _, carried, r_rates = model(x[:, 9:], state=saved, return_spikes=True)
                torch.testing.assert_close(torch.cat([left, right], dim=1), whole, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(model(other)[0][:, :9], whole[:, :9], rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(carried, state, rtol=1e-5, atol=2e-6)
                for l, r, s in zip(l_rates, r_rates, rates):
                    self.assertTrue(torch.equal(torch.cat([l, r], dim=1), s))
                reset = mask_state(carried, torch.tensor([True, False]))
                self.assertTrue(torch.equal(reset[-1][0][0][0], carried[-1][0][0][0]))
                self.assertEqual(reset[-1][0][0][1].count_nonzero().item(), 0)

                self.assertTrue(supported(asdict(cfg)))
                rec = {}
                torch.testing.assert_close(run(model, x, {}, rec=rec), whole, rtol=1e-5, atol=2e-6)
                cal = calibrate(rec)
                rounded = run(model, x, dict(w=4, a=8, v=16, b=16, s=16), cal)
                self.assertTrue(torch.isfinite(rounded).all())
                self.assertFalse(torch.equal(rounded, whole))

    def test_stateless_off_is_the_legacy_default(self):
        x = torch.randint(self.cfg.vocab, (2, 11))
        cfg = replace(self.cfg, mem_kq_levels=4, mem_delays=4, mem_gate="channel")
        old_cfg = asdict(cfg)
        del old_cfg["mem_kq_stateless"]
        model = SpikingLM(Config.from_dict(old_cfg))
        with torch.no_grad():
            model.mems["1"].w_out.weight.normal_(std=0.2)
            model.mems["1"].delay_dw.normal_(std=0.1)
        explicit_off = SpikingLM(replace(cfg, mem_kq_stateless=False))
        explicit_off.load_state_dict(model.state_dict(), strict=True)
        for net in (model, explicit_off):
            logits, loss, state, spikes = net(x, x, return_spikes=True)
            loss.backward()
            if net is model:
                expected = logits, loss, state, spikes
        torch.testing.assert_close((logits, loss, state, spikes), expected, rtol=0, atol=0)
        for p, other in zip(model.parameters(), explicit_off.parameters()):
            torch.testing.assert_close(p.grad, other.grad, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

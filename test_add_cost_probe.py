"""CPU tests for the add-cost probe. Install Icarus Verilog for the RTL test."""
from itertools import product
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from experiments.add_cost_probe import PRIMARY_LAYOUT, concurrent_lanes, evaluate, projection_bound, row_probabilities

ROOT = Path(__file__).resolve().parent
TESTBENCH = """
module testbench;
    reg clk = 0;
    reg signed [3:0] w0, w1, w2, a4, level;
    reg signed [7:0] a8;
    reg signed [31:0] acc0, acc1, acc2;
    wire [31:0] mac8, p80, p81, p0, p1, p2, binary, event_out, skipped;
    wire read_weight;
    reg [31:0] ref8, ref81, ref0, ref1, ref2, refb, refe, refs;
    reg ref_read;
    integer f, n, count;

    mac4x8 m(clk, w0, a8, acc0, mac8);
    packed_mac4x8 m2(clk, w0, w1, a8, acc0, acc1, p80, p81);
    packed_mac4x4 p(clk, w0, w1, w2, a4, acc0, acc1, acc2, p0, p1, p2);
    binary_event b(clk, w0, acc0, binary);
    level_event e(clk, w0, level, acc0, event_out);
    zero_skip_mac4x4 s(clk, w0, a4, acc0, skipped, read_weight);

    initial begin
        f = $fopen("vectors.txt", "r");
        if (!f) $fatal(1, "missing vectors");
        count = 0;
        n = $fscanf(f, "%d %d %d %d %d %d %d %d %d %h %h %h %h %h %h %h %h %d",
                    w0, w1, w2, a4, a8, level, acc0, acc1, acc2, ref8, ref81, ref0, ref1, ref2, refb, refe, refs, ref_read);
        while (n == 18) begin
            #1 clk = 1;
            #1;
            if ({mac8, p80, p81, p0, p1, p2, binary, event_out, skipped, read_weight}
                 !== {ref8, ref8, ref81, ref0, ref1, ref2, refb, refe, refs, ref_read})
                $fatal(1, "mismatch at vector %d", count);
            clk = 0;
            count = count + 1;
            n = $fscanf(f, "%d %d %d %d %d %d %d %d %d %h %h %h %h %h %h %h %h %d",
                        w0, w1, w2, a4, a8, level, acc0, acc1, acc2,
                        ref8, ref81, ref0, ref1, ref2, refb, refe, refs, ref_read);
        end
        $display("%0d vectors matched", count);
        $finish;
    end
endmodule
"""


class AddCostTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "install Icarus Verilog for RTL checks")
    def test_integer_reference(self):
        # mac4x8 is shared by the DSP and -nodsp mappings. Check all W4A4
        # tuples and random W4A8 activations, levels and accumulators.
        rng = random.Random(12)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "check.v").write_text(TESTBENCH)
            with (root / "vectors.txt").open("w") as file:
                for i, (w0, w1, w2, a4) in enumerate(product(range(-8, 8), repeat=4)):
                    a8, level = rng.randrange(-128, 128), rng.randrange(-4, 5)
                    acc0, acc1, acc2 = (rng.randrange(-2**31, 2**31) for _ in range(3))
                    if i % 64 == 0:
                        acc0 = rng.choice((-2**31, 2**31 - 1))
                    inputs = (w0, w1, w2, a4, a8, level, acc0, acc1, acc2)
                    expected = (acc0 + w0 * a8, acc1 + w1 * a8, acc0 + w0 * a4, acc1 + w1 * a4, acc2 + w2 * a4,
                                acc0 + w0, acc0 + w0 * level, acc0 + (w0 * a4 if a4 else 0))
                    print(*inputs, *(f"{value & 0xffffffff:08x}" for value in expected), int(a4 != 0), file=file)
            subprocess.run(["iverilog", "-g2012", "-s", "testbench", "-o", "test",
                            str(ROOT / "fpga/add_cost_lanes.v"), "check.v"], cwd=root, check=True)
            result = subprocess.check_output(["vvp", "test"], cwd=root, text=True)
            self.assertIn("65536 vectors matched", result)

    def test_histogram_and_activity_extremes(self):
        layers = [(0.5, 64, 16.), (0.1, 128, 16.)]
        self.assertEqual(row_probabilities(layers, [4, 1]),
                         {"low_activity": [0.125, 0.1], "high_activity": [0.5, 0.1]})
        observed = row_probabilities(layers, [4, 1], [[0.75, 0.125, 0., 0.125, 0.]])
        self.assertEqual(observed, {"measured_histogram": [0.25, 0.1]})
        with self.assertRaises(ValueError):
            row_probabilities(layers, [4, 1], [[0.75, 0.25, 0., 0., 0.]])

    def test_layouts_and_workloads(self):
        dsp = dict(lut=32, carry4_units=8, ff=32, dsp=1)
        lanes = {
            "mac4x8_dsp": dict(resources=dsp, updates_per_cycle=1),
            "packed_mac4x8": dict(resources=dict(lut=75, carry4_units=19, ff=64, dsp=1), updates_per_cycle=2),
            "packed_mac4x4": dict(resources=dict(lut=115, carry4_units=34, ff=96, dsp=1), updates_per_cycle=3),
            "binary_event": dict(resources=dict(dsp, dsp=0)),
            "level_event": dict(resources=dict(lut=53, carry4_units=11, ff=32, dsp=0)),
        }
        budget, layouts = concurrent_lanes(lanes, 0.2)
        self.assertEqual(len(layouts), 6)
        self.assertEqual(layouts[PRIMARY_LAYOUT]["macs_per_cycle"], 14438)
        self.assertEqual(layouts["1_yosys"]["events_per_cycle"]["level_event"], (1042944 - 7219 * 32) // 53)
        for width in (1, 2, 3):
            mapped = layouts[f"{width}_yosys"]
            ideal = layouts[f"{width}_dsp_zero_fabric"]
            self.assertEqual(mapped["macs_per_cycle"], 7219 * width)
            self.assertEqual(ideal["macs_per_cycle"], 7219 * width)
            self.assertEqual(ideal["spare_resources"]["lut"], budget["lut"])
            self.assertGreater(ideal["events_per_cycle"]["level_event"], mapped["events_per_cycle"]["level_event"])
        run = evaluate("g45_kd25", layouts, None, [200], [0.25])
        self.assertEqual(set(run["layouts"]), set(layouts))
        self.assertGreater(run["dense_projection_macs_per_token"], 25e6)  # Includes the full head.
        for layout in run["layouts"].values():
            self.assertEqual(set(layout["ranges_by_activity"]), {"low_activity", "high_activity"})
            for row in layout["cases"]:
                bound = "serial" if row["streams"] == 1 else "parallel"
                self.assertEqual(row["c_effective_primary"], row[f"c_effective_{bound}"])

    def test_bounds(self):
        traffic = dict(dense_resident_bytes=2 * 23808, event_resident_bytes=4 * 23808,
                       hbm_bytes=0, event_hbm_bytes=0)
        row = projection_bound(100, 60, 120, (10, 20), traffic, 1, 250, 1)
        self.assertEqual(row["parallel_cycles"], 10)
        self.assertEqual(row["serial_cycles"], 13)
        self.assertEqual(row["c_effective_parallel"], 0)
        self.assertEqual(row["c_event_compute"], 0.25)
        self.assertEqual(row["primary_cycles"], 13)
        row = projection_bound(100, 60, 120, (10, 20), traffic, 0.25, 250, 1)
        self.assertEqual(row["onchip_weight_cycles"], 24)
        self.assertEqual(row["parallel_cycles"], 24)
        self.assertAlmostEqual(row["c_effective_parallel"], 14 / 12)


if __name__ == "__main__":
    unittest.main()

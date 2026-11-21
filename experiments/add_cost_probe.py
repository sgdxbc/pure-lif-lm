"""Synthesize U55C arithmetic lanes and model projection throughput at 1 and 16 streams.

Install fpga/requirements-add-cost.txt, then run from the repo root:
    python3 experiments/add_cost_probe.py --out runs/add_cost_probe.json

Optional --histograms JSON maps a run name to one probability vector [p0,...,pK]
per quant relay, in validation-log order. Without these, bracket row activity from
the logged mean magnitude. See the output assumptions for the bandwidth model.
"""
import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fpga_est

YOWASP = "0.71.0.0.post1285"
YOSYS_COMMIT = "b4c586fd3"
# U55C DS978. Count CARRY8 as two four-bit carry units for this area bound.
BOARD = {"lut": 1303680, "carry4_units": 325920, "ff": 2607360, "dsp": 9024}
WEIGHT_BANKS = {"bram36": 2016, "uram": 960, "data_bytes_per_bank_cycle": 8}
LANES = {
    "mac4x8_dsp": ("mac4x8", "", 1),
    "mac4x8_fabric": ("mac4x8", "-nodsp", 1),
    "packed_mac4x8": ("packed_mac4x8", "", 2),
    "packed_mac4x4": ("packed_mac4x4", "", 3),
    "binary_event": ("binary_event", "-nodsp", 1),
    "level_event": ("level_event", "-nodsp", 1),
    "zero_skip_mac4x4": ("zero_skip_mac4x4", "-nodsp", 1),
}
PRIMARY_LAYOUT = "2_yosys"


def synthesize(yosys):
    yosys = shutil.which(yosys)
    if not yosys:
        raise ValueError(f"install yowasp-yosys=={YOWASP} or give its path with --yosys")
    version = subprocess.check_output([yosys, "-V"], text=True).strip()
    if f"git sha1 {YOSYS_COMMIT}" not in version:
        raise ValueError(f"expected yowasp-yosys=={YOWASP}, received {version}")
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        shutil.copyfile(ROOT / "fpga/add_cost_lanes.v", path / "lanes.v")
        for name, (top, flags, updates) in LANES.items():
            script = (f"read_verilog lanes.v; synth_xilinx -family xcup -top {top} "
                      f"-noiopad -noclkbuf -flatten {flags}; write_json lane.json")
            subprocess.run([yosys, "-Q", "-q", "-p", script], cwd=path, check=True)
            cells = json.loads((path / "lane.json").read_text())["modules"][top]["cells"]
            counts = Counter(c["type"] for c in cells.values())
            used = {"lut": sum(v for k, v in counts.items() if k.startswith("LUT")),
                    "carry4_units": counts["CARRY4"] + 2 * counts["CARRY8"],
                    "ff": sum(v for k, v in counts.items() if k.startswith("FD")),
                    "dsp": counts["DSP48E2"]}
            if used["dsp"] != (0 if flags else 1):
                raise ValueError(f"{name} unexpected DSP mapping {dict(counts)}")
            results[name] = dict(primitives=dict(sorted(counts.items())), resources=used, updates_per_cycle=updates)
    return version, results


def lane_count(budget, used):
    return min(budget[k] // count for k, count in used.items() if count)


def concurrent_lanes(lanes, reserved):
    budget = {k: int(n * (1 - reserved)) for k, n in BOARD.items()}
    layouts = {}
    for lane in ("packed_mac4x8", "mac4x8_dsp", "packed_mac4x4"):
        products = lanes[lane]["updates_per_cycle"]
        measured = lanes[lane]["resources"]
        zero_fabric = {k: measured[k] if k == "dsp" else 0 for k in budget}
        for accumulator, used in (("yosys", measured), ("dsp_zero_fabric", zero_fabric)):
            dense = lane_count(budget, used)
            spare = {k: budget[k] - dense * used[k] for k in budget}
            events = {kind: lane_count(spare, lanes[kind]["resources"]) for kind in ("binary_event", "level_event")}
            if min(dense, *events.values()) <= 0:
                raise ValueError(f"{lane}/{accumulator} leaves no room for both engines")
            layouts[f"{products}_{accumulator}"] = dict(
                dense_lane=lane, accumulator_case=accumulator, macs_per_dsp=products, dense_lanes=dense,
                dense_lane_resources=used, spare_resources=spare,
                macs_per_cycle=dense * products, events_per_cycle=events)
    return budget, layouts


def load_workload(run):
    path = ROOT / "runs" / run
    args = json.loads((path / "args.json").read_text())
    evals = [r for r in map(json.loads, (path / "log.jsonl").read_text().splitlines()) if "val_rates" in r]
    eval_ = evals[-1]
    key = "val_rates_ema" if "val_rates_ema" in eval_ else "val_rates"
    if args.get("arch") != "tf" or args.get("wq", "none") != "none" or args.get("out_classes"):
        raise ValueError(f"{run} needs full-head, non-ternary model_tf projection counts")
    n_relays = 3 if args.get("relay", "none") != "none" else 0
    has_binary_mlp = args.get("mlp", "gelu") != "gelu"
    per_layer = n_relays + int(has_binary_mlp)
    if not per_layer or len(eval_[key]) != per_layer * args["n_layers"]:
        raise ValueError(f"{run} needs saved binary or quant relay rates")
    levels = ([args.get("relay_levels", 1)] * n_relays + [1] * int(has_binary_mlp)) * args["n_layers"]
    hw = fpga_est.tf_hw(args, rates=eval_[key], bits=4)
    return hw, levels, {"step": eval_["step"], "rate_key": key}


def row_probabilities(layers, levels, histograms=None):
    """Extremes are sharp for each row family given its mean magnitude and cap."""
    means = [rate for rate, _, _ in layers]
    if any(not 0 <= r <= k for r, k in zip(means, levels, strict=True)):
        raise ValueError("saved rate outside its quantized range")
    if max(levels) == 1:
        return {"binary_rates": means}
    if histograms is None:
        return {"low_activity": [r / k for r, k in zip(means, levels)],
                "high_activity": [min(1., r) for r in means]}
    hist = iter(histograms)
    probs = []
    for r, k in zip(means, levels):
        if k == 1:
            probs.append(r)
            continue
        p = next(hist, ())
        if (len(p) != k + 1 or not all(0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1, abs_tol=1e-5)
                or not math.isclose(sum(i * v for i, v in enumerate(p)), r, abs_tol=1e-4)):
            raise ValueError("histogram does not match the saved rate")
        probs.append(sum(p[1:]))
    if next(hist, None) is not None:
        raise ValueError("too many relay histograms")
    return {"measured_histogram": probs}


def projection_traffic(hw, probabilities, streams):
    layers = hw["sparse_layers"]
    sparse_sizes = [n * row_bytes for _, n, row_bytes in layers]
    sparse_reads = [size * (1 - (1 - p) ** streams) for size, p in zip(sparse_sizes, probabilities, strict=True)]
    dense_reads = hw["dense_bytes"]
    u55c = fpga_est.DEVICES["u55c"]
    regions = [("state", streams * hw["state_bytes"], 2 * streams * hw["state_bytes"]),
               ("kv", streams * hw["kv_bytes"], streams * hw["kv_bytes"]),
               ("dense_weights", hw["weight_bytes_onchip"] - sum(sparse_sizes), dense_reads)]
    regions.extend(("sparse_weights", size, reads) for size, reads in zip(sparse_sizes, sparse_reads))
    hbm, _, split = fpga_est.place(regions, 0.9e6 * u55c["onchip_mb"], hw["table_weights"] / 2,
                                   streams * hw["per_stream_bytes"])
    sparse_hbm = split["sparse_weights"]["hbm_bytes_per_step"]
    return dict(dense_resident_bytes=dense_reads - split["dense_weights"]["hbm_bytes_per_step"],
                event_resident_bytes=max(0., sum(sparse_reads) - sparse_hbm),
                hbm_bytes=hbm, event_hbm_bytes=sparse_hbm)


def projection_bound(dense, events, spikes, rates, traffic, bank_fraction, clock, streams):
    mac_rate, event_rate = rates
    mac_cycles, event_cycles = streams * dense / mac_rate, streams * events / event_rate
    resident_bw = (WEIGHT_BANKS["bram36"] + WEIGHT_BANKS["uram"]) * WEIGHT_BANKS["data_bytes_per_bank_cycle"]
    resident_bw *= bank_fraction
    hbm_bw = fpga_est.DEVICES["u55c"]["bw_gbs"] * 600 / clock  # 60% peak, bytes/cycle
    read_dense = traffic["dense_resident_bytes"] / resident_bw
    read_events = traffic["event_resident_bytes"] / resident_bw
    read_hbm = traffic["hbm_bytes"] / hbm_bw
    base = max(mac_cycles, read_dense, read_hbm - traffic["event_hbm_bytes"] / hbm_bw)
    memory = max(read_dense + read_events, read_hbm)
    parallel = max(mac_cycles, event_cycles, memory)
    serial = max(mac_cycles + event_cycles, memory)
    scale = mac_rate / (streams * spikes)
    primary = serial if streams == 1 else parallel
    return dict(resident_bytes_per_cycle=resident_bw, onchip_weight_cycles=read_dense + read_events,
                hbm_cycles=read_hbm, dense_cycles=mac_cycles, event_cycles=event_cycles,
                parallel_cycles=parallel, serial_cycles=serial, primary_cycles=primary,
                parallel_tokens_s=streams * clock * 1e6 / parallel, primary_tokens_s=streams * clock * 1e6 / primary,
                c_event_compute=event_cycles * scale, c_effective_parallel=(parallel - base) * scale,
                c_effective_serial=(serial - base) * scale, c_effective_primary=(primary - base) * scale)


def evaluate(run, layouts, histograms, clocks, bank_fractions):
    hw, levels, source = load_workload(run)
    layers = hw["sparse_layers"]
    choices = row_probabilities(layers, levels, histograms)
    dense = 2 * hw["dense_bytes"]
    if hw["sparse_adds"] <= 0:
        raise ValueError(f"{run} has no measured spike updates for normalizing c")
    event_lane = "level_event" if max(levels) > 1 else "binary_event"
    results = {}
    for name, layout in layouts.items():
        rates = layout["macs_per_cycle"], layout["events_per_cycle"][event_lane]
        cases, ranges = [], {}
        for activity, probabilities in choices.items():
            updates = sum(p * n * rb * 2 for p, (_, n, rb) in zip(probabilities, layers))
            ranges[activity] = {}
            for streams in (1, 16):
                traffic = projection_traffic(hw, probabilities, streams)
                stream_cases = []
                for clock, bank in product(clocks, bank_fractions):
                    bounds = projection_bound(dense, updates, hw["sparse_adds"], rates, traffic, bank, clock, streams)
                    stream_cases.append(dict(activity=activity, streams=streams, clock_mhz=clock, bank_fraction=bank,
                                             event_updates_per_token=updates, **traffic, **bounds))
                cases.extend(stream_cases)
                ranges[activity][str(streams)] = {
                    key: [min(c[key] for c in stream_cases), max(c[key] for c in stream_cases)]
                    for key in ("c_event_compute", "c_effective_primary", "c_effective_parallel", "c_effective_serial")}
        results[name] = dict(ranges_by_activity=ranges, cases=cases)
    return dict(rate_source=source, event_lane=event_lane, dense_projection_macs_per_token=dense,
                other_estimator_macs_per_token=hw["dense_macs"] - dense,
                spike_sum_weight_updates_per_token=hw["sparse_adds"], nonzero_probabilities=choices,
                layouts=results)


def format_range(values):
    low, high = (f"{v:.3f}" for v in values)
    return low if low == high else f"{low}–{high}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", default="yowasp-yosys")
    parser.add_argument("--runs", nargs="+", default=["g40_laspk", "g45_kd25", "g43_q4t163"])
    parser.add_argument("--histograms", type=Path)
    parser.add_argument("--reserve", type=float, default=0.2)
    parser.add_argument("--clocks", nargs="+", type=float, default=[200, 250, 300])
    parser.add_argument("--bank-fractions", nargs="+", type=float, default=[0.10, 0.25, 0.5, 1.0])
    parser.add_argument("--out", type=Path, default=Path("runs/add_cost_probe.json"))
    args = parser.parse_args()
    if (not 0 <= args.reserve < 1 or not all(0 < b <= 1 for b in args.bank_fractions)
            or not all(math.isfinite(f) and f > 0 for f in args.clocks)):
        parser.error("invalid control reservation, clock or bank fraction")
    histograms = json.loads(args.histograms.read_text()) if args.histograms else {}
    version, lanes = synthesize(args.yosys)
    budget, layouts = concurrent_lanes(lanes, args.reserve)
    runs = {name: evaluate(name, layouts, histograms.get(name), args.clocks, args.bank_fractions) for name in args.runs}
    assumptions = {
        "card": "AMD Alveo U55C", "resources": BOARD, "control_reserved_fraction": args.reserve,
        "weight_banks": WEIGHT_BANKS, "active_bank_fractions": args.bank_fractions, "assumed_clocks_mhz": args.clocks,
        "hbm": "60% of 460 GB/s, fpga_est divisible placement with 90% on-chip capacity",
        "compute": "Dense projections include full-head readout. W4A8 at 1 or 2 MACs/DSP-cycle, packed W4A4 at 3. Event updates in LUT fabric. II=1 assumes independent accumulator inputs or a feedback path that sustains one update per cycle",
        "accumulators": {
            "yosys": "Synthesized fabric accumulators and registers, including signed packing and extraction where used",
            "dsp_zero_fabric": "Ideal resource envelope charging only DSPs for dense lanes. P feedback can hold one 32-bit sum. At 2/3 MACs per DSP this is a counterfactual with free packing, extraction and external accumulation, since P cannot hold independent 32-bit sums for all outputs",
        },
        "area": "LUT and carry budget applied independently; carry halves assume ideal pairing, FF packing and routing unmodeled",
        "reads": "One weight row per nonzero level; perfect shared-row broadcast over independent streams, no bank conflicts",
        "memory": "One 64-data-bit read per active bank per cycle. Active-bank fractions model delivered bandwidth; 10% is a stress case. Resident state ports, accumulator traffic and scale reads unmodeled",
        "bounds": "Projection-only. Primary at 1 stream is serial engines, with memory allowed to overlap. The 1-stream parallel counterfactual requires independent work absent from a single token chain. At 16 streams the primary overlap bound assumes independent stream work and scheduling that exposes it",
        "c": "Per spike-sum weight update, in each dense layout's DSP MAC units. Event c charges event-engine time. Effective c charges marginal time vs dense work at the same placement",
        "limits": "Cell counts give neither achievable clock nor dynamic power. No measured board timing or energy",
    }
    report = dict(yowasp_yosys=YOWASP, yosys_version=version, assumptions=assumptions, lanes=lanes,
                  resource_budget=budget, primary_layout=PRIMARY_LAYOUT, layouts=layouts, runs=runs)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(version)
    print(f"Primary layout {PRIMARY_LAYOUT}. Packed zero-fabric rows are ideal resource envelopes.")
    print("| MACs/DSP | Accumulator case | Workload / activity | Event-engine c | Effective c, 1 serial | Effective c, 16 overlap |")
    print("|---:|---|---|---:|---:|---:|")
    for name, layout in layouts.items():
        for run, values in runs.items():
            for activity, ranges in values["layouts"][name]["ranges_by_activity"].items():
                cells = [format_range(ranges["1"]["c_event_compute"]),
                         format_range(ranges["1"]["c_effective_primary"]),
                         format_range(ranges["16"]["c_effective_primary"])]
                print("| " + " | ".join([str(layout["macs_per_dsp"]), layout["accumulator_case"],
                                        f"{run} / {activity}", *cells]) + " |")


if __name__ == "__main__":
    main()

"""Best-case logical traffic for surprise-gating one GLA block."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpga_est import tf_hw

CELLS = 32
TARGET = 0.20


def upper_bound(fixed_weight_bytes, refresh_weight_bytes, fixed_stream_bytes,
                refresh_stream_bytes, streams, detector_bytes):
    fixed = fixed_weight_bytes / streams + fixed_stream_bytes
    refresh = refresh_weight_bytes / streams + refresh_stream_bytes
    full = fixed + refresh
    quiet = fixed + detector_bytes
    saving = (full - quiet) / full
    return {
        "streams": streams,
        "always_refresh_bytes_per_token": full,
        "suppressible_bytes_per_token": refresh,
        "minimum_detector_bytes_per_token": detector_bytes,
        "best_case_bytes_per_token": quiet,
        "maximum_saving_fraction": saving,
        "required_suppressible_bytes_at_this_baseline": TARGET * full + detector_bytes,
        "clears_bound": saving >= TARGET,
    }


def report(args):
    if (args.get("attn"), args.get("mlp"), args.get("relay"), args.get("wq"), args.get("out_classes")) != (
        "linear", "gelu", "none", "none", 0
    ):
        raise ValueError("the bound assumes dense GLA with a flat vocabulary head")
    full = tf_hw(args, bits=4)
    retained = tf_hw(dict(args, n_layers=args["n_layers"] - 1), bits=4)
    # bank_traffic in probe_regime.py charges 13 bytes read and 4 bytes written
    # per cell, plus one selection byte. The toy prediction tables aren't used.
    detector_bytes = CELLS * (13 + 4) + 1
    rows = [upper_bound(
        retained["weight_bytes_onchip"],
        full["weight_bytes_onchip"] - retained["weight_bytes_onchip"],
        retained["per_stream_bytes"] + 2 * retained["state_bytes"],
        2 * (full["state_bytes"] - retained["state_bytes"]),
        streams, detector_bytes,
    ) for streams in (1, 16)]
    return {"run": args["out"], "bits": 4, "refreshable_blocks": 1,
            "min_saving_fraction": TARGET, "rows": rows,
            "build_text_probe": any(row["clears_bound"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    shape = json.loads((ROOT / "runs/g48_gla31/args.json").read_text())
    result = json.dumps(report(shape), indent=2) + "\n"
    if args.out:
        args.out.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()

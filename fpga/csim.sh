#!/bin/bash
# C simulation of the FPGA kernel (fpga/lif_lm.cpp compiled with g++), checked bit for bit against hw_ref.cpp
# (docs/experiments.md 5.44). From the repository root:
#   bash fpga/csim.sh HW_DIR [NB] [STEPS] [PROMPT]
# HW_DIR: hw_export.py's output (manifest.txt, tables.bin; windows.bin when written with --windows, else random
# tokens). NB streams (default 4), STEPS steps (default 32), PROMPT prompt tokens for the generation check (default 8).
# Needs the arbitrary-precision types of Vitis HLS (ap_int.h): on a machine with Vitis, $XILINX_HLS/include; else
# AMD's open-source copy, github.com/Xilinx/HLS_arbitrary_Precision_Types (AP_INC=its include directory).
# Checks: (1) every stream reads the same tokens in both (teacher forcing): the bus after the last layer at every step
# (int32) and the greedy prediction after every step must be identical; (2) the kernel generates from a prompt, and
# hw_ref, reading the prompt and the generated tokens, must predict exactly the tokens the kernel generated.
set -e
cd "$(dirname "$0")/.."
HW=$1; NBS=${2:-4}; STEPS=${3:-32}; PROMPT=${4:-8}
AP_INC=${AP_INC:-${XILINX_HLS:+$XILINX_HLS/include}}
[ -n "$AP_INC" ] && [ -f "$AP_INC/ap_int.h" ] || { echo "set AP_INC to a directory with ap_int.h"; exit 1; }
OUT=$(mktemp -d)
python3 fpga/pack.py "$HW" --out "$OUT/pack" >/dev/null
g++ -O3 -march=native -fopenmp -ffp-contract=off -o "$OUT/hw_ref" hw_ref.cpp
g++ -std=c++17 -O2 -ffp-contract=off -Wno-unknown-pragmas -DNB=$NBS -I "$OUT/pack" -I "$AP_INC" \
    -o "$OUT/csim" fpga/lif_lm.cpp fpga/csim_main.cpp
python3 - "$HW" "$OUT" $NBS $STEPS <<'EOF'
import sys, os, numpy as np
hw, out, nb, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
vocab = int([l.split()[2] for l in open(os.path.join(hw, "manifest.txt")) if l.startswith("param vocab ")][0])
wf = os.path.join(hw, "windows.bin")
if os.path.exists(wf):
    w = np.fromfile(wf, dtype=np.int32).reshape(-1, 257)[:nb, :steps + 1]
else:
    w = np.random.default_rng(0).integers(0, vocab, (nb, steps + 1)).astype(np.int32)
assert w.shape == (nb, steps + 1), "need NB windows of STEPS + 1 tokens"
w.tofile(os.path.join(out, "win.bin"))                                    # for hw_ref: N x (LEN + 1)
np.ascontiguousarray(w[:, :steps]).tofile(os.path.join(out, "tok.bin"))   # for the kernel: NB x STEPS
EOF
HWREF_BUS=$OUT/ref_bus.bin HWREF_GREEDY=$OUT/ref_greedy.bin "$OUT/hw_ref" "$HW" "$OUT/win.bin" $NBS $STEPS >/dev/null
"$OUT/csim" "$OUT/pack" "$OUT/tok.bin" $STEPS $STEPS 1 "$OUT/k_greedy.bin" "$OUT/k_bus.bin"
"$OUT/csim" "$OUT/pack" "$OUT/tok.bin" $STEPS $PROMPT 0 "$OUT/k_gen.bin"
python3 - "$OUT" $NBS $STEPS $PROMPT <<'EOF'
import sys, numpy as np
out, nb, steps, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rb, kb = np.fromfile(out + "/ref_bus.bin", np.int32), np.fromfile(out + "/k_bus.bin", np.int32)
rg, kg = np.fromfile(out + "/ref_greedy.bin", np.int32), np.fromfile(out + "/k_greedy.bin", np.int32)
print(f"teacher forcing, {nb} streams x {steps} steps: bus identical: {np.array_equal(rb, kb)} "
      f"({int((rb != kb).sum())} of {rb.size} values differ); greedy predictions identical: {np.array_equal(rg, kg)}")
gen = np.fromfile(out + "/k_gen.bin", np.int32).reshape(nb, steps)
tok = np.fromfile(out + "/tok.bin", np.int32).reshape(nb, steps)
seq = np.concatenate([tok[:, :prompt], gen[:, prompt - 1:]], 1)            # prompt, then what the kernel generated
np.ascontiguousarray(seq[:, :steps + 1]).astype(np.int32).tofile(out + "/win2.bin")
np.save(out + "/gen.npy", gen)
EOF
HWREF_GREEDY=$OUT/ref_greedy2.bin "$OUT/hw_ref" "$HW" "$OUT/win2.bin" $NBS $STEPS >/dev/null
python3 - "$OUT" $NBS $STEPS $PROMPT <<'EOF'
import sys, numpy as np
out, nb, steps, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
gen = np.load(out + "/gen.npy")
ref = np.fromfile(out + "/ref_greedy2.bin", np.int32).reshape(nb, steps)
ok = np.array_equal(ref[:, prompt - 1:], gen[:, prompt - 1:])
print(f"generation from {prompt} prompt tokens: hw_ref predicts the kernel's {steps - prompt + 1} generated tokens "
      f"exactly: {ok}")
EOF
rm -rf "$OUT"

"""Pack a network exported by hw_export.py (runs/X/hw/manifest.txt + tables.bin) for the FPGA kernel (fpga/lif_lm.cpp):
the weight matrices in the order and layout the kernel reads them from HBM, the small per-neuron tables, and a C++
header with the network's sizes and offsets, which the kernel is compiled with. From the repository root:
  python3 fpga/pack.py runs/X/hw [--out runs/X/fpga]      (writes weights.bin, params.bin, aux.bin, model_cfg.h)

weights.bin, 512-bit words (64 bytes):
  - every matrix (O x I) in the order of one token's computation: per layer the plastic synapses' kq, v and W_o, then
    the population's W_in, leak_in, leak_out and W_out; then the readout's short-list and class rows (h_cls), then the
    word rows (h_word) class by class, so that one class's rows are one contiguous range (CLASS_WORDS words each).
    A matrix is cut into blocks of RB = 16 rows (the last one padded with zero rows) and its inputs into groups of 128
    (padded with zero columns when 128 does not divide I: the integer sum over a group is unchanged, so this is the
    same as hw_ref's one group per row). For each block, group by group: one word with the 16 rows' float32 comb
    values, then for each of the 16 rows its 128 weights: one word of 4-bit two's complement values (weight c in bits
    4c..4c+3) when every value of the matrix fits in 4 bits, else two words of 8-bit values (64 each).
  - the input rows (emb_w): per row d / 64 words of 8-bit values and one word with its d / 128 float32 scales (emb_sc,
    already times 2^BF): EMB_WORDS words per row.
params.bin: 32-bit values (int32 or float32 bit patterns), the per-neuron and per-channel tables, at the offsets in
  model_cfg.h (P_* constants, in 32-bit units).
aux.bin: int32: code (vocab x rows per word: the input rows each word fires), short_word (the word of each short-list
  neuron), class_word (the word of each class slot, -1 for an unused one); offsets in model_cfg.h (A_*).
"""
import argparse, os
import numpy as np

RB = 16


def read(hwdir):
    tens, par = {}, {}
    blob = open(os.path.join(hwdir, "tables.bin"), "rb").read()
    for line in open(os.path.join(hwdir, "manifest.txt")):
        f = line.split()
        if f[0] == "param":
            par[f[1]] = f[2]
            continue
        name, dt, nd = f[1], f[2], int(f[3])
        shape = tuple(int(x) for x in f[4:4 + nd])
        off = int(f[4 + nd])
        n = int(np.prod(shape))
        tens[name] = np.frombuffer(blob, dtype=np.dtype(dt), count=n, offset=off).reshape(shape)
    return tens, par


def fits4(w):
    return np.abs(w.astype(np.int32)).max() <= 7


def pack_matrix(w, comb, bits=0):
    """w: int8 (O, I); comb: float32 (O, groups). Returns (uint8 array of whole words, bits); bits 0: 4 when every value
    fits in 4 bits, else 8."""
    O, I = w.shape
    ng = -(-I // 128)
    assert comb.shape == (O, ng), (comb.shape, O, ng)
    bits = bits or (4 if fits4(w) else 8)
    nb = -(-O // RB)
    wp = np.zeros((nb * RB, ng * 128), np.int8)
    wp[:O, :I] = w
    cp = np.zeros((nb * RB, ng), np.float32)
    cp[:O] = comb
    out = []
    for blk in range(nb):
        rows = slice(blk * RB, (blk + 1) * RB)
        for g in range(ng):
            out.append(cp[rows, g].astype(np.float32).view(np.uint8))                    # one word: 16 floats
            wg = wp[rows, g * 128:(g + 1) * 128]                                           # (16, 128)
            if bits == 4:
                u = (wg.astype(np.int16) & 0xF).astype(np.uint8)
                out.append((u[:, 0::2] | (u[:, 1::2] << 4)).reshape(-1))                  # 16 words of 64 bytes
            else:
                out.append(wg.view(np.uint8).reshape(-1))                                   # 32 words
    return np.concatenate(out), bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hw", help="hw_export.py's output directory")
    p.add_argument("--out", default="")
    a = p.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.hw.rstrip("/"))), "fpga")
    os.makedirs(out, exist_ok=True)
    T, P = read(a.hw)
    d, H, L, R = int(P["d"]), int(P["H"]), int(P["L"]), int(P["leak_rank"])
    nh, mk, dv = int(P["mem_heads"]), int(P["mem_keys"]), int(P["mem_dv"])
    C, S, V = int(P["out_classes"]), int(P["out_short"]), int(P["vocab"])
    M = int(P.get("class_m", 0))
    mem = [int(x) for x in P["mem_layers"].split(",")] if P["mem_layers"] != "none" else []
    assert R > 0 and C > 0 and int(P["hard_reset"]) == 1 and int(P["G"]) == 128, "the kernel's network type"
    assert mem == list(range(L)), "plastic synapses in front of every population"

    words, cfg, prm, poff = [], [], [], {}
    nwords = [0]

    def add_matrix(tag, wname, cname):
        b, bits = pack_matrix(T[wname], T[cname])
        cfg.append(f"#define W_{tag} {nwords[0]}L\n#define BITS_{tag} {bits}")
        words.append(b)
        nwords[0] += len(b) // 64
        return bits

    def add_param(tag, arr, dtype):
        arr = np.ascontiguousarray(np.asarray(arr).astype(dtype)).reshape(-1)
        poff[tag] = sum(len(x) for x in prm)
        prm.append(arr.view(np.uint32))

    tags = []
    for l in range(L):
        for t, w, c in ((f"M{l}_KQ", f"m{l}_kq_w", f"m{l}_kq_comb"), (f"M{l}_V", f"m{l}_v_w", f"m{l}_v_comb"),
                        (f"M{l}_O", f"m{l}_o_w", f"m{l}_o_comb"), (f"P{l}_IN", f"p{l}_in_w", f"p{l}_in_comb"),
                        (f"P{l}_LI", f"p{l}_li_w", f"p{l}_li_comb"), (f"P{l}_LO", f"p{l}_lo_w", f"p{l}_lo_comb"),
                        (f"P{l}_OUT", f"p{l}_out_w", f"p{l}_out_comb")):
            add_matrix(t, w, c)
            tags.append(t)
        add_param(f"M{l}_AIN", T[f"m{l}_ain"], np.float32)
        add_param(f"M{l}_KQB", T[f"m{l}_kq_b"], np.float32)
        add_param(f"M{l}_KQBETA", T[f"m{l}_kq_beta"], np.int32)
        add_param(f"M{l}_LAM", T[f"m{l}_lam"], np.int32)
        add_param(f"M{l}_VAIN", T[f"m{l}_v_ain"], np.float32)
        add_param(f"M{l}_VTOS", T[f"m{l}_v_toS"], np.float32)
        add_param(f"M{l}_GAIN", T[f"m{l}_gain"], np.float32)
        add_param(f"M{l}_RAIN", T[f"m{l}_r_ain"], np.float32)
        add_param(f"P{l}_AIN", T[f"p{l}_ain"], np.float32)
        add_param(f"P{l}_INB", T[f"p{l}_in_b"], np.float32)
        add_param(f"P{l}_MAIN", T[f"p{l}_m_ain"], np.float32)
        add_param(f"P{l}_BLOGIT", T[f"p{l}_blogit"], np.float32)
        add_param(f"P{l}_G1", T[f"p{l}_g1"], np.int32)
    add_matrix("H_CLS", "h_cls_w", "h_cls_comb")
    # word rows: class by class, each class padded to whole blocks
    hw, hc = T["h_word_w"], T["h_word_comb"]
    cw0 = nwords[0]
    per = None
    wbits = 4 if fits4(hw) else 8
    for c in range(C):
        b, bits = pack_matrix(hw[c * M:(c + 1) * M], hc[c * M:(c + 1) * M], wbits)
        per = len(b) // 64 if per is None else per
        assert len(b) // 64 == per
        words.append(b)
        nwords[0] += per
    cfg.append(f"#define W_H_WORD {cw0}L\n#define BITS_H_WORD {bits}\n#define CLASS_WORDS {per}")
    # input rows: d / 64 words of int8 + 1 word of scales
    emb, sc = T["emb_w"], T["emb_sc"]
    nrow, ew = emb.shape[0], d // 64
    assert d % 128 == 0 and sc.shape[1] == d // 128 <= 16
    e = np.zeros((nrow, (ew + 1) * 64), np.uint8)
    e[:, :d] = emb.view(np.uint8)
    e[:, ew * 64:ew * 64 + 4 * sc.shape[1]] = sc.astype(np.float32).view(np.uint8)
    cfg.append(f"#define W_EMB {nwords[0]}L\n#define EMB_WORDS {ew + 1}")
    words.append(e.reshape(-1))
    nwords[0] += (ew + 1) * nrow
    add_param("H_AIN", T["h_ain"], np.float32)
    add_param("SIG", T["sig"], np.int32)

    # aux: code, short_word, class_word
    word_cls, slot = T["word_cls"], T["slot"]
    short_word = np.full(S, -1, np.int32)
    class_word = np.full(C * M, -1, np.int32)
    for w_ in range(V):
        if word_cls[w_] < 0:
            short_word[slot[w_]] = w_
        else:
            class_word[slot[w_] - S] = w_
    code = T["code"].astype(np.int32)
    aux = [code.reshape(-1), short_word, class_word]
    a_off = np.cumsum([0] + [len(x) for x in aux])

    np.concatenate(words).tofile(os.path.join(out, "weights.bin"))
    np.concatenate(prm).tofile(os.path.join(out, "params.bin"))
    np.concatenate(aux).astype(np.int32).tofile(os.path.join(out, "aux.bin"))
    vint = [int(P[f"p{l}_vint"]) for l in range(L)]
    kint = [int(P[f"m{l}_kq_vint"]) for l in range(L)]
    sint = [int(P[f"m{l}_S_int"]) for l in range(L)]
    h = ["// generated by fpga/pack.py from " + os.path.abspath(a.hw) + "; do not edit",
         "#pragma once",
         f"#define D {d}\n#define H {H}\n#define NL {L}\n#define R {R}\n#define NH {nh}\n#define MK {mk}\n#define DV {dv}",
         f"#define VOCAB {V}\n#define NSHORT {S}\n#define NCLS {C}\n#define CM {M}\n#define NCODE {code.shape[1]}",
         f"#define BF {int(P['BF'])}\n#define VB {int(P['V_BITS'])}\n#define SB {int(P['S_BITS'])}",
         f"#define SIGSTEP {int(P['SIG_STEP'])}\n#define SIGN ({int(P['SIG_RANGE'])} << SIGSTEP)",
         f"#define MEM_EPS {float(P['mem_eps'])!r}f",
         f"#define RB {RB}\n#define NWORDS {nwords[0]}L\n#define NPARAMS {sum(len(x) for x in prm)}L",
         f"#define A_CODE {a_off[0]}L\n#define A_SHORT {a_off[1]}L\n#define A_CLASS {a_off[2]}L\n#define NAUX {a_off[3]}L",
         "static const int VINT[NL] = {" + ", ".join(map(str, vint)) + "};",
         "static const int KQINT[NL] = {" + ", ".join(map(str, kint)) + "};",
         "static const int SINT[NL] = {" + ", ".join(map(str, sint)) + "};"]
    h += cfg
    for k, v in poff.items():
        h.append(f"#define P_{k} {v}L")
    # per-layer offset tables, so the kernel can index by layer
    for t in ("KQ", "V", "O"):
        h.append(f"static const long W_M_{t}[NL] = {{" + ", ".join(f"W_M{l}_{t}" for l in range(L)) + "};")
        h.append(f"static const int BITS_M_{t}[NL] = {{" + ", ".join(f"BITS_M{l}_{t}" for l in range(L)) + "};")
    for t in ("IN", "LI", "LO", "OUT"):
        h.append(f"static const long W_P_{t}[NL] = {{" + ", ".join(f"W_P{l}_{t}" for l in range(L)) + "};")
        h.append(f"static const int BITS_P_{t}[NL] = {{" + ", ".join(f"BITS_P{l}_{t}" for l in range(L)) + "};")
    for t in ("AIN", "KQB", "KQBETA", "LAM", "VAIN", "VTOS", "GAIN", "RAIN"):
        h.append(f"static const long P_M_{t}[NL] = {{" + ", ".join(f"P_M{l}_{t}" for l in range(L)) + "};")
    for t in ("AIN", "INB", "MAIN", "BLOGIT", "G1"):
        h.append(f"static const long P_P_{t}[NL] = {{" + ", ".join(f"P_P{l}_{t}" for l in range(L)) + "};")
    open(os.path.join(out, "model_cfg.h"), "w").write("\n".join(h) + "\n")
    print(f"wrote {out}: weights.bin {nwords[0] * 64 / 1e6:.1f} MB ({nwords[0]} words), params.bin "
          f"{sum(len(x) for x in prm) * 4 / 1e6:.2f} MB, aux.bin {a_off[3] * 4 / 1e6:.2f} MB, model_cfg.h")


if __name__ == "__main__":
    main()

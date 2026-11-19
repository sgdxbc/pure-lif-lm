"""Export a trained spiking network as the integer tables of the FPGA prototype (docs/experiments.md 5.44), for the
bit-exact reference hw_ref.cpp (and later the FPGA kernel, which must give the same numbers). From the repository root:
  python3 hw_export.py runs/X [--ema] [--out runs/X/hw]       (writes OUT/manifest.txt and OUT/tables.bin)

What the hardware computes for one token of one stream (hw_ref.cpp is the exact definition; every rounding below is
"floor(x + 0.5)" in float32, or "(x + 2^(k-1)) >> k" for integers):
  bus      the current bus, int32 fixed point with BF fractional bits (BF from the calibrated range of the bus).
           Start: the sum of the word's input rows (in_groups: two rows, or one own row for a short-list word).
  matrices every weight matrix as int8 values (4-bit or 8-bit range, as trained) with one float32 scale per output row
           and group of 128 inputs (per row when 128 does not divide the inputs). A matrix-vector product: the input
           is rounded to A_BITS-bit integers (8, or wider) with one scale per group of 128 channels (A8g128; calibrated,
           static), each group's
           integer sum (int32, exact) is multiplied by one float32 "comb" = weight scale x activation scale (x any
           constant folded in: 1/threshold, 2^BF, (1 - lambda)), and the groups are summed in order in float32.
  neurons  potentials in V-bit fixed point (V_BITS, integer bits from the calibration, threshold folded so they fire at
           1): x = round(beta * v) + I, fire if x >= 1, then v = x * (1 - gamma) on a spike (hard reset) or
           v = x - 1 (soft reset, the plastic synapses' key and query neurons). beta and (1 - gamma) are 16-bit
           fractions; with the leak synapses beta = sigmoid(beta_logit + U m) comes from a table over the argument in
           steps of 2^-10 (no transcendental function is computed at run time).
  plastic  S (per group: mem_keys x mem_dv) in S_BITS-bit fixed point: every step S = round(lambda * S), then each
           firing key neuron's row adds the value (rounded to int8 per group of 128 channels, as A8g128), each firing
           query neuron's row is added to the read-out r; r is normalised per group in float32
           (r / sqrt(mean(r^2) + eps)), times the gain, rounded to int8 and sent into the bus through W_o.
  readout  two-level (out_classes, out_short) or plain: float32 logits of the short-list and class neurons, and of the
           word neurons of one class.
Calibration (the ranges): the unrounded step-by-step pass of quant_sim.py on 16 validation windows (shifted by half a
window from the ones evaluated), as quant_sim.py does.
"""
import argparse, json, math, os, re
import numpy as np
import torch
from sample import load
import quant_sim as Q

G = 128          # group of inputs per scale (weights and activations)
BETA_BITS = 16   # fractional bits of beta, 1 - gamma, lambda
SIG_STEP = 10    # sigmoid table: argument in steps of 2^-SIG_STEP
SIG_RANGE = 16   # over [-16, 16]


def int_weights(w, bits):
    """(int8 values, float32 scales (rows, groups), group size): model.quantize_weight's integers and scales."""
    w = w.detach().float().cpu()
    qmax = 2 ** (bits - 1) - 1
    o, i = w.shape
    g = G if i % G == 0 else i
    wg = w.reshape(o, i // g, g)
    s = wg.abs().amax(-1).clamp(min=1e-12) / qmax                       # (o, i // g)
    q = torch.round(wg / s.unsqueeze(-1)).clamp(-qmax, qmax).reshape(o, i)
    return q.to(torch.int8).numpy(), s.numpy().astype(np.float64), g


A_BITS = 8      # activations (matrix inputs): signed integers of this many bits (export(a_bits=...))


def act_scale(amax, n):
    """Per-group (G channels) activation scales of n channels from per-channel maxima (one group if G does not divide)."""
    amax = amax.detach().float().cpu().numpy().astype(np.float64)
    g = G if n % G == 0 else n
    return amax.reshape(n // g, g).max(-1).clip(min=1e-12) / (2 ** (A_BITS - 1) - 1), g


class Writer:
    def __init__(self, out):
        self.out = out
        self.blob = open(os.path.join(out, "tables.bin"), "wb")
        self.lines, self.off = [], 0

    def t(self, name, a, dtype):
        a = np.ascontiguousarray(np.asarray(a).astype(dtype))
        self.lines.append(f"tensor {name} {np.dtype(dtype).name} {a.ndim} {' '.join(map(str, a.shape))} {self.off}")
        self.blob.write(a.tobytes())
        self.off += a.nbytes

    def p(self, name, v):
        self.lines.append(f"param {name} {v}")

    def close(self):
        self.blob.close()
        open(os.path.join(self.out, "manifest.txt"), "w").write("\n".join(self.lines) + "\n")


def b16(x):
    return np.round(np.asarray(x, dtype=np.float64) * 2 ** BETA_BITS).astype(np.int64)


@torch.no_grad()
def export(model, cal, out, v_bits=24, s_bits=20, a_bits=12):
    global A_BITS
    A_BITS = a_bits
    c = model.cfg
    assert c.steps_per_token == 1, "one step per token"
    assert not getattr(c, "in_rank", 0) and getattr(c, "mem_src", "bus") == "bus", "in_rank / mem_src: not exported"
    assert not getattr(c, "spike_soft", 0.0), "spike_soft > 0: not a spiking network"
    assert not getattr(c, "bus_norm", False), "bus_norm: not implemented here"
    assert getattr(c, "mem_kind", "plastic") == "plastic", "mem_kind gla: not a spiking network"
    os.makedirs(out, exist_ok=True)
    W = Writer(out)
    pats = [re.compile(x) for x in c.wbits8.split(",") if x]
    bits_of = lambda name: 8 if (not c.wbits or any(p.search(name) for p in pats)) else c.wbits
    d, H, L = c.d_model, c.d_hidden, c.n_layers
    # the bus: int32 with BF fractional bits, 25% headroom over the largest calibrated value anywhere
    bmax = max([float(x.max()) for x in cal["bus"]] + [float(cal["head"].max())] +
               [float(x.max()) for x in (cal.get("mem_bus") or [])])
    bus_int = max(0, math.ceil(math.log2(bmax * 1.25)))
    BF = 30 - bus_int
    for k, v in dict(d=d, H=H, L=L, vocab=c.vocab, BF=BF, V_BITS=v_bits, S_BITS=s_bits, A_BITS=a_bits, BETA_BITS=BETA_BITS,
                     SIG_STEP=SIG_STEP, SIG_RANGE=SIG_RANGE, G=G, hard_reset=int(c.hard_reset),
                     leak_rank=c.leak_rank, mem_heads=c.mem_heads, mem_keys=c.mem_keys, mem_dv=c.mem_dv,
                     mem_eps=c.mem_eps, out_classes=c.out_classes, out_short=c.out_short,
                     in_groups=c.in_groups).items():
        W.p(k, v)
    W.p("mem_layers", ",".join(map(str, getattr(model, "mem_layers", []))) or "none")
    two = float(2 ** BF)

    # input rows and the rows each word fires
    name = "embed.weight"
    q, s, g = int_weights(model.embed.weight, bits_of(name))
    assert g == G
    W.t("emb_w", q, np.int8)
    W.t("emb_sc", s * two, np.float32)
    if c.in_groups > 1:
        rows = model.embed.rows(torch.arange(c.vocab, device=model.embed.code.device)).cpu().numpy()
    else:
        rows = np.arange(c.vocab).reshape(-1, 1)
    W.t("code", rows, np.int32)

    def matrix(tag, mod_w, name, comb_extra, a_scale, a_group):
        """int8 weights and comb (rows, groups) = weight scale x activation scale x comb_extra (per row)."""
        q, s, g = int_weights(mod_w, bits_of(name))
        assert g == a_group, (name, g, a_group)
        comb = s * np.asarray(a_scale, dtype=np.float64)[None, :] * np.asarray(comb_extra, dtype=np.float64).reshape(-1, 1)
        W.t(tag + "_w", q, np.int8)
        W.t(tag + "_comb", comb, np.float32)

    def ainv(tag, sc):
        W.t(tag, 1.0 / sc, np.float32)

    def vfrac(intb, bits):
        return bits - 1 - intb

    # sigmoid table: beta = round(sigmoid(k * 2^-SIG_STEP) * 2^16), k = -N..N
    N = SIG_RANGE * 2 ** SIG_STEP
    k = np.arange(-N, N + 1, dtype=np.float64) / 2 ** SIG_STEP
    W.t("sig", b16(1 / (1 + np.exp(-k))), np.int32)

    mem_layers = getattr(model, "mem_layers", [])
    for li, layer in enumerate(model.layers):
        if li in mem_layers:
            j = mem_layers.index(li)
            mem = model.mems[str(li)]
            h, dv = c.mem_heads, c.mem_dv
            pre = f"m{li}"
            sa, ga = act_scale(cal["mem_bus"][j], d)
            ainv(pre + "_ain", sa * two)                                   # bus (fixed point) -> int8
            th = mem.kq_log_theta.exp().double().cpu().numpy()
            matrix(pre + "_kq", mem.kq_in.weight, f"mems.{li}.kq_in.weight", 1 / th, sa, ga)
            W.t(pre + "_kq_b", mem.kq_in.bias.double().cpu().numpy() / th, np.float32)
            W.t(pre + "_kq_beta", b16(torch.sigmoid(mem.kq_beta_logit.double()).cpu().numpy()), np.int32)
            ki = cal["kq_v_int"][j]
            W.p(pre + "_kq_vint", ki)
            lam = torch.sigmoid(mem.decay_beta_logit.double()).cpu().numpy()            # (h,)
            W.t(pre + "_lam", b16(lam), np.int32)
            wa = np.repeat(1 - lam, dv) if c.mem_write_avg else np.ones(h * dv)          # folded into the scale
            matrix(pre + "_v", mem.v_in.weight, f"mems.{li}.v_in.weight", wa, sa, ga)
            sv, gv = act_scale(cal["mem_v"][j], h * dv)
            ainv(pre + "_v_ain", sv)                                       # value (float) -> int8
            si = cal["mem_S_int"][j]
            SF = vfrac(si, s_bits)
            W.p(pre + "_S_int", si)
            W.t(pre + "_v_toS", sv * 2.0 ** SF, np.float32)                # int8 value -> S units
            W.t(pre + "_gain", mem.gain.double().cpu().numpy(), np.float32)
            sr, gr = act_scale(cal["mem_r"][j], h * dv)
            ainv(pre + "_r_ain", sr)                                       # normalised read-out -> int8
            matrix(pre + "_o", mem.w_out.weight, f"mems.{li}.w_out.weight", np.full(d, two), sr, gr)
        pre = f"p{li}"
        sa, ga = act_scale(cal["bus"][li], d)
        ainv(pre + "_ain", sa * two)
        th = layer.theta().double().cpu().numpy()
        matrix(pre + "_in", layer.w_in.weight, f"layers.{li}.w_in.weight", 1 / th, sa, ga)
        W.t(pre + "_in_b", layer.w_in.bias.double().cpu().numpy() / th, np.float32)
        if c.leak_rank:
            matrix(pre + "_li", layer.leak_in.weight, f"layers.{li}.leak_in.weight", np.ones(c.leak_rank), sa, ga)
            sm, gm = act_scale(cal["m"][li], c.leak_rank)
            ainv(pre + "_m_ain", sm)
            matrix(pre + "_lo", layer.leak_out.weight, f"layers.{li}.leak_out.weight", np.ones(H), sm, gm)
            W.t(pre + "_blogit", layer.beta_logit.double().cpu().numpy(), np.float32)
        else:
            W.t(pre + "_beta", b16(layer.beta().double().cpu().numpy()), np.int32)
        W.t(pre + "_g1", b16(1 - layer.gamma().double().cpu().numpy()), np.int32)   # 1 - gamma
        W.p(pre + "_vint", cal["v_int"][li])
        q, s, g = int_weights(layer.w_out.weight, bits_of(f"layers.{li}.w_out.weight"))   # spikes: no act. scale
        W.t(pre + "_out_w", q, np.int8)
        W.t(pre + "_out_comb", s * two, np.float32)

    # readout
    sa, ga = act_scale(cal["head"], d)
    ainv("h_ain", sa * two)
    if c.out_classes > 0:
        hd = model.head
        matrix("h_cls", hd.w_cls, "head.w_cls", np.ones(hd.w_cls.size(0)), sa, ga)
        matrix("h_word", hd.w_word, "head.w_word", np.ones(hd.w_word.size(0)), sa, ga)
        W.t("word_cls", hd.word_cls.cpu().numpy(), np.int32)
        W.t("slot", hd.slot.cpu().numpy(), np.int32)
        W.t("used", hd.used.cpu().numpy(), np.uint8)
        W.p("class_m", hd.m)
    else:
        matrix("h", model.head.weight, "head.weight", np.ones(c.vocab), sa, ga)
    W.close()
    return BF


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    p.add_argument("--v_bits", type=int, default=24)
    # defaults from quant_sim on g29_hw (docs 5.44): 12-bit activations with group scales and 20-bit plastic
    # synapses lose nothing (-0.0005); 8 / 16 bits lost 0.025, 10 / 18 bits 0.005
    p.add_argument("--s_bits", type=int, default=20, help="plastic synapses S (at most 24)")
    p.add_argument("--a_bits", type=int, default=12, help="activations, the matrices' inputs (at most 15)")
    p.add_argument("--windows", type=int, default=0, help="also write OUT/windows.bin: quant_sim.py's N validation "
                   "windows of 256 tokens (int32, N x 257: each window and its last target), for hw_ref")
    a = p.parse_args()
    args = json.load(open(os.path.join(a.run, "args.json")))
    model, _ = load(os.path.join(a.run, "ckpt.pt"), ema=bool(a.ema and args.get("ema", 0)), device=torch.device(a.device))
    rec = {}
    xc, _ = Q.windows(16, 256, 128, a.device, args.get("data", "data"))
    Q.run(model, xc, {}, rec=rec)
    out = a.out or os.path.join(a.run, "hw")
    BF = export(model, Q.calibrate(rec), out, a.v_bits, a.s_bits, a.a_bits)
    if a.windows:
        x, y = Q.windows(a.windows, 256, 0, "cpu", args.get("data", "data"))
        torch.cat([x, y[:, -1:]], 1).numpy().astype(np.int32).tofile(os.path.join(out, "windows.bin"))
    print(f"wrote {out}/manifest.txt, tables.bin ({os.path.getsize(os.path.join(out, 'tables.bin')) / 1e6:.1f} MB); bus "
          f"fractional bits {BF}")


if __name__ == "__main__":
    main()

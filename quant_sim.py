"""Low-precision simulation (README 5.20): how much validation loss a trained spiking network loses when its weights,
the current bus and the membrane potentials are kept in the low-precision integers that an FPGA or a dataflow
accelerator would use. Nothing is retrained: the trained weights are rounded, and the network is run step by step
with every quantity rounded the way the hardware would. Every setting is measured on the same validation windows, so
the differences are paired. From the repository root:
  python3 quant_sim.py --device cuda [--ema] runs/X        (writes runs/X/quant.json)

  W8 / W4 / W4g128   weights as symmetric integers with one scale per output row, or per group of 128 inputs
                     (a matrix whose input width is not a multiple of 128 keeps one scale per row)
  A8                 8-bit integers, one scale per channel, for the bus (the input of W_in, of the leak synapses and of
                     the readout) and for the rank-r intermediate of the leak synapses. Scales are calibrated on
                     separate windows; a per-channel scale folds into the next weights, so it costs nothing in hardware.
  V16 / V24          membrane potentials in fixed point after dividing by each neuron's threshold (the threshold folds
                     into W_in and the bias, so the neuron fires at V >= 1), with 16 or 24 bits; the integer bits per
                     layer come from the calibration, the rest are fractional. Rounded after the update and the reset.
  V16p               as V16, but the integer bits cover the 99.99th percentile of |V| in the calibration instead of
                     its maximum (a few neurons drift thousands of thresholds away); values beyond saturate.
  B12 / B16          the leak factor beta (every step, after the sigmoid: a lookup table in hardware) and the reset
                     fraction gamma, with 12 or 16 fractional bits.
"fp32" runs the same step-by-step code without rounding; "model" is the network's own forward pass (bfloat16, fused
kernel) on the same windows, as a check that the step-by-step code is the same network.
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from sample import load

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    ("fp32", {}),
    ("W8", dict(w=8)), ("W4", dict(w=4)), ("W4g128", dict(w=4, group=128)),
    ("A8", dict(a=8)),
    ("V24", dict(v=24)), ("V16", dict(v=16)),
    ("B16", dict(b=16)), ("B12", dict(b=12)),
    ("W8 A8 V24 B16", dict(w=8, a=8, v=24, b=16)),
    ("W8 A8 V16 B12", dict(w=8, a=8, v=16, b=12)),
    ("W4g128 A8 V24 B16", dict(w=4, group=128, a=8, v=24, b=16)),
    ("W4g128 A8 V16 B12", dict(w=4, group=128, a=8, v=16, b=12)),
    # V16p: integer bits from the 99.99th percentile of |V| instead of the maximum; rarer values saturate
    ("V16p", dict(v=16, vp=1)), ("W8 A8 V16p B12", dict(w=8, a=8, v=16, vp=1, b=12)),
]


def qw(w, bits, group=0):
    """Symmetric integer weights, one scale per output row or per group of `group` inputs."""
    if not bits:
        return w
    qmax = 2 ** (bits - 1) - 1
    o, i = w.shape
    if group and i % group == 0:
        wg = w.reshape(o, i // group, group)
        s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
        return (torch.round(wg / s).clamp(-qmax, qmax) * s).reshape(o, i)
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax
    return torch.round(w / s).clamp(-qmax, qmax) * s


def qa(x, amax, bits):
    """Symmetric integers with one scale per channel (last dimension); amax from the calibration."""
    qmax = 2 ** (bits - 1) - 1
    s = amax.clamp(min=1e-12) / qmax
    return torch.round(x / s).clamp(-qmax, qmax) * s


def qfix(x, frac, intb):
    """Signed fixed point: `intb` integer bits, `frac` fractional bits."""
    s = 2.0 ** -frac
    lim = 2.0 ** intb
    return (torch.round(x / s) * s).clamp(-lim, lim - s)


def qfrac(p, bits):
    return torch.round(p * 2 ** bits) / 2 ** bits


@torch.no_grad()
def run(model, x, q, cal=None, rec=None):
    """Step-by-step forward pass with the rounding in q. rec: dict that collects the calibration maxima."""
    c = model.cfg
    assert c.rec_rank == 0 and c.delay_k == 0 and not c.spike_bus, \
        "not implemented for same-layer or delayed synapses or the spiking bus"
    T, w, g_ = c.steps_per_token, q.get("w", 0), q.get("group", 0)
    A, Vb, Bb = q.get("a", 0), q.get("v", 0), q.get("b", 0)
    f32 = lambda t: t.float()
    bus = qw(f32(model.embed.weight), w, g_)[x]
    if T > 1:
        bus = bus.repeat_interleave(T, dim=1)
    B, LT, _ = bus.shape
    for li, layer in enumerate(model.layers):
        xin = qa(bus, cal["bus"][li], A) if A else bus
        th = layer.theta().float()
        I = F.linear(xin, qw(f32(layer.w_in.weight), w, g_), f32(layer.w_in.bias)) / th   # threshold folded in: fire at 1
        if c.leak_rank:
            m = F.linear(xin, qw(f32(layer.leak_in.weight), w, g_))
            if A:
                m = qa(m, cal["m"][li], A)
            beta = torch.sigmoid(f32(layer.beta_logit) + F.linear(m, qw(f32(layer.leak_out.weight), w, g_)))
        else:
            beta = layer.beta().float()
        gam = layer.gamma().float()
        if Bb:
            beta, gam = qfrac(beta, Bb), qfrac(gam, Bb)
        if c.leak_gate:   # current scaled by (1 - beta), from the same lookup table
            I = I * (1 - beta)
        if Vb:
            intb = cal["v_int_p" if q.get("vp") else "v_int"][li]
            frac = Vb - 1 - intb
        v = bus.new_zeros(B, I.size(-1))
        vmax, vsamp = 0.0, []
        out = []
        for t in range(LT):
            xt = (beta[:, t] if beta.dim() == 3 else beta) * v + I[:, t]
            if Vb:
                xt = qfix(xt, frac, intb)
            s = (xt >= 1).float()
            v = xt * (1 - gam * s) if c.hard_reset else xt - s
            if Vb:
                v = qfix(v, frac, intb)
            if rec is not None:
                vmax = max(vmax, xt.abs().max().item())
                if t % 8 == 0:
                    vsamp.append(xt.abs().flatten())
            out.append(s)
        S = torch.stack(out, 1)
        if rec is not None:
            rec.setdefault("bus", []).append(bus.abs().amax((0, 1)))
            if c.leak_rank:
                rec.setdefault("m", []).append(m.abs().amax((0, 1)))
            rec.setdefault("vmax", []).append(vmax)
            vs = torch.cat(vsamp)
            rec.setdefault("vp9999", []).append(vs.kthvalue(max(1, int(0.9999 * vs.numel()))).values.item())
            rec.setdefault("rate", []).append(S.mean().item())
        bus = bus + F.linear(S, qw(f32(layer.w_out.weight), w, g_))
    if T > 1:
        bus = bus.view(B, -1, T, bus.size(-1)).mean(2)
    if rec is not None:
        rec["head"] = bus.abs().amax((0, 1))
    hin = qa(bus, cal["head"], A) if A else bus
    return F.linear(hin, qw(f32(model.head.weight), w, g_))


def windows(n, seq, offset, dev, data="data"):
    arr = np.memmap(os.path.join(HERE, data, "valid.bin"), dtype=np.uint16, mode="r")
    starts = np.linspace(0, len(arr) - seq - 2, n + 1).astype(np.int64)[:-1] + offset
    x = torch.from_numpy(np.stack([arr[i:i + seq] for i in starts]).astype(np.int64)).to(dev)
    y = torch.from_numpy(np.stack([arr[i + 1:i + seq + 1] for i in starts]).astype(np.int64)).to(dev)
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory with ckpt.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true", help="moving-average weights where the checkpoint has them")
    p.add_argument("--windows", type=int, default=256, help="validation windows of 256 tokens")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--configs", default="", help="comma-separated subset of the setting names")
    a = p.parse_args()
    dev = torch.device(a.device)
    args = json.load(open(os.path.join(a.run, "args.json")))
    has_ema = bool(a.ema and args.get("ema", 0))
    data = args.get("data", "data")   # the run's own data
    model, _ = load(os.path.join(a.run, "ckpt.pt"), ema=has_ema, device=dev)
    seq = 256
    # calibration on separate windows (shifted by half a window), unrounded
    rec = {}
    xc, _ = windows(16, seq, seq // 2, dev, data)
    run(model, xc, {}, rec=rec)
    cal = {"bus": rec["bus"], "m": rec.get("m"), "head": rec["head"],
           "v_int": [max(0, math.ceil(math.log2(max(vm, 1.0) * 1.25))) for vm in rec["vmax"]],
           "v_int_p": [max(0, math.ceil(math.log2(max(vm, 1.0) * 1.25))) for vm in rec["vp9999"]]}
    x, y = windows(a.windows, seq, 0, dev, data)

    def loss_of(fn):
        tot = 0.0
        for i in range(0, len(x), a.batch):
            logits = fn(x[i:i + a.batch])
            tot += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y[i:i + a.batch].reshape(-1),
                                   reduction="sum").item()
        return tot / y.numel()

    def own(xb):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            return model(xb)[0]

    res = {"ema": has_ema, "windows": a.windows, "tokens": y.numel(), "v_int_bits": cal["v_int"],
           "v_int_bits_p9999": cal["v_int_p"], "calibration_vmax": rec["vmax"], "calibration_v_p9999": rec["vp9999"],
           "rate": rec["rate"], "loss": {}}
    res["loss"]["model"] = loss_of(own)
    print(f"{'model (bf16, fused kernel)':22s} {res['loss']['model']:.4f}", flush=True)
    keep = set(s.strip() for s in a.configs.split(",") if s.strip())
    for name, q in CONFIGS:
        if keep and name not in keep and name != "fp32":
            continue
        res["loss"][name] = loss_of(lambda xb: run(model, xb, q, cal))
        d = res["loss"][name] - res["loss"]["fp32"]
        print(f"{name:22s} {res['loss'][name]:.4f}  ({d:+.4f} vs fp32)", flush=True)
    json.dump(res, open(os.path.join(a.run, "quant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

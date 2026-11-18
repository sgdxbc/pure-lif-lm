"""Low-precision simulation (docs/experiments.md 5.20): how much validation loss a trained spiking network loses when its weights,
the current bus and the membrane potentials are kept in the low-precision integers that an FPGA or a dataflow
accelerator would use. Nothing is retrained: the trained weights are rounded, and the network is run step by step
with every quantity rounded the way the hardware would. Every setting is measured on the same validation windows, so
the differences are paired. From the repository root:
  python3 quant_sim.py --device cuda [--ema] runs/X        (writes runs/X/quant.json)

  W8 / W4 / W4g128   weights as symmetric integers with one scale per output row, or per group of 128 inputs
                     (a matrix whose input width is not a multiple of 128 keeps one scale per row)
  A8g128 / A8t       as A8 with one scale per group of 128 channels / per vector (for weights trained as integers,
                     where a per-channel scale cannot be folded in; the group's scale multiplies its integer sum)
  A8                 8-bit integers, one scale per channel, for the bus (the input of W_in, of the leak synapses and of
                     the readout) and for the rank-r intermediate of the leak synapses. Scales are calibrated on
                     separate windows; a per-channel scale folds into the next weights, so it costs nothing in hardware.
  V16 / V24          membrane potentials in fixed point after dividing by each neuron's threshold (the threshold folds
                     into W_in and the bias, so the neuron fires at V >= 1), with 16 or 24 bits; the integer bits per
                     layer come from the calibration, the rest are fractional. Rounded after the update and the reset.
  V16p               as V16, but the integer bits cover the 99.99th percentile of |V| in the calibration instead of
                     its maximum (a few neurons drift thousands of thresholds away); values beyond saturate.
  B12 / B16          the leak factor beta (every step, after the sigmoid: a lookup table in hardware) and the reset
                     fraction gamma, with 12 or 16 fractional bits (and the plastic synapses' decay lambda and their
                     key and query neurons' leak).
Networks with plastic synapses (mem_heads; docs/experiments.md 5.44): their key and query neurons' potentials as V,
their bus input, value channels and normalised read-out as A (8-bit integers per channel), and
  S12 / S16 / S24    the synapses S in fixed point; the integer bits from the calibrated maximum of |S| (fewer than 0
                     when it stays below 1/2, leaving more fractional bits).
  The read-out's normalisation r / sqrt(mean(r^2) + eps) is computed exactly (a lookup table or a few iterations in
  hardware) and rounded with A. Coded input (in_groups) and the two-level readout (out_classes) use their own forward
  pass with the rounded weights. A network trained with low-precision weights (wbits) keeps its trained integers:
  the W settings are left out.
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


def qa(x, amax, bits, group=0):
    """Symmetric integers with one scale per channel (last dimension); amax per channel from the calibration.
    group > 0: one scale per group of that many channels (the largest amax of the group; one for all channels when
    it does not divide their number); group < 0: one scale for all channels."""
    qmax = 2 ** (bits - 1) - 1
    n = amax.numel()
    if group < 0 or (group and n % group):
        amax = amax.max().expand(n)
    elif group:
        amax = amax.view(-1, group).amax(-1, keepdim=True).expand(-1, group).reshape(n)
    s = amax.clamp(min=1e-12) / qmax
    return torch.round(x / s).clamp(-qmax, qmax) * s


def qfix(x, frac, intb):
    """Signed fixed point: `intb` integer bits, `frac` fractional bits."""
    s = 2.0 ** -frac
    lim = 2.0 ** intb
    return (torch.round(x / s) * s).clamp(-lim, lim - s)


def qfrac(p, bits):
    return torch.round(p * 2 ** bits) / 2 ** bits


def fire(I, beta, gam, Vb, intb, hard, rec=None, key=None):
    """LIF neurons over the sequence, threshold folded in (fire at 1). I: (B, L, D) current already divided by the
    threshold; beta: (D) or (B, L, D); gam: (D) (hard reset) or None (soft reset, V -= 1 on a spike). Vb > 0: the
    potentials in Vb-bit fixed point with intb integer bits, rounded after the update and after the reset."""
    frac = Vb - 1 - intb if Vb else 0
    v = I.new_zeros(I.size(0), I.size(-1))
    vmax, vsamp, out = 0.0, [], []
    for t in range(I.size(1)):
        xt = (beta[:, t] if beta.dim() == 3 else beta) * v + I[:, t]
        if Vb:
            xt = qfix(xt, frac, intb)
        st = (xt >= 1).float()
        v = xt * (1 - gam * st) if hard else xt - st
        if Vb:
            v = qfix(v, frac, intb)
        if rec is not None:
            vmax = max(vmax, xt.abs().max().item())
            if t % 8 == 0:
                vsamp.append(xt.abs().flatten())
        out.append(st)
    if rec is not None:
        vs = torch.cat(vsamp)
        rec.setdefault(key + "vmax", []).append(vmax)
        rec.setdefault(key + "vp9999", []).append(vs.kthvalue(max(1, int(0.9999 * vs.numel()))).values.item())
    return torch.stack(out, 1)


def vint(cal, key, i, q):
    return cal[key + ("v_int_p" if q.get("vp") else "v_int")][i] if q.get("v") else 0


@torch.no_grad()
def plastic(mem, bus, q, cal, rec, j, A, Bb):
    """The plastic synapses in front of a population (model.PlasticSynapses), step by step: key and query neurons
    (soft reset), S[t] = lam S[t-1] + k[t] v[t]^T, r[t] = S[t]^T q[t], normalised per group, times the gain, into the
    bus through W_o. With "s" bits, S in fixed point (integer bits from the calibrated maximum of |S|)."""
    c = mem.cfg
    h, m, dv = c.mem_heads, c.mem_keys, c.mem_dv
    w, g_, Sb, ag = q.get("w", 0), q.get("group", 0), q.get("s", 0), q.get("ag", 0)
    f32 = lambda t: t.float()
    B, L, _ = bus.shape
    xin = qa(bus, cal["mem_bus"][j], A, ag) if A else bus
    th = mem.kq_log_theta.exp().float()
    I = F.linear(xin, qw(f32(mem.kq_in.weight), w, g_), f32(mem.kq_in.bias)) / th
    kb = torch.sigmoid(f32(mem.kq_beta_logit))
    lam = torch.sigmoid(f32(mem.decay_beta_logit))                                       # (h,)
    if Bb:
        kb, lam = qfrac(kb, Bb), qfrac(lam, Bb)
    s = fire(I, kb, None, q.get("v", 0), vint(cal, "kq_", j, q), False, rec, "kq_")
    k = s[..., :h * m].reshape(B, L, h, m)
    qq = s[..., h * m:].reshape(B, L, h, m)
    Wv = f32(mem.v_in.weight)
    if c.mem_write_avg:   # (1 - lambda) per group, folded into W_v
        Wv = Wv * (1 - torch.sigmoid(f32(mem.decay_beta_logit))).repeat_interleave(dv).unsqueeze(1)
    val = F.linear(xin, qw(Wv, w, g_)).view(B, L, h, dv)
    if A:
        val = qa(val.reshape(B, L, h * dv), cal["mem_v"][j], A, ag).view(B, L, h, dv)
    if Sb:
        intb = cal["mem_S_int"][j]
        frac = Sb - 1 - intb
    S = bus.new_zeros(B, h, m, dv)
    smax, r = 0.0, []
    for t in range(L):
        S = lam.view(1, h, 1, 1) * S + k[:, t].unsqueeze(-1) * val[:, t].unsqueeze(-2)
        if Sb:
            S = qfix(S, frac, intb)
        if rec is not None:
            smax = max(smax, S.abs().max().item())
        r.append((qq[:, t].unsqueeze(-1) * S).sum(-2))                                   # (B, h, dv)
    r = torch.stack(r, 1)                                                                # (B, L, h, dv)
    r = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + c.mem_eps)).reshape(B, L, h * dv) * f32(mem.gain)
    if rec is not None:
        rec.setdefault("mem_bus", []).append(bus.abs().amax((0, 1)))
        rec.setdefault("mem_v", []).append(val.reshape(B, L, h * dv).abs().amax((0, 1)))
        rec.setdefault("mem_r", []).append(r.abs().amax((0, 1)))
        rec.setdefault("mem_Smax", []).append(smax)
    if A:
        r = qa(r, cal["mem_r"][j], A, ag)
    return F.linear(r, qw(f32(mem.w_out.weight), w, g_))


@torch.no_grad()
def run(model, x, q, cal=None, rec=None):
    """Step-by-step forward pass with the rounding in q. rec: dict that collects the calibration maxima."""
    from torch.func import functional_call
    c = model.cfg
    T, w, g_ = c.steps_per_token, q.get("w", 0), q.get("group", 0)
    A, Vb, Bb, ag = q.get("a", 0), q.get("v", 0), q.get("b", 0), q.get("ag", 0)
    f32 = lambda t: t.float()
    if c.in_groups > 1:   # coded input: the sum of the rows of the input neurons that fire
        bus = functional_call(model.embed, {"weight": qw(f32(model.embed.weight), w, g_)}, (x,))
    else:
        bus = qw(f32(model.embed.weight), w, g_)[x]
    if T > 1:
        bus = bus.repeat_interleave(T, dim=1)
    B, LT, _ = bus.shape
    mem_layers = getattr(model, "mem_layers", [])
    for li, layer in enumerate(model.layers):
        if li in mem_layers:
            j = mem_layers.index(li)
            bus = bus + plastic(model.mems[str(li)], bus, q, cal, rec, j, A, Bb)
        xin = qa(bus, cal["bus"][li], A, ag) if A else bus
        th = layer.theta().float()
        I = F.linear(xin, qw(f32(layer.w_in.weight), w, g_), f32(layer.w_in.bias)) / th   # threshold folded in: fire at 1
        if c.leak_rank:
            m = F.linear(xin, qw(f32(layer.leak_in.weight), w, g_))
            if A:
                m = qa(m, cal["m"][li], A, ag)
            beta = torch.sigmoid(f32(layer.beta_logit) + F.linear(m, qw(f32(layer.leak_out.weight), w, g_)))
        else:
            beta = layer.beta().float()
        gam = layer.gamma().float()
        if Bb:
            beta, gam = qfrac(beta, Bb), qfrac(gam, Bb)
        S = fire(I, beta, gam, Vb, vint(cal, "", li, q), c.hard_reset, rec, "")
        if rec is not None:
            rec.setdefault("bus", []).append(bus.abs().amax((0, 1)))
            if c.leak_rank:
                rec.setdefault("m", []).append(m.abs().amax((0, 1)))
            rec.setdefault("rate", []).append(S.mean().item())
        bus = bus + F.linear(S, qw(f32(layer.w_out.weight), w, g_))
    if T > 1:
        bus = bus.view(B, -1, T, bus.size(-1)).mean(2)
    if rec is not None:
        rec["head"] = bus.abs().amax((0, 1))
    hin = qa(bus, cal["head"], A, ag) if A else bus
    if c.out_classes > 0:   # two-level readout: log P(w) for every word
        return functional_call(model.head, {"w_cls": qw(f32(model.head.w_cls), w, g_),
                                            "w_word": qw(f32(model.head.w_word), w, g_)}, (hin,))
    return F.linear(hin, qw(f32(model.head.weight), w, g_))


def calibrate(rec):
    """Ranges from an unrounded run (rec): per-channel maxima for the 8-bit integers; integer bits of the fixed-point
    potentials (the maximum, or its 99.99th percentile, with 25% headroom; at least 0) and of the plastic synapses
    (the maximum with 25% headroom; negative when they stay below 1, leaving more fractional bits)."""
    ib = lambda vm: max(0, math.ceil(math.log2(max(vm, 1.0) * 1.25)))
    cal = {"bus": rec["bus"], "m": rec.get("m"), "head": rec["head"],
           "v_int": [ib(vm) for vm in rec["vmax"]], "v_int_p": [ib(vm) for vm in rec["vp9999"]]}
    if "mem_bus" in rec:
        cal.update(mem_bus=rec["mem_bus"], mem_v=rec["mem_v"], mem_r=rec["mem_r"],
                   kq_v_int=[ib(vm) for vm in rec["kq_vmax"]], kq_v_int_p=[ib(vm) for vm in rec["kq_vp9999"]],
                   mem_S_int=[math.ceil(math.log2(max(sm, 1e-6) * 1.25)) for sm in rec["mem_Smax"]])
    return cal


def configs_for(cfg):
    """The settings for this network. A network trained with low-precision weights (wbits) is loaded with its
    weights already rounded as trained (sample.load), so the W settings are left out (rounding them again to another
    grid would only add error). With plastic synapses: their values S in fixed point, S24 / S16 / S12."""
    out = [(n, q) for n, q in CONFIGS if not (cfg.wbits and q.get("w"))]
    full = dict(a=8, v=24, b=16) if cfg.wbits else dict(w=8, a=8, v=24, b=16)
    tag = "A8 V24 B16" if cfg.wbits else "W8 A8 V24 B16"
    if cfg.wbits:
        # the per-channel scales of A8 cannot be folded into weights trained as 4-bit integers without rounding
        # them again; one scale per group of 128 channels (as the weights' groups) or per vector factors out of
        # each group's integer sum instead
        out += [("A8 V24 B16", full), ("A8g128", dict(a=8, ag=128)), ("A8t", dict(a=8, ag=-1)),
                ("A8g128 V24 B16", dict(full, ag=128))]
    if cfg.mem_heads:
        out += [("S24", dict(s=24)), ("S16", dict(s=16)), ("S12", dict(s=12))]
        out += [(tag + " S16", dict(full, s=16)), (tag + " S12", dict(full, s=12))]
        if cfg.wbits:
            out.append(("A8g128 V24 B16 S16", dict(full, ag=128, s=16)))
    return out


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
    cal = calibrate(rec)
    x, y = windows(a.windows, seq, 0, dev, data)

    def loss_of(fn):
        tot = 0.0
        for i in range(0, len(x), a.batch):
            logits = fn(x[i:i + a.batch])
            tot += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y[i:i + a.batch].reshape(-1),
                                   reduction="sum").item()
        return tot / y.numel()

    bf16 = dev.type == "cpu" or torch.cuda.get_device_capability(dev)[0] >= 8   # Ampere and later

    def own(xb):   # older GPUs without bfloat16 (the cluster's nv partition): float32
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            return model(xb)[0]

    res = {"ema": has_ema, "windows": a.windows, "tokens": y.numel(), "v_int_bits": cal["v_int"],
           "v_int_bits_p9999": cal["v_int_p"], "calibration_vmax": rec["vmax"], "calibration_v_p9999": rec["vp9999"],
           "rate": rec["rate"], "loss": {}}
    res["loss"]["model"] = loss_of(own)
    print(f"{'model (bf16, fused kernel)':22s} {res['loss']['model']:.4f}", flush=True)
    keep = set(s.strip() for s in a.configs.split(",") if s.strip())
    if model.cfg.mem_heads:
        res.update(kq_v_int_bits=cal["kq_v_int"], mem_S_int_bits=cal["mem_S_int"], calibration_mem_Smax=rec["mem_Smax"])
    for name, q in configs_for(model.cfg):
        if keep and name not in keep and name != "fp32":
            continue
        res["loss"][name] = loss_of(lambda xb: run(model, xb, q, cal))
        d = res["loss"][name] - res["loss"]["fp32"]
        print(f"{name:22s} {res['loss'][name]:.4f}  ({d:+.4f} vs fp32)", flush=True)
    json.dump(res, open(os.path.join(a.run, "quant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

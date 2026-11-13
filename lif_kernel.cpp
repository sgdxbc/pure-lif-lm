// LIF neuron scan kernel (CPU, OpenMP).
//
// Discrete-time LIF, one simulation step per token:
//   V_pre[t]  = beta * V_post[t-1] + I[t]          (leaky integration of input current)
//   S[t]      = Heaviside(V_pre[t] - theta)        (fire)
//   V_post[t] = V_pre[t] - theta * S[t]            (soft reset, "subtract")
//            or V_pre[t] * (1 - S[t])              (hard reset, "to zero")
//
// Backward uses a surrogate derivative for the Heaviside:
//   z = (V - theta) / theta,   dS/dV ~= sg(z) / theta
// Layouts: I, S, V_pre are (B, L, D) contiguous; beta, theta are (D); v0 is (B, D).
// I / S / dS / dI may be float32 or bfloat16 (spikes are exactly 0/1 either way); all internal
// arithmetic and the saved membrane potential V_pre are float32.
// Work is split over (batch, chunk-of-neurons); each task walks through time sequentially.

#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <cmath>
#include <vector>

static constexpr int64_t CHUNK = 256;

static inline float tof(float x) { return x; }
static inline float tof(c10::BFloat16 x) { return static_cast<float>(x); }
template <typename T> static inline T fromf(float x) { return static_cast<T>(x); }

// ------------------------------------------------------------------------------------ forward
template <bool HARD, typename T>
static void fwd_impl(int64_t B, int64_t L, int64_t D, const T* Ip, const float* bp, const float* tp,
                     const float* v0p, T* Sp, float* Vpp, float* vlp) {
    const int64_t nchunk = (D + CHUNK - 1) / CHUNK;
#pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < B * nchunk; ++task) {
        const int64_t b = task / nchunk;
        const int64_t d0 = (task % nchunk) * CHUNK;
        const int64_t n = std::min(D, d0 + CHUNK) - d0;
        alignas(64) float v[CHUNK];
        const float* __restrict bt = bp + d0;
        const float* __restrict th = tp + d0;
        for (int64_t k = 0; k < n; ++k) v[k] = v0p[b * D + d0 + k];
        for (int64_t t = 0; t < L; ++t) {
            const int64_t off = (b * L + t) * D + d0;
            const T* __restrict it = Ip + off;
            T* __restrict st = Sp + off;
            float* __restrict xt = Vpp + off;
#pragma omp simd
            for (int64_t k = 0; k < n; ++k) {
                const float x = bt[k] * v[k] + tof(it[k]);
                xt[k] = x;
                const float s = x >= th[k] ? 1.f : 0.f;
                st[k] = fromf<T>(s);
                v[k] = HARD ? x * (1.f - s) : x - th[k] * s;
            }
        }
        for (int64_t k = 0; k < n; ++k) vlp[b * D + d0 + k] = v[k];
    }
}

std::vector<torch::Tensor> lif_forward(torch::Tensor I, torch::Tensor beta, torch::Tensor theta,
                                       torch::Tensor v0, int64_t hard_reset) {
    TORCH_CHECK(I.is_contiguous() && I.dim() == 3);
    TORCH_CHECK(beta.dtype() == torch::kFloat32 && theta.dtype() == torch::kFloat32 && v0.dtype() == torch::kFloat32);
    const int64_t B = I.size(0), L = I.size(1), D = I.size(2);
    auto S = torch::empty_like(I);
    auto Vp = torch::empty({B, L, D}, I.options().dtype(torch::kFloat32));
    auto vlast = torch::empty({B, D}, I.options().dtype(torch::kFloat32));
    const float *bp = beta.data_ptr<float>(), *tp = theta.data_ptr<float>(), *v0p = v0.data_ptr<float>();
    float *Vpp = Vp.data_ptr<float>(), *vlp = vlast.data_ptr<float>();
#define FWD(T) do { \
        if (hard_reset) fwd_impl<true, T>(B, L, D, I.data_ptr<T>(), bp, tp, v0p, S.data_ptr<T>(), Vpp, vlp); \
        else            fwd_impl<false, T>(B, L, D, I.data_ptr<T>(), bp, tp, v0p, S.data_ptr<T>(), Vpp, vlp); } while (0)
    if (I.dtype() == torch::kFloat32) FWD(float);
    else if (I.dtype() == torch::kBFloat16) FWD(c10::BFloat16);
    else TORCH_CHECK(false, "I must be float32 or bfloat16");
#undef FWD
    return {S, Vp, vlast};
}

// ----------------------------------------------------------------------------------- backward
template <int SG>
static inline float surrogate_t(float z, float alpha) {
    if constexpr (SG == 0) {          // derivative of sigmoid(alpha * z)
        const float u = std::fmin(std::fmax(-alpha * z, -30.f), 30.f);   // keep exp finite (-ffast-math)
        const float s = 1.f / (1.f + std::exp(u));
        return alpha * s * (1.f - s);
    } else if constexpr (SG == 1) {   // SuperSpike / fast sigmoid: 1 / (1 + alpha|z|)^2
        const float a = 1.f + alpha * std::fabs(z);
        return 1.f / (a * a);
    } else {                          // triangle with half-width 1/alpha, peak alpha
        const float a = 1.f - alpha * std::fabs(z);
        return a > 0.f ? alpha * a : 0.f;
    }
}

// One backward time step for a chunk of neurons. FIRST: t == 0 (previous membrane comes from v0).
template <bool HARD, bool DETACH, int SG, bool FIRST, typename T>
static inline void bwd_step(int64_t n, const T* __restrict gS, const float* __restrict x_t,
                            const float* __restrict x_prev, const float* __restrict v0,
                            const float* __restrict bt, const float* __restrict th,
                            T* __restrict gI, float* __restrict gpost,
                            float* __restrict accb, float* __restrict acct, float alpha) {
#pragma omp simd
    for (int64_t k = 0; k < n; ++k) {
        const float x = x_t[k];
        const float thk = th[k];
        const float inv = 1.f / thk;
        const float s = x >= thk ? 1.f : 0.f;
        const float sg = surrogate_t<SG>((x - thk) * inv, alpha) * inv;   // dS/dV
        float dx, dth;   // dV_post/dV_pre, dV_post/dtheta
        if constexpr (HARD) {
            dx = 1.f - s; dth = 0.f;
            if constexpr (!DETACH) { dx -= x * sg; dth = x * sg * x * inv; }
        } else {
            dx = 1.f; dth = -s;
            if constexpr (!DETACH) { dx -= thk * sg; dth += sg * x; }
        }
        const float g_s = tof(gS[k]);
        const float gp = gpost[k];
        const float gx = g_s * sg + gp * dx;   // dL/dV_pre[t] = dL/dI[t]
        gI[k] = fromf<T>(gx);
        float vprev;
        if constexpr (FIRST) {
            vprev = v0[k];
        } else {
            const float xp = x_prev[k];
            const float sp = xp >= thk ? 1.f : 0.f;
            vprev = HARD ? xp * (1.f - sp) : xp - thk * sp;
        }
        accb[k] += gx * vprev;
        acct[k] += g_s * (-sg * x * inv) + gp * dth;
        gpost[k] = bt[k] * gx;   // becomes dL/dV_post[t-1]
    }
}

template <bool HARD, bool DETACH, int SG, typename T>
static void bwd_impl(int64_t B, int64_t L, int64_t D, const T* gSp, const float* Vpp,
                     const float* bp, const float* tp, const float* v0p, const float* gvlp,
                     T* gIp, float* gbp, float* gtp, float* gv0p, float alpha) {
    const int64_t nchunk = (D + CHUNK - 1) / CHUNK;
#pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < B * nchunk; ++task) {
        const int64_t b = task / nchunk;
        const int64_t d0 = (task % nchunk) * CHUNK;
        const int64_t n = std::min(D, d0 + CHUNK) - d0;
        alignas(64) float gpost[CHUNK], accb[CHUNK], acct[CHUNK];
        for (int64_t k = 0; k < n; ++k) { gpost[k] = gvlp[b * D + d0 + k]; accb[k] = 0.f; acct[k] = 0.f; }
        for (int64_t t = L - 1; t >= 1; --t) {
            const int64_t off = (b * L + t) * D + d0;
            bwd_step<HARD, DETACH, SG, false, T>(n, gSp + off, Vpp + off, Vpp + off - D, nullptr,
                                                 bp + d0, tp + d0, gIp + off, gpost, accb, acct, alpha);
        }
        {
            const int64_t off = (b * L) * D + d0;
            bwd_step<HARD, DETACH, SG, true, T>(n, gSp + off, Vpp + off, nullptr, v0p + b * D + d0,
                                                bp + d0, tp + d0, gIp + off, gpost, accb, acct, alpha);
        }
        for (int64_t k = 0; k < n; ++k) {
            gbp[b * D + d0 + k] = accb[k];
            gtp[b * D + d0 + k] = acct[k];
            gv0p[b * D + d0 + k] = gpost[k];   // = beta * dL/dV_pre[0]
        }
    }
}

// Returns gI (B,L,D; dtype of gS), gbeta_partial (B,D), gtheta_partial (B,D), gv0 (B,D)
std::vector<torch::Tensor> lif_backward(torch::Tensor gS, torch::Tensor Vp, torch::Tensor beta,
                                        torch::Tensor theta, torch::Tensor v0, torch::Tensor gvlast,
                                        int64_t hard_reset, double alpha_d, int64_t detach_reset,
                                        int64_t sg_type) {
    TORCH_CHECK(gS.is_contiguous() && Vp.is_contiguous() && Vp.dtype() == torch::kFloat32);
    const int64_t B = Vp.size(0), L = Vp.size(1), D = Vp.size(2);
    auto gI = torch::empty_like(gS);
    auto fopt = Vp.options();
    auto gb = torch::empty({B, D}, fopt), gt = torch::empty({B, D}, fopt), gv0 = torch::empty({B, D}, fopt);
    const float a = (float)alpha_d;
#define CALL(H, DT, S, T) bwd_impl<H, DT, S, T>(B, L, D, gS.data_ptr<T>(), Vp.data_ptr<float>(), \
        beta.data_ptr<float>(), theta.data_ptr<float>(), v0.data_ptr<float>(), gvlast.data_ptr<float>(), \
        gI.data_ptr<T>(), gb.data_ptr<float>(), gt.data_ptr<float>(), gv0.data_ptr<float>(), a)
#define SGSWITCH(H, DT, T) switch (sg_type) { case 0: CALL(H, DT, 0, T); break; case 1: CALL(H, DT, 1, T); break; default: CALL(H, DT, 2, T); }
#define DISPATCH(T) if (hard_reset) { if (detach_reset) { SGSWITCH(true, true, T) } else { SGSWITCH(true, false, T) } } \
                    else            { if (detach_reset) { SGSWITCH(false, true, T) } else { SGSWITCH(false, false, T) } }
    if (gS.dtype() == torch::kFloat32) { DISPATCH(float) }
    else if (gS.dtype() == torch::kBFloat16) { DISPATCH(c10::BFloat16) }
    else TORCH_CHECK(false, "gS must be float32 or bfloat16");
#undef DISPATCH
#undef SGSWITCH
#undef CALL
    return {gI, gb, gt, gv0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lif_forward, "LIF forward");
    m.def("backward", &lif_backward, "LIF backward");
}

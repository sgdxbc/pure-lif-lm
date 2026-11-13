// LIF neuron scan kernels (CPU, OpenMP).
//
// Discrete-time LIF, one simulation step per token:
//   V_pre[t]  = beta * V_post[t-1] + I[t] (+ R s[t-1])   (leaky integration; optional same-layer synapses)
//   S[t]      = Heaviside(V_pre[t] - theta)             (fire)
//   V_post[t] = V_pre[t] - theta * S[t]                 (soft reset, "subtract")
//            or V_pre[t] * (1 - gamma * S[t])           (multiplicative reset; gamma = 1 is the classic
//                                                         hard reset "to zero", gamma -> 0 keeps the membrane)
//
// Backward uses a surrogate derivative for the Heaviside:
//   z = (V - theta) / theta,   dS/dV ~= sg(z) / theta
// Layouts: I, S, V_pre are (B, L, D) contiguous; beta, theta, gamma are (D); v0 / s0 are (B, D).
// I / S / dS / dI may be float32 or bfloat16 (spikes are exactly 0/1 either way); all internal
// arithmetic and the saved membrane potential V_pre are float32.
//
// Two kernels:
//  * lif_forward / lif_backward: no same-layer synapses. Every neuron is independent, so work is split
//    over (batch, chunk-of-neurons) and each task walks through time on its own.
//  * lif_rec_forward / lif_rec_backward: with low-rank same-layer synapses R = A @ Bm acting on the
//    previous step's spikes. Time is the outer loop; each step does a small matmul, then an element-wise
//    update parallelised over (batch, neuron). The weight gradients of A and Bm are formed afterwards in
//    Python from dI and S with two large matmuls.

#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <cmath>
#include <vector>

static constexpr int64_t CHUNK = 256;

static inline float tof(float x) { return x; }
static inline float tof(c10::BFloat16 x) { return static_cast<float>(x); }
template <typename T> static inline T fromf(float x) { return static_cast<T>(x); }

template <bool HARD>
static inline float reset(float x, float s, float th, float g) {
    return HARD ? x * (1.f - g * s) : x - th * s;
}

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

// Backward of one time step for n neurons.  g_s: dL/dS[t] (from above), gpost: in = dL/dV_post[t],
// out = dL/dV_post[t-1].  x_prev == nullptr means t == 0 (previous membrane is v0, already post-reset).
template <bool HARD, bool DETACH, int SG, typename T, typename TG>
static inline void bwd_elem(int64_t n, const TG* __restrict gS, const T* __restrict carry,
                            const float* __restrict x_t, const float* __restrict x_prev,
                            const float* __restrict v0, const float* __restrict bt,
                            const float* __restrict th, const float* __restrict gm,
                            T* __restrict gI, float* __restrict gpost, float* __restrict accb,
                            float* __restrict acct, float* __restrict accg, float alpha) {
#pragma omp simd
    for (int64_t k = 0; k < n; ++k) {
        const float x = x_t[k];
        const float thk = th[k];
        const float gk = HARD ? gm[k] : 0.f;
        const float inv = 1.f / thk;
        const float s = x >= thk ? 1.f : 0.f;
        const float sg = surrogate_t<SG>((x - thk) * inv, alpha) * inv;   // dS/dV
        float dx, dth, dgm;   // dV_post/dV_pre, dV_post/dtheta, dV_post/dgamma
        if constexpr (HARD) {
            dx = 1.f - gk * s; dth = 0.f; dgm = -s * x;
            if constexpr (!DETACH) { dx -= gk * x * sg; dth = gk * x * sg * x * inv; }
        } else {
            dx = 1.f; dth = -s; dgm = 0.f;
            if constexpr (!DETACH) { dx -= thk * sg; dth += sg * x; }
        }
        const float g_s = tof(gS[k]) + (carry ? tof(carry[k]) : 0.f);
        const float gq = gpost[k];
        const float gx = g_s * sg + gq * dx;   // dL/dV_pre[t] = dL/dI[t]
        gI[k] = fromf<T>(gx);
        float vprev;
        if (x_prev == nullptr) {
            vprev = v0[k];
        } else {
            const float xp = x_prev[k];
            vprev = reset<HARD>(xp, xp >= thk ? 1.f : 0.f, thk, gk);
        }
        accb[k] += gx * vprev;
        acct[k] += g_s * (-sg * x * inv) + gq * dth;
        accg[k] += gq * dgm;
        gpost[k] = bt[k] * gx;
    }
}

// =============================================================== kernel 1: independent neurons
template <bool HARD, typename T>
static void fwd_impl(int64_t B, int64_t L, int64_t D, const T* Ip, const float* bp, const float* tp,
                     const float* gp, const float* v0p, T* Sp, float* Vpp, float* vlp) {
    const int64_t nchunk = (D + CHUNK - 1) / CHUNK;
#pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < B * nchunk; ++task) {
        const int64_t b = task / nchunk;
        const int64_t d0 = (task % nchunk) * CHUNK;
        const int64_t n = std::min(D, d0 + CHUNK) - d0;
        alignas(64) float v[CHUNK];
        const float* __restrict bt = bp + d0;
        const float* __restrict th = tp + d0;
        const float* __restrict gm = gp + d0;
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
                v[k] = reset<HARD>(x, s, th[k], gm[k]);
            }
        }
        for (int64_t k = 0; k < n; ++k) vlp[b * D + d0 + k] = v[k];
    }
}

static void check_params(const torch::Tensor& beta, const torch::Tensor& theta, const torch::Tensor& gamma,
                         const torch::Tensor& v0) {
    TORCH_CHECK(beta.dtype() == torch::kFloat32 && theta.dtype() == torch::kFloat32 &&
                gamma.dtype() == torch::kFloat32 && v0.dtype() == torch::kFloat32);
    TORCH_CHECK(beta.is_contiguous() && theta.is_contiguous() && gamma.is_contiguous() && v0.is_contiguous());
}

std::vector<torch::Tensor> lif_forward(torch::Tensor I, torch::Tensor beta, torch::Tensor theta,
                                       torch::Tensor gamma, torch::Tensor v0, int64_t hard_reset) {
    TORCH_CHECK(I.is_contiguous() && I.dim() == 3);
    check_params(beta, theta, gamma, v0);
    const int64_t B = I.size(0), L = I.size(1), D = I.size(2);
    auto S = torch::empty_like(I);
    auto Vp = torch::empty({B, L, D}, I.options().dtype(torch::kFloat32));
    auto vlast = torch::empty({B, D}, I.options().dtype(torch::kFloat32));
    const float *bp = beta.data_ptr<float>(), *tp = theta.data_ptr<float>(), *gp = gamma.data_ptr<float>();
    const float* v0p = v0.data_ptr<float>();
    float *Vpp = Vp.data_ptr<float>(), *vlp = vlast.data_ptr<float>();
#define FWD(T) do { \
        if (hard_reset) fwd_impl<true, T>(B, L, D, I.data_ptr<T>(), bp, tp, gp, v0p, S.data_ptr<T>(), Vpp, vlp); \
        else            fwd_impl<false, T>(B, L, D, I.data_ptr<T>(), bp, tp, gp, v0p, S.data_ptr<T>(), Vpp, vlp); } while (0)
    if (I.dtype() == torch::kFloat32) FWD(float);
    else if (I.dtype() == torch::kBFloat16) FWD(c10::BFloat16);
    else TORCH_CHECK(false, "I must be float32 or bfloat16");
#undef FWD
    return {S, Vp, vlast};
}

template <bool HARD, bool DETACH, int SG, typename T>
static void bwd_impl(int64_t B, int64_t L, int64_t D, const T* gSp, const float* Vpp, const float* bp,
                     const float* tp, const float* gmp, const float* v0p, const float* gvlp, T* gIp,
                     float* gbp, float* gtp, float* ggp, float* gv0p, float alpha) {
    const int64_t nchunk = (D + CHUNK - 1) / CHUNK;
#pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < B * nchunk; ++task) {
        const int64_t b = task / nchunk;
        const int64_t d0 = (task % nchunk) * CHUNK;
        const int64_t n = std::min(D, d0 + CHUNK) - d0;
        alignas(64) float gpost[CHUNK], accb[CHUNK], acct[CHUNK], accg[CHUNK];
        for (int64_t k = 0; k < n; ++k) {
            gpost[k] = gvlp[b * D + d0 + k]; accb[k] = 0.f; acct[k] = 0.f; accg[k] = 0.f;
        }
        for (int64_t t = L - 1; t >= 0; --t) {
            const int64_t off = (b * L + t) * D + d0;
            bwd_elem<HARD, DETACH, SG, T, T>(n, gSp + off, (const T*)nullptr, Vpp + off,
                                             t > 0 ? Vpp + off - D : nullptr, v0p + b * D + d0,
                                             bp + d0, tp + d0, gmp + d0, gIp + off, gpost, accb, acct,
                                             accg, alpha);
        }
        for (int64_t k = 0; k < n; ++k) {
            gbp[b * D + d0 + k] = accb[k];
            gtp[b * D + d0 + k] = acct[k];
            ggp[b * D + d0 + k] = accg[k];
            gv0p[b * D + d0 + k] = gpost[k];   // = beta * dL/dV_pre[0]
        }
    }
}

#define DISPATCH_BWD(IMPL, T, ...)                                                                    \
    do {                                                                                               \
        if (hard_reset) {                                                                              \
            if (detach_reset) { switch (sg_type) { case 0: IMPL<true, true, 0, T>(__VA_ARGS__); break; \
                case 1: IMPL<true, true, 1, T>(__VA_ARGS__); break; default: IMPL<true, true, 2, T>(__VA_ARGS__); } } \
            else { switch (sg_type) { case 0: IMPL<true, false, 0, T>(__VA_ARGS__); break;            \
                case 1: IMPL<true, false, 1, T>(__VA_ARGS__); break; default: IMPL<true, false, 2, T>(__VA_ARGS__); } } \
        } else {                                                                                       \
            if (detach_reset) { switch (sg_type) { case 0: IMPL<false, true, 0, T>(__VA_ARGS__); break; \
                case 1: IMPL<false, true, 1, T>(__VA_ARGS__); break; default: IMPL<false, true, 2, T>(__VA_ARGS__); } } \
            else { switch (sg_type) { case 0: IMPL<false, false, 0, T>(__VA_ARGS__); break;           \
                case 1: IMPL<false, false, 1, T>(__VA_ARGS__); break; default: IMPL<false, false, 2, T>(__VA_ARGS__); } } \
        }                                                                                              \
    } while (0)

// Returns gI (B,L,D; dtype of gS), gbeta_partial, gtheta_partial, ggamma_partial (B,D), gv0 (B,D)
std::vector<torch::Tensor> lif_backward(torch::Tensor gS, torch::Tensor Vp, torch::Tensor beta,
                                        torch::Tensor theta, torch::Tensor gamma, torch::Tensor v0,
                                        torch::Tensor gvlast, int64_t hard_reset, double alpha_d,
                                        int64_t detach_reset, int64_t sg_type) {
    TORCH_CHECK(gS.is_contiguous() && Vp.is_contiguous() && Vp.dtype() == torch::kFloat32);
    check_params(beta, theta, gamma, v0);
    const int64_t B = Vp.size(0), L = Vp.size(1), D = Vp.size(2);
    auto gI = torch::empty_like(gS);
    auto fopt = Vp.options();
    auto gb = torch::empty({B, D}, fopt), gt = torch::empty({B, D}, fopt);
    auto gg = torch::empty({B, D}, fopt), gv0 = torch::empty({B, D}, fopt);
    const float a = (float)alpha_d;
#define ARGS(T) B, L, D, gS.data_ptr<T>(), Vp.data_ptr<float>(), beta.data_ptr<float>(), theta.data_ptr<float>(), \
        gamma.data_ptr<float>(), v0.data_ptr<float>(), gvlast.data_ptr<float>(), gI.data_ptr<T>(), \
        gb.data_ptr<float>(), gt.data_ptr<float>(), gg.data_ptr<float>(), gv0.data_ptr<float>(), a
    if (gS.dtype() == torch::kFloat32) DISPATCH_BWD(bwd_impl, float, ARGS(float));
    else if (gS.dtype() == torch::kBFloat16) DISPATCH_BWD(bwd_impl, c10::BFloat16, ARGS(c10::BFloat16));
    else TORCH_CHECK(false, "gS must be float32 or bfloat16");
#undef ARGS
    return {gI, gb, gt, gg, gv0};
}

// ============================================================ kernel 2: same-layer synapses
template <bool HARD, typename T>
static void rec_fwd_step(int64_t B, int64_t L, int64_t D, int64_t t, const T* Ip, const T* recp,
                         const float* bp, const float* tp, const float* gp, float* vp, T* Sp, float* Vpp) {
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const int64_t off = (b * L + t) * D;
        const T* __restrict it = Ip + off;
        const T* __restrict rc = recp + b * D;
        float* __restrict v = vp + b * D;
        T* __restrict st = Sp + off;
        float* __restrict xt = Vpp + off;
#pragma omp simd
        for (int64_t d = 0; d < D; ++d) {
            const float x = bp[d] * v[d] + tof(it[d]) + tof(rc[d]);
            xt[d] = x;
            const float s = x >= tp[d] ? 1.f : 0.f;
            st[d] = fromf<T>(s);
            v[d] = reset<HARD>(x, s, tp[d], gp[d]);
        }
    }
}

std::vector<torch::Tensor> lif_rec_forward(torch::Tensor I, torch::Tensor beta, torch::Tensor theta,
                                           torch::Tensor gamma, torch::Tensor v0, torch::Tensor s0,
                                           torch::Tensor A, torch::Tensor Bm, int64_t hard_reset) {
    TORCH_CHECK(I.is_contiguous() && I.dim() == 3);
    check_params(beta, theta, gamma, v0);
    TORCH_CHECK(s0.dtype() == I.dtype() && A.dtype() == I.dtype() && Bm.dtype() == I.dtype());
    const int64_t B = I.size(0), L = I.size(1), D = I.size(2);
    auto S = torch::empty_like(I);
    auto Vp = torch::empty({B, L, D}, I.options().dtype(torch::kFloat32));
    auto v = v0.clone();
    torch::Tensor sprev = s0;
    const float *bp = beta.data_ptr<float>(), *tp = theta.data_ptr<float>(), *gp = gamma.data_ptr<float>();
    for (int64_t t = 0; t < L; ++t) {
        auto rec = at::mm(at::mm(sprev, A), Bm).contiguous();   // (B, D): current through same-layer synapses
#define STEP(T) do { \
            if (hard_reset) rec_fwd_step<true, T>(B, L, D, t, I.data_ptr<T>(), rec.data_ptr<T>(), bp, tp, gp, \
                                                  v.data_ptr<float>(), S.data_ptr<T>(), Vp.data_ptr<float>()); \
            else rec_fwd_step<false, T>(B, L, D, t, I.data_ptr<T>(), rec.data_ptr<T>(), bp, tp, gp, \
                                        v.data_ptr<float>(), S.data_ptr<T>(), Vp.data_ptr<float>()); } while (0)
        if (I.dtype() == torch::kFloat32) STEP(float);
        else if (I.dtype() == torch::kBFloat16) STEP(c10::BFloat16);
        else TORCH_CHECK(false, "I must be float32 or bfloat16");
#undef STEP
        sprev = S.select(1, t);
    }
    return {S, Vp, v};
}

template <bool HARD, bool DETACH, int SG, typename T>
static void rec_bwd_step(int64_t B, int64_t L, int64_t D, int64_t t, const T* gSp, const T* carryp,
                         const float* Vpp, const float* bp, const float* tp, const float* gmp,
                         const float* v0p, T* gIp, float* gpost, float* accb, float* acct, float* accg,
                         float alpha) {
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const int64_t off = (b * L + t) * D;
        bwd_elem<HARD, DETACH, SG, T, T>(D, gSp + off, carryp + b * D, Vpp + off,
                                         t > 0 ? Vpp + off - D : nullptr, v0p + b * D, bp, tp, gmp,
                                         gIp + off, gpost + b * D, accb + b * D, acct + b * D,
                                         accg + b * D, alpha);
    }
}

// Returns gI (B,L,D), gbeta_partial, gtheta_partial, ggamma_partial (B,D), gv0 (B,D), gs0 (B,D)
std::vector<torch::Tensor> lif_rec_backward(torch::Tensor gS, torch::Tensor Vp, torch::Tensor beta,
                                            torch::Tensor theta, torch::Tensor gamma, torch::Tensor v0,
                                            torch::Tensor gvlast, torch::Tensor A, torch::Tensor Bm,
                                            int64_t hard_reset, double alpha_d, int64_t detach_reset,
                                            int64_t sg_type) {
    TORCH_CHECK(gS.is_contiguous() && Vp.is_contiguous() && Vp.dtype() == torch::kFloat32);
    check_params(beta, theta, gamma, v0);
    const int64_t B = Vp.size(0), L = Vp.size(1), D = Vp.size(2);
    auto gI = torch::empty_like(gS);
    auto fopt = Vp.options();
    auto gpost = gvlast.clone();
    auto accb = torch::zeros({B, D}, fopt), acct = torch::zeros({B, D}, fopt), accg = torch::zeros({B, D}, fopt);
    auto carry = torch::zeros({B, D}, gS.options());      // dL/dS[t] arriving through the same-layer synapses
    auto At = A.t().contiguous(), Bt = Bm.t().contiguous();
    const float a = (float)alpha_d;
    for (int64_t t = L - 1; t >= 0; --t) {
#define ARGS(T) B, L, D, t, gS.data_ptr<T>(), carry.data_ptr<T>(), Vp.data_ptr<float>(), beta.data_ptr<float>(), \
            theta.data_ptr<float>(), gamma.data_ptr<float>(), v0.data_ptr<float>(), gI.data_ptr<T>(), \
            gpost.data_ptr<float>(), accb.data_ptr<float>(), acct.data_ptr<float>(), accg.data_ptr<float>(), a
        if (gS.dtype() == torch::kFloat32) DISPATCH_BWD(rec_bwd_step, float, ARGS(float));
        else if (gS.dtype() == torch::kBFloat16) DISPATCH_BWD(rec_bwd_step, c10::BFloat16, ARGS(c10::BFloat16));
        else TORCH_CHECK(false, "gS must be float32 or bfloat16");
#undef ARGS
        carry = at::mm(at::mm(gI.select(1, t), Bt), At).contiguous();   // dL/dS[t-1] via R
    }
    return {gI, accb, acct, accg, gpost, carry};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lif_forward, "LIF forward");
    m.def("backward", &lif_backward, "LIF backward");
    m.def("rec_forward", &lif_rec_forward, "LIF forward with same-layer synapses");
    m.def("rec_backward", &lif_rec_backward, "LIF backward with same-layer synapses");
}

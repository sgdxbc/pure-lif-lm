// FPGA kernel (Vitis HLS, AMD Alveo U55C) of the spiking language model: first version, docs/experiments.md 5.44
// ("one U55C, weights in HBM"). It runs NB streams in lockstep, one token per stream and step, for n_steps steps, all
// inside one call, with every neuron's state on chip; the weights are read from HBM once per step for all NB streams.
// The arithmetic is exactly hw_ref.cpp's (the definition): the kernel must reproduce its numbers bit for bit, which
// fpga/csim.sh checks in C simulation (the same source compiled with g++).
//
// Inputs (fpga/pack.py writes them): w (weights.bin: every matrix in the order of the computation, in blocks of RB rows
// and groups of 128 inputs, a word of float32 combs then 4-bit or 8-bit weights; the input rows), prm (params.bin:
// per-neuron and per-channel tables), aux (aux.bin: the input rows of each word, the word of each readout slot).
// tok_in: NB x n_steps tokens. mode 0 (generation): the first n_prompt are read, then each stream reads the token it
// generated (greedy: the most likely short-list or class neuron, then the most likely word of that class). mode 1
// (check): every step reads tok_in, and dump receives the bus after the last layer (NB x n_steps x D int32).
// tok_out: NB x n_steps, the greedy prediction after each step.
//
// First version: correct, with the obvious parallelism (all NB streams and a whole 512-bit word of weights per step
// of the matrix engine); not yet tuned (one HBM port, sequential phases).
#include <ap_int.h>
#include <stdint.h>
#include <math.h>
#include "model_cfg.h"

#ifndef NB
#define NB 16
#endif

typedef ap_uint<512> word_t;

#define MAXI (H > D ? H : D)
#define TOPN (NSHORT + NCLS)
#define MAXO_A (H > TOPN ? H : TOPN)
#define MAXO (MAXO_A > 2 * NH * MK ? MAXO_A : 2 * NH * MK)
#define HM (NH * MK)
#define HD (NH * DV)

static inline int64_t rnd(float x) { return (int64_t)floorf(x + 0.5f); }
static inline int64_t rshift16(int64_t x) { return (x + ((int64_t)1 << 15)) >> 16; }
static inline int64_t clampi(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline float f32_of(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }

// on-chip copies of the small tables (params.bin)
static uint32_t P[NPARAMS];
static inline float pf(long i) { return f32_of(P[i]); }
static inline int32_t pi(long i) { return (int32_t)P[i]; }

// y[b][o] = sum over the groups, in order, of (float)(integer sum of the group) * comb, for streams b < ns
static void matvec(const word_t* w, long off, int O, int I, int bits, int ns, int8_t act[NB][MAXI],
                   float y[NB][MAXO]) {
  const int ng = (I + 127) / 128, nb = (O + RB - 1) / RB, wpg = bits == 4 ? 1 : 2;
  long p = off;
blocks:
  for (int blk = 0; blk < nb; blk++) {
    float part[RB][NB];
    for (int r = 0; r < RB; r++)
      for (int b = 0; b < NB; b++) part[r][b] = 0.f;
  groups:
    for (int g = 0; g < ng; g++) {
      word_t cw = w[p++];
    rows:
      for (int r = 0; r < RB; r++) {
        int32_t acc[NB];
        for (int b = 0; b < NB; b++) acc[b] = 0;
        for (int k = 0; k < wpg; k++) {
          word_t ww = w[p++];
          const int n = bits == 4 ? 128 : 64;
          for (int c = 0; c < n; c++) {
            int wt = bits == 4 ? (int)(ap_int<4>)ww.range(4 * c + 3, 4 * c) : (int)(ap_int<8>)ww.range(8 * c + 7, 8 * c);
            int col = g * 128 + k * 64 + c;
            if (col < I)
              for (int b = 0; b < NB; b++)
                if (b < ns) acc[b] += wt * (int)act[b][col];
          }
        }
        float cm = f32_of((uint32_t)cw.range(32 * r + 31, 32 * r));
        for (int b = 0; b < NB; b++) part[r][b] = part[r][b] + (float)acc[b] * cm;
      }
    }
    for (int r = 0; r < RB; r++) {
      int o = blk * RB + r;
      if (o < O)
        for (int b = 0; b < NB; b++) y[b][o] = part[r][b];
    }
  }
}

static void act_from_bus(const int32_t bus[D], long ainv, int8_t a[MAXI]) {
  for (int c = 0; c < D; c++) a[c] = (int8_t)clampi(rnd((float)bus[c] * pf(ainv + c / 128)), -127, 127);
}
static void act_from_float(const float* x, long ainv, int n, int g, int8_t* a) {
  for (int c = 0; c < n; c++) a[c] = (int8_t)clampi(rnd(x[c] * pf(ainv + c / g)), -127, 127);
}
static void add_to_bus(int32_t bus[D], const float* y) {
  for (int c = 0; c < D; c++) bus[c] = (int32_t)clampi((int64_t)bus[c] + rnd(y[c]), INT32_MIN, INT32_MAX);
}

// LIF update (hw_ref.cpp lif): beta per neuron from prm (betap >= 0) or per step (betas); hard reset with 1 - gamma
// from prm (g1p >= 0) or soft reset
static void lif(int n, const float* I, int32_t* v, long betap, const int32_t* betas, long g1p, int intb, uint8_t* spk) {
  const int VF = VB - 1 - intb;
  const int64_t lo = -((int64_t)1 << (VB - 1)), hi = ((int64_t)1 << (VB - 1)) - 1, one = (int64_t)1 << VF;
  const float sc = ldexpf(1.0f, VF);
  for (int o = 0; o < n; o++) {
    int64_t Ii = clampi(rnd(I[o] * sc), lo, hi);
    int64_t b = betap >= 0 ? pi(betap + o) : betas[o];
    int64_t x = clampi(rshift16(b * (int64_t)v[o]) + Ii, lo, hi);
    uint8_t s = x >= one;
    if (s) x = g1p >= 0 ? clampi(rshift16(x * (int64_t)pi(g1p + o)), lo, hi) : clampi(x - one, lo, hi);
    v[o] = (int32_t)x;
    spk[o] = s;
  }
}

// state of the NB streams
static int32_t V[NB][NL][H];
static int32_t KQ[NB][NL][2 * HM];
static int16_t S[NB][NL][NH][MK][DV];
static int32_t BUS[NB][D];
// scratch
static int8_t ACT[NB][MAXI];
static float Y[NB][MAXO];
static float Y2[NB][MAXO];
static uint8_t SPK[NB][MAXI];

static int greedy(const word_t* w, const int32_t* aux, int b) {
  // the most likely short-list or class neuron (first on ties)
  int best = 0;
  for (int i = 1; i < TOPN; i++)
    if (Y[b][i] > Y[b][best]) best = i;
  if (best < NSHORT) return aux[A_SHORT + best];
  int c = best - NSHORT;
  static int8_t a1[NB][MAXI];
  static float y1[NB][MAXO];
  for (int k = 0; k < D; k++) a1[0][k] = ACT[b][k];
  matvec(w, W_H_WORD + (long)c * CLASS_WORDS, CM, D, BITS_H_WORD, 1, a1, y1);
  int bp = -1;
  for (int p = 0; p < CM; p++) {
    if (aux[A_CLASS + (long)c * CM + p] < 0) continue;
    if (bp < 0 || y1[0][p] > y1[0][bp]) bp = p;
  }
  return aux[A_CLASS + (long)c * CM + bp];
}

static void step(const word_t* w, const int32_t* aux, const int x[NB]) {
  // input rows
  for (int b = 0; b < NB; b++) {
    for (int c = 0; c < D; c++) BUS[b][c] = 0;
    for (int k = 0; k < NCODE; k++) {
      int row = aux[A_CODE + (long)x[b] * NCODE + k];
      long base = W_EMB + (long)row * EMB_WORDS;
      word_t sw = w[base + EMB_WORDS - 1];
      for (int q = 0; q < EMB_WORDS - 1; q++) {
        word_t ew = w[base + q];
        for (int j = 0; j < 64; j++) {
          int c = q * 64 + j;
          float sc = f32_of((uint32_t)sw.range(32 * (c / 128) + 31, 32 * (c / 128)));
          BUS[b][c] += (int32_t)rnd((float)(int)(ap_int<8>)ew.range(8 * j + 7, 8 * j) * sc);
        }
      }
    }
  }
layers:
  for (int l = 0; l < NL; l++) {
    // plastic synapses in front of population l
    for (int b = 0; b < NB; b++) act_from_bus(BUS[b], P_M_AIN[l], ACT[b]);
    matvec(w, W_M_KQ[l], 2 * HM, D, BITS_M_KQ[l], NB, ACT, Y);
    matvec(w, W_M_V[l], HD, D, BITS_M_V[l], NB, ACT, Y2);
    const int SF = SB - 1 - SINT[l];
    const int64_t slo = -((int64_t)1 << (SB - 1)), shi = ((int64_t)1 << (SB - 1)) - 1;
    const float unit = ldexpf(1.0f, -SF);
    for (int b = 0; b < NB; b++) {
      for (int o = 0; o < 2 * HM; o++) Y[b][o] = Y[b][o] + pf(P_M_KQB[l] + o);
      lif(2 * HM, Y[b], KQ[b][l], P_M_KQBETA[l], 0, -1, KQINT[l], SPK[b]);
      int8_t va[HD];
      act_from_float(Y2[b], P_M_VAIN[l], HD, HD % 128 == 0 ? 128 : HD, va);
      float z[HD];
      for (int hh = 0; hh < NH; hh++) {
        int64_t lam = pi(P_M_LAM[l] + hh);
        for (int k = 0; k < MK; k++)
          for (int ch = 0; ch < DV; ch++) S[b][l][hh][k][ch] = (int16_t)clampi(rshift16(lam * S[b][l][hh][k][ch]), slo, shi);
        for (int k = 0; k < MK; k++) {
          if (!SPK[b][hh * MK + k]) continue;   // key neuron k fired: write its row
          for (int ch = 0; ch < DV; ch++) {
            int cc = hh * DV + ch;
            int64_t vs = rnd((float)va[cc] * pf(P_M_VTOS[l] + cc / (HD % 128 == 0 ? 128 : HD)));
            S[b][l][hh][k][ch] = (int16_t)clampi(S[b][l][hh][k][ch] + vs, slo, shi);
          }
        }
        int64_t r[DV];
        for (int ch = 0; ch < DV; ch++) r[ch] = 0;
        for (int k = 0; k < MK; k++) {
          if (!SPK[b][HM + hh * MK + k]) continue;   // query neuron k fired: read its row
          for (int ch = 0; ch < DV; ch++) r[ch] += S[b][l][hh][k][ch];
        }
        float ms = 0.f;
        for (int ch = 0; ch < DV; ch++) {
          float rf = (float)r[ch] * unit;
          z[hh * DV + ch] = rf;
          ms = ms + rf * rf;
        }
        ms = ms / (float)DV;
        float inv = 1.0f / sqrtf(ms + MEM_EPS);
        for (int ch = 0; ch < DV; ch++) z[hh * DV + ch] = z[hh * DV + ch] * inv * pf(P_M_GAIN[l] + hh * DV + ch);
      }
      act_from_float(z, P_M_RAIN[l], HD, HD % 128 == 0 ? 128 : HD, ACT[b]);
    }
    matvec(w, W_M_O[l], D, HD, BITS_M_O[l], NB, ACT, Y);
    for (int b = 0; b < NB; b++) add_to_bus(BUS[b], Y[b]);

    // population l
    for (int b = 0; b < NB; b++) act_from_bus(BUS[b], P_P_AIN[l], ACT[b]);
    matvec(w, W_P_IN[l], H, D, BITS_P_IN[l], NB, ACT, Y);
    matvec(w, W_P_LI[l], R, D, BITS_P_LI[l], NB, ACT, Y2);
    for (int b = 0; b < NB; b++) act_from_float(Y2[b], P_P_MAIN[l], R, R % 128 == 0 ? 128 : R, ACT[b]);
    matvec(w, W_P_LO[l], H, R, BITS_P_LO[l], NB, ACT, Y2);
    for (int b = 0; b < NB; b++) {
      static int32_t betas[H];
      const float sc = ldexpf(1.0f, SIGSTEP);
      for (int o = 0; o < H; o++) {
        Y[b][o] = Y[b][o] + pf(P_P_INB[l] + o);
        betas[o] = pi(P_SIG + clampi(rnd((pf(P_P_BLOGIT[l] + o) + Y2[b][o]) * sc), -SIGN, SIGN) + SIGN);
      }
      lif(H, Y[b], V[b][l], -1, betas, P_P_G1[l], VINT[l], SPK[b]);
      for (int o = 0; o < H; o++) ACT[b][o] = (int8_t)SPK[b][o];   // W_out's input: the spikes, 0 or 1
    }
    matvec(w, W_P_OUT[l], D, H, BITS_P_OUT[l], NB, ACT, Y);
    for (int b = 0; b < NB; b++) add_to_bus(BUS[b], Y[b]);
  }
  for (int b = 0; b < NB; b++) act_from_bus(BUS[b], P_H_AIN, ACT[b]);
  matvec(w, W_H_CLS, TOPN, D, BITS_H_CLS, NB, ACT, Y);
}

extern "C" void lif_lm(const word_t* w, const uint32_t* prm, const int32_t* aux, const int32_t* tok_in,
                       int32_t* tok_out, int32_t* dump, int n_steps, int n_prompt, int mode) {
#pragma HLS INTERFACE m_axi port = w offset = slave bundle = gmem0 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = prm offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = aux offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = tok_in offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = tok_out offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = dump offset = slave bundle = gmem1
  for (long i = 0; i < NPARAMS; i++) P[i] = prm[i];
  for (int b = 0; b < NB; b++) {
    for (int l = 0; l < NL; l++) {
      for (int o = 0; o < H; o++) V[b][l][o] = 0;
      for (int o = 0; o < 2 * HM; o++) KQ[b][l][o] = 0;
      for (int hh = 0; hh < NH; hh++)
        for (int k = 0; k < MK; k++)
          for (int ch = 0; ch < DV; ch++) S[b][l][hh][k][ch] = 0;
    }
  }
  int x[NB];
  for (int b = 0; b < NB; b++) x[b] = tok_in[(long)b * n_steps];
  for (int t = 0; t < n_steps; t++) {
    step(w, aux, x);
    for (int b = 0; b < NB; b++) {
      if (mode == 1)
        for (int c = 0; c < D; c++) dump[((long)b * n_steps + t) * D + c] = BUS[b][c];
      int nxt = greedy(w, aux, b);
      tok_out[(long)b * n_steps + t] = nxt;
      if (t + 1 < n_steps) x[b] = (mode == 1 || t + 1 < n_prompt) ? tok_in[(long)b * n_steps + t + 1] : nxt;
    }
  }
}

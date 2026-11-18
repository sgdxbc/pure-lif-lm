// hw_ref.cpp: the bit-exact integer reference of the FPGA prototype (docs/experiments.md 5.44). It runs a network
// exported by hw_export.py (manifest.txt + tables.bin) token by token, with exactly the integer and float32 arithmetic
// the FPGA kernel will use (hw_export.py's docstring describes it; this file is the definition), and prints the
// validation loss of the windows it is given. The FPGA kernel must reproduce its numbers exactly.
//
//   g++ -O3 -march=native -fopenmp -ffp-contract=off -o hw_ref hw_ref.cpp
//   ./hw_ref DIR WINDOWS.bin N LEN [OUT_LOGP.bin]
// WINDOWS.bin: int32 tokens, N windows of LEN + 1 tokens; each window is read from rest (all potentials and synapses
// zero), and log P(token t+1) is computed for t = 0..LEN-1. OUT_LOGP.bin: those N x LEN values (float64).
// For checking the FPGA kernel (fpga/csim.sh): HWREF_BUS=FILE writes the bus after the last layer at every step
// (N x LEN x d int32), HWREF_GREEDY=FILE the greedy prediction after every step (N x LEN int32: the most likely
// short-list or class neuron, then the most likely word of that class; the first one on ties).
//
// Rounding: float -> integer is floor(x + 0.5f) in float32; an integer product with a 16-bit fraction is
// (x + 2^15) >> 16. float32 sums run in the order written (no fused multiply-add: -ffp-contract=off).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Tensor { std::string dtype; std::vector<int64_t> shape; size_t off; };

struct Tables {
  std::map<std::string, Tensor> t;
  std::map<std::string, std::string> p;
  std::vector<char> blob;
  void load(const std::string& dir) {
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) { fprintf(stderr, "cannot read %s/manifest.txt\n", dir.c_str()); exit(1); }
    std::string line;
    while (std::getline(mf, line)) {
      std::istringstream ss(line);
      std::string kind, name;
      ss >> kind >> name;
      if (kind == "param") { std::string v; ss >> v; p[name] = v; continue; }
      Tensor x; int nd; ss >> x.dtype >> nd;
      x.shape.resize(nd);
      for (auto& s : x.shape) ss >> s;
      ss >> x.off;
      t[name] = x;
    }
    std::ifstream bf(dir + "/tables.bin", std::ios::binary | std::ios::ate);
    size_t n = bf.tellg();
    bf.seekg(0);
    blob.resize(n);
    bf.read(blob.data(), n);
  }
  bool has(const std::string& n) const { return t.count(n) > 0; }
  template <class T> const T* get(const std::string& n, const char* dt) const {
    auto it = t.find(n);
    if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); }
    if (it->second.dtype != dt) { fprintf(stderr, "%s is %s, not %s\n", n.c_str(), it->second.dtype.c_str(), dt); exit(1); }
    return reinterpret_cast<const T*>(blob.data() + it->second.off);
  }
  const int8_t* i8(const std::string& n) const { return get<int8_t>(n, "int8"); }
  const int32_t* i32(const std::string& n) const { return get<int32_t>(n, "int32"); }
  const uint8_t* u8(const std::string& n) const { return get<uint8_t>(n, "uint8"); }
  const float* f32(const std::string& n) const { return get<float>(n, "float32"); }
  int64_t dim(const std::string& n, int i) const { return t.at(n).shape.at(i); }
  int64_t pi(const std::string& n) const { return atoll(p.at(n).c_str()); }
  float pf(const std::string& n) const { return (float)atof(p.at(n).c_str()); }
};

static inline int64_t rnd(float x) { return (int64_t)std::floor(x + 0.5f); }
static inline int64_t rshift16(int64_t x) { return (x + (int64_t(1) << 15)) >> 16; }
static inline int64_t clampi(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }

// activations: A_BITS-bit integers (stored as int16), one scale per group of g channels:
// a = clamp(round(x * ainv[group]), -AMAX, AMAX), AMAX = 2^(A_BITS-1) - 1
typedef int16_t act_t;
static int AMAX = 127;
static void act_from_bus(const int32_t* bus, const float* ainv, int n, int g, act_t* a) {
  for (int c = 0; c < n; c++) a[c] = (act_t)clampi(rnd((float)bus[c] * ainv[c / g]), -AMAX, AMAX);
}
static void act_from_float(const float* x, const float* ainv, int n, int g, act_t* a) {
  for (int c = 0; c < n; c++) a[c] = (act_t)clampi(rnd(x[c] * ainv[c / g]), -AMAX, AMAX);
}

// y[o] = sum over the groups, in order, of (float)(integer sum of the group) * comb[o][group]
static void matvec(const int8_t* w, const float* comb, const act_t* a, int O, int I, int g, float* y) {
  int ng = I / g;
  for (int o = 0; o < O; o++) {
    const int8_t* wr = w + (size_t)o * I;
    float s = 0.f;
    for (int gi = 0; gi < ng; gi++) {
      int32_t acc = 0;
      for (int c = gi * g; c < (gi + 1) * g; c++) acc += (int32_t)wr[c] * (int32_t)a[c];
      s = s + (float)acc * comb[(size_t)o * ng + gi];
    }
    y[o] = s;
  }
}

struct Net {
  const Tables& T;
  int d, H, L, R, h, m, dv, BF, VB, SB, G, N, SIGSTEP, hard, S_short, C, M, vocab, nrows_code;
  std::vector<int> memlayer;   // per layer: index among the plastic-synapse layers, or -1
  int nmem;
  float eps;
  explicit Net(const Tables& t) : T(t) {
    d = T.pi("d"); H = T.pi("H"); L = T.pi("L"); R = T.pi("leak_rank"); h = T.pi("mem_heads"); m = T.pi("mem_keys");
    dv = T.pi("mem_dv"); BF = T.pi("BF"); VB = T.pi("V_BITS"); SB = T.pi("S_BITS"); G = T.pi("G");
    SIGSTEP = T.pi("SIG_STEP"); N = T.pi("SIG_RANGE") << SIGSTEP; hard = T.pi("hard_reset");
    S_short = T.pi("out_short"); C = T.pi("out_classes"); vocab = T.pi("vocab"); eps = T.pf("mem_eps");
    M = C > 0 ? T.pi("class_m") : 0;
    AMAX = (T.p.count("A_BITS") ? (int)T.pi("A_BITS") : 8);
    AMAX = (1 << (AMAX - 1)) - 1;
    nrows_code = T.dim("code", 1);
    memlayer.assign(L, -1);
    nmem = 0;
    std::string ml = T.p.at("mem_layers");
    if (ml != "none") {
      std::stringstream ss(ml);
      std::string tok;
      while (std::getline(ss, tok, ',')) memlayer[atoi(tok.c_str())] = nmem++;
    }
  }
  int gof(int n) const { return n % G == 0 ? G : n; }
};

// one stream's state and scratch
struct Stream {
  std::vector<int32_t> V;      // L x H potentials
  std::vector<int32_t> KQ;     // nmem x 2hm key and query potentials
  std::vector<int32_t> S;      // nmem x h x m x dv plastic synapses
  std::vector<int32_t> bus;
  std::vector<act_t> a, a2;
  std::vector<float> y, y2, z;
  std::vector<uint8_t> spk;
  std::vector<int64_t> r;
  Stream(const Net& n) {
    V.assign((size_t)n.L * n.H, 0);
    KQ.assign((size_t)n.nmem * 2 * n.h * n.m, 0);
    S.assign((size_t)n.nmem * n.h * n.m * n.dv, 0);
    bus.assign(n.d, 0);
    size_t big = std::max<size_t>({(size_t)n.H, (size_t)n.d, (size_t)2 * n.h * n.m, (size_t)n.h * n.dv,
                                   (size_t)(n.S_short + n.C), (size_t)n.vocab, (size_t)n.M});
    a.resize(big); a2.resize(big); y.resize(big); y2.resize(big); z.resize(big); spk.resize(big);
    r.resize((size_t)n.h * n.dv);
  }
};

// LIF update of n neurons. I: float current (threshold folded in); v: potentials (VB-bit, intb integer bits);
// beta: 16-bit fractions per neuron (or nullptr: per-step betas in betas); g1: 1 - gamma (hard reset) or nullptr
// (soft reset). Writes spikes.
static void lif(int n, const float* I, int32_t* v, const int32_t* beta, const int32_t* betas, const int32_t* g1,
                int VB, int intb, uint8_t* spk) {
  int VF = VB - 1 - intb;
  int64_t lo = -(int64_t(1) << (VB - 1)), hi = (int64_t(1) << (VB - 1)) - 1, one = int64_t(1) << VF;
  float sc = std::ldexp(1.0f, VF);
  for (int o = 0; o < n; o++) {
    int64_t Ii = clampi(rnd(I[o] * sc), lo, hi);
    int64_t b = beta ? beta[o] : betas[o];
    int64_t x = clampi(rshift16(b * (int64_t)v[o]) + Ii, lo, hi);
    uint8_t s = x >= one;
    if (s) x = g1 ? clampi(rshift16(x * (int64_t)g1[o]), lo, hi) : clampi(x - one, lo, hi);
    v[o] = (int32_t)x;
    spk[o] = s;
  }
}

static void add_to_bus(int32_t* bus, const float* y, int d) {
  for (int c = 0; c < d; c++)
    bus[c] = (int32_t)clampi((int64_t)bus[c] + rnd(y[c]), INT32_MIN, INT32_MAX);
}

// one token: from the word x to the readout's int8 input (left in st.a)
static void step(const Net& n, Stream& st, int x) {
  const Tables& T = n.T;
  // input rows
  const int8_t* ew = T.i8("emb_w");
  const float* es = T.f32("emb_sc");
  const int32_t* code = T.i32("code");
  int ng = n.d / n.G;
  std::fill(st.bus.begin(), st.bus.end(), 0);
  for (int k = 0; k < n.nrows_code; k++) {
    int row = code[(size_t)x * n.nrows_code + k];
    for (int c = 0; c < n.d; c++)
      st.bus[c] += (int32_t)rnd((float)ew[(size_t)row * n.d + c] * es[(size_t)row * ng + c / n.G]);
  }
  for (int l = 0; l < n.L; l++) {
    std::string L_ = std::to_string(l);
    int j = n.memlayer[l];
    if (j >= 0) {   // plastic synapses in front of population l
      std::string P = "m" + L_;
      int hm = n.h * n.m, HD = n.h * n.dv;
      act_from_bus(st.bus.data(), T.f32(P + "_ain"), n.d, n.gof(n.d), st.a.data());
      matvec(T.i8(P + "_kq_w"), T.f32(P + "_kq_comb"), st.a.data(), 2 * hm, n.d, n.gof(n.d), st.y.data());
      const float* kb = T.f32(P + "_kq_b");
      for (int o = 0; o < 2 * hm; o++) st.y[o] = st.y[o] + kb[o];
      lif(2 * hm, st.y.data(), st.KQ.data() + (size_t)j * 2 * hm, T.i32(P + "_kq_beta"), nullptr, nullptr, n.VB,
          T.pi(P + "_kq_vint"), st.spk.data());
      // values, as int8, then in S units
      matvec(T.i8(P + "_v_w"), T.f32(P + "_v_comb"), st.a.data(), HD, n.d, n.gof(n.d), st.y2.data());
      act_from_float(st.y2.data(), T.f32(P + "_v_ain"), HD, n.gof(HD), st.a2.data());
      const float* toS = T.f32(P + "_v_toS");
      const int32_t* lam = T.i32(P + "_lam");
      int SB = n.SB;
      int64_t slo = -(int64_t(1) << (SB - 1)), shi = (int64_t(1) << (SB - 1)) - 1;
      int SF = SB - 1 - (int)T.pi(P + "_S_int");
      int32_t* S = st.S.data() + (size_t)j * n.h * n.m * n.dv;
      int gv = n.gof(HD);
      for (int hh = 0; hh < n.h; hh++) {
        int32_t* Sh = S + (size_t)hh * n.m * n.dv;
        for (int q = 0; q < n.m * n.dv; q++) Sh[q] = (int32_t)clampi(rshift16((int64_t)lam[hh] * Sh[q]), slo, shi);
        for (int k = 0; k < n.m; k++) {
          if (!st.spk[hh * n.m + k]) continue;            // key neuron k of this group fired: write its row
          for (int ch = 0; ch < n.dv; ch++) {
            int cc = hh * n.dv + ch;
            int64_t vs = rnd((float)st.a2[cc] * toS[cc / gv]);
            Sh[k * n.dv + ch] = (int32_t)clampi(Sh[k * n.dv + ch] + vs, slo, shi);
          }
        }
        for (int ch = 0; ch < n.dv; ch++) st.r[hh * n.dv + ch] = 0;
        for (int k = 0; k < n.m; k++) {
          if (!st.spk[hm + hh * n.m + k]) continue;       // query neuron k fired: read its row
          for (int ch = 0; ch < n.dv; ch++) st.r[hh * n.dv + ch] += Sh[k * n.dv + ch];
        }
      }
      // normalise per group, times the gain
      const float* gain = T.f32(P + "_gain");
      float unit = std::ldexp(1.0f, -SF);
      for (int hh = 0; hh < n.h; hh++) {
        float ms = 0.f;
        for (int ch = 0; ch < n.dv; ch++) {
          float rf = (float)st.r[hh * n.dv + ch] * unit;
          st.z[hh * n.dv + ch] = rf;
          ms = ms + rf * rf;
        }
        ms = ms / (float)n.dv;
        float inv = 1.0f / std::sqrt(ms + n.eps);
        for (int ch = 0; ch < n.dv; ch++) st.z[hh * n.dv + ch] = st.z[hh * n.dv + ch] * inv * gain[hh * n.dv + ch];
      }
      act_from_float(st.z.data(), T.f32(P + "_r_ain"), HD, n.gof(HD), st.a2.data());
      matvec(T.i8(P + "_o_w"), T.f32(P + "_o_comb"), st.a2.data(), n.d, HD, n.gof(HD), st.y.data());
      add_to_bus(st.bus.data(), st.y.data(), n.d);
    }
    // population l
    std::string P = "p" + L_;
    act_from_bus(st.bus.data(), T.f32(P + "_ain"), n.d, n.gof(n.d), st.a.data());
    matvec(T.i8(P + "_in_w"), T.f32(P + "_in_comb"), st.a.data(), n.H, n.d, n.gof(n.d), st.y.data());
    const float* b = T.f32(P + "_in_b");
    for (int o = 0; o < n.H; o++) st.y[o] = st.y[o] + b[o];
    std::vector<int32_t> betas;
    const int32_t* beta = nullptr;
    if (n.R) {   // leak synapses: beta from the sigmoid table
      matvec(T.i8(P + "_li_w"), T.f32(P + "_li_comb"), st.a.data(), n.R, n.d, n.gof(n.d), st.z.data());
      act_from_float(st.z.data(), T.f32(P + "_m_ain"), n.R, n.gof(n.R), st.a2.data());
      matvec(T.i8(P + "_lo_w"), T.f32(P + "_lo_comb"), st.a2.data(), n.H, n.R, n.gof(n.R), st.y2.data());
      const float* bl = T.f32(P + "_blogit");
      const int32_t* sig = T.i32("sig");
      float sc = std::ldexp(1.0f, n.SIGSTEP);
      betas.resize(n.H);
      for (int o = 0; o < n.H; o++) betas[o] = sig[clampi(rnd((bl[o] + st.y2[o]) * sc), -n.N, n.N) + n.N];
    } else {
      beta = T.i32(P + "_beta");
    }
    lif(n.H, st.y.data(), st.V.data() + (size_t)l * n.H, beta, betas.data(), n.hard ? T.i32(P + "_g1") : nullptr,
        n.VB, T.pi(P + "_vint"), st.spk.data());
    // W_out: the outgoing weights of the neurons that fired, summed per group of 128 neurons
    const int8_t* wo = T.i8(P + "_out_w");
    const float* oc = T.f32(P + "_out_comb");
    int go = n.gof(n.H), ngo = n.H / go;
    std::vector<int> fired;
    for (int o = 0; o < n.H; o++) if (st.spk[o]) fired.push_back(o);
    for (int c = 0; c < n.d; c++) {
      float s = 0.f;
      size_t f = 0;
      for (int gi = 0; gi < ngo; gi++) {
        int32_t acc = 0;
        for (; f < fired.size() && fired[f] < (gi + 1) * go; f++) acc += wo[(size_t)c * n.H + fired[f]];
        s = s + (float)acc * oc[(size_t)c * ngo + gi];
      }
      st.y[c] = s;
    }
    add_to_bus(st.bus.data(), st.y.data(), n.d);
  }
  act_from_bus(st.bus.data(), T.f32("h_ain"), n.d, n.gof(n.d), st.a.data());
}

// log P(target) from the readout (two-level or plain), in double
static double logp(const Net& n, Stream& st, int target) {
  const Tables& T = n.T;
  auto lse = [](const float* v, int k, const uint8_t* used) {
    double mx = -1e300;
    for (int i = 0; i < k; i++) if (!used || used[i]) mx = std::max(mx, (double)v[i]);
    double s = 0;
    for (int i = 0; i < k; i++) if (!used || used[i]) s += std::exp((double)v[i] - mx);
    return mx + std::log(s);
  };
  if (n.C == 0) {
    matvec(T.i8("h_w"), T.f32("h_comb"), st.a.data(), n.vocab, n.d, n.gof(n.d), st.y.data());
    return (double)st.y[target] - lse(st.y.data(), n.vocab, nullptr);
  }
  int top = n.S_short + n.C;
  matvec(T.i8("h_cls_w"), T.f32("h_cls_comb"), st.a.data(), top, n.d, n.gof(n.d), st.y.data());
  double ltop = lse(st.y.data(), top, nullptr);
  int cls = T.i32("word_cls")[target], slot = T.i32("slot")[target];
  if (cls < 0) return (double)st.y[slot] - ltop;
  double lc = (double)st.y[n.S_short + cls] - ltop;
  size_t r0 = (size_t)cls * n.M;
  matvec(T.i8("h_word_w") + r0 * n.d, T.f32("h_word_comb") + r0 * (n.d / n.gof(n.d)), st.a.data(), n.M, n.d,
         n.gof(n.d), st.y2.data());
  return lc + (double)st.y2[slot - n.S_short - (int)r0] - lse(st.y2.data(), n.M, T.u8("used") + r0);
}

// greedy prediction from the readout's input (st.a): the most likely short-list or class neuron, then the most likely
// word of that class (first on ties); plain readout: the most likely word
static int greedy(const Net& n, Stream& st, const std::vector<int>& short_word, const std::vector<int>& class_word) {
  const Tables& T = n.T;
  if (n.C == 0) {
    matvec(T.i8("h_w"), T.f32("h_comb"), st.a.data(), n.vocab, n.d, n.gof(n.d), st.y.data());
    return (int)(std::max_element(st.y.begin(), st.y.begin() + n.vocab) - st.y.begin());
  }
  int top = n.S_short + n.C;
  matvec(T.i8("h_cls_w"), T.f32("h_cls_comb"), st.a.data(), top, n.d, n.gof(n.d), st.y.data());
  int best = 0;
  for (int i = 1; i < top; i++) if (st.y[i] > st.y[best]) best = i;
  if (best < n.S_short) return short_word[best];
  int c = best - n.S_short;
  size_t r0 = (size_t)c * n.M;
  matvec(T.i8("h_word_w") + r0 * n.d, T.f32("h_word_comb") + r0 * (n.d / n.gof(n.d)), st.a.data(), n.M, n.d,
         n.gof(n.d), st.y2.data());
  int bp = -1;
  for (int p = 0; p < n.M; p++) {
    if (class_word[r0 + p] < 0) continue;
    if (bp < 0 || st.y2[p] > st.y2[bp]) bp = p;
  }
  return class_word[r0 + bp];
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s DIR WINDOWS.bin N LEN [OUT_LOGP.bin]\n", argv[0]); return 1; }
  Tables T;
  T.load(argv[1]);
  Net net(T);
  int N = atoi(argv[3]), LEN = atoi(argv[4]);
  std::vector<int32_t> tok((size_t)N * (LEN + 1));
  FILE* f = fopen(argv[2], "rb");
  if (!f || fread(tok.data(), 4, tok.size(), f) != tok.size()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
  fclose(f);
  std::vector<double> lp((size_t)N * LEN);
  const char* bus_out = getenv("HWREF_BUS");
  const char* greedy_out = getenv("HWREF_GREEDY");
  std::vector<int32_t> bus_dump(bus_out ? (size_t)N * LEN * net.d : 0), gr(greedy_out ? (size_t)N * LEN : 0);
  std::vector<int> short_word(net.S_short, -1), class_word((size_t)net.C * net.M, -1);
  if (net.C) {
    const int32_t* wc = T.i32("word_cls");
    const int32_t* sl = T.i32("slot");
    for (int v = 0; v < net.vocab; v++) {
      if (wc[v] < 0) short_word[sl[v]] = v;
      else class_word[sl[v] - net.S_short] = v;
    }
  }
#pragma omp parallel for schedule(dynamic, 1)
  for (int w = 0; w < N; w++) {
    Stream st(net);
    for (int t = 0; t < LEN; t++) {
      step(net, st, tok[(size_t)w * (LEN + 1) + t]);
      if (bus_out) std::copy(st.bus.begin(), st.bus.end(), bus_dump.begin() + ((size_t)w * LEN + t) * net.d);
      lp[(size_t)w * LEN + t] = logp(net, st, tok[(size_t)w * (LEN + 1) + t + 1]);
      if (greedy_out) gr[(size_t)w * LEN + t] = greedy(net, st, short_word, class_word);
    }
  }
  if (bus_out) { FILE* o = fopen(bus_out, "wb"); fwrite(bus_dump.data(), 4, bus_dump.size(), o); fclose(o); }
  if (greedy_out) { FILE* o = fopen(greedy_out, "wb"); fwrite(gr.data(), 4, gr.size(), o); fclose(o); }
  double tot = 0;
  for (double v : lp) tot -= v;
  printf("windows %d x %d tokens: loss %.6f\n", N, LEN, tot / lp.size());
  if (argc > 5) {
    FILE* o = fopen(argv[5], "wb");
    fwrite(lp.data(), 8, lp.size(), o);
    fclose(o);
  }
  return 0;
}

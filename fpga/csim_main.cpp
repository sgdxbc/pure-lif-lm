// C simulation driver of the FPGA kernel (fpga/lif_lm.cpp compiled with g++): reads pack.py's files and a token file,
// runs the kernel, writes its outputs. fpga/csim.sh compares them with hw_ref.cpp.
//   csim DIR TOKENS.bin N_STEPS N_PROMPT MODE OUT_TOKENS.bin [OUT_BUS.bin]
// TOKENS.bin: int32, NB streams x N_STEPS tokens.
#include <ap_int.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "model_cfg.h"
#ifndef NB
#define NB 16
#endif
typedef ap_uint<512> word_t;
extern "C" void lif_lm(const word_t* w, const uint32_t* prm, const int32_t* aux, const int32_t* tok_in,
                       int32_t* tok_out, int32_t* dump, int n_steps, int n_prompt, int mode);

template <class T> static std::vector<T> slurp(const std::string& path, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s DIR TOKENS.bin N_STEPS N_PROMPT MODE OUT_TOKENS.bin [OUT_BUS.bin]\n", argv[0]); return 1; }
  std::string dir = argv[1];
  int n_steps = atoi(argv[3]), n_prompt = atoi(argv[4]), mode = atoi(argv[5]);
  std::vector<uint8_t> raw = slurp<uint8_t>(dir + "/weights.bin", (size_t)NWORDS * 64);
  std::vector<word_t> w(NWORDS);
  for (long i = 0; i < NWORDS; i++)
    for (int j = 0; j < 64; j++) w[i].range(8 * j + 7, 8 * j) = raw[(size_t)i * 64 + j];
  std::vector<uint32_t> prm = slurp<uint32_t>(dir + "/params.bin", NPARAMS);
  std::vector<int32_t> aux = slurp<int32_t>(dir + "/aux.bin", NAUX);
  std::vector<int32_t> tok = slurp<int32_t>(argv[2], (size_t)NB * n_steps);
  std::vector<int32_t> out((size_t)NB * n_steps), dump(mode == 1 ? (size_t)NB * n_steps * D : 1);
  lif_lm(w.data(), prm.data(), aux.data(), tok.data(), out.data(), dump.data(), n_steps, n_prompt, mode);
  FILE* o = fopen(argv[6], "wb");
  fwrite(out.data(), 4, out.size(), o);
  fclose(o);
  if (mode == 1 && argc > 7) { o = fopen(argv[7], "wb"); fwrite(dump.data(), 4, dump.size(), o); fclose(o); }
  printf("kernel: %d streams x %d steps done\n", NB, n_steps);
  return 0;
}

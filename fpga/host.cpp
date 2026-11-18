// Host program of the FPGA kernel (XRT native API; docs/experiments.md 5.44). It loads pack.py's files into the board's
// HBM, runs the kernel, and measures generation speed and board power:
//   host XCLBIN PACK_DIR TOKENS.bin PROMPT STEPS1 STEPS2 [OUT_TOKENS.bin]
// TOKENS.bin: int32, NB streams x STEPS2 tokens (only the first PROMPT of each stream are read). The kernel is run in
// generation mode twice, for STEPS1 and STEPS2 steps; the difference of the two times over STEPS2 - STEPS1 is the
// time of one step (one token in each of the NB streams) without the fixed cost of a call (loading the small tables,
// clearing the state). Board power is read every 100 ms during the longer run (XRT's electrical report: the card's
// own sensors, the whole card), and energy per token = average power x time per step / NB.
// Check mode: host XCLBIN PACK_DIR TOKENS.bin check STEPS OUT_TOKENS.bin OUT_BUS.bin runs the kernel with teacher
// forcing (mode 1) for comparison with hw_ref.cpp (as fpga/csim.sh does in C simulation).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "model_cfg.h"
#ifndef NB
#define NB 16
#endif

template <class T> static std::vector<T> slurp(const std::string& path, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
  fclose(f);
  return v;
}

// the card's power in watts from XRT's electrical report (JSON text): "power_consumption_watts"
static double board_watts(const xrt::device& dev) {
  std::string j = dev.get_info<xrt::info::device::electrical>();
  size_t k = j.find("\"power_consumption_watts\"");
  if (k == std::string::npos) return -1;
  k = j.find(':', k);
  size_t q = j.find('"', k);
  return atof(j.c_str() + (q != std::string::npos && q < k + 4 ? q + 1 : k + 1));
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s XCLBIN PACK_DIR TOKENS.bin PROMPT STEPS1 STEPS2 [OUT_TOKENS.bin]\n"
                    "       %s XCLBIN PACK_DIR TOKENS.bin check STEPS OUT_TOKENS.bin OUT_BUS.bin\n", argv[0], argv[0]);
    return 1;
  }
  std::string dir = argv[2];
  bool check = std::string(argv[4]) == "check";
  int steps2 = check ? atoi(argv[5]) : atoi(argv[6]);
  auto dev = xrt::device(0);
  auto uuid = dev.load_xclbin(argv[1]);
  auto krn = xrt::kernel(dev, uuid, "lif_lm");
  auto load = [&](int arg, const void* src, size_t bytes) {
    xrt::bo bo(dev, bytes, krn.group_id(arg));
    if (src) { bo.write(src); bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    return bo;
  };
  std::vector<uint8_t> w = slurp<uint8_t>(dir + "/weights.bin", (size_t)NWORDS * 64);
  std::vector<uint32_t> prm = slurp<uint32_t>(dir + "/params.bin", NPARAMS);
  std::vector<int32_t> aux = slurp<int32_t>(dir + "/aux.bin", NAUX);
  std::vector<int32_t> tok = slurp<int32_t>(argv[3], (size_t)NB * steps2);
  auto bw = load(0, w.data(), w.size()), bp = load(1, prm.data(), prm.size() * 4), ba = load(2, aux.data(), aux.size() * 4);
  auto bt = load(3, tok.data(), tok.size() * 4), bo = load(4, nullptr, tok.size() * 4);
  auto bd = load(5, nullptr, check ? (size_t)NB * steps2 * D * 4 : 64);
  std::vector<int32_t> out((size_t)NB * steps2);
  if (check) {
    krn(bw, bp, ba, bt, bo, bd, steps2, steps2, 1).wait();
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo.read(out.data());
    std::vector<int32_t> dump((size_t)NB * steps2 * D);
    bd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bd.read(dump.data());
    FILE* f = fopen(argv[6], "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
    f = fopen(argv[7], "wb"); fwrite(dump.data(), 4, dump.size(), f); fclose(f);
    printf("check run: %d streams x %d steps written\n", NB, steps2);
    return 0;
  }
  int prompt = atoi(argv[4]), steps1 = atoi(argv[5]);
  auto timed = [&](int steps, std::vector<double>* watts) {
    std::atomic<bool> stop(false);
    std::thread mon;
    if (watts) mon = std::thread([&] {
      while (!stop) { watts->push_back(board_watts(dev)); std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
    });
    auto t0 = std::chrono::steady_clock::now();
    krn(bw, bp, ba, bt, bo, bd, steps, prompt, 0).wait();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop = true;
    if (mon.joinable()) mon.join();
    return s;
  };
  double idle = board_watts(dev);
  timed(steps1, nullptr);   // warm-up
  double t1 = timed(steps1, nullptr);
  std::vector<double> watts;
  double t2 = timed(steps2, &watts);
  double avg = 0;
  int n = 0;
  for (double x : watts) if (x > 0) { avg += x; n++; }
  avg = n ? avg / n : -1;
  double per_step = (t2 - t1) / (steps2 - steps1);
  printf("NB %d streams: %.3f ms per step (one token per stream), %.0f tokens/s, %.1f tokens/s per stream\n", NB,
         per_step * 1e3, NB / per_step, 1.0 / per_step);
  printf("board power: idle %.1f W, during generation %.1f W (%d readings): %.3f mJ per token\n", idle, avg, n,
         avg * per_step / NB * 1e3);
  printf("calls: %d steps %.3f s, %d steps %.3f s\n", steps1, t1, steps2, t2);
  if (argc > 7) {
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo.read(out.data());
    FILE* f = fopen(argv[7], "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
  }
  return 0;
}

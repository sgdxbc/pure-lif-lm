# Project notes for Claude

## Goal and principle (set by the project owner)
- Goal: keep the network purely made of spiking (LIF) neurons, and improve model quality by any means
  that respects that, with the aim of matching a Transformer.
- "Spiking networks are small" is a stereotype, not a constraint of this project. Scaling up, even beyond the
  size of the Transformer baseline, is a legitimate and possibly necessary choice.
- Compare against the Transformer mainly at equal training data. Record parameter count, speed and state size,
  and use them for compute planning, but do not reject a direction because it is bigger or slower.
- "Pure spiking" as currently agreed: all state and nonlinearity live in LIF neurons; neurons interact only through
  spikes via synapses (plain, low-rank factorised, delayed, or leak-modulating synapses are all fine). No attention,
  no normalisation layers, no non-spiking nonlinear units. Two-compartment neurons are not used for now.

## Working conventions
- Experiments run on a 4-core CPU with the C++/OpenMP kernels in lif_kernel.cpp; queue scripts are phase*.sh.
- Wait for a queue script with an exact match, e.g. `pgrep -x -f "/bin/bash ./phase6.sh"`; plain `pgrep -f`
  also matches the shell that launched it.
- Checkpoints (ckpt.pt) are not committed (over GitHub's size limit); logs, args and probe JSON files are.
- The README (in Chinese) is the record of results; update it with each phase.

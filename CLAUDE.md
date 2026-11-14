# Project notes for Claude

## Goal and principle (set by the project owner)
- Goal: keep the network purely made of spiking (LIF) neurons, and improve model quality by any means
  that respects that, with the aim of matching a Transformer's *inference* performance.
- "Spiking networks are small" is a stereotype, not a constraint of this project. Scaling up, even beyond the
  size of the Transformer baseline, is a legitimate and possibly necessary choice.
- "Matching a Transformer" is a rough goal: the end result only has to be reasonably interpretable as reaching
  Transformer level in some sense; no particular Transformer or metric is fixed in advance.
- What counts is how the trained spiking network performs at inference (validation loss, recall, samples)
  compared with the Transformer. How it was trained does not matter: amount of training data, training length,
  and distillation from a Transformer (or any other teacher) are all allowed; just record them.
- Record parameter count, speed and state size, and use them for compute planning, but do not reject a direction
  because it is bigger or slower.
- "Pure spiking" as currently agreed: all state and nonlinearity live in LIF neurons; neurons interact only through
  spikes via synapses (plain, low-rank factorised, delayed, or leak-modulating synapses are all fine). No attention,
  no normalisation layers, no non-spiking nonlinear units. Two-compartment neurons are not used for now.

## Working conventions
- Experiments run on a 4-core CPU with the C++/OpenMP kernels in lif_kernel.cpp; queue scripts are phase*.sh.
- Wait for a queue script with an exact match, e.g. `pgrep -x -f "/bin/bash ./phase6.sh"`; plain `pgrep -f`
  also matches the shell that launched it.
- Background jobs only survive while this cloud session has a background task of its own running: when a turn
  ends with none, the container can be stopped and restarted, killing every job (this happened on 2026-11-14).
  Keep a waiter (run_in_background) armed for as long as experiments run, and re-arm it when it times out.
  Restarted jobs resume from the last ckpt.pt with --resume (saved at every eval).
- Checkpoints (ckpt.pt) are not committed (over GitHub's size limit); logs, args and probe JSON files are.
- The README (in Chinese) is the record of results; update it with each phase.

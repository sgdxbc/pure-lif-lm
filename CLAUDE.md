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

## Running on the GPU cluster (added 2026-11-14)
- The GPUs are on a university SLURM cluster that the owner reaches through a Cisco VPN and SSH. The login node
  has internet access; compute nodes may not (check before relying on it). The cloud session cannot reach the
  cluster (its egress gateway blocks VPN tunnels and SSH), so GPU work is done by a Claude session started on the
  login node, and the two sessions coordinate only through this git repository.
- Storage (from the cluster's documentation): the home directory is the large-capacity storage, on a network
  file system shared by all nodes; /tmp is fast node-local storage. Keep the clone, data/, the Python environment
  and caches in home. Training reads small random slices of the memory-mapped data/*.bin, which the node's page
  cache usually serves well from the network file system; copy them to /tmp inside a job only if data loading
  turns out to be a bottleneck. /tmp on a compute node is not visible elsewhere and may be wiped after the job.
- Claude Code's own binary lives in the shared home too: do not run Claude sessions on two login nodes at once,
  since an auto-update started on one can delete the binary the other is running.
- Never train on the login node: downloads and git there, everything else in sbatch/srun jobs. gpu_setup.sh
  assumes it runs where a GPU is visible; download/tokenise the data first, then run its checks in a GPU job.
- Order of work once the GPU runs: (1) environment + gpu_setup.sh (kernel check against the reference, speed at
  12M-117M); (2) the training-method screen in phase8.sh with DEVICE=cuda (tell the owner, so the CPU copy of
  phase 8 in the cloud session can be stopped); (3) a stronger Transformer teacher for distillation (bigger
  and/or longer than tf_full); (4) larger spiking models with leak synapses, on more of the 552M training tokens
  (all runs so far used 115M).
- Results: commit logs, args and probe JSON as usual; pull before editing README, since the cloud session may
  still be writing up phase 7 (25.25M + leak64 on CPU, finishing around 2026-11-14 23:00 UTC).

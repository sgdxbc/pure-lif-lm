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
- Login node limits: 1 GB of virtual memory per process (torch cannot even be imported) and a 10 MB /tmp quota per
  user. Use it only for git, downloads, sbatch/squeue and small torch-free scripts (e.g. slurm/args_of.py);
  tokenising runs in a CPU job (slurm/prepare.sbatch). Every job sources slurm/env.sh (venv ~/venv-lif, caches).
- Partitions, GPU types and per-user limits are in runs/gpu_setup_report.md: `gpu` allows 3 h per job, `gpu-long`
  3 days; at most 8 GPUs at once, of which one h100-96 and one a100-80. Do not use the old `gpu:nv` cards (no bf16).
  Measured on one H100: 25M + leak64 trains at ~665k tok/s (CPU: ~10k), 117M at ~307k tok/s.
- Conventions: one run = one job of slurm/train.sbatch (train, then full-validation eval and both memory probes
  into the run directory; PROBES=0 skips them). A wave of runs is submitted by a gpuN.sh script at the repo
  root, with run directories runs/gN_*. `slurm/args_of.py runs/X` prints the flags that repeat run X exactly.
- Division of work: the cluster session submits and watches jobs, fixes job-level problems, and commits results
  (runs/gN_* without ckpt.pt, slurm/logs/) with `git pull --rebase` before each push. The cloud session plans the
  waves, writes the analysis into README and edits the shared code and docs. The cloud session can send messages
  to the cluster session and read its transcript; the cluster session cannot message the cloud session, so it
  reports through commits (say in the commit message if something needs a decision).
- In the cloud session, each send_message call asked the owner for approval until the allow rules in
  .claude/settings.json on main (added by the owner on 2026-11-14) were loaded; that file is read when a session
  starts. Keep messages few and hand work over through commits where possible.

# Project notes for Claude

## Goal and principle (set by the project owner)
- Vision (owner, 2026-11-19; README section 0 has it in full): show that deploying on FPGAs / dataflow hardware is a
  strong competitor to GPUs in some scenario (no unacceptable weakness, a clear highlight), so that someone deploying
  from scratch seriously considers it, especially when GPUs are scarce. Language models because of their influence.
  Spiking networks because mainstream networks favour GPUs, and because the LIF neuron offers an ultra-light
  recurrence + nonlinearity (memory = its own potential, one decay and one addition per step, no recurrent matrix;
  nonlinearity = a threshold, 0/1 output, so reading it takes only additions). Change the public image of spiking
  networks (small, ultra-sparse, neuromorphic chips): define their own scaling behaviour and make powerful hardware
  their home ground. Compute is limited, so establish trends clearly for better-resourced researchers.
  Judge every candidate by quality and target-hardware cost together, against an equal-quality Transformer on its
  best hardware; sparsity is a result, not a goal.
- Priorities (owner, 2026-11-20, after the second self-audit): scaling laws are not urgent; first find a competitive
  structure. Stay with spiking networks and keep looking for structures where the spikes themselves give an
  advantage (on the same FPGA at equal quality), rather than settling for "a small fixed-state network on an FPGA".
- Stage goals (revised by the owner on 2026-11-15, README section 0):
  1. Quality: match classic small language models (GPT-2 124M-355M, Pythia 160M-410M class) on general text, and be
     basically usable on some tasks.
  2. Scaling: show that quality keeps improving as more training compute is put in (loss against compute, next to a
     Transformer trained on the same data). Real large-scale training is left for later.
  3. Hardware: on the target hardware (FPGA, dataflow accelerators) show some concrete advantage, ideally not tied
     to a narrow scenario, and not fall clearly behind where there is no advantage. Compare against the
     Transformer's corresponding numbers on the hardware where it performs best (a GPU is fine, measured with a mature
     inference stack); do not force a Transformer onto non-GPU hardware it is ill-suited to (owner, 2026-11-15).
  4. Structure: pure LIF is no longer a hard requirement (running on the target hardware is). Keep LIF neurons as
     fully as possible, keep the structure simple, and leave room for extensions and wider deployment.
- Considered and set aside for now (2026-11-15): matching a current frontier 7-9B model. Training at that level
  from scratch is far beyond this compute; the route for later is converting an open model (Qwen3.5-9B, itself a
  3:1 Gated DeltaNet / attention hybrid) and distilling, as SpikingBrain and RADLADS did.
- Long-term direction (the earlier goal): improve model quality by any means, with the aim of matching a
  Transformer's *inference* performance.
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
  no normalisation layers, no non-spiking nonlinear units. Two-compartment neurons are not used for now. This is
  still the default, but since 2026-11-15 deviations are allowed when they serve the goals: record and justify
  each, and prefer ones that keep the hardware benefits (spikes or small integers between neurons, fixed-size state).
- Hardware targets (agreed 2026-11-15, README section 0): FPGAs and dataflow accelerators; neuromorphic chips are
  not targeted for now (real-valued bus, far more synapses per neuron than they are built for). Long-term aim:
  parity with a Transformer of the same parameter count; claimable sooner: per-token cost with many concurrent
  streams and long text, once quality on long text is shown. Report the hardware cost (hw_cost.py for spiking
  networks, docs/experiments.md 5.20; experiments/fpga_compare.py for model_tf.py networks, 5.51) with every wave, and
  compare networks on the same FPGA at equal quality (a loss against FPGA cost plot), not one network against itself.
- The owner has left it to the cloud session to decide how to proceed (2026-11-15): plan and launch waves without
  asking first, and report results and decisions afterwards. Confirmed again for the new stage goals: keep going on
  your own until they are reached, exploring any improvement freely. The spiking network may have more parameters
  than the Transformer it is compared with, as long as it reaches the goals. The owner suggests (without firm
  grounds) trying improvements to the training method first.

## Repository layout (tidied 2026-11-15)
- README.md (Chinese): goals (section 0), current status, network structure, file guide. Keep it short and current.
- docs/experiments.md (Chinese): the chronological lab record, sections 3-5 with the old numbering (5.1-5.30...),
  so "5.28" in code comments, commit messages and scripts means a section of this file. New results go here as new
  5.x sections; update README's "当前进展" when the overall picture changes.
- docs/running.md (Chinese): how to prepare data, train, evaluate and estimate hardware cost; every option.
- experiments/: one launch script per wave (gpuN.sh; the early CPU queue scripts phase*.sh, ablate*.sh), with an index
  in experiments/README.md (script -> section). Scripts cd to the repo root themselves.
- runs/: one directory per run (runs/README.md explains prefixes and files). slurm/: job scripts, cluster notes.
- Python modules stay at the repo root (the slurm job scripts call them by path).
- Keep the code small: an option that was tried and did not help is deleted rather than kept (its results stay in
  docs/experiments.md; the code stays in git history). Removed so far (2026-11-15): rec_rank, delay_k, spike_bus,
  leak_gate (last in commit 490e9e6); 2026-11-16: mem_rule "delta" (commit c138343), mem_clock (commit c0868f9), mem_count_norm (commit bf7fed7);
  2026-11-19: in_rank, mem_src "spikes", spike_soft with soft_hold/soft_end (commit 5fcacfd).
  model.Config.from_dict still loads checkpoints that saved them switched off. Check a new memory mechanism on the CPU
  first with experiments/recall_toy.py (synthetic key-value recall) or experiments/lag_toy.py (copy the token k
  positions back), then on TinyStories: the toys show whether a mechanism can work, not whether it helps on text.
  Old diagnostics (neuron_drift.py, leak_stats.py, leak_ablate.py, analyze.py) are not maintained.
- Commit and push straight to main (owner, 2026-11-17). exp/reset-recurrence-scale is frozen at d6f5356: do not push
  to it or delete it.

## Working conventions
- Early experiments ran on a 4-core CPU with the C++/OpenMP kernels in lif_kernel.cpp (queue scripts
  experiments/phase*.sh); since 2026-11-14 the GPU cluster (below) runs everything.
- Background jobs only survive while this cloud session has a background task of its own running: when a turn
  ends with none, the container can be stopped and restarted, killing every job (this happened on 2026-11-14).
  Keep a waiter (run_in_background) armed for as long as experiments run, and re-arm it when it times out.
  Restarted jobs resume from the last ckpt.pt with --resume (saved at every eval).
- Checkpoints (ckpt.pt) are not committed (over GitHub's size limit); logs, args and probe JSON files are.
- docs/experiments.md (in Chinese) is the record of results; update it with each wave.

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
  into the run directory; PROBES=0 skips them). A wave of runs is submitted by an experiments/gpuN.sh script
  (`bash experiments/gpuN.sh`), with run directories runs/gN_*. `slurm/args_of.py runs/X` prints the flags that repeat run X exactly.
- Division of work: the cluster session submits and watches jobs, fixes job-level problems, and commits results
  (runs/gN_* without ckpt.pt, slurm/logs/) with `git pull --rebase` before each push. The cloud session plans the
  waves, writes the analysis into docs/experiments.md (and README's status) and edits the shared code and docs. The cloud session can send messages
  to the cluster session and read its transcript; the cluster session cannot message the cloud session, so it
  reports through commits (say in the commit message if something needs a decision).
- Messaging: since the cloud session restarted with the allow rules in .claude/settings.json (main branch), its
  send_message calls go through without the owner's approval. Scheduled wake-ups (send_later / triggers) are
  refused for the cloud session (it is limited to the owner's trusted devices), so keep a background task
  running instead of relying on them.
- The cluster session (`claude remote-control`) runs from a batch job, `sbatch slurm/claude_session.sbatch`
  (partition `long`, 3-day limit; since 2026-11-15, job 1143785), in tmux on a compute node, with no process left
  on the login node. Start, look at its screen, and renew as described in that script and slurm/notes.txt; only
  one Claude may run in the clone at a time (a second remote-control in the same directory exits at once).
- Why not an interactive `srun --pty`: its srun client stays on the login node relaying Claude's terminal output,
  and the login node SIGKILLs processes after 5 min of CPU time (or when the owner's login session ends); slurmstepd
  then kills Claude too, while the allocation stays (job 1143176 died that way after 29.5 min; nothing was
  committed for 4.5 h).
- After a restart, the new remote-control process reattaches the old session, with its context, once a message
  is waiting for it: if only an empty session appears, send the old session (session_017FuF8wVUfCZdqUSmMn5iLC,
  "xlogin1-memoized-panda") a message. `claude remote-control -c` does not work here. If the cluster goes quiet for
  much longer than a wave should take, check the session with get_session (connection_status) and ask the owner
  to restart it.

## Working rules (from the 2026-11-19 self-audit, docs/audits.md)
- Everything the owner reads is in Chinese, one-line status notes included, in plain words (no run names, EMA, KL or
  accum unless explained). While the owner is away, message only for a decision or a notable result; otherwise keep
  the notes in docs/experiments.md and give one plain summary when the owner returns.
- Results: decide on final numbers, or once a paired difference has been stable over at least three evaluations;
  mark single evaluations as 初步 and keep them out of bold headlines. Seed noise rests on one pair of runs
  (g35_base, _s1) until the three seeds of g40_laspk (wave 44) finish, so treat differences under about 0.02 as
  possibly noise. Trend claims only from paired runs with the same teacher and settings; compare variants only on the
  same base network. When the owner states a hope or a vision, list the evidence for it and against it separately.
- Before asking the owner to do anything, check that it can work with a probe that needs nothing from them.
- Waiting: the Bash tool's run_in_background has a 30-minute default limit, so always pass timeout >= the alarm's
  minutes + 5 (in ms). A container restart kills every background task: re-arm the alarm. Scheduled wake-ups
  (send_later) are refused for this session and the cluster session cannot message it: only the alarm wakes it.
  Keep the expected finish time of every running job; if one is over 30 minutes late with no report, read the
  cluster session's history (list_events) and ask it. The tool stops background commands after 2 hours whatever the
  timeout, so keep alarms at 115 minutes or less.
- At most one commit per wake cycle (commits are also the cluster's wake-up signal); check every number before
  committing, and commit judgement rules before (never together with) the results they judge. Before a wave, train
  every new combination for a few hundred steps on a small CPU network (gradients, spikes, loss), check the launch
  script's flags against the base run's args.json, give each run a dry run that includes one evaluation on its card
  type (163M spiking networks: accum 2 on 80-96 GB cards, 4 on 40-47 GB; accum 1 ran out of memory), and keep a next
  job queued behind every slot.
- Before ending a turn, read the owner-facing text once for English sentences and run names.

## Periodic self-audits (owner, 2026-11-19)
- After about every 24 hours of work or every 3 waves, whichever comes first, start three fresh subagents in
  parallel: (1) code and repository (recent diffs for bugs; docs, indexes and scripts against the code), (2) research
  (conclusions against the evidence and against the vision in README section 0; confounds, overclaims, where GPU
  time goes, what is neglected), (3) process (the session transcript under /root/.claude/projects/: communication,
  delays, mistakes, the rules above). Brief them with the files to read, not with your own conclusions.
- Check what they report before acting on it, record the findings and what was done in docs/audits.md, and give
  the owner a Chinese summary that separates what was fixed from what needs the owner's decision.

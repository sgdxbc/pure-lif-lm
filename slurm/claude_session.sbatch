#!/bin/bash
#SBATCH -p long -c 2 --mem=8G -t 3-00:00:00 -J claude-session -o slurm/logs/%x-%j.out
# Runs the cluster Claude session (claude remote-control) on a compute node, leaving no process on the login node.
# Why: with `srun --pty bash` the srun client stays on the login node, relaying all of Claude's terminal output, and
# the login node kills any process that has used 5 minutes of CPU time; when srun dies, slurmstepd kills the step and
# Claude with it (job 1143176: step SIGKILLed after 29.5 min while the allocation stayed; see slurm/notes.txt).
# Usage, on the login node from the repository root, after the previous Claude session has been stopped (never run
# two at once):
#   sbatch slurm/claude_session.sbatch
# To look at its terminal (only briefly: the attach is itself an srun on the login node), detach with Ctrl-b d:
#   srun --jobid=<jobid> --overlap --pty tmux attach -t cc
# The job ends when Claude exits. Renew before the 3-day limit: stop it (scancel <jobid>), then submit again.
cd "${SLURM_SUBMIT_DIR:-$HOME/pure-lif-lm}"
echo "$(date) $(hostname) starting claude remote-control"
if command -v tmux >/dev/null; then
  tmux new -d -s cc "claude remote-control"
  sleep 5
  while tmux has-session -t cc 2>/dev/null; do sleep 60; done
else  # no tmux on this node: give Claude a pseudo-terminal with script(1); its screen goes to the job log
  script -qfc "claude remote-control" /dev/null
fi
echo "$(date) claude remote-control exited"

#!/bin/bash
#SBATCH -p long -c 2 --mem=8G -t 3-00:00:00 -J claude-session -o slurm/logs/%x-%j.out
# Runs the cluster Claude session (claude remote-control) on a compute node, leaving no process on the login node.
# Why: with `srun --pty bash` the srun client stays on the login node, relaying all of Claude's terminal output, and
# the login node kills any process that has used 5 minutes of CPU time; when srun dies, slurmstepd kills the step and
# Claude with it (job 1143176: step SIGKILLed after 29.5 min while the allocation stayed; see slurm/notes.txt).
# Usage, on the login node from the repository root, after the previous Claude session has been stopped (a second
# `claude remote-control` in the same directory exits at once, as job 1143780 did):
#   sbatch slurm/claude_session.sbatch
# To look at its terminal (only briefly: the attach is itself an srun on the login node), detach with Ctrl-b d:
#   srun --jobid=<jobid> --overlap --pty tmux -L claude attach -t cc
# The job ends when Claude exits, and its log then shows Claude's last screen. Renew before the 3-day limit:
# stop it (scancel <jobid>), then submit again.
export PATH="$HOME/.local/bin:$PATH"   # where claude is installed; batch shells do not read ~/.bashrc
cd "${SLURM_SUBMIT_DIR:-$HOME/pure-lif-lm}"
echo "$(date) $(hostname) starting $(command -v claude) remote-control"
if command -v tmux >/dev/null; then
  conf=$(mktemp)
  echo "set -g remain-on-exit on" > "$conf"   # keep the pane after Claude exits, to read its last screen
  tmux -L claude -f "$conf" new -d -s cc -x 160 -y 50 "claude remote-control"
  while [ "$(tmux -L claude display -p -t cc '#{pane_dead}' 2>/dev/null)" = 0 ]; do sleep 30; done
  echo "$(date) claude remote-control exited with status $(tmux -L claude display -p -t cc '#{pane_dead_status}' 2>/dev/null); last screen:"
  tmux -L claude capture-pane -p -t cc -S -200 2>/dev/null
  tmux -L claude kill-server 2>/dev/null
  rm -f "$conf"
else  # no tmux on this node: give Claude a pseudo-terminal with script(1); its screen goes to the job log
  script -qfc "claude remote-control" /dev/null
  echo "$(date) claude remote-control exited"
fi

#!/bin/bash
#SBATCH -p long -c 2 --mem=8G -t 3-00:00:00 -J claude-session -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@900
# Runs the cluster Claude session (claude remote-control) on a compute node, leaving no process on the login node.
# Why: with `srun --pty bash` the srun client stays on the login node, relaying all of Claude's terminal output, and
# the login node kills any process that has used 5 minutes of CPU time; when srun dies, slurmstepd kills the step and
# Claude with it (job 1143176: step SIGKILLed after 29.5 min while the allocation stayed; see slurm/notes.txt).
# Usage, on the login node from the repository root, after the previous Claude session has been stopped (a second
# `claude remote-control` in the same directory exits at once, as job 1143780 did):
#   sbatch slurm/claude_session.sbatch
# To look at its terminal (only briefly: the attach is itself an srun on the login node), detach with Ctrl-b d:
#   srun --jobid=<jobid> --overlap --pty tmux -L claude attach -t cc
#
# Renewal, by itself and with nothing on the login node (compute nodes can run sbatch): 15 minutes before the 3-day
# limit Slurm sends this script SIGTERM (--signal above). It then queues its successor (this script again, starting
# once this job has ended), stops Claude and ends. The successor reattaches Claude to the same session
# (`claude remote-control --session-id`, the ID read from slurm/claude_session.id, which is not committed), so the
# conversation goes on; only the background tasks of the old Claude process are gone and have to be started again.
# If Claude exits by itself after running at least an hour, the successor is queued too; after a shorter run it is
# not (a fault that made Claude exit at once would otherwise start one job after another).
# If the reattach exits within two minutes, a new session is started instead (it shows up in the Claude app).
# Without slurm/claude_session.id, a new session. To stop the chain: touch slurm/claude_session.stop, and scancel a
# successor that is already queued (squeue -u $USER -n claude-session).
export PATH="$HOME/.local/bin:$PATH"   # where claude is installed; batch shells do not read ~/.bashrc
cd "${SLURM_SUBMIT_DIR:-$HOME/pure-lif-lm}"
idf=slurm/claude_session.id
cmd="claude remote-control"
[ -s $idf ] && cmd="claude remote-control --session-id $(tr -d '[:space:]' < $idf)"
start=$(date +%s); renew=0
trap 'echo "$(date) the time limit is near: renewing"; renew=1' TERM
queue_next() {
  if [ -e slurm/claude_session.stop ]; then echo "slurm/claude_session.stop exists: no successor"; return; fi
  local next
  next=$(sbatch --parsable -p long --dependency=afterany:$SLURM_JOB_ID slurm/claude_session.sbatch) \
    && echo "$(date) queued the next session job: $next (starts when job $SLURM_JOB_ID has ended)"
}
pane_alive() { [ "$(tmux -L claude display -p -t cc '#{pane_dead}' 2>/dev/null)" = 0 ]; }
echo "$(date) $(hostname) job $SLURM_JOB_ID starting: $cmd ($(command -v claude))"
if command -v tmux >/dev/null; then
  conf=$(mktemp)
  echo "set -g remain-on-exit on" > "$conf"   # keep the pane after Claude exits, to read its last screen
  tmux -L claude -f "$conf" new -d -s cc -x 160 -y 50 "$cmd"
  if [ "$cmd" != "claude remote-control" ]; then   # did the reattach work?
    for _ in 1 2 3 4; do pane_alive && sleep 30; done
    if ! pane_alive; then
      echo "$(date) the reattach exited at once; its screen:"; tmux -L claude capture-pane -p -t cc -S -50 2>/dev/null
      tmux -L claude kill-server 2>/dev/null; sleep 2
      echo "$(date) starting a new session instead"; cmd="claude remote-control"
      tmux -L claude -f "$conf" new -d -s cc -x 160 -y 50 "$cmd"
    fi
  fi
  while pane_alive && [ $renew = 0 ]; do sleep 30 & wait $!; done
  if [ $renew = 1 ]; then
    queue_next
    for _ in 1 2; do pane_alive && tmux -L claude send-keys -t cc C-c && sleep 10; done   # let Claude stop
  elif [ $(( $(date +%s) - start )) -ge 3600 ]; then
    echo "$(date) claude remote-control exited after $(( ($(date +%s) - start) / 60 )) minutes"; queue_next
  else
    echo "$(date) claude remote-control exited within an hour of starting: no successor"
  fi
  echo "$(date) last screen (exit status $(tmux -L claude display -p -t cc '#{pane_dead_status}' 2>/dev/null)):"
  tmux -L claude capture-pane -p -t cc -S -200 2>/dev/null
  tmux -L claude kill-server 2>/dev/null
  rm -f "$conf"
else  # no tmux on this node: give Claude a pseudo-terminal with script(1); its screen goes to the job log
  script -qfc "$cmd" /dev/null & pid=$!
  while kill -0 $pid 2>/dev/null && [ $renew = 0 ]; do sleep 30 & wait $!; done
  if [ $renew = 1 ]; then queue_next; kill -INT $pid 2>/dev/null; sleep 10; kill $pid 2>/dev/null
  elif [ $(( $(date +%s) - start )) -ge 3600 ]; then queue_next
  fi
  echo "$(date) claude remote-control exited"
fi

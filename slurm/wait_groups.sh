#!/bin/bash
# Wait until one of the given run-name groups has no job left in the queue, or a job log reports a failure.
# Usage: slurm/wait_groups.sh PREFIX... (e.g. "g1_tf_full|g1_p2_large|g1_p7" "g1_117m" "g1_s_")
# Prints which group finished (groups already reported are listed in slurm/.reported).
cd "$(dirname "$0")/.."
touch slurm/.reported
while true; do
  q=$(squeue -h -u $USER -o %j)
  for g in "$@"; do
    grep -qxF "$g" slurm/.reported && continue
    if ! echo "$q" | grep -qE "^($g)"; then echo "$g" >> slurm/.reported; echo "GROUP DONE: $g"; exit 0; fi
  done
  # RELAY=1: runs continue by themselves after the 3 h limit (slurm/train.sbatch RELAY), so a time-limit stop is not a
  # failure; their real failures leave runs/NAME/FAILED and "failed with exit code" in the job log.
  pat="failed with exit code\|Traceback"; [ "${RELAY:-0}" = "1" ] || pat="$pat\|DUE TO TIME LIMIT"
  f=$(grep -l "$pat" slurm/logs/g[0-9]*_*.out 2>/dev/null | grep -vxFf slurm/.reported)
  if [ -n "$f" ]; then echo "$f" >> slurm/.reported; echo "FAILED: $f"; exit 0; fi
  sleep 60
done

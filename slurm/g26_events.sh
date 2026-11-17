#!/bin/bash
# Waits for the next wave-26 event and prints it: a new kept checkpoint (ckpt_step*.pt, once its size is stable),
# runs/g26_tf163/done, or a FAILED file in either chain. Events already reported are listed in slurm/.g26seen.
cd "$(dirname "$0")/.."
touch slurm/.g26seen
while true; do
  for f in runs/g26_snn212/ckpt_step*.pt runs/g26_tf163/ckpt_step*.pt runs/g26_tf163/done runs/g26_*/FAILED; do
    [ -e "$f" ] || continue
    grep -qxF "$f" slurm/.g26seen && continue
    if [[ $f == *.pt ]]; then s1=$(stat -c %s "$f"); sleep 60; [ "$(stat -c %s "$f")" = "$s1" ] || continue; fi
    echo "$f" >> slurm/.g26seen; echo "EVENT: $f"; ls -l "$f"; exit 0
  done
  sleep 120
done

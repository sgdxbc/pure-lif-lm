# Whole-text query witness replay

## Windows and groups

Use the g44_all3 EMA at step 20,000 and the same four query heads and FP32
settings as item 7. Draw 1,024 windows without replacement from the aligned,
nonoverlapping 1,025-token blocks of FineWeb validation (seed 5665). Each window
starts with fresh state and contributes 1,024 next-token predictions. Reserve
32 further randomly drawn windows for timing. The target population is the
predictions in this window protocol.

Assign each position to one group, in this order.

1. Item-6-eligible names. Use its within-document unique-source and wrong-name
   rules, its four teacher heads (raw g13_tf163 weights; it has no EMA), and its teacher share and rank tests.
2. Other unique-source repeats with a competing name under the same lexical
   rules, including name candidates that fail the teacher screen.
3. Everything else.

Score the original NLL at every position. For each group report its count,
prevalence and original NLL sum divided by the number of predictions across
all groups. This last quantity is the ceiling if the group alone improves to
perfect prediction.

## Sampling and replay

For each of groups 1 and 2, classify original wins and losses using the same
four-head score and original decay as item 7. A tie is a loss. Draw a simple
random sample without replacement within each group-by-rank stratum (up to 128
wins or 512 losses per group, seed 5666). Record each stratum's population and
sample size, each selected position and its inclusion probability (sample
size / population). Wins have zero gain. Run the existing fixed-key oracle
on sampled losses under original survival. Give any loss without a verified
witness zero gain. The third group also has zero gain.

Replay each verified witness in its own copy of the original full window.
At one query position substitute the four saved count vectors, then run the
remaining model normally. Copies can share a GPU batch. Score the answer at
the edited position only, paired with the unedited copy in the same batch
layout. Report how often a fixed-key win remains a win after replay and keep
the signed log-probability gain even if its new rank fails or the answer gets
worse.

Divide each sampled gain by its inclusion probability and sum within
window. Divide the sum across windows by the number of predictions in all
windows. Report the overall estimate and its group contributions. Form a
95% percentile interval from 10,000 paired window bootstrap draws (seed 5667)
of the weighted window contributions. Keep unsampled and zero-gain windows
in the bootstrap.

Training for whole-text quality reopens only if this answer-aware replay
gains at least 0.012 nat per ordinary token and the paired lower bound is
strictly positive. Otherwise close the learned-retrieval line for whole-text
purposes. The estimate measures the specified intervention in the first two
groups with the checkpoint's existing query budgets.

## Time budget

Time the first 128 CPU oracle solves on the separate timing windows before
committing to the main sample. Use up to 64 losses from each group first and
fill any shortfall from the other group. Report the number achieved, CPU
process time, wall time and observed replay time. These timing positions are
excluded from the estimate. If the pilot finds no witness, time an identity
replay to estimate forward cost. After scoring the main baseline, allow twice the
observed per-loss wall time for solving and replaying, plus a two-minute
reserve. If the projected cost exceeds the time left, reduce both loss quotas
proportionally before drawing. Keep at least one per nonempty loss stratum
when the budget permits. The probabilities in the report use the reduced
quotas. Log the time estimate and quotas before running the main oracle.
If even the minimum cannot be completed, or the selected sample overruns its
internal deadline, the job emits no decision from an incomplete sample.

Use one H100-47, four CPU cores and 16 GiB host RAM. The GNU timeout below
enforces the one-hour limit, including startup. The script uses a 3,450-second
internal deadline.

```bash
source slurm/env.sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout --kill-after=5s 3595s python3 experiments/broad_replay.py --device cuda --out runs/broad_replay.json
```

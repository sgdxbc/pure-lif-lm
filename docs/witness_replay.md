# Replay of the saved query witnesses

## Passages and interventions

Use g44_all3 EMA at update 20,000, the original 5.63 LAMBADA tokenization
and FP32 memory tracing. Preserve the diagnostic's batch membership and
padding. Reproduce all 24 fixed-key witness wins before interpreting the
recomputed forward passes. Score all 24 all3 losses in `runs/write_read.json`, with their
original wrong-name positions. Use the original-survival witnesses exactly
as stored, in the recorded head order (9,3), (10,5), (11,2), (11,3).

At the final context token, immediately before each layer's plastic memory
scan, replace the selected heads' query counts. Use three arms.

- **Original** replays the counts captured in the unedited forward pass.
  Require bitwise identical answer-position logits and reproduce the 24
  original ranking losses.
- **Witness** uses the stored count vectors.
- **Sham** draws a legal 0–4 count vector independently per passage and
  head, with the original count budget and exactly the witness's L1
  distance from the original. Seed the draw from 5664, the passage's
  recorded position and the layer and head IDs. Randomize the coordinate
  order. After each coordinate, a suffix feasibility table tracks the
  remaining upward and downward moves. Draw uniformly among next counts
  that permit completion. This gives a reproducible feasible draw without
  reference to the keys or answer score. Report coincidences with the
  witness, including heads with zero edit.

Use the native memory scan, including the model's actual decay. The edit
changes the selected scan input at one position per passage. The scan
uses the keys and values produced by that arm up to that layer. The rest
of the model runs normally, including the computation of later keys.
Verify each arm's per-head budget and each sham's L1 distance against the
unedited counts.

## Outcomes and decision

Measure the original four-head ranking using the realized (possibly edited)
query and keys from each layer in each arm. Normalize each head by its total
causal absolute coefficient mass, including the query position, and average
the four head shares. A win requires the source share to exceed every wrong
name's share. Report how many of the 24 saved wins survive the full witness
forward and record each passage's arm margins.

Use the model's full-vocabulary log-softmax at the query position to score
the original answer token. For each comparison, take the mean of the paired
witness-minus-control log-probability differences across all 24 losses,
including failed flips. Report 95% percentile intervals from 10,000
passage-paired bootstrap draws with seed 5664. Reopen discussion of learned
retrieval only if both mean gains are at least 0.20 nat and both lower
bounds are strictly positive. Otherwise close spending on these four
heads. This answer-aware intervention estimates headroom on the selected
failures.

## Cluster command

From the cluster checkout in an H100-47 allocation, with checkpoints and
LAMBADA cached, budget 5–15 minutes, 8 GiB GPU memory and 16 GiB host RAM.
These are estimates. The job uses inference only.

```bash
source slurm/env.sh
python3 experiments/witness_replay.py --device cuda --out runs/witness_replay.json
```

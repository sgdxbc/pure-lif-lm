# Source-targeted query pilot

## Initialization and shared task

Start all arms from g44_all3 EMA at update 20,000. Train the query half of
`kq_in` for heads (9,3), (10,5), (11,2) and (11,3), including their
biases. Keep all other weights and the key half outside AdamW, including
its weight decay. The task is 0.1 next-token cross-entropy plus 0.9
unit-temperature logit distillation from g13_tf163. Use AdamW at 6e-5
with weight decay 0.1 and gradient clipping at 1. Train 1,000 updates of
32 × 1,024 tokens with microbatches of eight.

## Fixed windows, targets and calibration

Sample 32,032 disjoint FineWeb blocks of 1,025 tokens using seed 5662.
Exclude overlaps with the alignment plan's training and calibration
windows, the write/read screen blocks and the first ten million tokens.
Reserve the first 32 windows for calibration. Freeze ordered offsets and
targets once during preparation. All three arms consume this same plan.

Use the screen's name-source subset, which was the basis of the coverage
gate. The next token has a unique earlier source in its visible document,
the source is name-like, and there is a competing name-like write. Apply
the same FP32 teacher screen to the same four teacher heads. The source
must receive at least 0.20 of the group share and strictly beat every
wrong name. Use every eligible query. For sham, select the wrong name
closest to the source in lag, breaking ties toward the earlier position.
Log the distribution of lag differences and the count per update.

The true arm adds negative log probability of the source position under
the student group's smoothed distribution over the full causal row. The
sham arm uses the same loss on the nearest wrong name at the same query.
The plain arm uses the shared task. Use the same per-head causal smoothing
as the alignment pilot, with 0.001 total uniform pseudocount before
normalization. Average each auxiliary over eligible positions in the
effective batch. An empty batch uses the shared task. On the reserved
effective batch, compute the unclipped gradient norm of each loss over
the trainable query rows. Give true and sham separate fixed weights so
each weighted auxiliary gradient norm is 10% of the task gradient norm
at initialization. Record weights, norms and achieved ratios.

## Decision and descriptive checks

Shuffle eligible LAMBADA with seed 5663 after excluding both original 5.63
splits and the alignment pilot's 256 passages. Keep the first 512 or all
eligible passages if fewer. Use the original 5.63 eligibility. The margin
subset has a name-like source and at least one wrong name under 5.66's
name detector. Record its IDs before training and report both counts.

Report manipulation first. For each arm report mean smoothed source share
over the decision set and over the margin subset, mean correct-minus-best
wrong share margin and strict win fraction on that subset. On these same
passages report per-head key drift from the initial EMA. Use the fraction
of changed count vectors and mean absolute coordinate change for all
causal writes, the source writes and the wrong-name writes. Record key
and query firing before and after on the alignment pilot's validation
windows.

1. The true-minus-plain margin has a strictly positive 95% lower bound
   from paired passage bootstrap. An empty margin subset fails this stage.
2. The true arm improves mean unassisted answer log-probability by at
   least 0.20 nat over both plain and sham, each with a strictly positive
   paired 95% lower bound. Full validation NLL is at most 0.012 higher
   than plain. Full LAMBADA accuracy is at most 2 percentage points lower.
   Measured-rate estimated cost is within 1% at both 1 and 16 streams
   under the same 4-bit U55C model as the alignment pilot.

Use 10,000 paired percentile bootstrap draws with seed 5662. Both stages
must pass to earn an independent-seed confirmation. Report both fresh
64-case copy tests and the original four-head oracle redirect headroom on
the decision set as descriptive checks. Load each saved final checkpoint
and report the unsmoothed name-competition win rate on the 57 eligible
passages of the original 5.63 held-out set.

## Cluster commands

Source `slurm/env.sh` in the cluster checkout with SciPy available. Prepare
and calibrate once on H100-47. Run each train command in its own H100-47 allocation. Summarize
on CPU.

```bash
python3 experiments/query_pilot.py prepare --device cuda --out runs/query_pilot
python3 experiments/query_pilot.py calibrate --device cuda --out runs/query_pilot
python3 experiments/query_pilot.py train --arm plain --device cuda --out runs/query_pilot
python3 experiments/query_pilot.py train --arm sham --device cuda --out runs/query_pilot
python3 experiments/query_pilot.py train --arm true --device cuda --out runs/query_pilot
python3 experiments/query_pilot.py summarize --device cpu --out runs/query_pilot
```

Allow 35–60 minutes for preparation, up to 8 GiB GPU memory and 24 GiB
host RAM. Allow about 20–30 minutes per arm including evaluation and
checkpoint I/O, 24 GiB GPU memory and 24 GiB host RAM. Calibration needs
about 1–3 minutes with the same memory allowance. These are estimates
based on the earlier screen and alignment pilot.

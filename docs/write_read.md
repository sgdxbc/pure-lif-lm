# Fixed-key retrieval and source-supervision coverage

## Write, read and retention

Use exactly the 57 name-answer passages with a wrong name in the recorded
5.66 name-competition output and the same four selected heads per model.
Use g44_all3, g39_snngla, g38_la and g13_tf163 with the original weight
selection and FP32 tracing. Stop if measured win counts differ from the
recorded 33, 31, 38 and 41 respectively. Save per-passage results. Replay
the original padding and batch composition for numerical comparability.

For each source/wrong pair record key norms, zero keys, exact key equality
and cosine. For all3 also record active-set Jaccard and exact count-vector
collisions. Undefined cosine or Jaccard is missing and has separate
coverage. On the existing lag- and frequency-matched pairs, compare source
versus wrong to that same source versus matched control. Average within
passage first and resample passages using the previous bootstrap.
Compare within-model cosine excesses with the teacher on common passages.

Report original and pre-decay winner margins, including signed dot-product
margins per head. For the recurrent models set survival to one and retain
the original per-head absolute-score normalization and equal head weighting.
Use the teacher's original softmax throughout. Count loss-to-win and
win-to-loss changes and bootstrap the paired win-rate difference. This
re-ranking varies survival with fixed realized queries and keys.

## Fixed-key oracle

For each all3 baseline loss, run the oracle with original survival and with
survival one. Each proposed query has integer counts from 0 through 4 and
the observed sum of counts in each head. A silent head contributes zero.
The oracle must beat every wrong name with the same set of four queries.
Normalize against every causal write, including the query position.

For each head let the denominator vector be the sum of keys weighted by
survival over all causal writes. Divide query variables by their score
denominator to obtain a linear relaxation of the simultaneous margin
problem. Permit a zero contribution in a head when its count budget can
fit entirely on coordinates with zero denominator. Round the relaxed
queries to the integer budgets. Improve the rounded and observed queries
with a bounded search transferring one count between two coordinates in
one head. Accept a witness only after checking its counts and strictly
positive margins against all wrong names using exact arithmetic on the
stored floating-point survival factors. Also require the rounded FP32
coefficient ranking to win. The bound applies to the mathematical margin
with the stored factors.

For a bound, use a convex combination of the wrong-name constraints. The
minimum margin is at most this weighted mean. Use the relaxation's dual
weights, uniform weights and individual wrong names as candidate
combinations. For each combination maximize the weighted margin in each
head independently. For a trial ratio, the maximum numerator minus ratio
times denominator is obtained by assigning the count budget to coordinates
in descending order of their adjusted coefficients, capped at four per
coordinate. A nonpositive maximum proves that ratio is an upper bound.
Use integer arithmetic on the dyadic survival weights and rational
competitor weights. The average of these per-head upper bounds bounds the
simultaneous margin. A bound at most zero certifies impossibility. A
positive bound with no verified witness leaves the case unresolved.

Record all outcomes by passage. A verified winner under original survival
shows read-side headroom. A loss restored by removing decay with the observed
query supports a retention effect. An oracle win only without decay also
supports retention when the original oracle is certified impossible.
Impossibility even without decay supports a write-code limit under the
matched query budget. Mark overlapping indications as mixed and cases
without a resolved indication as inconclusive. The oracle uses the answer
and describes the capacity of the fixed key and query-count representation.

## FineWeb coverage screen

Use 4,096 disjoint 1,025-token blocks from the FineWeb training stream with
seed 5661. Exclude blocks overlapping any previous alignment-pilot training
or calibration window in its saved plan, and the first ten million tokens
used for name matching. The denominator is 1,024 next-token positions per
block. Search the visible current document for a unique earlier occurrence
of the target token. Exclude EOS targets and immediate repeats as in the
alignment pilot. Use the name-competition proxy on the visible document
prefix strictly before the query. Use the same token-boundary and
sentence-initial rules. A competing write differs from the target in token
ID and case-folded spelling.

Use the selected teacher heads and average their causal softmax rows.
Count a position if the group's source share is at least 0.20 and strictly
beats every competing name. Report the full funnel and the nested
name-source subset, counts per million tokens and source lag in tokens
with quantiles and fixed lag bins. Record the screened block offsets.

For 1,000 updates of 32 sequences, call coverage adequate when the
name-source subset's window-bootstrap lower bound projects at least
4,000 eligible positions across 32,000 windows (about 122 per million tokens), and at least 90% of the
screen's sequential groups of 32 windows contain a position in that
subset. This targets four useful auxiliary positions per update on
average while checking that positions are spread across batches. The
screen measures availability for a conditional pilot.

## Run

Use an H100-47 allocation with `slurm/env.sh` sourced and SciPy available.
Allow about 10–20 minutes, up to 16 GiB GPU memory and 24 GiB host RAM with
cached inputs. The job runs one model at a time.

```bash
python3 experiments/write_read.py --device cuda --out runs/write_read.json
```

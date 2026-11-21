# Name competition in the 5.63 memory diagnostic

Use the 192 held-out LAMBADA passage IDs in
`runs/memory_maps/summary.json`, in their recorded order. Reconstruct their
original diagnostic token sequences and use the four zero-based layer/head
pairs already selected for each model in that file. Run g44_all3, g39_snngla,
g38_la and teacher g13_tf163 with the same checkpoint weight selection as 5.63.
Optional alignment-pilot checkpoints use their saved final weights and all3's
four heads. Each model gets one causal FP32 pass per passage.

## Eligibility and matching

A lexical candidate is a complete ASCII alphabetic word of at least two
letters occupying one GPT-2 token. Require word boundaries on both sides.
An apostrophe before a separate possessive suffix is a boundary. A name-like
candidate is title case. At the start of the prefix, after a sentence-ending
mark or on a new line, require a second title-case occurrence of that word
anywhere in the visible prefix and zero lowercase occurrences. Ignore opening
quotes and brackets when checking for sentence starts. This recovers repeated
sentence-initial names while excluding most capitals from sentence position.

Hand audit the two saved LAMBADA examples, `lambada:860` and
`lambada:2821`, at all detected context positions. Five of seven detections
are person names. The first passage gives Sun and Mind (false positives),
Matt twice, Petr and Jenna. The second gives Joey. Observed precision is
5/7 on this sample. Ritter takes two tokens and Richard occurs once at sentence start.
This lexical proxy can miss names and include capitalized common nouns.

The correct write is the diagnostic's unique earlier occurrence of the answer.
Require it to be name-like for the primary analysis. Wrong names are all
earlier name-like token positions with a different spelling from the answer,
ignoring case. Repeated occurrences count as separate writes. The control pool
is earlier complete lowercase lexical tokens excluding the answer and
spellings identified as names in the visible prefix. The query position is
excluded from both pools.

Compute unigram counts from the first 10,000,000 uint16 tokens of the shared
FineWeb training stream. The frequency bin is floor(log2(count + 1)) for the
exact token ID. For each wrong write, use the closest control in the same bin
within 16 positions of its lag from the query. Break ties by smallest absolute
count difference, then earliest position. Controls may be reused. Skip
unmatched wrong writes for the competition effect. Matching uses the passage
and the training counts alone and is shared across all models.

## Metrics and interpretation

Normalize the absolute coefficients in each selected head's query row by its
total causal absolute mass, including the query position. A silent row has
zero shares. Average the four head shares with equal weight. Use the largest
wrong-name share for the strict probability that the correct write beats the
best wrong write. Ties count as failures. Include every name-answer passage
with a wrong name in this metric.

For the competition effect, subtract each matched control's share from its
wrong-name share. Average the pair differences within each passage, then
average across passages with at least one matched pair. Report the eligible
pair and passage counts, coverage for both metrics, and the number of
passages with no wrong name, overall and among name answers. Report named
positions, pairs, shares and passage-level metrics for audit.

Use 10,000 passage bootstrap draws with seed 0 and percentile 95% intervals.
All comparisons with the teacher resample the same passages for both models.
Treat the Petr/Matt pattern as general for a model only when its competition
effect has a strictly positive 95% lower bound. Compare each model's
competition effect and correct-write win rate with the teacher's on the same
eligible cases. A positive result describes the sampled passages under the
lexical and matching rules.

## Cluster command

From the cluster checkout inside a GPU allocation with `slurm/env.sh`
sourced, run the following command.

```bash
python3 experiments/name_competition.py --device cuda --out runs/name_competition.json
```

Budget about 5–10 minutes on an H100-47 with cached LAMBADA and checkpoint
access. Allocate 16 GiB host RAM and allow up to 8 GiB GPU memory as an
estimate. Models load one at a time and tracing retains only selected layers.
To include the alignment-pilot arms, add one `--ckpt path/to/ckpt.pt` per arm
and choose a separate output path.

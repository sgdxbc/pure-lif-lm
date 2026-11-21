# Round 2 · Learned binding on text

Freeze the EMA weights of g48_gla31, including its flat vocabulary head.
Encode tokens in independent 128-token blocks with fresh body state. Train
only the memory adapter and add its decoded read to the final body feature.
The adapter sees all the causal features in one passage and starts empty on
the next passage. Neither the write policy nor the read policy receives the
generator's roles, target positions, names or masks as inputs.

Insert a two-person vignette into a FineWeb passage and complete a later
natural sentence that refers to one of their roles. The selected name is the
next token after the completion prefix. In the paired story, swap the names
in the vignette and hold the rest of the tokens fixed. Lay intervening
vignettes with different people and places at evenly spaced positions within
the gap. Counterbalance the queried role, draw the place and names without
replacement within each story, and reject passages already containing either
answer token or where the source touches the query's reset block. Keep all
passages within single source documents. The gap is
measured from the correct name in the source to the predicted token.

Use 32 / 24 / 32 disjoint names and four disjoint vignette/completion templates
per split (defined in the script). Each name with a leading space must round
trip as one GPT-2 BPE token, keep that identity in each vignette, and remain
a separate token after the completion prefix. Sample up to 2,048 distinct
training documents (FineWeb train, first 64M tokens) at gaps 128 and 256, with
2 and 8 competitors except that the 128-token condition uses 2. Within
FineWeb valid use even document indices for development and odd indices for
held-out. Use 128 distinct documents per cell at gap 256 on development and
at gaps 256 and 2,048 on held-out, with 2 and 8 competitors at each gap.
Use 128 additional untouched documents per evaluation split for ordinary
NLL. The name pairs in the two copies of each passage share all filler and
decoy writes. Reserve long held-out documents before sampling shorter cells.
Seeds are 3101 (train generator), 3102 (initialisation), 3103
(development), 3104 (held-out), and 3105 (bootstrap).

Both learned arms have 128 addresses and 32 value channels. A scalar
thresholded write gate is learned from the current feature. In the spike arm
the previous token leaves a 16-address binary trace, and a write updates only
its coincident addresses with the current 16-address event, when the gate
opens. Its read has 16 binary addresses. Use straight-through gradients for
the gate and the addressed events. The dense arm uses continuous keys
computed from both positions, a continuous query, and the same learned
hard write gate. Both add the outer product of key and value to a persistent
matrix with a write-mass column. Divide each read's value sum by its mass
plus 0.25. There is no decay. The digital reference uses the trained spike
arm's encoder and decoder and the same inputs. Keep a bounded FIFO of
successful writes, take up to four records by positive overlap, with recency
breaking ties, and read their similarity-weighted values.

Use one pass over the same training pairs for each learned arm, with AdamW
at 0.001. Optimise the two-name conditional log loss averaged over the
original and swapped stories, plus 0.25 times exact vocabulary NLL on
four fixed positions in the unmodified version of each donor passage.
Score ordinary evaluation NLL at 16 evenly spaced positions per donor.
Every arm operates on the same positions. Also report the frozen body
without a memory adapter as a reference.

Cap persistent mutable state and scratch for addressing, updates and reads
at 32 KiB per stream. Count FP32 accumulators, values and key scratch, packed
trace/record masks, scan metadata and FIFO bookkeeping. Exclude frozen or
learned weights and activations shared with the body. Report per-token
logical state bytes read/written and reset bytes. Count all key and gate
projections, selected value projections, the read decoder, addressed adds
or MACs, and the address-selection or popcount/ranking work separately.
Charge queries at every token of the ordinary development passages even
where the script scores only selected positions. Include digital tags and
searches. The configured dense memory
has the same capacity limit and the same matrix shape. Check on development
that the spike arm is within the dense arm's work budget for the counted
bytes and for adds plus MACs (one operation each). Report selection and
ranking separately. Treat these logical counts as a screen for comparable
work, with bank access and FPGA timing left for follow-up.

Report correct-minus-swapped name log odds for each story, averaged within
its document pair, and two-choice accuracy with half credit for ties.
Before accessing held-out data, stop if neither trained arm beats 50% on
the 256-token development probe with eight competitors. Advancing requires
the development work check and, at 2,048 tokens with eight competitors,
at least 0.20 nat/name above the stronger of the dense and digital controls.
Use 10,000 document-paired bootstrap replicates and require the two-sided
95% lower bound above zero against the stronger control. Ordinary held-out
NLL may exceed the dense arm's by at most 0.02 nat/token.

A pass earns confirmation from an independent training seed. Confirmation
also tests rotating recent traces among name events while retaining their
event counts and using continuous k-winner key values without a coincidence
gate, at a matched number of active addresses. These ablations are deferred.
The first round is capped at one H100-hour including feature extraction,
training and evaluation. Default execution uses development data only. Run
held-out evaluation once after this spec is recorded and only from a
development checkpoint that clears the stop rule.

## One development revision after the failed assay

The original 256×8 stop stands. Both learned arms scored 50%. Their reported
training name NLLs average the whole pass. A write adds to shared rows of
an outer-product matrix. High write frequency can cause interference, but
the ordinary-text frequency alone cannot identify the cause of the failure.

Use experiments/transfer_binding_r1.py in a fresh output directory. Keep the
original body, adapters, value projection from the current token, name and
ordinary-text objectives, work accounting and held-out rules. Add the same
penalty to both learners. On paired story prefixes through the prediction
position and on unmodified donor text, separately measure the mean binary
gate opening rate r. Add the mean over the two domains of 10 × (r − 0.04)²
to the loss. Use the gate's existing sigmoid surrogate for the derivative.
A spike write also needs a nonempty coincidence. Report its effective rate
alongside gate openings. No name or source-position labels control writes.

First train both arms only at gap 128 with two competitors. Use 2,048 donors
and regenerate assignments, templates and filler placement on subsequent
passes over them. Keep donor text fixed across passes. Use the original fit
seed and exclusions. Use a maximum of 4,096 updates or four minutes of
training, whichever comes first. Score once on 128 development documents at
128×2. Reserve the original development documents (both 256-token cells and
ordinary NLL) before drawing these additional development passages. The
larger fit proceeds only if at least one learned arm scores strictly above
60% two-choice accuracy. A failure closes the revision before held-out.

If the easy check passes, carry both arms' weights into the original mix of
128×2, 256×2 and 256×8. Draw another 2,048 donors disjoint from the easy fit,
with fresh assignments on subsequent passes. Train for at most 65,536
updates or 35 minutes. Each stage starts fresh AdamW at a peak of 0.001.
Warm up over the first 5% of stage progress and cosine-decay to 0.0001.
Progress is the larger of elapsed training-time fraction and update fraction.
The two arms see identical batches and use the same schedule. Evaluation
uses the final weights and occurs once at each stage boundary.

Log interval-average training name NLL, frozen-body name NLL on the same
pairs, paired mean log odds, ordinary NLL, gate opening and effective write
rates to training.jsonl every 100 updates and at each stage end. Report the
last interval's metrics. The original development stop at 256×8, the work
gate and the held-out decision thresholds apply after the mixed fit. A second
failure closes the arm. Charge the original development run's elapsed time
against the shared H100-hour along with both stages and any later evaluation.

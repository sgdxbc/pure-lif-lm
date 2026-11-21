# Direct event readout rejection screen

Capture packed final-MLP events and next-token labels from the g45_kd25 EMA
checkpoint on 8,192 FineWeb training windows at length 1,024. Draw one fixed
window order with seed 563, excluding both the fit and calibration windows
listed in the original head_screen manifest. Use the first 512, 2,048 and
8,192 windows for the learning curve (524,288, 2,097,152 and 8,388,608 tokens).
Capture once. The run stores packed events and uint16 labels in a temporary
directory, then removes them when fitting completes.

At each size, estimate a fixed bias from add-one-smoothed word frequencies.
Rank edges per event by smoothed excess co-occurrence with the next token and
fix fan-outs of 32, 128 and 512. Fit each set of edge deltas from scratch
with two passes of full-vocabulary next-token cross-entropy. Use AdamW at
0.03 for the first pass and 0.01 for the second, with batch size 2,048 and
shuffled contiguous blocks. CUDA uses a BF16 masked dense projection for
training efficiency; only the supported deltas are optimized. The original
8,192-token head_screen calibration split supplies every held-out score.
It supplies no event-head training data or checkpoint selection.

Report fitted NLL and KL to the flat head, untouched NLL and KL floors,
events per token, and touched-word counts for each fit size and fan-out.
Compute the sparse target likelihood and its explicit full-vocabulary
reference in FP64. For the KL floor, give the touched logits and the total
untouched probability their ideal values. The remaining error is the flat
head's untouched probability mass times the KL from its conditional
distribution on those words to the fixed bias conditional. The optimistic
true-token NLL floor independently allows arbitrary logits for touched
words and arbitrary aggregate mass for untouched words.

## Rejection rule

Compare KL on the same calibration windows to the minimum recorded
calibration_kl among the conventional linear, shortlist and nonlinear
heads in runs/head_screen/g45_kd25.json. Those heads' best epochs were
chosen on these windows, which slightly favors their reported KL. Use only
the largest fit size for the decision. The smaller sizes show data scaling.

- Retire the tested direct-event design if every fan-out has both fitted KL
  and untouched KL floor more than 0.012 nat/token above the conventional
  reference. This rules out parity for the tested assignments and bias even
  with perfect touched scores. It is a rejection screen for 5.63 parity,
  whose final quality gate uses full-validation true-token NLL.
- If all fitted KL values miss parity and any untouched KL floor is at most
  0.012 above the reference, record **not retired, fitting-limited** and stop.
- If any fitted KL is at most 0.012 above the reference, record a parity
  candidate. A separate plan must cover full-validation NLL, LAMBADA and
  whole-card fpga_est comparisons at 1 and 16 streams under the 5.63 rule.

Count logical storage reads/writes and footprint with 16-bit word indices
and edge deltas, epoch-tagged stream-local accumulators, a broadcast of
shared lists, and a touched-word CDF plus fixed bias alias table. No FPGA
latency or whole-card energy is inferred from those counts.

## Cluster command

Run from the repository root on one H100-47. The registered command needs
about 2.20 GB of extra disk at 2,048 channels (under 3.30 GB at 3,072).
The program stops before scoring the decision if its 3,500-second budget
expires. Allow one hour for the job.

```bash
python3 experiments/event_readout.py --device cuda \
  --run runs/g45_kd25 --cache runs/head_screen/g45_kd25 \
  --head-report runs/head_screen/g45_kd25.json --out runs/event_readout/kd25.json
```

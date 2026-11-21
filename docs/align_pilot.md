# Section 5.63 alignment pilot

Start from g44_all3 EMA at step 20,000. All three arms optimize only weight
and bias rows in the key/query input projections of zero-based heads (9,3),
(10,5), (11,2) and (11,3). AdamW has learning rate 0.00006, decay 0.1 and
gradient clipping at 1. Train 1,000 updates of 32 × 1,024 tokens, accumulating
four microbatches of eight sequences from identical FineWeb training windows.
The common task loss is 0.1 CE plus 0.9 unit-temperature logit distillation
from g13_tf163. Eval uses the final weights of each continuation.

Sample up to eight query positions per training sequence at which the next
token occurs exactly once before the query in the same document. Exclude
EOS targets and immediate repeats. Average the teacher's softmax rows from
heads (10,5), (11,1), (11,7) and (11,0). For each student head normalize
the causal, decayed spike-count coefficients with 0.001 total pseudocount
spread uniformly over that row, then average the normalized distributions.
The aligned arm adds KL from the teacher group to the student group. The sham
arm permutes teacher probabilities independently within the existing diagnostic
lag bands, keeping the same weight. The plain arm runs the task loss alone.
Calibrate the shared weight on a separate matched training batch with auxiliary
gradient norm 10% of task gradient norm summed across its four microbatches.
Log the norms, achieved ratios, weighted loss coefficient, teacher and sham
source mass, and firing before and after for each selected key/query row
(mean rate per threshold neuron plus fraction of nonzero count).

Before running training, select and save the fresh LAMBADA set in plan.json.
Use exactly the diagnostic eligibility, exclude the diagnostic development
and held-out passage IDs, then use the first 256 of a seeded shuffle, or all
eligible passages if fewer. Save the IDs, tokens and source/sham locations.
The file also fixes all arms' train window offsets and both fresh 64-case
copy tests. Natural copy positions exclude the earlier diagnostic's 64
positions. The same seed per microbatch fixes query and sham sampling.

Report manipulation first. Compare source mass and group KL on the fresh
passages and explicitly report if aligned source mass fails to rise over
plain. Record silent head rows. The decision requires aligned unassisted
answer log-prob to improve at least 0.20 nat over each control, both paired
95% bootstrap lower bounds above zero, full validation NLL at most 0.012 worse
than plain, full LAMBADA accuracy at most two percentage points below plain,
and measured-rate estimated 4-bit U55C cost within 1% of plain at both one
and 16 streams. On both copy sets report intact log-prob and its drop under
source replacement. Reapply the original oracle and report source-minus-
unassisted headroom plus source versus sham. A pass earns independent-seed
confirmation.

Prepare on CPU. Calibrate once on a GPU, then run each train command in its
own H100-47 allocation with a one-hour budget per arm, including evaluation.
Use a fresh common output directory. Run summarize after all arms finish.

```bash
python3 experiments/align_pilot.py prepare --device cpu --out runs/align_pilot
python3 experiments/align_pilot.py calibrate --device cuda --out runs/align_pilot
python3 experiments/align_pilot.py train --arm plain --device cuda --out runs/align_pilot
python3 experiments/align_pilot.py train --arm sham --device cuda --out runs/align_pilot
python3 experiments/align_pilot.py train --arm aligned --device cuda --out runs/align_pilot
python3 experiments/align_pilot.py summarize --device cpu --out runs/align_pilot
```

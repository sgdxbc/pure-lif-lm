# 5.64 draft · Adaptive regime detector

Generate streams of eight equiprobable cues. In regime 0 the clean successor is
the cue itself. In regime 1 the successor changes for cues 0–3 (add four);
cues 4–7 are uninformative. Start in either regime with equal probability and
alternate after independent uniform sojourns of 64–256 observations. Replace
the clean successor with a uniform different symbol with probability 0.05.
Predict before reading each successor. Each learner estimates the two tables
from the same 512 labeled development examples per cue per regime, using one
pseudocount per output. Pool counts where the two estimated modes agree. All subsequent prediction and detector training see
only cue and observed successor, with labels used to calculate evaluation
metrics. Observations drive the bank with the log-likelihood ratio clipped to
[-1, 1] after dividing by the log odds for the generator's clean/noise rates.

A bank of 32 cells integrates evidence with learned per-cell leak and two
symmetric thresholds. Each vote changes only on a crossing in the opposite
direction. The downstream predictor retains its active table until the
majority changes, using mixture weights 0.99/0.01. Ties, including the start, use equal
weights. Compare that event-gated bank with an always-updated readout of the
same voltages (logistic of twice their mean) and a digital sequential
likelihood-ratio filter (two-state Bayesian update with hazard 1/160).
All three use the same fitted successor tables. Report updates for the
downstream predictor separately from per-cell checking and events.

Train the same bank with surrogate BPTT and with forward eligibility traces
modulated by the next-token NLL derivative. Both use the same initialization,
sequences and batch SGD at learning rate 0.5. Nearest-boundary sigmoid derivatives serve as the
surrogate. The common loss supplies the same straight-through vote credit
to each cell. This deliberately tests a small bank with independent internal
states. For this bank, forward sensitivities and BPTT implement the same
surrogate gradient. Count measured array storage for detector learning state
at lengths 64 and 256, including saved BPTT history or live forward traces.
Report fixed shared batch input and parameter bytes separately. Report the
initial bank on the same development evaluation stream to show the effect of fitting.

Count packed logical per-observation traffic for the two table rows, the
membership bit, voltages, votes, parameter reads and unsuppressed state
writes. Include one input cue and successor byte each. Report digital filter
state and all three predictor refresh rates. This counts the mandatory
voltage update even during silence. Sojourns and corruption labels are
generated outside the detector and never counted as detector inputs.

Development seeds 201, 204 and 202 cover table fitting, detector training
and separate development metrics. Default training is 12 minibatches of 32 streams at length 512, and
development evaluation is 128 streams at length 1,024. Proposed held-out
seeds are 2001, 2002 and 2003, with 256 streams of length 1,024 each, requiring
at least 1,000 switches per seed. Pool held-out observations for NLL and
updates. Count recovery in informative observations from the true switch
until the correct table is first selected, counting an open interval as a
failure at its end. For false switches, use corruptions separated by four
observations from another corruption or real switch. A false switch changes
the prediction to the wrong regime immediately after the corruption.

Advance the detector if NLL exceeds the always-updated comparison by at most
0.02 nat, downstream updates fall by at least 75%, median recovery takes no
more than four informative observations, and the isolated false-switch rate
is below 5%. Publish total byte traffic, including the digital reference.
A saving confined to the cheap two-state decoder is insufficient to claim
a language-model efficiency gain. The local rule advances if its NLL is
within 0.02 nat of BPTT and its measured detector learning state stays flat
from length 64 to 256. A passing toy earns one text transfer test.

Budget 30 CPU minutes on four cores and at most one development revision.
Run the default development command now. Use `--held-out` only after
the held-out spec is recorded.

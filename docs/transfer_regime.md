# Round 2 · Surprise-gated text state

Gate implementation on a traffic bound. Use the trained g48_gla31 shape
at 4-bit weights as the reference for a small language model. Give the
candidate a generous unit of avoidable work: allow a surprise trigger to
skip all weight and state traffic for one full GLA block on a quiet token.
The other blocks, input embedding row and flat vocabulary head remain hot.
For a group of concurrent tokens, share one scan of dense weights and
charge a state read and write and an embedding row read per stream. Count
logical bytes accessed, regardless of on-chip residence. These are not
HBM transfers or a measured estimate of energy.

For the best case, suppress every refresh, set state-cache serving traffic
and evidence extraction cost to zero, and pay the silent detector's state
and parameter accesses at every token. Use the 32-cell event bank from
round 1. This adds 32 × 17 bytes for membrane read/write, leak, threshold and
vote, and one byte for reading its downstream selection. Report the bound
at 1 and 16 concurrent streams. Report the maximum permitted saving and
the traffic that would have to be suppressible for a 20% net reduction.
If the bound is below 20%, stop with the CPU bound report.

If a later design clears that bar, first measure the prediction benefit of
an expensive slow document state with always-refresh and frozen-state
controls. Derive causal surprise from the observed token's pre-update
probability. Compare learned hysteresis with a scalar moving-average
surprise trigger tuned to the same refresh rate. Give all arms the same
cached causal features and slow-state update. Score next-token probabilities
before the next state update. Charge evidence extraction, detector state
and parameter traffic, slow-state refresh and lookup, and the unchanged
parts of the language model. Report downstream refresh rate, NLL, recovery
after switches and false refreshes after isolated rare tokens.

For that conditional test, use distinct FineWeb training documents to fit
the slow state. Split FineWeb validation by document index parity for
development and held-out. Draw ordinary within-document excerpts and join
pairs of different documents at sentence boundaries for unmarked topic
switches. Preserve the same token stream for every arm. Use development
seed 3201 and held-out seed 3202, with 128 ordinary excerpts and 128 switches
per evaluation split and up to 1,024 tokens per excerpt. Freeze threshold
selection on development before scoring held-out and score false alarms on
ordinary excerpts with isolated low-probability tokens. No text is needed
for the traffic bound.

Advance the conditional test only if the bank is within 0.02 nat/token of
always-refresh, saves at least 20% in whole-model traffic, and beats the
scalar trigger by at least 0.02 nat/token at matched traffic. The CPU
budget for this test is 30 minutes after caching. Computing the traffic
bound should take seconds.

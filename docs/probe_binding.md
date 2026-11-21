# 5.64 draft · One-shot binding by coincidence

Write an ordered pair of symbols followed by its value. The key is emitted as
four distinct addresses out of 128 for each constituent. In the coincidence
arm, the first symbol leaves a one-phase trace and each of the 16 ordered
contacts between that trace and the second symbol increments the arriving
value's synapse. The shuffled control cyclically rotates first-symbol traces
across the bindings while preserving event counts, write times and values.

An episode selects both orders of 4–16 distinct unordered pairs from eight
symbols, draws independent values from 32 choices with distinct values for
reversals, and queries one original binding. Eight extra pairs from the same
symbols occupy evenly spaced positions during the gap as interfering writes.
Other gap positions emit unmarked filler. Score both an intact query and a
query missing one randomly selected event in a randomly selected constituent.
Each episode starts with empty memory. All references see the same streams.

References are an exact indexed table, a digital overlap lookup with ordered
codes, and a decaying outer-product memory. The outer-product reference hashes
the same ordered contacts to 246 float32 rows and writes to the column indexed
by the value. It applies exponential decay lazily per row, with the row's last
update time stored alongside it. Infinite retention only updates the active
value and needs no timestamps. Select the decay using mean clean and dropped
development accuracy from 16, 64, 256, 2,048 and infinity. Break ties toward
longer retention. Include the whole development sweep in the results.

The cap is 32 KiB of mutable state per episode. Count bytes at packed logical
field widths. Each marked event packet or filler tick contributes one phase
byte and four address bytes when present, counted separately from state traffic.
A binding also sends one value byte. Queries send two phase packets.
Charge trace updates, hash probes, tags, values, timestamps and the scratch
score vector including initialization and winner selection. All arms reset
their allocated state at episode start, charged separately. Host array
overhead, cache lines and FPGA arbitration are outside this byte model.

Development uses 100 episodes per cell with seed 101, 8/16 bindings, and gaps
32/64. Proposed held-out seeds are 1001, 1002 and 1003, each with 1,000 episodes
per gap at 32 bindings and gaps 256/2,048. Use paired clean/drop queries in each
episode and score the timing control on the clean query. Advance if pooled clean
accuracy is at least 90% at each gap, dropped accuracy is at least 75% at
2,048, and the timing ablation costs at least 20 percentage points at each gap.
Also require higher clean recall than the development-selected decay at both
gaps. If the digital overlap lookup achieves equal or better intact and dropped
accuracy with equal or lower write and query traffic, the capability still
stands and the hardware claim closes. A survivor gets one repeated-name text
transfer test without explicit write and query markers.

Budget 20 CPU minutes on four cores and at most one development revision.
Run the default development command now. Use `--held-out DEVELOPMENT_JSON`
only after the held-out spec is recorded.

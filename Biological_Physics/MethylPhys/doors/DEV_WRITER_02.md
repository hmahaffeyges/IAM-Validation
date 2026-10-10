# DEV-WRITER-02 — does the writer's discrimination set the copy error, context by context? (development; sealed 2026-10-10, nothing read)

**Why.** DEV-WRITER-01 found the writer's measured discrimination (median D = 63.7) gives a single-step copy error 0.0155, while healthy cells
hold 0.032: about twice as many errors. Whether that factor is real, and whether the writer sets the copy error at all, is tested here across the
256 flanking contexts, where the writer's discrimination varies by more than tenfold. The factor 2 was noticed AFTER both numbers were known and
is therefore not predicted here; only the dependence on D is.

**Prediction (Model A', nothing fitted).** In flanking context c the copy error healthy cells hold is eps_c = k/(1+D_c), with D_c the DNMT1
HM/UM specificity of context c (Adam et al. 2023, pairwise ratios of the 256 k_NNCGNN; 29 for ACCGGA to >300 for GGCGAC, average 87), measured on
the enzyme outside any cell. In logarithms: **log eps_c = log k + 1 x log(1/(1+D_c))**, slope 1.

**Measurement.** Per-context copy error on the 153 healthy Loyfer WGBS samples behind the canon holding energy (PROC-CHANNEL-01), each sample its
own reference; Stage Q's rule unchanged (molecules with >= 6 calls, >= 80 % methylated; isolated errors). CpG index -> hg19 position -> NNCGNN.
One ordinary least-squares fit of log eps_c on log(1/(1+D_c)) per sample; the 153 slopes are the result.
**Control arm, decisive (simulation below).** The same fit in UNMETHYLATED territory (molecules with no methylated calls), where the writer is not
choosing: a sequence-dependent technical artefact (conversion, sequencing error, mappability) acts on the calls and moves BOTH slopes; the writer
moves only the methylated one. The statistic is the DIFFERENCE, methylated slope minus control slope.

**Simulation (`development/sims/writer_context_01.py`, output `writer_context_01_output.txt`), before any reading.**
Counting noise is negligible: with 16-20 M opportunities per sample the slope's spread across samples is 0.004. The slope alone is NOT decisive:
a nuisance correlated with D (rho 0.9) of size 1.0 in log eps fakes slope 1.36 with no writer effect. The difference of slopes is:
| nuisance size, correlation with D | methylated slope | control slope | difference |
|---|---|---|---|
| 0.3, 0.0 | 1.04 | 0.01 | 1.03 |
| 0.6, 0.5 | 1.42 | 0.47 | 0.96 |
| 0.6, 0.9 | 1.78 | 0.78 | 1.00 |
| 1.0, 0.9 | 2.40 | 1.33 | 1.07 |
The difference holds at 1.0 whatever the artefact does.

**Bars (fixed before reading).** Median difference of slopes over the 153 samples:
- **0.5 to 1.5: met.** The writer's discrimination, measured outside the cell, sets the copy error cells hold. k is then read from the intercept
  and reported (k near 1 = the writer's single step; k near 2 = two chances to lose the site per copy), as a measurement, not a prediction.
- **below 0.2: not met.** The copy error does not follow the writer's discrimination; the holding energy is set elsewhere.
- **between: undecided.**
Reported whatever the outcome; no constant changes either way. The 256 simulated D values are replaced by Adam's measured ratios at scoring;
if those are not in the supplementary file in usable form, the test is parked before any cell file is read.

**Amendment, before any cell file is read (2026-10-10 evening).** (1) Enzyme side in hand: Adam 2023 Data Set 1 (`data/DEV_WRITER_02/`,
`enzyme_table.py`), D = HM/UM per NNCGNN, 28.9 (ACCGGA) to 317.5 (GGCGAC), mean 87.3: the paper's printed values. (2) Strands. A .pat file merges
both strands at each CpG; the opposite strand reads the reverse-complement context, and DNMT1 discriminates on the strand it methylates. The
prediction per top-strand context is therefore the strand average, **eps_c = k x ½[1/(1+D_c) + 1/(1+D_rc(c))]**, fitted as log eps_c on log of
that bracket, slope 1. Contexts and their reverse complements carry the same prediction, so the fit has 136 independent points (120 pairs, 16
palindromes); slopes are computed on those 136. The bars are unchanged.

**Reader, scorer and planted test, committed before any cell file is read.** `data/DEV_WRITER_02/context_eps.py` (box; channel.py's copy-error
and de novo definitions split by hg19 NNCGNN context, with a built-in check that the context sums reproduce PROC-CHANNEL-01 per sample exactly),
`score_writer_02.py` (the sealed statistic), `planted_test_02.py` (output `planted_test_02_output.txt`): synthetic molecules with errors planted at
eps_c = 2 x bracket_c → copy-error slope 0.992, control slope 0.026: PASSED. **The same test shows this reader returns k about 25 % low**
(1.49 for a planted 2.0), because it counts only isolated errors on molecules >= 80 % methylated. The slope, the sealed statistic, is unaffected.
k is therefore reported twice: as read, and divided by the planted recovery (1.49/2.0 at these rates). The canon eps0 = 0.032 comes from this same
reader, so the "about twice the writer's error" of DEV-WRITER-01 is read through the same bias.

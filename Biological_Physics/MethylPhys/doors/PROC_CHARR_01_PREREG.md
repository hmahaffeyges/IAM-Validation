# PROC-CHARR-01 — pre-registration (written 2026-10-01, before any brook charr read is aligned)

**Question.** With one lab, one kit, one sequencer and every fish prepared the same way, is the gate-error reading a property of each fish
rather than of its library — and does +2 °C during sperm maturation move it, and by how much in kT?

**Data.** Wellband & Bernatchez, GSE227271 / PRJNA944546. Brook charr (*Salvelinus fontinalis*), 40 males, milt (sperm), WGBS, NEB Ultra II,
HiSeq X, paired 2×150, 2–4 lane-runs per fish (114 runs). Two lines (selected 5–6 generations for no early maturation + growth; control) ×
ambient vs **+2 °C tracking the daily/seasonal cycle, Sept–Dec (final gamete maturation)**; spawning years 2017 (14 fish) and 2018 (26 fish).

**Processing (fixed now).** fastp-equivalent trimming (Trim Galore, paired); Bismark (bowtie2, directional, paired) to the brook charr RefSeq
assembly ASM2944872v1 (GCF_029448725.1); unique best alignments; deduplicated; read pair = one molecule (overlap counted once); 5′ M-bias
bases ignored as set by the pooled M-bias plot of the first fish, before any error is computed. **Equal depth:** the same number of read pairs is
taken from every fish, split evenly over its runs (first N pairs of each run). N is set from a timing pilot (alignment speed only, no error
statistic computed) so the whole set finishes in ≤ 4 h; N is written into this file before the full run.

**Statistic (identical to PROC-SALMON-01).** Qualifying molecule: ≥ 6 CpG calls, ≥ 80 % methylated. Isolated error: unmethylated CpG flanked by
two methylated CpGs on the same molecule; opportunities n − 2. ε = errors/opportunities per fish. Instrument: conversion failure c (methylated
non-CpG / all non-CpG), sequencing error s (mismatch rate at bases conversion cannot touch); ε_corr = ε − s; c printed beside every reading.
Genotype mask: site dropped for a fish if > 30 % of its ≥ 5 qualifying molecules there carry an error. E = ln((1 − ε_corr)/ε_corr) kT.

**Predictions.**
- **P0 (instrument consistency):** ε_corr from two halves of each fish's runs agree, ICC ≥ 0.80 over 40 fish.
- **P1 (the Methow lesson — the fish, not the library):** across fish, |Spearman ρ| between ε_corr and (a) conversion failure, (b) duplicate
  fraction, (c) masked-site fraction is < 0.30 for each, AND the between-fish SD of ε_corr exceeds the median run-half |difference|.
  If any |ρ| ≥ 0.30, P1 fails and no fish-level difference (P3, P4) is interpreted.
- **P2 (scale, from Methow sperm):** Methow steelhead sperm (RRBS, 10 °C) read 4.02 kT. Pass if the median ambient fish reads 3.8–4.3 kT
  (ε_corr 0.0134–0.0219). Failure is reported with the RRBS-vs-WGBS caveat.
- **P3 (temperature, IAM's law):** two hypotheses. *Fixed in kT* (what Methow and human cells showed): warm − ambient Δε_corr = 0.
  *Fixed in joules*: E scales as 1/T, so +2 K at ~283 K lowers E by 0.7 % and raises ε_corr by ≈ +3 %. Test: line- and year-adjusted
  linear model, ε_corr ~ temperature + line + year; report the warm/ambient ratio with 95 % CI. Supports *fixed in kT* if the CI includes 1.00
  and excludes 1.03; supports *fixed in joules* if it includes 1.03 and excludes 1.00; otherwise **underpowered**, stated as such.
- **P4 (line, no direction):** same model, line term, two-sided, α 0.05. Descriptive if P1 fails.

**Stated limits now.** Temperature differed by 2 °C for ~3 months only; 2017 groups are unbalanced (1 fish in two cells); the lines differ
genetically (the mask handles single-site variants only). A temperature null here bounds the effect at this dose; it does not exclude larger
temperature effects.

**Depth, set from the timing pilot (2026-10-01, before any error statistic was computed).** Pilot: GSM7094125, 2,000,000 read pairs, Bismark
294 s on ~10 cores, mapping efficiency 47.1 %, duplicates 4.8 %; no extraction run. Box 2 has ~28 cores free beside the two chains, so
**N = 5,000,000 read pairs per fish**, 3 fish at a time (estimated ≤ 4 h for 40 fish).

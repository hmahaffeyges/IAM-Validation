# PROC-MOLECULE-01 — outcome (2026-10-01). Pre-registration: PROC_MOLECULE_01_PREREG.md (sha eb71b9146a8ed367).

12 ENCODE files, 4,000 windows; molecules ≥ 80 % methylated with ≥ 6 CpGs: 0.2–1.5 M per sample. Flag: P(K ≥ k | n−2, ε₀ = 0.0227) < 0.001.

| prediction | result | verdict |
|---|---|---|
| P1 healthy flagged fraction within 2× of the binomial prediction (all six) | observed / predicted = 3.4–4.8 | **FAIL** |
| P2 1 % cancer detected in ≥ 18/20 mixes, all three pairs | 0/20 in every pair (10 %: K562 5/20, GM12878 12/20, HepG2 19/20) | **FAIL** |

**Why P1 failed.** Errors per molecule are binomial (variance/mean 0.91–0.95 of the binomial value) — the model's form is right. Its rate is not:
on these molecules the isolated-error rate is 0.029–0.036 in healthy cells, not ε₀ = 0.0227. ε₀ came from a different statistic (all-site copy error,
no ≥ 80 %-methylated read filter); selecting methylated reads keeps reads from partly methylated regions, which carry more isolated errors. Lesson:
the floor and the reading must be defined by the same statistic on the same molecule selection.

**Why P2 failed.** Per-molecule thresholding throws away most of the signal: cancer reads 3× healthy on the flagged fraction (≈ 0.0013 vs 0.0005),
but only ~1 molecule in 2,000 is flagged, so at 1 % tumour the excess is a few molecules against a background SD of 14–23.

**Exploratory (not pre-registered): total isolated-error count, no threshold.** Same mixes: detected in 20/20 at 10 %, 18–20/20 at 3 %,
1–6/20 at 1 %, 0/20 at 0.1 %, with 0.5–1.5 M molecules. The limit scales as 1/√N: plasma sequenced to tens of millions of qualifying molecules
would reach roughly 10× lower (an estimate, not measured). Rates: healthy 0.029–0.036, cancer lines 0.041–0.047 per opportunity.

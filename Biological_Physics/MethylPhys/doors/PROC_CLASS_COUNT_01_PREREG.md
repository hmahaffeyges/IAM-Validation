# PROC-CLASS-COUNT-01 — pre-registration: how many entropy levels do cells form, measured without any floor

**Written 2026-09-30, before the test is run.** The whole-methylome entropy (PROC-HMIN-REFIT-02) puts five classes within 0.025 bits.
The class floors may instead live on the loci that make each cell what it is. This test measures that without using any floor to
choose loci, so the answer cannot be built into the question. Measures only; changes nothing.

**Data.** Every atlas v2 cell with ≥ 2 Loyfer WGBS samples (one platform, true 0 and 1). Per-locus entropy = mean of the first-order
bias-corrected and the Bayesian estimates (they bracket the truth at ±0.01, PROC-HMIN-REFIT-02), depth ≥ 10.

**Three measures per cell** (no floor, no class, no population in any of them):
- **M1 whole methylome** — mean corrected H over the atlas universe (as REFIT-02).
- **M2 entropy profile** — the fraction of the cell's loci in each of 10 equal H bins (0–1 bit): the shape, not only the mean.
- **M3 defining loci, held out** — split each cell's samples in two (fixed seed). On half 1 of every cell, a cell's defining loci are the
  2,000 loci where its mean β differs most from the median of all other cells (|difference| ≥ 0.25). The cell's M3 is the mean corrected H
  of its half-2 samples at those loci; then the halves are swapped and the two averaged. Choosing and reading never share a sample.

**Questions, for each measure.**
- Q1 Do the eight classes explain the spread between cells? Between-class / within-class variance ratio, against 10,000 random
  relabellings of the same cells with the same class sizes.
- Q2 How many levels do the cells form, with no labels? Gaussian mixtures with 1 to 8 components on M1 and M3 (BIC), and on M2
  (hierarchical clustering, silhouette for 2–10 clusters).
- Q3 Do the unlabelled groups match the classes? Adjusted Rand index against the draft-rule class, with the relabelling null.

**What each outcome says.** Eight classes supported: Q1 beyond the relabelling null on M3 and Q2 finding about eight levels that match
(Q3 high). One level: Q1 at the null, BIC best at 1. Anything else is reported as the number the data give.

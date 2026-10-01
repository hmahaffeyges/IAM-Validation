# PROC-MOLECULE-01 — pre-registration (written 2026-09-30, before any per-molecule record was read)

**Question.** Averages cannot see a 0.1–1 % tumour fraction. Can IAM's law flag single molecules instead — with no training and no cohort?

**Data.** ENCODE WGBS (same files as PROC-ENCODE-01): K562, GM12878, HepG2 (2 each); CD34+ myeloid progenitor, B cell, CD14 monocyte, T cell, NK
cell, liver (1 each). 4,000 random 100-kb autosomal windows (seed 7). Per molecule: CpG count n, methylated count, isolated errors k (a CpG
disagreeing with both neighbours that agree with each other), longest unmethylated run.

**Physics model.** A healthy methylated molecule (≥ 80 % methylated, n ≥ 6) carries K ~ Binomial(n − 2, ε₀) isolated errors, ε₀ = 1/(1+e^(φM)),
φ = 0.1798 (ENCODE normal mean, PROC-ENCODE-01), M = 20.94 → ε₀ = 0.0227. **Flag:** P(K ≥ k | n − 2, ε₀) < 0.001. Fixed now; not tuned.

**Predictions.**
- P1 (physics predicts the healthy background): in each of the six healthy samples, the observed fraction of flagged molecules lies within a factor
  of 2 of the binomial prediction computed from that sample's own n distribution and ε₀.
- P2 (detection at 1 %): in-silico mixes of real molecules — K562 into CD34+ progenitor, GM12878 into B cell, HepG2 into liver — at 1 % tumour
  fraction, 20 random mixes each at the healthy sample's molecule count: flagged count exceeds the healthy background by ≥ 3 SD (SD from the 20
  pure-healthy resamples) in ≥ 18 of 20 mixes, in all three pairs.
Descriptive: the same at 10 %, 0.1 %, 0.01 %; and whether flagged molecules are enriched beyond the uniform-error expectation (clustered errors).

**Stated limits now.** Culture vs transformation is not separated (lines vs primary cells). ENCODE reads are 100–150 bp; plasma fragments are shorter.

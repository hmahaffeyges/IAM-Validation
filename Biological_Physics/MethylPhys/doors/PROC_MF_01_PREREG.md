# PROC-MF-01 — pre-registration: does a covariance-weighted matched filter lower the minimum detectable fraction of a foreign cell in blood?

**Written 2026-09-26, before any spiked specimen was scored with the filter.** No cohort is used. Truth is constructed;
the null is the 48 healthy arrays already scored today (12 per laboratory, GSE87571 / GSE42861 / GSE111629 /
GSE125105, mapped). Nothing below moves after results are visible.

## The question, in the author's words

*"We are using [the CMB tools] for failsafes but what about using them for celltype identification? cleaning up the
noise? sensitizing the cell type detection so we can find a needle in a haystack and then score it?"* — and, of the
group that abandoned blood-based detection: *"the reason they gave up was because of the noise … but they were not
cosmologists and they didnt have the CMB tools."*

## What is borrowed, and what it does not replace

**Borrowed:** the matched filter of CMB point-source and cluster detection. For a template **t** (the foreign cell's
profile minus the expected blood background at the same loci) and a noise covariance **N** estimated from healthy
specimens, the optimal linear estimator of the amplitude is

  f̂ = (tᵀ N⁻¹ r) / (tᵀ N⁻¹ t),  with  σ_f̂² = 1 / (tᵀ N⁻¹ t),

where **r** is the specimen's residual after the blood-only fit. It returns the fraction **and its significance**.
The current solver (NNLS, unit weights on every marker) is the special case N = I.

**Why it might work here and not for them:** PROC-COV-01 measured the healthy-blood residual to be a reproducible,
specimen-independent misfit, ~85 % of it removable, estimated leave-one-laboratory-out. Structured noise is what a
covariance-weighted filter is built against; random noise it only averages. If the residual were random the filter
would gain ~nothing over NNLS and this procedure would fail B1.

**Not replaced (the author's caution, 2026-09-26):** the ordinary instrument work. The filter runs on the repaired
solve block, on mapped betas, after twin resolution — every fix of this morning stays. Needlets and any locality
assumption on the sky are **not** borrowed: CpG address on the sphere is our projection, not a physical neighbourhood.

## Construction

- **Null:** the 48 healthy arrays (mapped). N is estimated from their residuals after the blood-only fit,
  **leave-one-laboratory-out**: the covariance used to score a laboratory's arrays never saw that laboratory.
  Shrinkage (Ledoit–Wolf) to keep N invertible with 36 arrays; the shrinkage intensity is whatever the estimator
  returns, not tuned.
- **Spikes:** Breast, Colon_epithelial_cells, Cortical_neurons, Prostate, each at 0.5, 1, 2, 5 %, composed by
  `synthetic_patient_generator.compose_cells` on a blood background drawn from the atlas blood cells, noise σ 0.044,
  10 seeds per fraction — **but also** spiked into the 48 *real* healthy arrays (atlas profile × f added to the mapped
  array, renormalised), because a constructed background is optimistic and the real one carries the misfit.
- **Detectors compared:** (a) NNLS on the repaired block, fraction read directly; (b) matched filter as above.
- **Threshold, fixed by rule not by value:** for each detector and each cell, the threshold is set on the 48
  *unspiked* healthy arrays at **≤ 1 false positive in 48** (the 47th of 48 ordered null values). The detection limit
  is the smallest spiked fraction at which **≥ 90 % of spikes** exceed that threshold.

## Bars

| bar | requirement | why |
|---|---|---|
| B1 | matched-filter detection limit is **lower than NNLS's** for at least 3 of the 4 cells on the **real-array** spikes | the claim itself; constructed-background spikes are reported but do not decide |
| B2 | the filter's σ_f̂ is **honest**: on the unspiked null, the fraction of |f̂/σ_f̂| > 2 is between 0.02 and 0.10 | a significance the null violates is a liar |
| B3 | filter fractions on constructed spikes are **unbiased**: median (f̂ − f) within ±0.005 at 2 % and 5 % | a filter that detects but misreads the amount is not an estimator |
| B4 | **no gain from the covariance alone**: shuffling N's off-diagonal structure (diagonal-only N) must give a detection limit between NNLS and the full filter, not equal to the full filter | if diagonal weights do everything, the structured covariance did nothing and the finding is "weight by variance", not "matched filter" |
| B5 | **negative control**: with N estimated on the *same* laboratory it scores (no leave-out), the limit must not be more than 20 % better than leave-one-out | the gain must not be the covariance memorising its own arrays |
| B6 | the blood composition on the 48 healthy arrays is **unchanged**: max |Δ fraction| over CD4/CD8/B/NK/mono < 0.005 between NNLS and the filter's background fit | the filter is a detection stage, not a new deconvolver |

**Decision rule.** B1–B3 and B6 met → the matched filter is adopted as the **detection stage ahead of the per-cell
A**, with the reported quantity per cell being (f̂, σ_f̂, detection: yes/no at the pre-registered false-positive
rate). B4 failing → adopt *diagonal* weighting and record that the structure bought nothing. B1 failing → record the
noise floor as real for this atlas and platform, and say so where the group who gave up is discussed. Any bar not
run is recorded NOT ASSESSED, never passed by argument.

## Evidence files (named before they exist)

In `kit/`: **PROC_MF_01.py**. In `kit/results/`: **PROC_MF_01.json** (every spike, every seed, both detectors) and **PROC_MF_01_null.json** (the 48 unspiked readings per detector). In `plates/`: **PROC_MF_01.png**. Linked from the outcome once they exist; not linked here because a link to a file that does not yet exist is a broken link.

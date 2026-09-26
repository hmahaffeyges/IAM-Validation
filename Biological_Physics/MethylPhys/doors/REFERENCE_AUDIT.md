# The per-cell A is computed on the wrong surface — measured 2026-09-26

Every cell's **own atlas mean** is, by construction, the reference healthy profile for that cell. On the
author's model an A-score of 1.0 means at the floor, for every cell, with the only per-cell difference being
which architecture class's H_min divides it. So a cell's own reference must read **A ≈ 1.0**. It does not.

## The audit: all 115 cells, each scored on its own reference

| A of the cell's own atlas mean | cells |
|---|---|
| 0.9 – 1.1 (as designed) | **11** |
| 0.5 – 0.9 | 39 |
| **below 0.5** — entropy far under the physical floor | **65** |
| above 1.1 | 0 |

Median **0.4574**, minimum 0.0099. An A below 1.0 means measured entropy below the Landauer floor, which is
not physically meaningful — so this is a calibration defect, not a reading.

## The cause, and it is arithmetic

The per-cell path scores each cell on its **discriminative marker panel**, and mean per-CpG entropy over
near-binary addresses is ≈ 0 by construction. **How near-binary a panel is varies widely between cells,
and that variation is the defect** — it is not a constant property of marker panels.

Fraction of a cell's marker addresses below 0.1 or above 0.9, measured across all 115 cells:

| min | p25 | median | p75 | max |
|---|---|---|---|---|
| 0.01 | 0.44 | 0.70 | 0.98 | 1.00 |

**corr(extreme fraction, A on markers) = -0.961** — the tightest relationship in
this audit. The 76 cells at or above 50 % extreme have median A **0.3173**;
the 39 below it, **0.6605**. So the number was measuring how sharply each
cell's markers happened to be chosen, not how well the cell holds its state — and cells with unusually
mild panels (Breast, 0.01) read plausibly by luck, which is why this stayed invisible until every cell
was audited.

**Correction, 2026-09-26:** the first version of this document said marker panels are "77 % to 100 %"
near-binary. That was generalised from the twelve worst cells and is contradicted by this document's own
comparison table below, which lists values down to 0.01. The distribution above is the measured one. The
same wrong figure reached the CHAIN_COMMISSIONING B-11 row, corrected with it, and the commit message of
`695d1c8`, which stands wrong in the record.

## The same cells, scored on identity loci instead

| cell | on its marker panel | on identity loci | markers near 0 or 1 |
|---|---|---|---|
| Cortical_neurons | 0.0099 | **1.0123** | 1.00 |
| Acinar | 0.1375 | **1.1427** | 0.99 |
| fibroblast | 0.2875 | **0.9937** | 1.00 |
| Colon_epithelial_cells | 0.3654 | **0.9597** | 0.77 |
| CD4_T-cells | 0.7447 | **0.9549** | 0.10 |
| HSC | 0.5976 | **0.9876** | 0.42 |
| Breast | 1.0638 | 0.8942 | 0.01 |

**Identity surface: 0.89 to 1.14, median 0.9876.** Marker surface: 0.0099 to 1.06, median 0.3654.
Breast read plausibly on markers only because its markers happen not to be extreme — 1 % of them — which is
why the defect was invisible until every cell was audited.

## The repository already contained the rule

The Jensen-bound check states that an identity-like unimodal panel has `H(mean beta) - mean H < 0.05` while a
bimodal marker-like panel exceeds it. Measured gaps on the marker panels here are **0.34 to 0.55**. The rule
was written and the class gauge obeys it; the per-cell path was never held to it.

## What this explains

- **Why immune at class level is the only reading that ever worked** — it is the one quantity computed on identity loci.
- **The 115-cell audit**, entirely.
- **Possibly the severity of the fraction confound** measured in [`PROC_SYNTH_01_OUTCOME.md`](PROC_SYNTH_01_OUTCOME.md): that was measured on the marker surface, and binary addresses are maximally sensitive to dilution. The 8.3x figure must be re-measured on the identity surface before it is quoted again.

## What it does not explain, and must not be used to excuse

The composition numbers are unaffected — the deconvolver selects its own markers and PROC-SYNTH-01 verified
it recovers a known mixture with 0.0000 error. Nothing here rescues PROC-PARTIAL-01, whose failure was a
+0.067 reference misfit measured directly in beta, independent of surface.

## What is needed, and it is a construction rather than a fix

Identity loci exist **per class only**. Using a cell's class loci would give A ≈ 1 for healthy references but
every cell in a class would share one surface and one observed value, so all 51 immune cells would read
identically — which destroys the per-cell resolution that is the point. **Per-cell identity loci must be
constructed**, the same way the eight class panels were, and the class panels' provenance is in
[`iamatlas_gauge_identity_loci_v1_0.json`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json) to mirror. That is the author's call to commission.

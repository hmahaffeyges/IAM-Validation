# CPG-VAL-137 — Sex-axis subtraction reduces the M-vs-F A_immune contrast

**Cohort:** GSE40279 Hannum 2013, n=656 healthy whole blood with sex metadata
**Date sealed:** 2026-06-06
**Outcome code:** O3_INVERTED

## Headline result

| Condition | d (M vs F) | N1 p-value |
|---|---|---|
| WITHOUT Stage 3 sex subtraction | -0.126 | 0.12 |
| WITH Stage 3 sex subtraction | -0.548 | 0.0 |
| **Δ\|d\|** | **-0.421** | — |

## External validation framing

The sex layer was fit on GSE50660 (n=464, different cohort). This VAL is an EXTERNAL test of whether that layer transfers to GSE40279 (different cohort, different age range, different ethnicity composition). A positive Δ|d| means the layer's sex coefficients learned on one cohort generalize to another — a cleaner test than VAL-136's cohort-internal smoking test.

## Interpretation

INVERTED — The M-vs-F A_immune contrast does not shrink (Δ|d| = -0.421). Possible reason: the immune marker panel selected by top discrimination is enriched for autosomal CpGs not strongly affected by sex, so the sex layer correction is small in this signal. The sex layer may still be doing work on chrX/chrY CpGs that are not in the immune-class marker panel.

## Cohort linkage

- Per-sample data: `CPG_VAL_137_per_sample.csv` (n=656 × 4 columns)
- Source β: `/tmp/geo_downloads/GSE40279_beta_matrix.npz`
- Sex layer source: `Biological_Physics/atlas_vault/walther_clinical_runtime/IAM_Cellular_Age/IAMAtlas_sex_layer.csv` (fit on GSE50660 n=464, 2026-06-06)

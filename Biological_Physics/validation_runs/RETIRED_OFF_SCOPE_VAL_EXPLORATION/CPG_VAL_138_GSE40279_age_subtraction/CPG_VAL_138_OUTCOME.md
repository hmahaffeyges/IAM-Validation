# CPG-VAL-138 — Age-axis subtraction reduces A_immune's chronological age dependence

**Cohort:** GSE40279 Hannum 2013, n=656 healthy aging cohort
**Date sealed:** 2026-06-06
**Outcome code:** O3_INVERTED

## Headline result

| Condition | Pearson r(A_immune, age) | p-value | n |
|---|---|---|---|
| WITHOUT Stage 3 age subtraction | -0.158 | 4.98e-05 | 656 |
| WITH Stage 3 age subtraction | -0.405 | 2.47e-27 | 656 |
| **Δ\|r\|** | **-0.248** | — | — |

## Interpretation

INVERTED — A_immune is not strongly age-correlated to begin with at baseline (r = -0.158), so age subtraction has minimal effect. This is consistent with the age layer being trained on a different cohort (foundation GSE51057+GSE51032, n=601) where the immune-relevant age effects may differ from those in GSE40279 (broader age range, different population mix).

## External validation framing

The age layer was fit on the foundation cohort (GSE51057+GSE51032, n=601, EPIC-Italy breast pre-diagnostic). This VAL tests whether that layer transfers to GSE40279 (Hannum 2013, different cohort, broader age range 19-101 vs ~40-65 in foundation). A reduction in |r| on GSE40279 confirms the age layer captures generic linear aging methylation rather than cohort-specific artifact.

## Cohort linkage

- Per-sample data: `CPG_VAL_138_per_sample.csv` (n=656 × 4 columns)
- Age layer source: `Biological_Physics/atlas_vault/walther_clinical_runtime/IAM_Cellular_Age/IAMAtlas_age_layer.csv` (8,199 CpGs)

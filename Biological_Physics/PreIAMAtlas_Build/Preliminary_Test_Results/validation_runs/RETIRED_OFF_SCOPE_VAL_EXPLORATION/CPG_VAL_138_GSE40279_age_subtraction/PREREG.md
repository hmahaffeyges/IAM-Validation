# CPG-VAL-138 — Pre-Registration

**VAL ID:** CPG-VAL-138
**Title:** Age-axis subtraction Δr on GSE40279 age dependence
**Date sealed:** 2026-06-06
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey

## Cohort

- **Source:** GSE40279 (Hannum 2013), n=656 healthy aging cohort, age range 19-101
- **Note:** This is the canonical aging cohort that the original Hannum clock was built on

## Signal

- **Primary signal:** Pearson r between A_immune (Stage 4) and chronological age (years)
- **Intervention under test:** Stage 3 age-axis foreground subtraction (β-level)
- **Age layer source:** `IAMAtlas_age_layer.csv` (8,199 CpGs, fit on foundation cohort GSE51057+GSE51032 n=601 HC)

## Decision rule

- **Pass condition:** |r_with_subtraction| < |r_without_subtraction|
- **Logic:** Age subtraction should remove the linear age-driven β component. A_immune residuals should track age less strongly after subtraction.

## Observed outcome (sealed 2026-06-06)

- **r (WITH age subtraction):** -0.405 (p = 2.47e-27, n = 656)
- **r (WITHOUT age subtraction):** -0.158 (p = 4.98e-05, n = 656)
- **Δ|r|:** -0.248
- **Outcome code:** O3_INVERTED

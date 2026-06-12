# CPG-VAL-140 — Pre-Registration

**VAL ID:** CPG-VAL-140
**Title:** Cellular age inversion via 8-class A-score linear model on GSE40279
**Date sealed:** 2026-06-06

## Cohort

- **Source:** GSE40279 (Hannum 2013), n=656 healthy aging cohort
- **Age range:** 19-101 years (median ~60)

## Signal

- **Primary signal:** Predicted chronological age from 8-class A-score vector (Stage 4 outputs from VAL-135)
- **Model:** Linear regression (simplest baseline; not the production Stage 6 inversion)
- **Production caveat:** This is NOT the production Stage 6 inversion against the 80-cell baseline (see Limitations). This VAL establishes whether the 8-class A-vector carries age signal at all, as a precursor to the proper 80-cell baseline inversion.

## Decision rule

- **Pass:** R² > 0.5 (8-class A-vector accounts for more than half of age variance)
- **Logic:** If the 8-class architectural decomposition is age-tracking, a simple linear model should recover substantial chronological age signal even without the per-class age curves.

## Observed outcome

- **R²:** 0.251
- **Pearson r:** 0.501
- **MAE:** 10.1 years
- **Outcome code:** O5_BASELINE_DOMINATED

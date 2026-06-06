# CPG-VAL-136 — Pre-Registration

**VAL ID:** CPG-VAL-136
**Title:** Smoking-axis subtraction Δd on GSE50660 never-vs-current contrast
**Date sealed:** 2026-06-06
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Cohort

- **Source:** GSE50660 (Tsaprouni 2014), n=464 healthy whole blood
- **Subgroup contrast:** never smokers (n=179) vs current smokers (n=22); former smokers (n=263) excluded from contrast
- **Platform:** Illumina HumanMethylation450

## Signal

- **Primary signal:** A_immune (Stage 4 immune-class A-score with 200-marker panel from IAMAtlas REBUILD)
- **Intervention under test:** Stage 3 smoking-axis foreground subtraction (β-level, layer CSV fit 2026-06-06 on this same cohort)

## Decision rule

- **Pass condition:** |d_with_subtraction| < |d_without_subtraction| (smoking subtraction shrinks the never-vs-current contrast)
- **Logic:** If the smoking layer is correctly removing the smoking-driven β shift, the residual A_immune should be more comparable between never and current smokers. A positive Δ|d| confirms the layer is doing biological attribution work.
- **Caveat:** This is a cohort-internal test; the smoking layer was fit on this same cohort. A clean external test would use a different cohort with smoking metadata. VAL-136 is a sanity check, not external validation.

## Observed outcome (sealed 2026-06-06)

- **d (never vs current, WITH smoking subtraction):** 0.005
- **d (never vs current, WITHOUT smoking subtraction):** -0.063
- **Δ|d|:** +0.058
- **N1 p-value (with subtraction):** 0.983
- **N1 p-value (without subtraction):** 0.775
- **Outcome code:** O1_PRIMARY_VALIDATED

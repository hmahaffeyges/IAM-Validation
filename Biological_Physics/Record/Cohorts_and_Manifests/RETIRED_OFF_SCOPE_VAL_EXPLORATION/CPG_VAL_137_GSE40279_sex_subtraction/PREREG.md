# CPG-VAL-137 — Pre-Registration

**VAL ID:** CPG-VAL-137
**Title:** Sex-axis subtraction Δd on GSE40279 M-vs-F contrast
**Date sealed:** 2026-06-06
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey

## Cohort

- **Source:** GSE40279 (Hannum 2013), n=656 healthy whole blood
- **Subgroup contrast:** Male (n=318) vs Female (n=338)

## Signal

- **Primary signal:** A_immune (Stage 4 immune-class A-score)
- **Intervention under test:** Stage 3 sex-axis foreground subtraction (β-level, layer CSV fit on GSE50660 n=464)

## Decision rule

- **Pass condition:** |d_with_subtraction| < |d_without_subtraction|
- **Logic:** If sex layer is correctly removing the sex-driven β shift on chrX/chrY/XCI CpGs and any autosomal sex-dimorphic CpGs, residual A_immune should be more sex-comparable.
- **External validation:** Sex layer was fit on GSE50660 (different cohort, different platform overlap); this is a CLEAN external test of the sex subtraction module.

## Observed outcome (sealed 2026-06-06)

- **d (M vs F, WITH sex subtraction):** -0.548
- **d (M vs F, WITHOUT sex subtraction):** -0.126
- **Δ|d|:** -0.421
- **Outcome code:** O3_INVERTED

# CPG_VAL_012_AD_PC_axes — Pre-Registration

**VAL ID:** CPG_VAL_012_AD_PC_axes
**Title:** AD-immune principal-component axes (AIBL)
**Date sealed:** 2026-06-03 (RETROSPECTIVE — see "Provenance note" below)
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Provenance note

This PREREG was sealed AFTER the analysis was run, as part of the AD-immune Phase 2 retrospective inventory. The substantive analysis was authored on 2026-06-02/03 across commits `570a34a`, `6cc9069`, `18167ce`, `fc87363`. The L9 null suite was added on 2026-06-03 in this commit. **A future version of this VAL under the v4 inventory protocol will require PREREG to be sealed BEFORE re-analysis, with hashes locked.** This retrospective document is honest about its position.

## Cohort

- **Source:** GSE153712 AIBL
- **n_case (ad):** 161
- **n_hc (hc):** 471

## Signal

- **Signal column:** `PC1 score (T-cell axis, fit on HC samples)`
- **Effect direction expected:** Hypothesis: PC2 will be T-cell axis like breast. Actual: PC1 is the T-cell axis in AIBL (different cohort structure).

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: ['N1_hc_label_permutation']
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Cohen's d (observed):** -0.356
- **Primary N1 null p-value:** 0.0
- **Result narrative:** PC1 d=-0.356 (p=8e-4), top loadings T-cell + neutrophil. PC3 secondary (d=+0.224, p=6e-6).

## Interpretation

PASS — observed PC1 deviation exceeds null by >3σ. Architectural T-cell exhaustion at covariance level.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + covariates
- `null_results.json` — N1 HC-label-permutation null result (+ N2 age-strata for VAL-011)
- `cohort_manifest.json` — cohort provenance, SHA-256, source URL
- `CPG_VAL_012_OUTCOME.md` — full outcome narrative
- Cohort-specific CSVs (per_celltype, residual_map, projections, etc.)

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Full 7-test L9 suite (N1-N7); only N1 (+ N2 for VAL-011) run here
- Sealed reproducer script (`cpg_val_012_ad_pc_axes.py`)
- results.json with sealed inputs/output hashes

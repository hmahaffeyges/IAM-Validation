# CPG_VAL_009_AD_AIBL_mahalanobis — Pre-Registration

**VAL ID:** CPG_VAL_009_AD_AIBL_mahalanobis
**Title:** AD-immune Mahalanobis hyper-volume (AIBL)
**Date sealed:** 2026-06-03 (RETROSPECTIVE — see "Provenance note" below)
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Provenance note

This PREREG was sealed AFTER the analysis was run, as part of the AD-immune Phase 2 retrospective inventory. The substantive analysis was authored on 2026-06-02/03 across commits `570a34a`, `6cc9069`, `18167ce`, `fc87363`. The L9 null suite was added on 2026-06-03 in this commit. **A future version of this VAL under the v4 inventory protocol will require PREREG to be sealed BEFORE re-analysis, with hashes locked.** This retrospective document is honest about its position.

## Cohort

- **Source:** GSE153712 AIBL
- **n_case (ad):** 161
- **n_hc (hc):** 471

## Signal

- **Signal column:** `mahalanobis_d (universal HC-centroid distance)`
- **Effect direction expected:** Modest positive effect (smaller than breast pre-dx +1.87)

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: ['N1_hc_label_permutation']
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Cohen's d (observed):** +0.200
- **Primary N1 null p-value:** 0.023
- **Result narrative:** d=+0.200 [95% CI +0.041, +0.452]; p_mwu=4.3e-04; MCI intermediate position confirmed

## Interpretation

PASS — observed signal exceeds null at p=0.023. Modest but real.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + covariates
- `null_results.json` — N1 HC-label-permutation null result (+ N2 age-strata for VAL-011)
- `cohort_manifest.json` — cohort provenance, SHA-256, source URL
- `CPG_VAL_009_OUTCOME.md` — full outcome narrative
- Cohort-specific CSVs (per_celltype, residual_map, projections, etc.)

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Full 7-test L9 suite (N1-N7); only N1 (+ N2 for VAL-011) run here
- Sealed reproducer script (`cpg_val_009_ad_aibl_mahalanobis.py`)
- results.json with sealed inputs/output hashes

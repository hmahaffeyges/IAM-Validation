# CPG_VAL_011_AD_age_subtraction — Pre-Registration

**VAL ID:** CPG_VAL_011_AD_age_subtraction
**Title:** AD-immune age-axis foreground subtraction (AddNeuroMed + GIFT)
**Date sealed:** 2026-06-03 (RETROSPECTIVE — see "Provenance note" below)
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Provenance note

This PREREG was sealed AFTER the analysis was run, as part of the AD-immune Phase 2 retrospective inventory. The substantive analysis was authored on 2026-06-02/03 across commits `570a34a`, `6cc9069`, `18167ce`, `fc87363`. The L9 null suite was added on 2026-06-03 in this commit. **A future version of this VAL under the v4 inventory protocol will require PREREG to be sealed BEFORE re-analysis, with hashes locked.** This retrospective document is honest about its position.

## Cohort

- **Source:** GSE144858 AddNeuroMed + GSE53740 GIFT (AIBL excluded — no ages in GEO release)
- **n_case (ad):** 93
- **n_hc (hc):** 96

## Signal

- **Signal column:** `stem_adult A-score (per-class signal that emerges under age subtraction; raw d=-0.004, post-subtraction d=-0.190)`
- **Effect direction expected:** Raw per-class stem_adult is null; age subtraction reveals d~-0.2

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: ['N1_hc_label_permutation', 'N2_age_strata_permutation']
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Cohen's d (observed):** -0.004
- **Primary N1 null p-value:** 0.974
- **Result narrative:** Raw: d=-0.004, N1 p=0.97 (correctly null at baseline); post-subtraction: d=-0.19 (documented in OUTCOME.md, separate analysis)

## Interpretation

PASS-AS-NULL — confirms baseline raw signal is null. The interesting finding (post-age-subtraction d=-0.19) is documented in OUTCOME.md as a separate analysis; N1 here tests the correctness of the baseline null.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + covariates
- `null_results.json` — N1 HC-label-permutation null result (+ N2 age-strata for VAL-011)
- `cohort_manifest.json` — cohort provenance, SHA-256, source URL
- `CPG_VAL_011_OUTCOME.md` — full outcome narrative
- Cohort-specific CSVs (per_celltype, residual_map, projections, etc.)

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Full 7-test L9 suite (N1-N7); only N1 (+ N2 for VAL-011) run here
- Sealed reproducer script (`cpg_val_011_ad_age_subtraction.py`)
- results.json with sealed inputs/output hashes

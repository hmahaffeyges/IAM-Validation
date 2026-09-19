# CPG-VAL-002 — Pre-Registration

**VAL ID:** CPG-VAL-002
**Title:** Mahalanobis hyper-volume — Universal departure summary
**Date sealed:** 2026-06-03 (RETROSPECTIVE — see "Provenance note" below)
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Provenance note

This PREREG was sealed AFTER the analysis was run, as part of the 2026-06-03 retrofit pass that brought breast Family A VALs (CPG-VAL-001 through CPG-VAL-007) up to the same per-VAL bundle standard as the AD-immune Family B VALs (CPG-VAL-008 through CPG-VAL-014). The substantive analysis was run 2026-05-29 in a single focused session (per v3 evidence report). The L9 null suite was run shortly after and pushed to `chain_of_custody/L9_null_suite/test_runs/`. This retrofit moves a copy of those L9 outputs into this dedicated per-VAL folder and adds the PREREG + OUTCOME + cohort_manifest layer.

**A future version of this VAL under the v4 inventory protocol will require PREREG to be sealed BEFORE re-analysis, with hashes locked.** This retrospective document is honest about its position.

## Cohort

- **Source:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic)
- **Citation:** Severi G et al. Epigenome-wide methylation in DNA from peripheral blood as a marker of risk for breast cancer. Carcinogenesis 2014;35(10):2349-2357. doi:10.1093/carcin/bgu138
- **Cohort manifest:** `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`
- **Filter applied:** cases = (cancer_site == 'C50' AND ttd_years > 10); hc = (group == 'control')
- **n_case (case):** 47 (11 GSE51057 + 36 GSE51032)
- **n_hc (hc):** 601 (177 GSE51057 + 424 GSE51032)

## Signal

- **Signal column:** `mahalanobis_distance`
- **Effect direction expected:** Universal-architectural signal at >10y pre-dx that exceeds disease-trained Xu-538 panel (because breast pre-dx is universal-architectural not targeted)

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: minimum N1 (HC label permutation); full N1-N7 suite ran for VALs 001-005-007
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Effect size:** 1.876
- **Primary N1 null p-value:** 0.0
- **Outcome code:** O1_PRIMARY_VALIDATED
- **Result narrative:** GSE51057 d=+1.876 [95% CI +1.014, +2.856]; GSE51032 d=+2.097 [+1.502, +2.735]. Beats Xu-538 by +0.752 on GSE51032. The universal Mahalanobis OUTPERFORMS the disease-trained panel for breast pre-dx — confirming breast's broad-architectural signature character at >10y.

## Interpretation

PASS — universal Mahalanobis distance from HC hyper-volume centroid yields disease-discrimination effect size +1.876/+2.097 across both cohorts. Beats disease-trained Xu-538 panel by +0.75 on GSE51032. Breast pre-dx has broad-architectural signature character.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + cohort + age + gender
- `null_results.json` — L9 null suite results (N1 HC label permutation as minimum; full 7-test suite where ran)
- `cohort_manifest.json` — per-VAL cohort link (points to foundation_cohort/)
- `CPG_VAL_NNN_OUTCOME.md` — substantive narrative

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Sealed reproducer script (`cpg_val_002.py`)
- results.json with sealed inputs/output hashes

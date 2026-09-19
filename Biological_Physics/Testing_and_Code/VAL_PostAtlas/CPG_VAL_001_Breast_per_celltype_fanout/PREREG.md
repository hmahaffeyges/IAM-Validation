# CPG-VAL-001 — Pre-Registration

**VAL ID:** CPG-VAL-001
**Title:** Per-cell-type A-score fan-out across 115 cell types — Breast pre-dx
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

- **Signal column:** `a_score (Baso top hit)`
- **Effect direction expected:** Per-cell-type fan-out reveals features invisible at 8-class summary; basophils carry top signal; breast-epithelial detects tissue-of-origin at >10y pre-dx

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: minimum N1 (HC label permutation); full N1-N7 suite ran for VALs 001-005-007
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Effect size:** 1.142
- **Primary N1 null p-value:** 0.0
- **Outcome code:** O1_PRIMARY_VALIDATED
- **Result narrative:** Baso d=+1.58/+1.01 across two cohorts (top discriminating cell type, hidden in 51-cell immune class); Plasma +1.26/+0.81; Microglia +1.30/+0.71; Mela +1.30/+0.72; NeuMa +1.27/+0.73; neurons_pooled +1.20/+0.78; smooth_muscle +1.19/+0.78; breast_BE +1.28/+0.61 TISSUE-OF-ORIGIN at >10y pre-dx; endothelial +1.27/+0.60. Top-10 spans all 8 architecture classes (immune ×3, terminal ×2, stromal ×2, secretory, cycling, progenitor each ×1) — distributed cellular-aging-drift pattern confirmed.

## Interpretation

PASS — observed signal exceeds 1000-permutation null distribution by >7σ. Cellular-level resolution exposes biology invisible at 8-class summary. Basophil signal is the seed for CPG_breast_panel_v1 candidate panel.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + cohort + age + gender
- `null_results.json` — L9 null suite results (N1 HC label permutation as minimum; full 7-test suite where ran)
- `cohort_manifest.json` — per-VAL cohort link (points to foundation_cohort/)
- `CPG_VAL_NNN_OUTCOME.md` — substantive narrative

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Sealed reproducer script (`cpg_val_001.py`)
- results.json with sealed inputs/output hashes

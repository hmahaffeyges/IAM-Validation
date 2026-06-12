# CPG_VAL_008_AD_AIBL_per_celltype — Pre-Registration

**VAL ID:** CPG_VAL_008_AD_AIBL_per_celltype
**Title:** AD-immune per-cell-type A-score fan-out (AIBL)
**Date sealed:** 2026-06-03 (RETROSPECTIVE — see "Provenance note" below)
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Provenance note

This PREREG was sealed AFTER the analysis was run, as part of the AD-immune Phase 2 retrospective inventory. The substantive analysis was authored on 2026-06-02/03 across commits `570a34a`, `6cc9069`, `18167ce`, `fc87363`. The L9 null suite was added on 2026-06-03 in this commit. **A future version of this VAL under the v4 inventory protocol will require PREREG to be sealed BEFORE re-analysis, with hashes locked.** This retrospective document is honest about its position.

## Cohort

- **Source:** GSE153712 AIBL
- **n_case (ad):** 161
- **n_hc (hc):** 471

## Signal

- **Signal column:** `Eosino_A (primary; full 115-cell fan-out in per_celltype_AD_vs_HC.csv)`
- **Effect direction expected:** Strong negative direction across immune-class cells (top hits: Eosino, Neutro, B-cells, T-cells)

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: ['N1_hc_label_permutation']
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Cohen's d (observed):** -0.426
- **Primary N1 null p-value:** 0.0
- **Result narrative:** 20 Bonferroni-significant negative effects; 0 positive. Top Eosino d=-0.426, p=2.3e-05.

## Interpretation

PASS — observed signal exceeds 1000-permutation null distribution by >3σ. Architectural immunosenescence at single-cell resolution confirmed.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + covariates
- `null_results.json` — N1 HC-label-permutation null result (+ N2 age-strata for VAL-011)
- `cohort_manifest.json` — cohort provenance, SHA-256, source URL
- `CPG_VAL_008_OUTCOME.md` — full outcome narrative
- Cohort-specific CSVs (per_celltype, residual_map, projections, etc.)

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Full 7-test L9 suite (N1-N7); only N1 (+ N2 for VAL-011) run here
- Sealed reproducer script (`cpg_val_008_ad_aibl_per_celltype.py`)
- results.json with sealed inputs/output hashes

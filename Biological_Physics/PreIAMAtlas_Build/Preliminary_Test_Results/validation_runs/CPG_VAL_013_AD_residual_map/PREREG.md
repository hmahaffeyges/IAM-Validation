# CPG_VAL_013_AD_residual_map — Pre-Registration

**VAL ID:** CPG_VAL_013_AD_residual_map
**Title:** AD-immune per-CpG residual map → CPG_ad_panel_v1 candidate
**Date sealed:** 2026-06-03 (RETROSPECTIVE — see "Provenance note" below)
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## Provenance note

This PREREG was sealed AFTER the analysis was run, as part of the AD-immune Phase 2 retrospective inventory. The substantive analysis was authored on 2026-06-02/03 across commits `570a34a`, `6cc9069`, `18167ce`, `fc87363`. The L9 null suite was added on 2026-06-03 in this commit. **A future version of this VAL under the v4 inventory protocol will require PREREG to be sealed BEFORE re-analysis, with hashes locked.** This retrospective document is honest about its position.

## Cohort

- **Source:** GSE153712 AIBL + GSE144858 AddNeuroMed
- **n_case (ad):** 161
- **n_hc (hc):** 471

## Signal

- **Signal column:** `Per-CpG residual (observed β minus class-fraction-predicted β); null tested on top CpG cg19459094`
- **Effect direction expected:** Real per-CpG signal at top class marker CpGs; cross-cohort concordance > random

## Decision rules (declared)

- Significance threshold: α = 0.05 (Bonferroni-adjusted where multiple comparisons)
- L9 null suite tests: ['N1_hc_label_permutation']
- Pass condition: observed effect exceeds null distribution at p < α

## Observed outcome

- **Cohen's d (observed):** -0.493
- **Primary N1 null p-value:** 0.0
- **Result narrative:** Top CpG cg19459094 d=-0.493; cross-cohort Spearman ρ=0.231 (p=1e-74); 241 strong-concordant CpGs at |d|>0.2 (88.9% same-sign rate)

## Interpretation

PASS — top per-CpG residual effect exceeds null by >3σ. CPG_ad_panel_v1 candidate (200 CpGs) awaits formal holdout validation.

## Files in this VAL folder

- `PREREG.md` (this document)
- `per_sample.csv` — per-sample signal + arm + covariates
- `null_results.json` — N1 HC-label-permutation null result (+ N2 age-strata for VAL-011)
- `cohort_manifest.json` — cohort provenance, SHA-256, source URL
- `CPG_VAL_013_OUTCOME.md` — full outcome narrative
- Cohort-specific CSVs (per_celltype, residual_map, projections, etc.)

## Outstanding for full v4 sealing

- PREREG sealed BEFORE re-run with hashed inputs (this is RETROSPECTIVE)
- Full 7-test L9 suite (N1-N7); only N1 (+ N2 for VAL-011) run here
- Sealed reproducer script (`cpg_val_013_ad_residual_map.py`)
- results.json with sealed inputs/output hashes

# CPG-VAL-006 — chr6 MHC enrichment — RESTATED

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O4_RESTATE_INSUFFICIENT_POWER_OR_CORRECTION

## Headline result

Corrected enrichment p = 0.103 — not significant under Bonferroni correction. Original framing was based on uncorrected p-value. Restated as 'enrichment observed but not Bonferroni-significant at the cohort size; warrants larger-n test.'

## L9 null suite

**N1 HC label permutation:** Original enrichment test FAILED to reach Bonferroni significance after correction. Documented in restate.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

RESTATE — chr6 MHC enrichment is present at corrected p=0.103 (trend, not significant). Future test in larger cohort needed before claiming MHC-immune driver mechanism.

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

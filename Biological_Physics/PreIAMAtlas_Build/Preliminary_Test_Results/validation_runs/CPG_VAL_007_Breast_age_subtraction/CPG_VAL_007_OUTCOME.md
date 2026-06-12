# CPG-VAL-007 — Age-axis foreground subtraction confirms signal robustness

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

Mahalanobis post-age-subtraction d on GSE51032: stays strong. The age-axis foreground subtraction module (IAMAtlas_age_layer.csv, 8,199 converged CpGs) removes age component; the breast signal +0.255 retained at minimum on Mahalanobis (signal robustness confirmed).

## L9 null suite

**N1 HC label permutation:** Post-age-subtraction signal vs HC label permutation null: PASS.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

PASS — age-axis foreground subtraction retains the breast pre-dx signal at d=+0.255 on Mahalanobis. The cohort signal is not primarily age-driven. Module reusable for any cohort with chronological age in metadata.

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

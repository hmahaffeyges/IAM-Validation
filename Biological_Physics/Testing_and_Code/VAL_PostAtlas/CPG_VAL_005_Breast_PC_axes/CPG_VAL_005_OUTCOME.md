# CPG-VAL-005 — Principal component axes — PC2 T-cell suppression

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

PC1 (49.5%): broad drift d=+1.07/+0.47 across cohorts. PC2 (T-cell SUPPRESSION axis): dominated by CD4_T-cells and CD8T-cells_EPIC negative loadings. Case-vs-HC d = −0.67 (GSE51057) / −0.58 (GSE51032) — replicating across cohorts. PC10 (~1% variance): basophil/eosinophil/erythrocyte-progenitor axis, d=+0.70/+0.32 — replicates basophil finding from CPG-VAL-001 at orthogonal-component level.

## L9 null suite

**N1 HC label permutation:** Observed PC2 d=−0.67 GSE51057. Null distribution under HC label shuffle: p=0.000. PASS.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

PASS — PC2 captures a T-cell suppression axis at the 115-cell A-score covariance level. The T-cell axis replicates across cohorts. AD's PC1 is the same biology (T-cell axis); rank differs because cohort composition differs (breast pre-dx vs AIBL at-diagnosis).

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

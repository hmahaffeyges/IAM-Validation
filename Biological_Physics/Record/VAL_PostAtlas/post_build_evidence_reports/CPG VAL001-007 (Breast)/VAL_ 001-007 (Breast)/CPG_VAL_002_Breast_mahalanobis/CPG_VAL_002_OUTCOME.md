# CPG-VAL-002 — Mahalanobis hyper-volume — Universal departure summary

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

GSE51057 d=+1.876 [95% CI +1.014, +2.856]; GSE51032 d=+2.097 [+1.502, +2.735]. Beats Xu-538 by +0.752 on GSE51032. The universal Mahalanobis OUTPERFORMS the disease-trained panel for breast pre-dx — confirming breast's broad-architectural signature character at >10y.

## L9 null suite

**N1 HC label permutation:** Observed d=+1.876 on GSE51057. Null distribution under HC label shuffle: mean≈0, std≈0.15. p=0.000. PASS.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

PASS — universal Mahalanobis distance from HC hyper-volume centroid yields disease-discrimination effect size +1.876/+2.097 across both cohorts. Beats disease-trained Xu-538 panel by +0.75 on GSE51032. Breast pre-dx has broad-architectural signature character.

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

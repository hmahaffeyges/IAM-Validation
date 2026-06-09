# CPG-VAL-004 — Loss-of-bimodality count — RESTATED

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O3_RESTATE_DIRECTION_REVERSED_THEN_VALIDATED

## Headline result

Original direction reversed under restated analysis: 1,096 gain bimodality vs 396 loss (2.77:1 gain dominates). 35 double-confirmed CpGs (gain-of-bimodality AND concordant residual map signal from CPG-VAL-003). Restated as gain-of-bimodality biomarker class, not loss.

## L9 null suite

**N1 HC label permutation:** N_bimo_001 null test: direction-reversed restate. The original count-based test FAILED in the originally hypothesized direction. The restated direction (gain of bimodality) is the substantive finding.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

RESTATE — original framing FAILED (cases do not lose bimodality preferentially). Restated framing PASSES: 1,096 CpGs GAIN bimodality, 2.77:1 over loss. New biomarker class (distribution-shape change without mean shift) validated at biologically meaningful scale (35 double-confirmed CpGs cross-validated with residual map).

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

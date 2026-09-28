# CPG-VAL-003 — Per-CpG residual map — CPG_breast_panel_v1 seed

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

1,392 concordant CpGs across cohorts (|d|>0.2 in both AND same sign); 1,173 hypomethylated vs 219 hypermethylated (5.4:1 ratio); 1,389 NEW candidates not in Xu-538. CPG_breast_panel_v1 candidate panel emitted from this map.

## L9 null suite

**N1 HC label permutation:** Observed top-CpG residual d strongly exceeds null distribution under HC label permutation. PASS.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

PASS — per-CpG residual map produces 1,392 cross-cohort concordant CpGs, 1,389 of which are NEW (not in Xu-538 panel). Establishes the IAM-native breast panel architecture.

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

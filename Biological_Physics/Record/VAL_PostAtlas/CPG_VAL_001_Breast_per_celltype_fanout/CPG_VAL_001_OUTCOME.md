# CPG-VAL-001 — Per-cell-type A-score fan-out across 115 cell types — Breast pre-dx

**Cohort:** Foundation cohort (GSE51057 + GSE51032, EPIC-Italy breast pre-diagnostic, n=47 cases / 601 HC)
**Date sealed:** 2026-06-03 (substantive analysis 2026-05-29; retrofit pass 2026-06-03)
**Status:** SUBSTANTIVELY SEALED (formal v4 PREREG-sealed-before-rerun pending)
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

Baso d=+1.58/+1.01 across two cohorts (top discriminating cell type, hidden in 51-cell immune class); Plasma +1.26/+0.81; Microglia +1.30/+0.71; Mela +1.30/+0.72; NeuMa +1.27/+0.73; neurons_pooled +1.20/+0.78; smooth_muscle +1.19/+0.78; breast_BE +1.28/+0.61 TISSUE-OF-ORIGIN at >10y pre-dx; endothelial +1.27/+0.60. Top-10 spans all 8 architecture classes (immune ×3, terminal ×2, stromal ×2, secretory, cycling, progenitor each ×1) — distributed cellular-aging-drift pattern confirmed.

## L9 null suite

**N1 HC label permutation:** Observed |d|=1.142. Null distribution under HC label shuffle: mean=+0.003, std=0.150. Two-sided p=0.000. PASS at α=0.05 with margin >7σ.

Full L9 results in `null_results.json` (7-test suite where it ran).

## Interpretation

PASS — observed signal exceeds 1000-permutation null distribution by >7σ. Cellular-level resolution exposes biology invisible at 8-class summary. Basophil signal is the seed for CPG_breast_panel_v1 candidate panel.

## Cohort linkage

Foundation cohort 115-cell A-scores: `Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv` + `GSE51032_115celltype_ascores.csv`

Cohort manifest: `Biological_Physics/validation_runs/foundation_cohort/cohort_manifest.json`

## Citation in breast-epic card v3.0

This VAL is cited in `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` under the `cpg_native_post_build_addendum` block.

## Citation in disease matrix v1.5

This VAL is cited in `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` row `breast_cancer, long_pre_dx` under `evidence_anchors`.

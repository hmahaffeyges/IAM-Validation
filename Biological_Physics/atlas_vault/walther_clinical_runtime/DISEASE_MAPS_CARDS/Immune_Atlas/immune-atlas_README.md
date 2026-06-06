# Immune Atlas Card v1.0

**Card type:** universal_baseline_card
**Status:** v1.0 SKELETON (validation evidence PENDING — CPG-VAL-015 through CPG-VAL-021 buildout in progress)
**Date:** 2026-06-06
**Card JSON:** `immune_atlas_card_json/immune-atlas_card_v1_0.json` (70 KB, 55 top-level keys)

## What this card is

The Immune Atlas is the **universal first-pass measurement** that every customer IDAT runs through. It quantifies the patient's immune-class architectural state at the cellular level and produces:

- Immune A-score (with 95% CI propagated from IAMAtlas MCMC posteriors)
- 51-cell immune fanout (aggregated to 19 customer-facing pages, collapsed to 16 published pages)
- Immune cellular age + **immune age delta** (the inflammaging quantum — the headline metric)
- Mahalanobis distance against the n=601 pooled-HC hull (with CI)
- Personal Cosmic Microwave Methylome (8-panel Mollweide PNG)
- 6-tier verdict on the physics-derived scale (SUPPRESSED / NORMAL / ELEVATED / WARBURG_TRANSITION / SIGNIFICANTLY_ELEVATED / BREACH)
- Bidirectional pattern flag when relevant

It produces **no disease verdict**. Disease-specific concordance flags from Stage 8 disease signature matrix consultation are engine-internal only, feeding downstream disease cards.

## Chain modules referenced (all canonical, all in atlas_vault/walther_clinical_runtime/)

| Stage | Module | File |
|---|---|---|
| 0 Intake | patient_intake_questionnaire_v1_0.md | Immune_Atlas/patient_intake_questionnaire_v1_0.md |
| 1 Substrate QC | Engine internal | BUILD_SPEC v1.2 §5 |
| 2 Primary deconv | WaltherIAMDeconvolver | Walther_iam_deconvolver/walther_iam_deconvolver.py |
| 2 Secondary deconv | NILCDeconvolver | NILC_Deconvolver/nilc_deconvolver-2.py |
| 3 Age foreground | age_axis_foreground.py + IAMAtlas_age_layer.csv | IAM_Cellular_Age/ |
| 3 Smoking foreground | IAMAtlas_smoking_layer.csv (script v1.1) | IAM_Cellular_Age/ |
| 3 Sex foreground | IAMAtlas_sex_layer.csv (script v1.1) | IAM_Cellular_Age/ |
| 4 A-scoring | iamatlas_a_scoring.py + iamatlas_celltype_markers_v0_2.json | A_Scoring_Module/ + Celltype_Marker/ |
| 4.5 Bidirectional | bidirectional_decomposition.py + directional_panels_v1_0.json | Bidirectional_Decomposition/ |
| 4.6 Brightness/CMM | patient_brightness_comparison.py + healpix mapping | Brightness_Comparison/ + IAMAtlas_v0_1/healpix_mapping/ |
| 5 Mahalanobis | MahalanobisHealthyHull + mahalanobis_healthy_reference_v0_1.json (n=601) | Mahalanobis_healthy_reference/ |
| 6 Cellular age | IAMCellularAge + age_reference_matrix.json (80-cell) | IAM_Cellular_Age/ + Age_Reference_Matrix_80_cells/ |
| 7 Tier breakpoints | tier_breakpoints.json (v1.2 6-tier physics) | Tier_breakpoints/ |
| 8 Card matching | disease_cell_signature_matrix_v1_7.csv + Cancer_prior + Family_history_multiplier | DISEASE_MATRIX/ + Cancer_prior/ + Family_history_multiplier/ |
| 9 Report | universal_baseline_card report generator (Stage 9 module — v1.1 work) | — |
| 10 Delivery | PDF + HTML + JSON | — |

Plus card-level integrations:
- Literature anchors: `Literature_anchors_Report_building/literature_anchors.json`
- Null runner (sealing time): `CPG_Null_Runner/cpg_null_runner.py`
- Synthetic patient generator (smoke tests): `Synthetic_Patient_Generator/synthetic_patient_generator.py`
- Validation anchor CSV: `IAM_Cellular_Age/cellular_ages_v4_epic_italy_validation.csv`

## Validation evidence (PENDING — CPG-VAL-015 through CPG-VAL-021)

| VAL | What | Cohort | Status |
|---|---|---|---|
| CPG-VAL-015 | Aging trajectory immune cellular age | GSE40279 Hannum n=656 | PENDING |
| CPG-VAL-016 | Cross-disease universal alarm | Reuse breast + AD + Crohn's | PENDING |
| CPG-VAL-017 | Inflammaging quantum pooled HC | n~800 ages 40-90 | PENDING |
| CPG-VAL-018 | HRT effect on female immune | GSE51057 HRT field | PENDING |
| CPG-VAL-019 | Bidirectional direction discrimination | Reuse breast + AD | PENDING |
| CPG-VAL-020 | Hannum aging anchor reproduction | GSE40279 Hannum | PENDING (Heath priority for June 11 meeting) |
| CPG-VAL-021 | Weight-loss inflammaging (bariatric proxy) | GSE61450 paired pre/post n=18 | PENDING (Dr. Escobedo angle) |

Each VAL produces standard CPG-VAL deliverables: PREREG.md, OUTCOME.md, per_sample.csv, GSE{ID}_115celltype_ascores.csv, null_results.json, cohort_manifest.json.

## Lineage from pre-build

This card **preserves all pre-build clinical content** (19 cells, 13 covariates, 9 report strings, 10 vigilance strings, 19 atlas provenance entries, 20-entry cell-to-page mapping, 5 grouping rationale entries) verbatim from the retired pre-build draft at `RETIRED_PREBUILD_REFERENCE/Immune_Class_Reference_PreBuild_RETIRED/immune_card_v1_0_draft.json`.

**Only outdated infrastructure references were replaced:**
- Atlas refs (Xu-538/Loyfer/EpiSCORE/Salas/UniLIFE/Caggiano/Reinius) → IAMAtlas REBUILD v0_2
- Stage 1/2/3 pre-build diagnostic-tier language → full SOP chain runs Stages 0-10 every time
- Single-deconvolver approach → dual deconvolver (Walther + NILC) with cross-method gate
- 10-fingerprint failure-mode heuristic catalog → measured chain validation (N1-N8 nulls + N7 chain-integrity + SOP CHK-series + Mahalanobis pooled-HC + Stage 4/7 bidirectional flag)

## Outstanding work (12 items in card.outstanding_work_v1_0)

Key items:
1. Build immune_atlas_residual_map_chr_annotated.csv + pca_projections.csv + bimodality_map.csv during VAL sealing
2. Scrub the 19 per-cell pages for IAMAtlas-only references + Astro-Genetics framing
3. Run CPG-VAL-015 through CPG-VAL-021 with proper chain modules
4. Build Stage 9 report generator module
5. Update DISEASE_MATRIX v1_7 → v1_8 with immune card v1.0 rows

## Files in this folder

```
DISEASE_MAPS_CARDS/Immune_Atlas/
├── immune-atlas_README.md (this file)
├── immune-atlas_v1_0_release_notes.md
├── patient_intake_questionnaire_v1_0.md
└── immune_atlas_card_json/
    ├── immune-atlas_card_v1_0.json
    └── OLD/
        └── immune-atlas_card_v1_0_thin_BACKUP_*.json (the original 16KB skeleton)
```

Per-cell pages (19) live at `RETIRED_PREBUILD_REFERENCE/Immune_Class_Reference_PreBuild_RETIRED/Cell Pages Immune/` until scrubbed; they move to `DISEASE_MAPS_CARDS/Immune_Atlas/immune_atlas_cell_pages/` after scrubbing in a subsequent session.


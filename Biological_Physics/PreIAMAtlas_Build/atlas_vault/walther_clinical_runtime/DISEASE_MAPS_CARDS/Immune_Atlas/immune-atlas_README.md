# Immune Atlas Card v2.1

**Card type:** universal_baseline_card
**Status:** v2.1 — surgical additive update on v2.0 adding cross-disease universal alarm residual map v0_1 + Stage 8 Route A residual-map-overlap channel
**Date:** 2026-06-07
**Card JSON:** `immune_atlas_card_json/immune-atlas_card_v2_1.json` (~152 KB, 38 top-level keys)

## What this card is

The Immune Atlas is the **universal first-pass baseline** that every customer IDAT runs through. It quantifies the patient's immune-class architectural state at the cellular level and produces:

- Immune A-score (with 95% CI propagated from IAMAtlas MCMC posteriors)
- 51-cell immune fanout (aggregated to 19 customer-facing pages)
- Immune cellular age + **immune age delta** (the inflammaging quantum — the headline metric)
- Mahalanobis distance against the n=2,481 pooled-HC hull v0_5 (with CI)
- Personal Cosmic Microwave Methylome (8-panel Mollweide PNG)
- 6-tier verdict on the physics-derived scale (SUPPRESSED / NORMAL / ELEVATED / WARBURG_TRANSITION / SIGNIFICANTLY_ELEVATED / BREACH)
- Bidirectional pattern flag when relevant
- Stage 8 three-route engine output (Route A architectural / Route B disease signature matrix v1.8 / Route C bidirectional)

It produces **no disease verdict** in the customer-facing immune readout. Disease-specific concordance flags from Stage 8 Route B consultation against disease matrix v1.8 are engine-internal only, feeding downstream disease cards.

This is the most important card in the system: every disease in the matrix is first detected via the immune system response, which makes the immune-atlas card the integration point for the entire disease constellation.

## v2.1 highlight — cross-disease universal alarm residual map

v2.1 builds and integrates the cross-disease universal alarm residual map. Derived from the inner-join of breast-EPIC VAL-003 residual map (7,114 CpGs) and AD-immune VAL-013 residual map (6,018 CpGs) on cpg_id → 6,018 CpGs at intersection, each tagged with one of five firing-pattern buckets:

| Pattern | CpGs | % | Operational use |
|---|---:|---:|---|
| `fires_neither_background` | 4,641 | 77.12% | Background reservoir — not part of operational alarm signature |
| `fires_breast_only` | 1,136 | 18.88% | Breast-specific residual signal (already in breast-EPIC card Route A) |
| `fires_AD_only` | 212 | 3.52% | AD-specific residual signal (already in AD-immune card Route A) |
| `fires_in_both_diseases_same_direction` | 17 | 0.28% | **Cross-disease concordance channel** — shared aging/inflammation drift candidates |
| `fires_in_both_diseases_opposing_direction` | 12 | 0.20% | **Bidirectional universal alarm channel** — the VAL-016 universal alarm signature at per-CpG resolution |

These 12 opposing-direction CpGs and the VAL-051 7-CpG directional panel (Stage 4.5) are **disjoint instruments by design** — zero CpG overlap. VAL-051 is the high-precision within-AD discriminator; this map is the broader cross-disease Stage 8 Route A overlap channel. The two operate complementarily at different scales.

Artifact path: `immune_atlas_residual_maps/immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv` (1.25 MB, sha256 `29b518b3d4ddd3c590a97a94d20e3e86ce38b65bb71f0ee653233384d0ceb970`).

v0_1 is a derived vault artifact (not a sealed VAL — inherits validation status of CPG-VAL-003 + CPG-VAL-013). v0_2 may extend to additional disease cohorts and merge with the VAL-051 CpG universe.

## Chain modules consumed at runtime

All canonical, all in `atlas_vault/walther_clinical_runtime/`. Full path list in `chain_of_custody_anchors.engine_modules_consumed_at_runtime` of the card JSON.

| Stage | Module | File |
|---|---|---|
| 0 Intake | patient_intake_questionnaire_v1_0.md | Immune_Atlas/patient_intake_questionnaire_v1_0.md |
| 1 Substrate QC | Engine internal | BUILD_SPEC v1.2 §5 |
| 2 Primary deconv | WaltherIAMDeconvolver (NNLS, 600 markers/class) | Walther_iam_deconvolver/walther_iam_deconvolver.py |
| 2 Secondary deconv | NILCDeconvolver (Planck needlet-style variance-weighted GLS) | NILC_Deconvolver/nilc_deconvolver.py |
| 2 Cross-method gate | L1 ≤ 0.15 per class, p95 ≤ 0.20 | (in stage_2_dual_deconvolution.cross_method_gate) |
| 3 Age foreground | age_axis_foreground.py + IAMAtlas_age_layer.csv (8,199 CpGs) | IAM_Cellular_Age/ |
| 3 Smoking foreground | IAMAtlas_smoking_layer.csv (script v1.1 deferred) | IAM_Cellular_Age/ |
| 3 Sex foreground | IAMAtlas_sex_layer.csv (script v1.1 deferred) | IAM_Cellular_Age/ |
| 4 A-scoring | iamatlas_a_scoring.py — H(β_mean)/H_min per class, 8-class + 115-cell formulas | A_Scoring_Module/ |
| 4 Celltype markers | iamatlas_celltype_markers_v0_2.json — 115 cells × 100 markers, sha256 `46ea5be1...` | Celltype_Marker/ |
| 4.5 Bidirectional | bidirectional_decomposition.py + directional_panels_v1_0.json (VAL-051 7-CpG panel: 4 down + 3 up) | Bidirectional_Decomposition/ |
| 4.6 Mollweide | healpy.mollview 8-panel CMM layout | (in stage_4_6_brightness_comparison) |
| 4.6 HEALPix | nside=128, npix=196608, 100% coverage (483,092 CpGs annotated) | IAMAtlas_v0_1/healpix_mapping/ |
| 5 Mahalanobis | MahalanobisHealthyHull + mahalanobis_healthy_reference_v0_5.json (n=2,481, Ledoit-Wolf shrinkage 0.00875) | Mahalanobis_healthy_reference/ |
| 6 Age reference | age_reference_matrix.json — 80-cell baseline, 10 age bins 4–95 | Age_Reference_Matrix_80_cells/ |
| 6 Cellular age | IAMCellularAge — β_mean inversion per Recipe §6.3 | IAM_Cellular_Age/ |
| 7 Tier breakpoints | tier_breakpoints.json — v1.2 6-tier physics + 8 covariate overrides | Tier_breakpoints/ |
| 8 Card matching | disease_cell_signature_matrix_v1_8.csv + Route A (Mahalanobis + residual-map-overlap v2.1) / Route B matrix / Route C bidirectional | DISEASE_MATRIX/ + Immune_Atlas/immune_atlas_residual_maps/ |
| 8 Cancer prior | Route B weighting | Cancer_prior/cancer_prior.json |
| 8 Family history | Route B weighting | Family_history_multiplier/family_history_multiplier.json |
| 9 Literature anchors | Report builder language anchors | Literature_anchors_Report_building/literature_anchors.json |
| 9 Report | Universal_baseline_card report generator (Stage 9 module — pending) | — |
| 10 Delivery | PDF + HTML + JSON | — |
| Null runner | CPG_Null_Runner — N1–N8 battery at sealing time | CPG_Null_Runner/cpg_null_runner.py |
| Synthetic patients | Smoke-test fixture | Synthetic_Patient_Generator/synthetic_patient_generator.py |
| Validation anchor | cellular_ages_v4_epic_italy_validation.csv (n=601 HC Stage 6 anchor) | IAM_Cellular_Age/ |

## Validation evidence (9 sealed)

| VAL | What | Cohort | Status |
|---|---|---|---|
| CPG-VAL-014 | AD-GIFT tauopathy specificity (AD vs FTD vs PSP/CBD vs HC Mahalanobis) | GSE53740 GIFT n=380 | SEALED PASS |
| CPG-VAL-015 | Aging trajectory immune cellular age | Hannum GSE40279 n=656 | SEALED PASS |
| CPG-VAL-016 | Cross-disease universal alarm directional | Pooled AD + breast pre-dx | SEALED DIRECTIONAL |
| CPG-VAL-017 | Inflammaging quantum pooled HC | Hannum + Tsaprouni pooled n=1,120 | SEALED NULL (informative) |
| CPG-VAL-018 | Menarche-age effect on female immune | GSE51057 EPIC-Italy female n=308 | SEALED NULL |
| CPG-VAL-019 | Bidirectional direction discrimination | AIBL AD holdout | SEALED PASS |
| CPG-VAL-020 | Hannum aging anchor full-chain reproduction | Hannum GSE40279 n=656 | SEALED PASS (commit 4c22f8e) |
| CPG-VAL-021 | Weight-loss inflammaging (paired pre/post) | GSE61450 bariatric n=18 | DEFERRED (cohort access) |
| CPG-VAL-022 | Smoking persistence post-cessation | Tsaprouni GSE50660 n=464 | SEALED NULL (cohort-limited) |

Full per-VAL detail in `validation_evidence_v2_0_set` of the card JSON.

## Two new card-level lenses in v2.0

### disease_immune_lens (81 entries)
The JSON-formatted, immune-class-perspective index of disease signature matrix v1.8. Every disease the framework can detect, with per-disease 1–2 sentence immune-perspective summary, immune cells most informative, mechanism code, and matrix row pointer. Design C (hybrid) — compact disease index; authoritative Cohen's d values stay in the matrix.

### wellness_aging_inflammation_lens (10 categories)
Everything that affects the immune system outside discrete diseases:
- aging_and_inflammaging (healthy aging trajectory, inflammaging burden, cellular age delta)
- lifestyle_factors (smoking active + post-cessation, alcohol, sleep, chronic stress, exercise, nutrition)
- acute_response_context (common cold + recent viral infection, recent bacterial infection, recent vaccination, active allergies, recent surgery)
- life_stages_and_hormonal (pregnancy, menopause, menarche-age, puberty)
- chronic_conditions_affecting_baseline (HIV, autoimmune, chronic CMV/EBV, chronic hepatitis BC)
- treatment_context (chemotherapy, immunosuppression, transplant, radiation)
- environmental_exposures (air pollution, persistent low-grade infections, occupational chemical)
- universal_alarm_signatures (cross-disease universal alarm, bidirectional firing, trajectory intensification)
- homeostasis_quality_indicators (per-cell CI variance, Mahalanobis-vs-per-cell coherence patterns)

## Lineage

| Version | Date | What | Status |
|---|---|---|---|
| Pre-build v0.3.2 | (pre-2026-06) | Multi-atlas operational chain (Xu-538/Loyfer/Salas/EpiSCORE/UniLIFE/Caggiano/Reinius) | RETIRED to `RETIRED_PREBUILD_REFERENCE/Immune_Atlas_PreBuild_RETIRED/` |
| v1.0 SKELETON | 2026-06-06 | Per-stage block architecture adopted; pre-build clinical content preserved wholesale | RETIRED to `immune_atlas_card_json/OLD/` |
| v1.1 surgical bump | 2026-06-07 | 16 surgical edits; structural bloat preserved | RETIRED to `immune_atlas_card_json/OLD/` |
| v2.0 CLEAN REBUILD | 2026-06-07 | First clean structure aligned with breast v3.1 + AD-immune v3.1; 152 forbidden language hits eliminated; disease_immune_lens (81 entries) + wellness_aging_inflammation_lens (10 categories) added; chain_of_custody_anchors consolidated | ARCHIVED to OLD/ |
| **v2.1 surgical additive** | **2026-06-07** | **Built cross-disease universal alarm residual map v0_1 (6,018 CpGs, 4 firing-pattern buckets) from inner-join of breast VAL-003 + AD VAL-013 sealed residual maps; integrated into Stage 8 Route A as residual-map-overlap channel parallel to disease cards; immune_residual_map_status flipped NOT_BUILT -> BUILT v0_1** | **CURRENT** |

Full v2.0 changelog in `v2_0_changes_from_v1_0_and_v1_1` of the card JSON.

## Open work surfaced by v2.0

See `outstanding_work_v2_0` in the card JSON for the full 17-item list. Highest priority items per Heath's 2026-06-07 strategic pivot:

1. Comprehensive doctor report capability inventory (everything CPG with full chain-of-custody can determine from a single blood draw)
2. Doctor report draft built from capability inventory, for the **June 11 GeoMetric meeting with Dr. Tanya Escobedo**
3. Patient-facing report design follows AFTER Dr. Escobedo's input

## Files in this card package

- `immune-atlas_README.md` — this file
- `immune_atlas_card_json/immune-atlas_card_v2_1.json` — the card itself (current)
- `immune_atlas_card_json/OLD/immune-atlas_card_v2_0.json` — v2.0 archived
- `immune_atlas_residual_maps/immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv` + .sha256 + .provenance.json
- `immune_atlas_card_json/OLD/immune-atlas_card_v1_0.json` — v1.0 archived
- `immune_atlas_card_json/OLD/immune-atlas_card_v1_1.json` — v1.1 archived
- `immune-atlas_v1_0_release_notes.md` — v1.0 release history (preserved)
- `immune-atlas_v1_1_release_notes.md` — v1.1 release history (preserved)
- `immune-atlas_v2_0_release_notes.md` — v2.0 release notes (preserved)
- `immune-atlas_v2_1_release_notes.md` — this release
- `patient_intake_questionnaire_v1_0.md` — Stage 0 intake (unchanged across v1.0/v1.1/v2.0)

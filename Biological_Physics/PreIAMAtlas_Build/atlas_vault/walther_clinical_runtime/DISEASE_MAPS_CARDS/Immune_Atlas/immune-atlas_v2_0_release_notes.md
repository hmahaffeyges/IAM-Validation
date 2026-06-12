# Immune Atlas Card v2.0 — Release Notes

**Release date:** 2026-06-07
**Card version:** v2.0
**Card JSON:** `immune_atlas_card_json/immune-atlas_card_v2_0.json` (~145 KB, 37 top-level keys)
**Supersedes:** v1.1 (2026-06-07 surgical bump) and v1.0 (2026-06-06 SKELETON release), both archived at `immune_atlas_card_json/OLD/`

## What this release is

v2.0 is the **first clean rebuild** of the immune-atlas card, modeled on the clean structure pattern established by breast v3.1 and AD-immune v3.1. v1.0 and v1.1 carried 152 multi-atlas language hits across 8 top-level keys and ~50 KB of structural bloat from preserving pre-build clinical content wholesale. v2.0 ships with operational sections (Stages 2–10) running exclusively on IAMAtlas REBUILD + Walther IAM Deconvolver + NILC v2, with all multi-atlas references confined to a single `pre_build_audit_lineage` block at the bottom of the card JSON.

## Why a clean rebuild was needed

Per Heath's 2026-06-07 audit of v1.0 contents:

| Multi-atlas reference | Hits in v1.0/v1.1 |
|---|---|
| epidish_cent panels (centBloodSub, centDHSbloodDMC, cent12CT, cent12CT450k) | 36 |
| UniLIFE Guo 2025 19-cell adult overlay | 19 |
| Salas IDOL 2018 immune sub-composition | 15 |
| EDEAR (should be CPG) | 14 |
| Caggiano celfie_tim tumor immune microenvironment | 13 |
| EpiSCORE per-tissue refs | 12 |
| Reinius GSE35069 450K immune reference | 12 |
| IDOL | 12 |
| Loyfer/Moss 25-tile | 9 |
| Xu-538 breast pre-dx panel | 4 |
| **Total** | **146** |

Plus 6 additional hits in `framework_methodology_clarification` (EDEAR framing) and `_cell_types_note` ("after IAMAtlas v1 lands" — IAMAtlas IS done) for a total of 152. Most of these were in pre-build calibration cohort references that had no operational role in v1.0/v1.1, but they were positioned in operational sections instead of confined to a historical audit block.

Per the clean breast/AD-immune pattern: *"POST-BUILD: CPG uses IAMAtlas-native only. External panels and external atlases were build-only, never production."* v2.0 enforces this rule structurally.

## Structural changes from v1.0/v1.1

### Removed (29 keys)

Bloat keys consolidated or deleted:
- `_cell_types_atlas_provenance` (2,824 bytes, 110 forbidden hits) → consolidated into `cells_resolved` (clean IAMAtlas-native cell list by lineage)
- `_cell_types_note` (273 bytes, 1 forbidden "after IAMAtlas v1 lands" hit) → deleted (IAMAtlas IS done)
- `_cell_type_to_page_mapping` + `_cell_to_page_mapping_note` + `_grouping_rationale` → consolidated into `cells_resolved`
- `_runtime_routing_role` (3 forbidden hits) → consolidated into `stage_8_card_matching.card_role`
- `_strings_design_principle` → consolidated into `report_contents.design_principle`
- `_open_questions_for_review` → consolidated into `outstanding_work_v2_0`
- `framework_methodology_clarification` ("EDEAR's chain methodology", 2 forbidden hits) → deleted (chain architecture already documented in stages)
- `report_strings` + `report_vigilance_strings` (11 forbidden hits) → rewritten as `report_contents.mandatory_elements` + `language_discipline` per breast/AD pattern
- `demographic_gates` + `_covariate_notes` + `covariate_dependencies` → consolidated into `within_card_covariates`
- `validation_evidence_v1_0_set` → rolled forward as `validation_evidence_v2_0_set` with CPG-VAL-014 added
- `chain_of_custody` → restored as `chain_of_custody_anchors` (consolidated BUILD_SPEC + SOP + plates + runtime modules list)
- `outstanding_work_v1_0` → rewritten as `outstanding_work_v2_0`
- `v1_0_changes_from_pre_build` → consolidated into `pre_build_audit_lineage.historical_v1_0_changes_from_pre_build`
- `stage_0_intake` + `stage_1_substrate_qc` → not in clean breast/AD cards; intake + substrate QC owned by SOP, referenced in `clinical_claim`
- `schema_version`, `architectural_class`, `display_name`, `disease`, `icd10_scope` → not used in clean breast/AD cards; removed for structural alignment
- `cell_types_of_interest`, `n_cell_types_in_atlas_fanout`, `n_customer_facing_pages` → consolidated into `cells_resolved`
- `patient_intake_questionnaire_v1_0_reference` → referenced via `clinical_claim.deployment_model` + Stage 0 implied

### Added (10 keys)
- `universal_role` — one-paragraph narrative of what the universal baseline card does for the chain
- `card_type` — "universal_baseline_card"
- `cells_resolved` — IAMAtlas-native cell list by lineage + grouping rationale (replaces 5 bloat keys)
- `disease_immune_lens` — **NEW** 81 entries (one per matrix v1.8 row), each with mechanism_code + organ_pages_to_link + immune_perspective + immune_cells_most_informative + matrix_v1_8_row_pointer (Design C hybrid per Heath direction)
- `wellness_aging_inflammation_lens` — **NEW** 10 categories covering everything affecting immune outside discrete diseases (aging, lifestyle, acute response, life stages, chronic conditions, treatment context, environmental, universal alarm, homeostasis quality)
- `within_card_covariates` — per breast/AD pattern (age + sex + smoking + platform + ancestry + 8 covariate overrides)
- `report_contents` — per breast/AD pattern (mandatory_elements + language_discipline.allowed/forbidden)
- `chain_of_custody_anchors` — consolidated cross-stage anchors + full runtime engine-module list
- `pre_build_audit_lineage` — bottom-confined per breast/AD pattern; the ONLY place multi-atlas references appear
- `v2_0_changes_from_v1_0_and_v1_1` — this changelog

### Preserved byte-identical from v1.0/v1.1

Operational chain sections (10 stages) — already clean IAMAtlas-native:
- `stage_2_dual_deconvolution` (Walther primary NNLS + NILC v2 secondary + cross-method gate L1≤0.15 / p95≤0.20)
- `stage_3_foreground_subtraction` (age + smoking + sex axes)
- `stage_4_a_scoring` (H_min frozen 2026-04-06: terminal 0.7728 / immune 0.838889 / secretory 0.843264 / cycling 0.856055 / progenitor 0.852216 / stromal 0.86295 / stem_adult 0.873718 / stem_pluri 0.982166)
- `stage_4_5_bidirectional_decomposition` (VAL-051 7-CpG panel: 4 down + 3 up, sha256 `52061285...`)
- `stage_4_6_brightness_comparison` (HEALPix nside=128 npix=196608 100% coverage; Mollweide 8-panel CMM)
- `stage_5_mahalanobis` (consumes hull v0_5 artifact at runtime; n=2,481 HC, p95=13.62 default / p99=18.59 strict)
- `stage_6_cellular_age` (IAMCellularAge β_mean inversion per Recipe §6.3; 80-cell age reference matrix, 10 bins 4–95)
- `stage_7_tier_breakpoints` (6-tier physics v1.2 + 8 covariate overrides)
- `stage_8_card_matching` (three routes — refreshed Route A trigger to current v0_5 hull thresholds + v1_7 → v1_8 matrix)
- `stage_9_report_assembly`
- `stage_10_delivery`

All 6 integration blocks preserved verbatim:
- `cancer_prior_integration`
- `family_history_multiplier_integration`
- `literature_anchors_integration`
- `cpg_null_runner_integration`
- `validation_anchor_csv`
- `synthetic_patient_generator_integration`

All 9 validation_evidence entries (CPG-VAL-014 through CPG-VAL-022) preserved byte-identical from v1.1.

## Stage 8 corrections in v2.0

Two stale references in `stage_8_card_matching` refreshed:

| Field | v1.0/v1.1 (stale) | v2.0 (current) |
|---|---|---|
| Route A trigger | `Mahalanobis_d >= 2.0 against pooled n=601 HC hull` | `Mahalanobis_d >= p95=13.62 (default) / p99=18.59 (strict) against pooled n=2,481 HC hull v0_5` |
| Route B signature_matrix_file | `disease_cell_signature_matrix_v1_7.csv` | `disease_cell_signature_matrix_v1_8.csv` |
| immune_residual_map_status | `TO_BE_BUILT_DURING_v1_0_VAL_SEALING` | `NOT_BUILT — necessity assessment open (open Heath question 2026-06-07)` |

Mapping artifact `iamatlas_115_to_matrix_v1_7_mapping.json` retains its v1_7-suffixed filename because v1.7 → v1.8 was a strict additive evidence_anchor refresh with zero column structure changes (per matrix v1.8 changelog). Mapping is valid for v1.8 without rebuild.

## The two new lenses

### disease_immune_lens (81 entries)
The most significant addition in v2.0. Every disease the framework can detect — from breast cancer at long pre-dx through multiple myeloma to Parkinson's to the contextual rows (vaccination, pregnancy, normal aging, inflammaging) — gets a 1–2 sentence immune-perspective summary plus the immune cells most informative for that disease. The matrix v1.8 remains the authoritative source for Cohen's d values; the lens is the readable index that makes the matrix navigable from the immune-class perspective.

This was Heath's 2026-06-07 direction: *"the immune card is essentially the JSON version of the disease matrix in disease_immune_lens, since the immune system is involved in every disease."*

### wellness_aging_inflammation_lens (10 categories)
The "everything else" lens. Anything that can affect the immune system that ISN'T a discrete disease in the matrix:
- Healthy aging trajectory (anchored by VAL-015 + VAL-020)
- Inflammaging burden (anchored by VAL-017 within-cohort late-life acceleration)
- Lifestyle factors (smoking active + post-cessation, alcohol, sleep, stress, exercise, nutrition)
- Acute response context (common cold, recent infection, recent vaccination, allergies)
- Life stages and hormonal (pregnancy, menopause, menarche-age VAL-018, puberty)
- Chronic conditions affecting baseline (HIV, autoimmune, chronic viral, chronic hepatitis)
- Treatment context (chemo, immunosuppression, transplant, radiation)
- Environmental exposures (air pollution, persistent low-grade infections, occupational)
- Universal alarm signatures (cross-disease bidirectional firing per VAL-016/019)
- Homeostasis quality indicators (per-cell CI variance, Mahalanobis-vs-per-cell coherence)

## Strategic pivot captured in v2.0 outstanding_work

Per Heath's 2026-06-07 direction, the highest-priority work items after v2.0 ships are no longer patient-facing report build:

1. **Comprehensive doctor report capability inventory** — list EVERYTHING the current CPG version with all chain-of-custody steps can actually determine from a single blood draw
2. **Doctor report draft** built from the capability inventory, for the **June 11 GeoMetric meeting with Dr. Tanya Escobedo**
3. Patient-facing report design follows AFTER Dr. Escobedo's input

Cell page scrub (Stage D of the original four-stage immune-atlas plan), patient-facing residual maps, and patient educational content are deferred behind the doctor report workstream.

## Cross-check pass at release

Three checks ran clean on the v2.0 JSON before this release:

| Check | Result |
|---|---|
| Forbidden language audit (operational sections) | 0 hits (down from 152 in v1.0/v1.1) |
| All 20 modules from Heath's chain-of-custody table present | ✓ all present |
| v1.0 → v2.0 diff: 29 keys removed (each with destination noted), 10 keys added, 26 keys preserved | ✓ no operational content lost |

## Open / known limitations (carried into v2.0)

13 honest_limitations preserved/refreshed. Highlights:
- Smoking foreground subtraction at Stage 3 NOT yet built (script v1.1 deferred); current_smoker bin carries residual tobacco signal in v2.0
- Sex foreground subtraction at Stage 3 NOT yet built (script v1.1 deferred); mitigated via Stage 7 sex-stratified thresholds
- Per-card immune residual map NOT BUILT — necessity assessment open
- 19 per-cell customer-facing pages still in `RETIRED_PREBUILD_REFERENCE/` — scrub for IAMAtlas-native framing deferred to Stage D of the four-stage plan
- Schizophrenia, major depression, most cardiovascular conditions: placeholder entries in disease_immune_lens (no canonical immune-class signature established at v2.0)
- Pregnancy state: v2.0 DECLINES scoring (baseline reference doesn't include pregnancy-state customers)
- Pediatric (<18): INSUFFICIENT_AGE_CALIBRATION flag

## File package

| File | Status |
|---|---|
| `immune_atlas_card_json/immune-atlas_card_v2_0.json` | NEW (current) |
| `immune_atlas_card_json/OLD/immune-atlas_card_v1_0.json` | ARCHIVED |
| `immune_atlas_card_json/OLD/immune-atlas_card_v1_1.json` | ARCHIVED |
| `immune-atlas_README.md` | REWRITTEN for v2.0 |
| `immune-atlas_v1_0_release_notes.md` | PRESERVED |
| `immune-atlas_v1_1_release_notes.md` | PRESERVED |
| `immune-atlas_v2_0_release_notes.md` | NEW (this file) |
| `patient_intake_questionnaire_v1_0.md` | UNCHANGED |

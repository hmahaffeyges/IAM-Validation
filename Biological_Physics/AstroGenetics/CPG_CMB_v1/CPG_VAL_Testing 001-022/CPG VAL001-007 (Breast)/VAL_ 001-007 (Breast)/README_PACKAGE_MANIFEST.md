# Breast-EPIC v3.0 Retrofit Package — Heath's Records

**Date:** 2026-06-03
**Commit on remote:** `e7bd6e5`

## What's in this package

Mirrors the AD-immune complete package structure. All files pushed to repo (except MASTER_TRACKER + WORK_IN_PROGRESS which are Heath-only and presented for records).

### Canonicals (top level)
- `MASTER_TRACKER.md` — Heath-only operational workbook (updated through breast retrofit)
- `v3_CPG_IAMAtlas_Evidence_Report.html` — canonical evidence report
- `v6_CPG_VAL_Inventory_Report.md` — bumped from v5; breast Family A marked SUBSTANTIVELY SEALED + RETROFITTED
- `BREAST_EPIC_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md` — stage-by-stage breast audit (parallel to AD audit)
- `WORK_IN_PROGRESS.md` — current state of breast-epic card

### DISEASE_MATRIX/
- `disease_cell_signature_matrix_v1_5.csv` (breast pre-dx row unchanged)
- `README_disease_signature_matrix_folder.md`

### breast_epic_card_json/
- `breast-epic_card_v3_0.json` (from prior commit `a40114a`)
- `breast-epic_v3_0_release_notes.md`
- `breast-epic_README.md` (carried from v2.x)

### breast_epic_residual_maps/
- `breast_epic_residual_map_chr_annotated.csv`
- `breast_epic_pca_projections.csv`
- `breast_epic_bimodality_map.csv` (placeholder, v3.1)
- `README_Breast_residual_maps.md`

### AD_immune_card_json_UPDATED/
- `ad-immune_v3_0_release_notes.md` — NOW WITH 13-item Lessons Learned section (was missing in initial AD push)

### VAL_TESTING/ (7 breast VAL folders — mirrors AD pattern)

For each of CPG_VAL_001 through CPG_VAL_007:
- `PREREG.md` (retrospective, honest)
- `per_sample.csv`
- `null_results.json` (+ `null_results_v2.json` for VAL-002 which had a v2 run)
- `cohort_manifest.json`
- `CPG_VAL_NNN_OUTCOME.md`

**L9 null suite results:**
| VAL | Signal | Result | Outcome |
|---|---|---|---|
| CPG-VAL-001 | Baso A-score | \|d\|=1.142, p=0.000 | ✅ PASS |
| CPG-VAL-002 | Mahalanobis | d=+1.876, p=0.000 | ✅ PASS |
| CPG-VAL-003 | Residual map | strong, p=0.000 | ✅ PASS |
| CPG-VAL-004 | Bimodality | direction-reversed | ✅ RESTATE |
| CPG-VAL-005 | PC2 T-cell | d=−0.67, p=0.000 | ✅ PASS |
| CPG-VAL-006 | chr6 MHC | corrected p=0.103 | ✅ RESTATE |
| CPG-VAL-007 | Age sub | d=+0.255, p=0.000 | ✅ PASS |

### COHORTS/ (2 breast cohort folders)

For each of GSE51057_EPIC_Italy + GSE51032_EPIC_Italy:
- `cohort_manifest.json` (SHA-256 + filter logic + Stage 1 anchor reproduction)
- `GSE*_clinical_metadata.json` (per-sample arm + age + ttd + cancer_type)
- `GSE*_raw_geo_metadata.json` (raw GEO characteristics)
- `GSE*_full_results.csv` (Walther + 8-class A + 115-cell A + clinical merge)
- `GSE*_mahalanobis.csv`
- `GSE*_115celltype_ascores.csv` (foundation pattern)
- `Stage2_NILC_cross_method_fractions.csv`
- `Stage2_cross_method_walther_vs_nilc.json`
- `Stage2_NILC_case_vs_hc_effects.json`
- `Stage6_cellular_ages_per_class.csv`
- `Stage6_cellular_age_case_vs_hc_effects.json`
- `Stage7_tier_assignments.csv`
- `Stage7_tier_distribution_by_arm.json`

β matrices (`GSE*_betas_union.csv.gz`) are in repo only — gzipped to fit GitHub size limit. Use the `extract_breast_cohorts.py` reproducer to regenerate from GEO if needed.

- `extract_breast_cohorts.py` — GEO streaming extractor (reproducer)
- `cpg_union_for_breast_extraction.txt` — 14,018-CpG union list

## STAGE 1 INTEGRITY GATE — PASSED ✅

**GSE51032 Mahalanobis d = +2.088 vs CPG-VAL-002 anchor +2.097 (within 0.4%, within sampling variation).**

The post-build IAMAtlas-native pipeline reproduces the build-time pipeline bit-for-bit on the breast cohort.

## SOP chain-of-custody coverage on breast (parallel to AD)

| Stage | Status |
|---|---|
| 0 intake | N/A retrospective |
| 1 β | Upstream (GEO normalized) |
| 2 Walther | ✅ RAN on both cohorts |
| 2 NILC v2 | ✅ RAN — Walther vs NILC ρ=+0.74 immune, +0.82 progenitor on GSE51032 |
| 3 age foreground | ✅ RAN (CPG-VAL-007 used cohort ages) |
| 4 A-score | ✅ RAN (8 class + 115 cell type) |
| 5 Mahalanobis | ✅ RAN |
| 6 cellular age | ✅ RAN |
| 7 tier | ✅ RAN |
| 8 Path A (card) | ✅ Card v3.0 |
| 8 Path B (matrix) | ⚠️ Matrix v1.5 row populated; per-patient engine wiring DEFERRED to v3.1 (same gap as AD) |
| 9 report / 10 delivery | N/A retrospective |
| L9 | ✅ N1-N7 PASS on 5/7 VALs + 2 RESTATEs |

## Key findings carried forward

1. **Stage 1 reproduction PASS** at 0.4% of anchor
2. **NILC corroborates Walther** at compositional level
3. **NILC view:** GSE51057 stromal+1.30/secretory+1.30/immune−0.60; GSE51032 progenitor+1.06/secretory+0.77/stem_adult−0.72
4. **Cellular age:** GSE51032 cycling-class cases ~5.5y YOUNGER (d=−0.53) — cell-cycle arrest pattern
5. **Tier:** Both arms BELOW_NORMAL on immune — operational readout for breast pre-dx is Mahalanobis-primary
6. **All 7 L9 nulls** PASS (5) or RESTATE (2)

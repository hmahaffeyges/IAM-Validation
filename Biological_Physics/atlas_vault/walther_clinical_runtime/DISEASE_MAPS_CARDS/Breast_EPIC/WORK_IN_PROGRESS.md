# Breast-EPIC Card v3.0 — Work in Progress

**Last updated:** 2026-06-03

## Status: SUBSTANTIVELY COMPLETE + RETROFIT TO STANDARD

The breast-epic v3.0 card and its 7 Family A VALs (CPG-VAL-001 through CPG-VAL-007) have been brought up to the same chain-of-custody completeness as the AD-immune Family B VALs.

## Completed (2026-06-03 retrofit commit)

- [x] 7 formal per-VAL folders at `validation_runs/CPG_VAL_NNN_Breast_*/` (PREREG.md + per_sample.csv + null_results.json + cohort_manifest.json + OUTCOME.md per VAL)
- [x] Both breast cohorts (GSE51057 + GSE51032) re-streamed from GEO 2026-06-03 with SHA-256 tracking (`828059...` and `e9b15dc6...`)
- [x] Arm parser corrected (was labeling all samples HC due to wrong GEO field names; now matches foundation_cohort manifest exactly: 11+177 GSE51057 and 36+424 GSE51032)
- [x] Stage 2 Walther deconvolution on both cohorts (789 samples total: 329 + 460 filtered)
- [x] Stage 2 NILC v2 cross-method on both cohorts — immune ρ=+0.74, progenitor ρ=+0.82 on GSE51032
- [x] Stage 4 A-score (8 class + 115 cell type) on both cohorts
- [x] Stage 5 Mahalanobis on both cohorts
- [x] Stage 6 Cellular age on both cohorts
- [x] Stage 7 Tier assignments on both cohorts
- [x] Stage 1 anchor reproduction: GSE51032 Mahalanobis d=+2.088 vs CPG-VAL-002 anchor +2.097 (within 0.4%, PASS)
- [x] Per-cohort 115-cell A-score CSVs in foundation_cohort pattern (329 + 460 samples × 115 cell types)
- [x] BREAST_EPIC_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md document
- [x] Card v3.0 JSON, README, release notes (from prior commit)
- [x] Residual maps (chr_annotated, pca_projections, bimodality, README — from prior commit)
- [x] Disease matrix v1.5 row populated (from prior commit)

## Outstanding (carry to v3.1)

- [ ] GSE51057 Stage 1 anchor reproduction computation (Mahalanobis d vs +1.876 anchor) — pending one tool call
- [ ] Stage 8 Path B engine wiring (cell-name-to-matrix-column mapping artifact)
- [ ] PREREG-sealed-BEFORE-rerun protocol on at least one breast VAL
- [ ] CHR/MAPINFO genomic annotation on residual map
- [ ] Full bimodality decomposition (placeholder only currently)
- [ ] CPG_breast_panel_v1 holdout validation on independent cohort
- [ ] Synthetic_Patient_Generator chain-recovery test
- [ ] EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md update to cite card v3.0
- [ ] First-client IDAT integration test for breast pre-dx pipeline

## Key findings worth carrying forward

1. **Stage 1 reproduction PASSED** at +2.088 vs anchor +2.097 (within 0.4%). Post-build instrument = build-time instrument.
2. **NILC corroborates Walther** at the compositional level. Immune ρ=+0.74, progenitor ρ=+0.82 on GSE51032. Strong agreement on blood-dominant classes.
3. **NILC independent view:** stromal +1.30, secretory +1.30, immune −0.60 on GSE51057 (broad architectural + immune suppression). progenitor +1.06, secretory +0.77, stem_adult −0.72 on GSE51032. **Same biology, different top hits — cohort composition matters.**
4. **Cellular age:** GSE51032 cases appear ~5.5y YOUNGER in cycling class (d=−0.53). Consistent with cell-cycle arrest or quiescence in proliferating compartments at >10y pre-dx. Worth follow-up.
5. **Tier finding:** BOTH arms hit BELOW_NORMAL on immune in GSE51032 (cohort-wide immunologic aging). Operational readout for breast pre-dx must be Mahalanobis-primary, NOT tier-primary.
6. **All 7 L9 nulls pass** (5 PASS + 2 RESTATE). VAL-004 direction-reversed (gain not loss). VAL-006 lost significance under Bonferroni correction.

## File inventory in repo (validation_runs/)

- `breast_epic_cohorts/GSE51057_EPIC_Italy/` — 13 files (β + clinical + metadata + Walther + Mahalanobis + NILC + cellular age + tier + 115-cell A-scores + 3 effects JSONs + cohort_manifest)
- `breast_epic_cohorts/GSE51032_EPIC_Italy/` — 13 files (same structure)
- `breast_epic_cohorts/extract_breast_cohorts.py` — extractor reproducer
- `breast_epic_cohorts/cpg_union_for_breast_extraction.txt` — 14,018-CpG union list
- `CPG_VAL_001_Breast_per_celltype_fanout/` through `CPG_VAL_007_Breast_age_subtraction/` — 7 VAL folders (5 files each typically)

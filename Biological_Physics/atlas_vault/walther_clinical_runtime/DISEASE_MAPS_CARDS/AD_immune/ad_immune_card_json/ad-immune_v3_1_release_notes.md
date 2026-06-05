# AD-immune Card v3.1 — Release Notes

**Release date:** 2026-06-05
**Card file:** `ad-immune_card_v3_1.json` (453 lines)
**Supersedes:** v3.0 (2026-06-02), 751 lines — archived at `OLD/ad-immune_card_v3_0.json`
**Companion:** `ad-immune_v3_0_release_notes.md` (v3.0 historical release notes + 13-item Lessons Learned, preserved)

---

## What v3.1 is

A full clean rewrite of the card JSON aligned to SOP v1.2 chain-of-custody stages. v3.0 was a "strict additive over v2.2" bump that preserved the entire v2.2 pre-build operational logic byte-for-byte (Stage 1 = 7-CpG Rule A panel framing, Stage 2 = Moss 2018 NNLS for cortical-neuron cfDNA, Stage 3 = Salas 2018 QC) and bolted a `cpg_native_post_build_addendum` block on top. v3.1 rewrites operational sections to use only current production methodology, with one important distinction from breast-epic v3.1:

**The 7-CpG Rule A directional panel IS legitimate operational AD scoring** (disease-trained, AUC 0.84 vs universal Mahalanobis AUC 0.62 on AIBL holdout). This is the OPPOSITE pattern from breast pre-dx where universal Mahalanobis BEATS the disease-trained Xu-538 panel by d=+0.75. Different diseases need different operational scoring routes.

In v3.1 Rule A is documented as `ad_disease_trained_panel` in operational sections — NOT confined to `pre_build_audit_lineage`. Moss / Salas / Loyfer references ARE confined to audit lineage.

---

## What's the actual operational chain v3.1 declares

| SOP Stage | Module | Card field |
|---|---|---|
| Stage 2 (Deconvolution) | Walther IAM Deconvolver (primary) + NILC v2 (cross-method) | `stage_2_deconvolution` |
| Stage 3 (Age foreground) | `age_axis_foreground.py` (MANDATORY for AD — Rule A panel has R²=0.26 with age) | `stage_3_age_foreground_subtraction` |
| Stage 4 (A-scoring) | `iamatlas_a_scoring.py` — 8 classes + 115 cell fan-out | `stage_4_a_scoring` |
| AD-specific | 7-CpG Rule A directional panel — age-adjusted Z output | `ad_disease_trained_panel` |
| Stage 5 (Mahalanobis) | `iamatlas_mahalanobis_scoring.py` | `stage_5_mahalanobis` |
| Stage 6 (Cellular age) | `iam_cellular_age_scoring.py` — AD immune class ~9y "younger" finding | `stage_6_cellular_age` |
| Stage 7 (Tier breakpoints) | Universal screen + AD-specific operational notes | `stage_7_tier_breakpoints` |
| Stage 8 (Card matching) | Three-route Boolean matching logic (AD / PSP-CBD / FTD) | `stage_8_card_matching` |

Stage 8 has three routes:

- **Route AD:** `Mahalanobis_d >= +0.40 AND age_adjusted_Rule_A_Z >= +1.0 AND immune_class_cellular_age_delta_yr <= -5` → `FIRED_AD`
- **Route PSP/CBD:** `Mahalanobis_d <= -0.20 AND BELOW_NORMAL on immune class` → `FIRED_PSP_CBD` (architectural compaction direction, opposite of AD)
- **Route FTD:** `Mahalanobis_d in [+0.10, +0.40] AND age_adjusted_Rule_A_Z in [+0.3, +1.0]` → `FIRED_FTD` (intermediate)

This three-way differential structure reflects the CPG-VAL-014 GIFT cohort finding: same Mahalanobis metric, three biologically distinct signatures by direction.

---

## What changed operationally vs v3.0

| v3.0 field | v3.1 field | Replacement |
|---|---|---|
| `stage_1_immune_flag` (Rule A described as Stage 1) | `ad_disease_trained_panel` block + `stage_2_deconvolution` runs FIRST | Walther deconvolution is Stage 2 for every card; Rule A is the AD-specific operational discriminator |
| `stage_2_localization` (Moss 2018 NNLS for cortical-neuron cfDNA) | `stage_2_deconvolution.primary_deconvolver` | Walther IAM Deconvolver on IAMAtlas REBUILD |
| `stage_3_subcomposition` (Salas 2018 QC) | `stage_2_deconvolution.secondary_deconvolver_for_cross_method_check` | NILC v2 cross-method check |
| Pre-build VAL refs (VAL-049/050/051/052/053/054b/057/090/091) in operational sections | `pre_build_audit_lineage.pre_build_val_series_for_audit_only` | Moved to audit lineage block (Rule A panel itself is operational; VAL-051 the SEALED VAL is the lineage anchor) |

---

## What stayed identical

- `card_id`, `disease`, `icd10_scope`, `differential_diagnosis_scope`
- 7-CpG Rule A panel composition (panel_id, n_cpgs=7, AUC 0.84 — these are the disease-trained AD discriminator)
- `h_min_by_class_frozen_2026_04_06` values (8 calibration anchors)
- `tier_thresholds` (universal screen calibration)
- `substrate.platforms_supported` (EPIC + 450K with coverage check + AddNeuroMed platform attenuation note)
- Three-cohort validation evidence (AIBL + AddNeuroMed + GIFT GSE53740)
- "Card never fires on one tile" rule

---

## Validation evidence reorganization

- `val_series: "CPG-VAL-008 through CPG-VAL-014"` (7 VALs)
- Three cohorts with full breakdowns (AIBL 726, AddNeuroMed 300, GIFT 384) + Stage 1 reproductions PASS on all three
- `headline_findings` array with 9 specific factual findings
- `sop_chain_of_custody_audit` pointer to `AD_IMMUNE_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md` (still applies — v3.1 didn't change computation)

---

## Three-way GIFT specificity (CPG-VAL-014) — the strongest evidence in the card

Same Mahalanobis metric, three biologically distinct signatures by direction:

| Disease | Mahalanobis d | p | Label |
|---|---|---|---|
| Alzheimer's disease | +0.681 | 0.001 | POSITIVE — hyper-volume departure outward |
| Frontotemporal dementia | +0.279 | 0.108 | INTERMEDIATE |
| PSP/CBD (4R-tauopathies) | -0.380 | 2×10⁻⁶ | NEGATIVE — BELOW_NORMAL compaction direction |

This is the strongest single piece of evidence in the AD card that the IAM-architectural distance is biologically meaningful. A generic "different from healthy" metric would not produce direction-resolved discrimination across diseases.

---

## Companion document updates

- `ad-immune_README.md` — added v3.1 entry to version log; emphasizes the Rule A operational distinction from breast
- `ad_immune_residual_maps/README_AD_residual_maps.md` — rewritten with three-route Stage 8 consumption + PC1-vs-PC2 rank explanation (same biology, different cohort composition)
- `WORK_IN_PROGRESS.md` — updated for v3.1 status; v3.0 archived at `OLD/`

---

## Carry-forward to v3.2

1. CPG_ad_panel_v1 (200 CpGs from CPG-VAL-013) — formal seal + holdout validation on independent cohort
2. Prospective primary-care validation cohort (this is the big one for AD clinical deployment)
3. Cross-ethnicity validation cohorts
4. CHR/MAPINFO genomic annotation on residual map
5. Stage 8 Path B engine wiring
6. Synthetic_Patient_Generator chain-recovery test
7. First-client IDAT integration test

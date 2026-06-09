# breast-epic_card v3.0 — Release Notes (2026-06-02)

**Card bumped:** v2.3 (2026-04-26) → v3.0 (2026-06-02)
**Authored by:** Heath W. Mahaffey + Walther (Claude) — Phase 1 documentation cleanup
**Operational logic:** unchanged from v2.3
**Evidence added:** post-IAMAtlas-build CPG-VAL series (CPG-VAL-001/002/003/005/007)

---

## What changed

This is a **strict-additive documentation bump.** The card's operational logic (Stage 1 immune flag, Stage 2 tissue-of-origin, Stage 3 sub-composition, tier thresholds, tissue-arm logic) is unchanged from v2.3 byte-for-byte. The change is purely the evidence layer: post-build CPG-VAL references have been appended to `validation_evidence_summary`, and a new top-level block `cpg_native_post_build_addendum` documents the post-build evidence trail.

| Section | v2.3 status | v3.0 status |
|---|---|---|
| `card_id`, `disease`, `icd10_scope`, `cookbook_master_readme` | original | unchanged |
| `stage_1_immune_flag` (Xu-538 panel + scoring + tier thresholds) | original | **unchanged byte-for-byte** |
| `stage_2_localization` (Moss tissue NNLS) | original | **unchanged byte-for-byte** |
| `stage_3_subcomposition` (Salas + UniLIFE) | original | **unchanged byte-for-byte** |
| `tissue_arm_VAL060` (TCGA-BRCA matched pair) | original | **unchanged byte-for-byte** |
| `validation_evidence_summary` (pre-build VAL-093/094/095/096) | 4 entries | **4 entries unchanged, 5 entries appended** (CPG-VAL-001/002/003/005/007) |
| `deployment_positioning_v23` | original | **unchanged byte-for-byte** |
| `known_limitations`, `universal_reference`, `lessons_learned` | original | **unchanged byte-for-byte** |
| `v22_changes`, `v23_changes` | original | **unchanged byte-for-byte** |
| `card_notes` | original | **unchanged byte-for-byte** |
| `cpg_native_post_build_addendum` | (didn't exist) | **NEW — explanatory block** |
| `v30_changes` | (didn't exist) | **NEW — change list** |

## Why a v3.0 bump rather than overwriting v2.3

Per Heath's standing rule: pre-existing card content is not deleted without conversation. The pre-build VAL evidence (VAL-093/094/095/096) is the lineage record of how IAMAtlas was built and calibrated — it stays in the card alongside the post-build CPG-VAL evidence. Both stories are true and complement each other.

The pre-build evidence shows: "this is the cohort + external panels that built the instrument."
The post-build evidence shows: "this is the IAMAtlas-native instrument operating on the same cohort, with sub-cellular resolution the pre-build pipeline couldn't reach."

The v2.3 file is preserved byte-identical at `breast_epic_card_json/OLD/breast-epic_card_v2_3.json` (SHA-256 `f4ea6d2b301dd8ed...8c0b43cc913` — matches original 2026-04-26 anchor).

## The 5 post-build VAL entries added

Full detail in the card JSON's `validation_evidence_summary` (entries 5 through 9). Brief inventory:

| ID | Headline | Tool | Null suite |
|---|---|---|---|
| CPG-VAL-001 | Per-cell-type fan-out — Baso d=+1.577/+1.010, BE +1.281/+0.614 tissue-of-origin at 10yr+ | `iamatlas_a_scoring.score_per_celltype()` | 7/7 PASS Sealed |
| CPG-VAL-002 | Mahalanobis hyper-volume d=+1.871/+2.088 (universal, not breast-trained) | `iamatlas_mahalanobis_scoring.py` | 7/7 PASS Sealed |
| CPG-VAL-003 | 1,392 concordant CpGs (5.4:1 hypomethylation field-effect) | Walther + per-CpG residual map | 7/7 PASS Sealed |
| CPG-VAL-005 | PC2 T-cell suppression axis d=−0.67/−0.58 | sklearn PCA on 115-cell HC covariance | 7/7 PASS Sealed |
| CPG-VAL-007 | Mahalanobis improves +0.255 GSE51032 after age-axis subtraction | `age_axis_foreground.py` | 7/7 PASS Sealed |

Two restated foundation VALs are documented separately in `cpg_native_post_build_addendum.null_suite_status_summary.restated`:
- CPG-VAL-004 (bimodality direction: 1,096 gain dominates 396 loss, 2.77:1 — the restated framing)
- CPG-VAL-006 (chr6 MHC look-elsewhere corrected p=0.103 — the restated framing)

## What remains TO BUILD per v3 inventory

Per `post_build_evidence/v3_CPG_VAL_Inventory_Report.md` §B.1, the breast-epic Family B confirmation series **CPG-VAL-008 through CPG-VAL-014** is still pending:

- CPG-VAL-008: CPG_breast_panel_v1 definition (1,389 new + 3 retained-from-Xu538 CpGs)
- CPG-VAL-009: CPG_breast_panel_v1 → A-score case-vs-HC on GSE51057 with held-out test
- CPG-VAL-010: Cross-platform replication on GSE51032 (full holdout)
- CPG-VAL-011: TTD-window stratification (>10yr / 5-10yr / 2-5yr / 0-2yr)
- CPG-VAL-012: Tissue arm TCGA-BRCA matched tumor-normal
- CPG-VAL-013: Mahalanobis specificity test on non-breast pre-dx cohort
- CPG-VAL-014: 35-CpG bimodality double-confirmed sub-panel test

These are formal sealed-VAL packaging plus some new analyses — not blocking for AD-immune card work, which can proceed in parallel.

## Pointers

- **Card JSON (v3.0):** `breast_epic_card_json/breast-epic_card_v3_0.json`
- **Card JSON (v2.3, verbatim):** `breast_epic_card_json/OLD/breast-epic_card_v2_3.json`
- **README (v2.3 operational logic, unchanged):** `breast_epic_card_json/breast-epic_README.md`
- **Operational data files (CPG-VAL-003/004/005 outputs):** `breast_epic_residual_maps/`
- **Foundation cohort A-scores:** `Biological_Physics/validation_runs/foundation_cohort/`
- **L9 null suite results for VAL-001-007:** `Biological_Physics/chain_of_custody/L9_null_suite/test_runs/`
- **Full post-build narrative:** `post_build_evidence/v2_CPG_IAMAtlas_Evidence_Report.html`
- **Single-source inventory:** `post_build_evidence/v3_CPG_VAL_Inventory_Report.md`

# Complete Card v3.1 + Matrix v1.7 Package

**Date:** 2026-06-05
**Commits:** `834bc29` (cards + matrix + residual maps) + `254fb7e` (companion files cleanup)
**Repo:** `hmahaffeyges/IAM-Validation` main branch

## What's in this package

```
Breast_EPIC/
├── BREAST_EPIC_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md  (applicability note added — applies to v3.1 too)
├── WORK_IN_PROGRESS.md                              (updated for v3.1 status)
├── breast_epic_card_json/
│   ├── breast-epic_card_v3_1.json                   ← v3.1 CARD (406 lines, clean rewrite)
│   ├── breast-epic_README.md                        (v3.1 version log entry added)
│   ├── breast-epic_v3_0_release_notes.md            (historical, preserved)
│   └── breast-epic_v3_1_release_notes.md            ← NEW v3.1 release notes (108 lines)
└── breast_epic_residual_maps/
    └── README_Breast_residual_maps.md               (rewritten with Stage 8 §66 consumption)

AD_immune/
├── AD_IMMUNE_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md   (applicability note added)
├── WORK_IN_PROGRESS.md                              (updated for v3.1 status)
├── ad_immune_card_json/
│   ├── ad-immune_card_v3_1.json                     ← v3.1 CARD (453 lines, clean rewrite)
│   ├── ad-immune_README.md                          (v3.1 version log entry added)
│   ├── ad-immune_v3_0_release_notes.md              (historical, preserved)
│   └── ad-immune_v3_1_release_notes.md              ← NEW v3.1 release notes (106 lines)
└── ad_immune_residual_maps/
    └── README_AD_residual_maps.md                   (rewritten with three-route Stage 8 consumption)

DISEASE_MATRIX/
├── disease_cell_signature_matrix_v1_7.csv           ← v1.7 (81 rows, strict additive over v1.6)
├── disease_cell_signature_matrix_engine_schema_v1_2.md
└── README_disease_signature_matrix_folder.md        (rewritten for v1.7)
```

## What v3.1 fixed

The v3.0 cards were "strict additive over v2.x" bumps — they preserved the v2.x pre-build operational logic byte-for-byte (Xu-538 panel as Stage 1, Moss 2018 NNLS as Stage 2, Salas 2018 as Stage 3) and bolted a small `cpg_native_post_build_addendum` on top. The body of v3.0 described pre-build operational logic; only the addendum acknowledged the post-build instrument exists.

**v3.1 is a full clean rewrite of the card JSON aligned to SOP v1.2 chain-of-custody stages.** Operational sections describe the actual current production methodology. Pre-build references are confined to a clearly-labeled `pre_build_audit_lineage` block at the bottom.

## What the engine consumes

Per SOP v1.2 Part II-C §65-§69, Stage 8 (card-level pattern matching) reads from each card:

- **substrate.platforms_supported** — eligibility gate
- **stage_8_card_matching.matching_logic** — Boolean rules over Stage 4/5/6/7 outputs
- **stage_8_card_matching.residual_map_reference** — file path + overlap threshold
- **within_card_covariates** — nuisance adjustments per SOP §68
- **report_contents.language_discipline** — allowed/forbidden customer-facing language

Field names preserved across v3.0→v3.1 where engine reads (tier_thresholds, h_min_by_class_frozen_2026_04_06, substrate.platforms_supported, validation_tier). Engine compatibility maintained.

## Stage 8 matching logic

**Breast-epic (two routes):**
- Route A (universal): `Mahalanobis_d >= 1.50 AND residual_overlap_rho >= 0.10 AND CI_lower > 0` → FIRED_long_pre_dx
- Route B (per-cell): `(basophil_A >= 1.20 OR breast_epithelial_A >= 1.10) AND PC2_T_cell_d <= -0.40` → FIRED_long_pre_dx

**AD-immune (three routes):**
- Route AD: `Mahalanobis_d >= +0.40 AND age_adjusted_Rule_A_Z >= +1.0 AND immune_cellular_age_delta <= -5` → FIRED_AD
- Route PSP/CBD: `Mahalanobis_d <= -0.20 AND BELOW_NORMAL on immune` → FIRED_PSP_CBD (architectural compaction, opposite of AD)
- Route FTD: `Mahalanobis_d in [+0.10, +0.40] AND age_adjusted_Rule_A_Z in [+0.3, +1.0]` → FIRED_FTD (intermediate)

## Disease matrix v1.7

Strict additive over v1.6: ONE new row added — `breast_cancer / long_pre_dx_post_build_v3_0` mirroring the AD v1.6 pattern with current-methodology evidence_anchors only (CPG-VAL-001 through 007). Original row 1 retained verbatim as audit lineage.

## What was NOT changed

- Underlying SOP chain-of-custody computation
- H_min values (frozen 2026-04-06)
- Validation findings (effect sizes, cohort sizes, residual map CpG counts)
- L9 null suite results
- CPG-VAL bundles
- v2.x and v3.0 cards (archived in OLD/ subdirectories)
- v3.0 release notes (preserved as historical)
- v3.0 SOP audit docs (applicability note added at top; still apply to v3.1)

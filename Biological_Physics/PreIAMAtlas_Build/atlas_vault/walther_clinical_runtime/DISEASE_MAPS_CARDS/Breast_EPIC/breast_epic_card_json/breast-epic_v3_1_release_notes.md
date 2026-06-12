# Breast-EPIC Card v3.1 — Release Notes

**Release date:** 2026-06-05
**Card file:** `breast-epic_card_v3_1.json` (406 lines)
**Supersedes:** v3.0 (2026-06-02), 1,012 lines — archived at `OLD/breast-epic_card_v3_0.json`
**Companion:** `breast-epic_v3_0_release_notes.md` (v3.0 historical release notes, preserved)

---

## What v3.1 is

A full clean rewrite of the card JSON aligned to SOP v1.2 chain-of-custody stages. v3.0 was a "strict additive over v2.3" bump that preserved the entire v2.3 pre-build operational logic byte-for-byte (Stage 1 invariant rule using Xu-538 panel, Stage 2 method using Moss 2018 NNLS deconvolution, Stage 3 sub-composition using Salas 2018 + EpiSCORE + UniLIFE) and bolted a small `cpg_native_post_build_addendum` block on top. The body of the v3.0 card described pre-build operational logic; the addendum acknowledged the post-build instrument exists. **v3.1 fixes that asymmetry** — operational sections describe the actual current production methodology, and pre-build references are confined to a clearly-labeled `pre_build_audit_lineage` block at the bottom.

This was triggered by Heath's directive 2026-06-05: *"these cards are supposed to be perfect and they are used for a specific purpose, read the SOP to familiarize yourself again, then create the files the correct way."*

---

## What's the actual operational chain v3.1 declares

Per SOP v1.2 chain-of-custody stages, in order:

| SOP Stage | Module | Card field |
|---|---|---|
| Stage 2 (Deconvolution) | Walther IAM Deconvolver (primary) + NILC v2 (cross-method) | `stage_2_deconvolution` |
| Stage 3 (Age foreground) | `age_axis_foreground.py` (mandatory when age available) | `stage_3_age_foreground_subtraction` |
| Stage 4 (A-scoring) | `iamatlas_a_scoring.py` — A = H(β) / H_min(class), 8 classes + 115 cell fan-out | `stage_4_a_scoring` |
| Stage 5 (Mahalanobis) | `iamatlas_mahalanobis_scoring.py` — Ledoit-Wolf shrinkage 0.00875, n_hc=601 | `stage_5_mahalanobis` |
| Stage 6 (Cellular age) | `iam_cellular_age_scoring.py` — 80-cell baseline, SAT handling | `stage_6_cellular_age` |
| Stage 7 (Tier breakpoints) | Universal screen — BELOW_NORMAL/NORMAL/MARGINAL/DETECTABLE/FLOOR_BREACH | `stage_7_tier_breakpoints` |
| Stage 8 (Card matching) | Two-route Boolean matching logic | `stage_8_card_matching` |

Stage 8 has two routes for breast-epic:

- **Route A (universal architectural):** `Mahalanobis_d >= 1.50 AND residual_map_overlap_rho >= 0.10 AND CI_lower > 0`
- **Route B (per-cell tissue-of-origin + immune suppression):** `(basophil_A >= 1.20 OR breast_epithelial_A >= 1.10) AND PC2_T_cell_axis_d <= -0.40`

Either route fires the card with `phase_emitted = "long_pre_dx"`.

---

## What changed operationally vs v3.0

| v3.0 field | v3.1 field | Replacement |
|---|---|---|
| `stage_1_immune_flag.panel` (Xu-538) | `stage_2_deconvolution.primary_deconvolver` | Walther IAM Deconvolver on IAMAtlas REBUILD |
| `stage_2_localization.method` (Moss 2018 NNLS) | `stage_2_deconvolution.primary_deconvolver` + `stage_4_a_scoring.fanout_115_cell` | Walther IAM Deconvolver with 115-cell fan-out via `iamatlas_a_scoring.py` |
| `stage_3_subcomposition` (Salas + UniLIFE) | `stage_2_deconvolution.secondary_deconvolver_for_cross_method_check` | NILC v2 cross-method check (NOT used for primary scoring) |
| `universal_stage_2_moss_deconvolution` | `stage_2_deconvolution` | — |
| `universal_stage_3_epidish_subcomposition` | `stage_2_deconvolution.secondary_deconvolver_for_cross_method_check` | — |
| Pre-build VAL refs (VAL-046/047/049/060/093/094/095/096/041) in operational sections | `pre_build_audit_lineage.pre_build_val_series_for_audit_only` | Moved to audit lineage block |
| `cpg_native_post_build_addendum` (small addendum block) | — | Folded into operational sections; addendum block no longer needed |

---

## What stayed identical

- `card_id`, `disease`, `icd10_scope`, `supersedes` chain
- `h_min_by_class_frozen_2026_04_06` values (8 calibration anchors — these are the literal H_min values)
- `tier_thresholds` (universal screen calibration from VAL-054b)
- `substrate.platforms_supported` (EPIC + EPIC v2 + 450K with coverage check)
- `report_contents.language_discipline` (allowed/forbidden language for customer-facing reports)
- "Card never fires on one tile" rule per SOP §65 invariant

---

## Validation evidence reorganization

v3.0 had `validation_evidence_summary` array with 9 entries mixing pre-build and post-build VALs. v3.1 has `validation_evidence` block with:

- `val_series: "CPG-VAL-001 through CPG-VAL-007"`
- `cohorts` array with both EPIC-Italy cohorts (GSE51057 + GSE51032) + filtered N + SHA-256 of β files
- `headline_findings` array with 10 specific factual findings keyed to specific CPG-VAL bundles
- `sop_chain_of_custody_audit` pointer to `BREAST_EPIC_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md` (still applies — v3.1 didn't change computation, only card framing)

---

## Companion document updates

- `breast-epic_README.md` — added v3.1 entry to version log, updated carry-forward header to v3.2
- `breast_epic_residual_maps/README_Breast_residual_maps.md` — rewritten with Stage 8 §66 consumption pattern documentation
- `WORK_IN_PROGRESS.md` — updated for v3.1 status; v3.0 archived at `OLD/`
- `disease_cell_signature_matrix_v1_6.csv` → `v1_7.csv` — added clean `breast_cancer / long_pre_dx_post_build_v3_0` row mirroring AD v1.6 pattern; original row 1 retained as audit lineage
- Cross-references v1.6 → v1.7 synced across evidence report + inventory reports + matrix README

---

## What v3.1 did NOT change

- The underlying SOP chain-of-custody computation. v3.1 is a card-definition cleanup; every CPG-VAL still runs through the same L1-L9 chain.
- The validation findings. Effect sizes, cohort sizes, residual map CpG counts, all unchanged.
- The H_min values (frozen 2026-04-06).
- The 7-VAL audit anchor (CPG-VAL-001 through CPG-VAL-007 still the validation series).
- The pre-build evidence trail. All 9 pre-build VALs (VAL-041/046/047/049/060/093/094/095/096) preserved in `pre_build_audit_lineage` block.

---

## Carry-forward to v3.2

Per `WORK_IN_PROGRESS.md` Outstanding section:

1. CPG_breast_panel_v1 (1,392 concordant CpGs from CPG-VAL-003) — formal seal + holdout validation
2. CHR/MAPINFO genomic annotation on residual map
3. Full bimodality decomposition (placeholder currently)
4. Cross-ethnicity validation cohorts
5. Stage 8 Path B engine wiring (cell-name-to-matrix-column mapping artifact)
6. Synthetic_Patient_Generator chain-recovery test
7. First-client IDAT integration test

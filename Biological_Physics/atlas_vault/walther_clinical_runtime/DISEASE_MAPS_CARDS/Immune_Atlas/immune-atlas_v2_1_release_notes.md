# Immune Atlas Card v2.1 — Release Notes

**Release date:** 2026-06-07
**Card version:** v2.1
**Card JSON:** `immune_atlas_card_json/immune-atlas_card_v2_1.json` (~152 KB, 38 top-level keys)
**Supersedes:** v2.0 (2026-06-07 first clean rebuild), archived to `immune_atlas_card_json/OLD/`

## What this release is

v2.1 is a surgical additive update on v2.0. It builds the **cross-disease universal alarm residual map v0_1** and integrates it into Stage 8 Route A as a residual-map-overlap channel parallel to the breast-EPIC and AD-immune disease cards. Per Heath's 2026-06-07 direction, the build was executed immediately after v2.0 push because the source residual maps already existed in identical format — no new data acquisition or cohort processing required.

## The new artifact

`immune_atlas_residual_maps/immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv` (1.25 MB, 6,018 CpGs × 17 columns, sha256 `29b518b3d4ddd3c590a97a94d20e3e86ce38b65bb71f0ee653233384d0ceb970`).

Derived from inner-join of:
- `breast_epic_residual_map_chr_annotated.csv` (CPG-VAL-003, 7,114 CpGs, breast pre-dx >10y EPIC-Italy GSE51057 + GSE51032)
- `ad_immune_residual_map_chr_annotated.csv` (CPG-VAL-013, 6,018 CpGs, AD AIBL + AddNeuroMed)

CHR + MAPINFO consistency verified across all 6,018 common CpGs (zero mismatches).

### Firing-pattern distribution

| Pattern | CpGs | % | Operational use |
|---|---:|---:|---|
| `fires_neither_background` | 4,641 | 77.12% | Background reservoir |
| `fires_breast_only` | 1,136 | 18.88% | Breast-specific signal |
| `fires_AD_only` | 212 | 3.52% | AD-specific signal |
| `fires_in_both_diseases_same_direction` | 17 | 0.28% | Cross-disease concordance channel |
| `fires_in_both_diseases_opposing_direction` | 12 | 0.20% | **VAL-016 bidirectional universal alarm at per-CpG resolution** |

The 12 opposing-direction CpGs are the operational signature of the cross-disease universal alarm Heath identified in CPG-VAL-016. Pooled (sign-aligned) tests on this subset would cancel; directional decomposition recovers them.

### Honest finding from cross-validation

The 12-CpG opposing-direction subset and the VAL-051 7-CpG directional panel (Stage 4.5) are **disjoint instruments** — they share zero CpGs. The VAL-051 panel was selected via Rule A criterion on AIBL-only training data using pre-build methodology, and its CpGs (cg16867657, cg25809905, cg22454769, cg09809672, cg26614073, cg00431549, cg02228185) are not in either post-build residual map's CpG universe. The 12 v0_1 CpGs (cg00932141, cg01006587, cg05257202, cg08200043, cg08886001, cg09622285, cg12565635, cg15459165, cg18332838, cg18418928, cg20700762, cg26131879) come from the post-build intersection.

These are complementary at different scales — not redundant:
- **VAL-051 7-CpG panel**: high-precision Stage 4.5 within-AD directional discriminator (AUC=0.84 AIBL holdout)
- **v0_1 12-CpG subset**: broader Stage 8 Route A cross-disease overlap channel

A future v0_2 may merge both CpG universes into a unified bidirectional instrument.

## Card changes from v2.0

### Modified
- `stage_8_card_matching.card_role` refreshed to reference the v2.1 residual-map-overlap channel
- `stage_8_card_matching.route_A_universal_architectural` restructured into two channels:
    - `trigger_mahalanobis_channel` (unchanged from v2.0)
    - `trigger_residual_map_overlap_channel_v2_1_NEW` with two sub-channels (`cross_disease_concordance_channel` + `bidirectional_universal_alarm_channel`)
- `stage_8_card_matching.immune_residual_map_status` flipped `NOT_BUILT — necessity assessment open` → `BUILT v0_1 (2026-06-07)` with full firing-pattern distribution
- `chain_of_custody_anchors.cross_disease_universal_alarm_residual_map_v0_1` NEW field with sha256 anchor
- `chain_of_custody_anchors.engine_modules_consumed_at_runtime` extended with the new residual map artifact (entry #13 after Stage 8 disease matrix)
- `honest_limitations` extended with 3 new v2.1-specific entries (v0_1 is derived vault artifact not sealed VAL; Route A thresholds provisional; VAL-051 + v0_1 disjoint)
- `outstanding_work_v2_0` → `outstanding_work_v2_1`: necessity-assessment item marked COMPLETED; three new v0_2 items added (extend to additional disease cohorts; optional merge with VAL-051 universe; first-cohort threshold recalibration)
- `validation_evidence_v2_0_set` → `validation_evidence_v2_1_set` (byte-identical content; version suffix bump)

### Added
- `v2_1_changes_from_v2_0` — this changelog

### Preserved byte-identical from v2.0 (35 of 37 keys)
- All operational chain stages 2-7 and 9-10 (untouched)
- Stage 4.5 bidirectional decomposition (VAL-051 7-CpG panel reference preserved — disjoint instrument per provenance)
- Stage 8 Route B disease signature matrix v1.8 + Route C bidirectional pattern (untouched)
- All 6 integration blocks (cancer_prior, family_history_multiplier, literature_anchors, cpg_null_runner, validation_anchor_csv, synthetic_patient_generator)
- cells_resolved, within_card_covariates, report_contents (untouched)
- disease_immune_lens (81 entries — untouched)
- wellness_aging_inflammation_lens (10 categories — untouched)
- pre_build_audit_lineage (untouched)
- v2_0_changes_from_v1_0_and_v1_1 (preserved as historical changelog alongside the new v2_1_changes_from_v2_0)

## Cross-check pass at release

| Check | Result |
|---|---|
| Forbidden language audit (operational sections, v2.1) | 0 hits |
| v2.0 → v2.1 diff | surgical additive: 0 keys removed; 1 added (`v2_1_changes_from_v2_0`); 2 renamed with version suffix (`outstanding_work_v2_1`, `validation_evidence_v2_1_set`); 35 byte-identical |
| Stage 8 Route A channel structure | 2 channels (Mahalanobis + residual-map-overlap), each with documented trigger + artifact reference |
| immune_residual_map_status flip | NOT_BUILT → BUILT v0_1 with full firing-pattern distribution documented |

## Open work for v0_2 (from outstanding_work_v2_1)

1. Cross-disease universal alarm residual map v0_2: extend to additional disease cohorts as sealed residual maps become available (CRC pre-dx, lung pre-dx, autoimmune) to broaden the cross-disease universe beyond the current 2-disease intersection
2. Cross-disease universal alarm residual map v0_2: optional merge with VAL-051 7-CpG panel CpG universe to create a unified bidirectional instrument
3. Stage 8 Route A residual-map-overlap channel threshold calibration on first patient cohort runs through immune-atlas v2.1 chain (provisional thresholds inherited from breast-EPIC v3.1)

## File package additions in v2.1

| File | Status |
|---|---|
| `immune_atlas_card_json/immune-atlas_card_v2_1.json` | NEW (current) |
| `immune_atlas_card_json/OLD/immune-atlas_card_v2_0.json` | ARCHIVED |
| `immune_atlas_residual_maps/immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv` | NEW (1.25 MB) |
| `immune_atlas_residual_maps/immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv.sha256` | NEW |
| `immune_atlas_residual_maps/immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.provenance.json` | NEW |
| `immune-atlas_README.md` | UPDATED for v2.1 |
| `immune-atlas_v2_1_release_notes.md` | NEW (this file) |
| `immune-atlas_v2_0_release_notes.md` | PRESERVED |

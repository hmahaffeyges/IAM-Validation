# Heath delivery — prostate-epic v0.3 sprint (CORRECT BASELINES)
**Date:** 2026-04-30 evening (third re-delivery)

## Why this delivery exists

Heath caught that the previous deliveries were edited from STALE baselines in `/mnt/project/` that lagged the cardio v0.3 sprint by hundreds of lines. The correct baselines were sitting in `/mnt/user-data/outputs/heath_only_v2026_04_29_v0_3_substrate_patch/` (the cardio v0.3 substrate-patch delivery from April 29 evening). This delivery redoes the v0.3 prostate edits onto those correct baselines.

## Line-count audit (every file, before → after)

| File | Heath's stated baseline | Actual baseline used | After v0.3 edits | Net delta |
|---|---|---|---|---|
| TESTING_CHECKLIST.md | 633 | **633 ✓** | 671 | +38 (CHK-2.7 + CHK-2.8 inserted after CHK-2.6) |
| LESSONS_LEARNED.md | 1,362 | **1,362 ✓** | 1,407 | +45 (prostate-LL-006/007/008 appended) |
| EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md | 963 | **963 ✓** | 963 | 0 (unchanged — no operational pipeline behavior change this sprint per F.3.7 spec) |
| README_MASTER_v2_4.md | 1,187 | **1,187 ✓** | 1,252 | +65 (v2.5 amendment line at top + prostate-epic per-card entry expanded with v0.3 promotion paragraph + multi_modal_validated tier definition expanded + new ## v2.5 amendment section appended at bottom) |
| GAPE_Reproduction_Paper_v1.md | 2,982 | **2,982 ✓** | 2,999 | +17 (§7.26 prostate sprint methodology evolutions inserted after §7.25 cardio sprint section) |
| GAPE_Evidence_Report_UPDATED.html | 18,203 | **18,203 ✓** | 19,231 | +1,028 (full VAL-117 + VAL-118 block with reproducibility triple per CHK-7.6 inserted after VAL-058 closing marker) |
| CROSS_CARD_CALIBRATION_TODO_v0_5.md | 886 | **886 ✓** | 899 | +13 (F.4 checkmark + 5 commit hashes for prostate-epic Wave 4) |

All edits surgical-additive. No content removed. v2.4 amendment chain content preserved verbatim. v0.2 prostate-epic per-card entry preserved verbatim and the v0.3 paragraph appended after it (not replacing).

## What changed in the four edits to existing files

### TESTING_CHECKLIST.md (+38 lines after CHK-2.6)
- CHK-2.7 Cell-of-origin atlas preregs MUST use magnitude-based |d| thresholds with direction labels (DISC-PROSTATE-002 / prostate-LL-007). Required wording for cell-of-origin atlas tile outcomes provided as template.
- CHK-2.8 CHK-3.1B coverage threshold pre-locks must match substrate floor, NOT default 95% (formalized from VAL-117 amendment). Substrate floor reference table provided.

### LESSONS_LEARNED.md (+45 lines appended after CCL-048 entry)
- prostate-LL-006 (DISC-PROSTATE-001) — Gene-promoter atlas family fitness depends on per-tissue cell-type distinctness. Extends LL-CARDIO-005 / DISC-CARDIO-004.
- prostate-LL-007 (DISC-PROSTATE-002) — Pre-registration discipline must use magnitude-based |d| thresholds for cell-of-origin atlases. Cookbook-wide rule formalized as CHK-2.7.
- prostate-LL-008 (DISC-PROSTATE-003) — ProstateRef LE tile reads tumor strongly NEGATIVE (luminal dedifferentiation signature). Operational diagnostic: A_LE BELOW VAL-117 healthy-floor q5 = 0.4190.

### README_MASTER_v2_4.md (+65 lines via three surgical edits)
- New `**Amended:** 2026-04-30` header line documenting v2.5 promotion
- Prostate-epic per-card entry (§5 in the per-card numbered list): v0.2 paragraph preserved verbatim; v0.3 promotion paragraph appended documenting VAL-117 + VAL-118 sealed findings, DISC-PROSTATE-001/002/003, tier promotion to multi_modal_validated_plus_multi_atlas_calibrated, and the 5-commit GitHub chain
- multi_modal_validated tier definition expanded to include prostate-epic v0.3 at the multi_modal_validated_plus_multi_atlas_calibrated sub-tier
- New `## v2.5 amendment` section appended at end of file with full sprint summary (Phase B calibration anchor VAL-117 + Phase C run-everything VAL-118 + CCL-041 amendment audit trail + DISC-PROSTATE discoveries + CHK gates added + atlas vault state + v0.2 limitations preserved + GitHub state)

### GAPE_Reproduction_Paper_v1.md (+17 lines via insertion of §7.26 after §7.25 cardio sprint section)
- §7.26 Prostate-epic v0.3 sprint methodology evolutions
  - §7.26.1 EpiSCORE gene-promoter atlas → 450K array CpG bridge engineering as a reusable infrastructure
  - §7.26.2 Magnitude-based |d| threshold rule with direction labels for cell-of-origin atlas preregs
  - §7.26.3 CHK-3.1B coverage threshold pre-locks must match substrate floor, NOT default 95%
  - §7.26.4 Two-stage streaming-write architecture for large β matrices

### GAPE_Evidence_Report_UPDATED.html (+1,028 lines via insertion after VAL-058 closing marker)
- Full VAL-117 + VAL-118 block with reproducibility triple per CHK-7.6:
  - Inline HTML-escaped Python source for val118_stage1_extract.py (2,129 chars), val118_stage2_score.py (15,958 chars), val117_prostateref_calibrate.py (17,786 chars) — CHK-7.6 Item 1
  - Inputs table with download URL + size + SHA-256 per file — CHK-7.6 Item 2
  - Environment block with Python version, numpy version, runtime, memory — CHK-7.6 Item 3
  - Expected headline output enumerated — CHK-7.6 Item 4
  - Full pre-registration chain with all four SHA-256 hashes (VAL-117 prereg + amendment, VAL-118 prereg + amendment)
  - ProstateRef per-tile signature table
  - Stage 3 multi-atlas immune signature table
  - CCL-041 amendment audit-trail explanation
  - Operational implication for v0.3 disease scoring + explicit non-claims

### CROSS_CARD_CALIBRATION_TODO_v0_5.md (+13 lines)
- Wave 4 prostate-epic checkmarked complete
- All five commit hashes documented (40ce175 → edf6229 → 58ecd16 → c5ee9d5 → 388e5b0)

## What was NEW (not edits to existing files)

### prostate_epic_README_v0_3.md (459 lines)
- v0.2 README content (162 lines) preserved VERBATIM
- v0.3 additions (297 lines) appended as new sections covering all 10 structured Phase E blocks (atlases_used_and_deferred, chk_3_1_thresholds_per_substrate, run_everything_phase_c_results, per_disease_scoring_policy, DISC-PROSTATE discoveries with title+body+Implication structure per E.6, validation_evidence_summary per VAL with cohort/n/substrate/design/QC/outcome/Cohen's d/interpretation/prereg SHA, cookbook-wide CCL cross-references, reproduction bundle, what we chose not to claim, what remains open)

### prostate_epic_card_v0_3.json (28 top-level keys)
- 20 v0.2 keys preserved verbatim
- 8 v0.3 structured blocks added: atlases_used_and_deferred, chk_3_1_thresholds_per_substrate, v0_3_run_everything_phase_c_results, per_disease_scoring_policy_v0_3, disc_prostate_v0_3, validation_evidence_summary_v0_3, cookbook_wide_ccl_cross_references_v0_3, reproducibility_anchors_v0_3

## GitHub commit chain

| Commit | Sealed | Files |
|---|---|---|
| `40ce175` | VAL-117 ProstateRef Phase B calibration anchor | val117_prostateref_calibrate.py, prereg.md, prereg_amendment.md, VAL-117_calibration_results.json, VAL-117_per_sample_calibration.csv, outcome.md, PREREG_SEAL.txt, PREREG_AMENDMENT_SEAL.txt |
| `edf6229` | VAL-118 first execution sealed O5 (preserved as discipline-discovery record) | val118_stage1_extract.py, val118_stage2_score.py, prereg.md, VAL-118_cohen_d_per_atlas.json, VAL-118_per_sample_run_everything.csv, outcome.md, PREREG_SEAL.txt |
| `58ecd16` | VAL-118 amendment sealed O1+O2(LE_NEGATIVE)+O4 | prereg_amendment.md, VAL-118_amendment_cohen_d_per_atlas.json, VAL-118_amendment_per_sample_run_everything.csv, outcome_amendment.md, PREREG_AMENDMENT_SEAL.txt |
| `c5ee9d5` | Phase D v0.2-vs-v0.3 outcome comparison | phase_d_v02_vs_v03.md |
| `388e5b0` | F.1 deliverables: cohort manifest, clinical metadata, stratified results, public README update | VAL-118_cohort_manifest.json, VAL-118_clinical_metadata.json, VAL-118_stratified_results.json, Biological_Physics/README.md (315→317 lines, prostate-epic v0.3 update paragraph at top of update history) |


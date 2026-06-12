# VAL-076 pre-registration — cervical-epic v0.1

**Card:** cervical-epic v0.1
**VAL ID:** VAL-076
**Date sealed:** 2026-04-25
**Cohort:** GSE143752 cervical EPIC 850K — El-Zein 2020 Quebec cohort (54 Healthy + 50 CIN1 + 40 CIN2 + 42 CIN3 = n=186 LBC samples)
**Specimen:** LBC pap smear cytology (exfoliated cervical cells)
**n total:** 186

## Design

First LBC primary-pathway validation for cervical-epic. Score Stage 1 immune Xu-538 A on bulk LBC β; compare Healthy vs each CIN grade individually and Healthy vs all-lesion pooled. Test for monotonic Healthy < CIN1 < CIN2 < CIN3 progression. NOTE: LBC is a novel specimen pathway for the Xu-538 panel (which was buffy-coat trained). Per CHK-0.5, this prereg includes panel transferability not yet established caveat. A null reading is a TRANSFERABILITY finding, not a 'no signal' finding.

## Pre-locked decision criteria

- O1_PASS_LBC_PRIMARY_ANCHOR: Healthy vs all-lesion d ≥ +0.5 with lower CI > 0 AND monotonic CIN1<CIN2<CIN3
- O2_PARTIAL_LBC: 0 < d < 0.5 with lower CI > 0
- O3_LBC_NULL: CI crosses zero — INTERPRET AS PANEL TRANSFERABILITY FINDING per CHK-0.5
- O5_LBC_NEGATIVE: d < 0
- O6_UNEXPECTED: data integrity flag fires

## Constants (all sealed)

- Panel: Xu-538 immune
- Panel SHA-256: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
- H_min: 0.838889 (immune class — universal Stage 1 rule per panc-LL-007)
- RNG seed: 20260425
- QC threshold: ≥ 400 valid Xu-538 CpGs per sample

## Mandatory checks per TESTING_CHECKLIST.md

- CHK-1.1 Sample_title verification: VERIFIED at runtime
- CHK-3.1 β distribution sanity check: REPORTED in results JSON
- CHK-3.2 Cross-cohort healthy baseline check vs VAL-073 anchor: REPORTED
- CHK-3.3 Panel coverage report: REPORTED
- CHK-3.4 Sample-group assignment spot check: VERIFIED
- CHK-3.5 Saturation flag check: REPORTED
- CHK-4.1 Biology consistency check: APPLIED before drafting outcome
- CHK-4.2 Cancellation hypothesis check (CCL-031): APPLIED

## Test 2 placeholder (CCL-030)

Test 2 (lymphoid-marker vs myeloid-marker sub-panel split on Xu-538) is PENDING OQ-2026-01 immune-atlas staging. Not runnable at v0.1. CCL-027 question (iv) is a placeholder.

## Reproduction
- Pre-reg SEAL file: VAL-076_PREREG_SEAL.txt (lock-time SHA of this prereg)
- Results JSON: VAL-076_results.json
- Outcome.md: VAL-076_outcome.md
- Python script: val_076_cervical_epic.py
- Cohort manifest: included in results JSON

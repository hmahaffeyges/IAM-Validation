# VAL-077 pre-registration — cervical-epic v0.1

**Card:** cervical-epic v0.1
**VAL ID:** VAL-077
**Date sealed:** 2026-04-25
**Cohort:** GSE287994 cervical EPIC 850K — Bowden 2025 Imperial London cohort (119 benign + 74 CIN3/CGIN + 54 cancer = n=247 LBC samples; 241 QC-passed)
**Specimen:** LBC pap smear cytology
**n total:** 247

## Design

Largest LBC cohort in the cervical-epic v0.1 plan. Score Stage 1 immune Xu-538 A on bulk LBC β; compare Benign vs CIN3-or-Cancer; HPV-stratified subgroup analysis. EPIC 850K platform — first cross-platform validation in the v0.1 plan. NOTE: panel transferability not yet established per CHK-0.5. Pre-reg includes mandatory data integrity check (CHK-3.1 β distribution sanity) before scoring is interpreted as biology.

## Pre-locked decision criteria

- O1_PASS_LBC_PRIMARY_ANCHOR: Benign vs disease d ≥ +0.5 with lower CI > 0
- O2_PARTIAL_LBC: 0 < d < 0.5 with lower CI > 0
- O3_LBC_NULL: CI crosses zero — INTERPRET AS PANEL TRANSFERABILITY FINDING
- O5_LBC_NEGATIVE: d < 0
- O6_UNEXPECTED: CHK-3.1 β distribution check fails (>40% in [0.4, 0.6] OR <20% at extremes = file is residual data, not raw β)

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
- Pre-reg SEAL file: VAL-077_PREREG_SEAL.txt (lock-time SHA of this prereg)
- Results JSON: VAL-077_results.json
- Outcome.md: VAL-077_outcome.md
- Python script: val_077_cervical_epic.py
- Cohort manifest: included in results JSON

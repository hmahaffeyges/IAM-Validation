# VAL-074 pre-registration — cervical-epic v0.1

**Card:** cervical-epic v0.1
**VAL ID:** VAL-074
**Date sealed:** 2026-04-25
**Cohort:** GSE46306 cervical HM450 — Farkas 2013 Stockholm cohort (20 HPV-negative normal + 17 CIN3 + 6 cancer = n=43 valid)
**Specimen:** Cervical tissue biopsy
**n total:** 44

## Design

Replication of VAL-073 GSE99511 tissue arm anchor on independent HM450 cervical cohort. Score Stage 1 immune Xu-538 A on bulk tissue; compare Normal vs CIN3, Normal vs Cancer, Normal vs Lesion (CIN3+Cancer) by unpaired Cohen's d. Test cross-cohort consistency with VAL-073 anchor pattern. Compute HPV+ vs HPV− subgroup d as exploratory (small n).

## Pre-locked decision criteria

- O1_PASS_PROGRESSION: Normal vs CIN3 d ≥ +0.5 with lower CI > 0 AND monotonic Normal < CIN3 < Cancer
- O2_PARTIAL: 0 < d < 0.5 with lower CI > 0
- O3_NULL: CI crosses zero (lower CI ≤ 0 ≤ upper CI)
- O5_NEGATIVE_DIRECTION: d < 0
- O6_UNEXPECTED: data integrity flag fires (saturation, baseline mismatch, panel coverage drift)

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
- Pre-reg SEAL file: VAL-074_PREREG_SEAL.txt (lock-time SHA of this prereg)
- Results JSON: VAL-074_results.json
- Outcome.md: VAL-074_outcome.md
- Python script: val_074_cervical_epic.py
- Cohort manifest: included in results JSON

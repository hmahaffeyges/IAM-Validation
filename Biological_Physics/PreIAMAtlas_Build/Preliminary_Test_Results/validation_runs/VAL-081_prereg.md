# VAL-081 pre-registration — cervical-epic v0.1

**Card:** cervical-epic v0.1
**VAL ID:** VAL-081
**Date sealed:** 2026-04-25
**Cohort:** GSE68339 cervical HM450 — Lando 2015 Norwegian Radium Hospital cohort (270 cervical SCC tumor biopsies, FIGO II/III, NO normals)
**Specimen:** Cervical tumor biopsy
**n total:** 270

## Design

Cancer-only cohort confirmation. Score Stage 1 immune Xu-538 A on all 270 tumor biopsies. External normal comparator: VAL-073 GSE99511 normal cervical tissue (n=28, mean A=0.6811 ± 0.0222). Compute fraction of VAL-081 tumors above VAL-073 normal p95 / p99. Compute external Cohen's d (VAL-081 tumors vs VAL-073 normals).

## Pre-locked decision criteria

- O1_PASS_CANCER_CONFIRMATION: external d ≥ +0.5 with lower CI > 0 AND ≥30% of tumors above VAL-073 normal p95
- O2_PARTIAL_CONFIRMATION: 0 < d < 0.5
- O3_NULL: CI crosses zero
- O5_NEGATIVE_DIRECTION: d < 0 — confirms VAL-074 cohort-direction-flip pattern
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
- Pre-reg SEAL file: VAL-081_PREREG_SEAL.txt (lock-time SHA of this prereg)
- Results JSON: VAL-081_results.json
- Outcome.md: VAL-081_outcome.md
- Python script: val_081_cervical_epic.py
- Cohort manifest: included in results JSON

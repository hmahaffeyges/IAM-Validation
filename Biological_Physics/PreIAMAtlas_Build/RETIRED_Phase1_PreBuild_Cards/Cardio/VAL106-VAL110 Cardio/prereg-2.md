# VAL-106 — Calibration VAL: TCGA HM450K sesame Level 3 platform CHK-3.1 threshold establishment

**Status:** PRE-REGISTERED, sealed before β-value access  
**Purpose:** Establish the platform-specific CHK-3.1 beta distribution check threshold for TCGA HM450K sesame Level 3 substrate per CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION  
**Calibration class:** PLATFORM_CALIBRATION (distinct from disease-card validation series)  
**Date:** 2026-04-28  
**RNG seed:** 20260428

## Why this VAL exists

CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION (formalized 2026-04-28 from VAL-101 + VAL-102-voided self-correction) requires that the CHK-3.1 beta distribution check threshold for any new platform must be set by a calibration VAL on a structurally-separated cohort, NOT by retroactive accommodation of the data that triggered the discovery of platform mismatch.

The TCGA HM450K sesame Level 3 platform threshold is currently TBD per the CCL-041 platform table. Until established, three planned cardio-epic VALs are blocked:
- **VAL-103** GSE69138 stroke 3-subtype (HM450K, n=404 discovery + n=185 replication)
- **VAL-104** GSE84395 PAH endothelial cells (HM450K, n=39)
- **VAL-105** GSE84274 aortic dissection / BAV (HM450K, n=24)

In addition, retroactive analysis of the cookbook itself shows VAL-062 / VAL-098 / VAL-099 ran on TCGA HM450K sesame Level 3 without an explicit CHK-3.1 step; the platform threshold established here will retroactively contextualize those VAL outcomes (without changing their sealed status, per cookbook discipline).

A third independent data point was observed during the GSE69138 supplementary file inspection on 2026-04-28: the GSE69138 ave_beta first 5 samples produced extreme 21.9-27.3% / middle 8.8-13.0% on the same CHK-3.1 methodology — confirming the HM450K-typical pattern of slightly less extreme bimodality than raw EPIC β.

## Pre-locked methodology (sealed before β-access)

### Calibration cohorts (structurally separated from active cardio-epic and hcc-epic test cohorts)

**Cohort 1 — TCGA-KIRC (Kidney Renal Clear Cell Carcinoma) adjacent-normal**
- Source: NIH GDC public access via TCGA-KIRC project
- Substrate: adjacent-normal kidney tissue from KIRC patients
- Platform: HM450K sesame Level 3 (standard TCGA pipeline)
- Pre-locked sample selection criterion: ALL adjacent-normal samples in TCGA-KIRC with `vial_id` containing "11" (TCGA convention for adjacent-normal solid tissue) and QC-passing β-matrix (≥400,000 valid β values per sample, the cookbook standard)
- Structural separation: kidney tissue has no overlap with cardio-epic test cohorts (whole blood, endothelial cells, aortic tissue) or hcc-epic test cohorts (liver tissue, ccfDNA)

**Cohort 2 — TCGA-PRAD (Prostate Adenocarcinoma) adjacent-normal**
- Source: NIH GDC public access via TCGA-PRAD project
- Substrate: adjacent-normal prostate tissue from PRAD patients
- Platform: HM450K sesame Level 3 (standard TCGA pipeline)
- Pre-locked sample selection criterion: ALL adjacent-normal samples in TCGA-PRAD with `vial_id` containing "11" and QC-passing β-matrix (≥400,000 valid β values per sample)
- Structural separation: prostate tissue has no overlap with cardio-epic or hcc-epic test cohorts

### CHK-3.1 measurement methodology

For each QC-passed sample, compute on β values across all autosomal CpGs:
1. Fraction of β values at extreme [<0.10 or >0.90] — denote f_extreme
2. Fraction of β values at middle [0.40, 0.60] — denote f_middle
3. Median β
4. Number of valid β values (non-NA, non-null)

Aggregate across the calibration cohorts:
- **Per-cohort statistics**: mean, median, SD, min, max of f_extreme and f_middle across samples
- **Cross-cohort statistics**: pooled mean, median, SD, min, max of f_extreme and f_middle across both cohorts combined
- **Convergence test**: do TCGA-KIRC and TCGA-PRAD produce statistically indistinguishable f_extreme distributions (Mann-Whitney U test, two-tailed, α=0.05)?

### Pre-locked threshold derivation rule (the calibration logic)

The calibration threshold is defined as the **lower bound of f_extreme observed in healthy adjacent-normal HM450K sesame Level 3 samples**, formalized as:

**TCGA HM450K sesame Level 3 CHK-3.1 thresholds:**
- `extreme_threshold = max(15.0, mean(f_extreme) - 2*SD(f_extreme))` (rounded down to nearest 0.5%)
- `middle_threshold = min(12.0, mean(f_middle) + 2*SD(f_middle))` (rounded up to nearest 0.5%)
- The 15.0% floor and 12.0% ceiling are pre-locked to prevent platform thresholds from drifting into degeneracy where they fail to flag actually-processed data (CCL-040 risk)

A test sample passes CHK-3.1 on TCGA HM450K sesame Level 3 substrate if:
- `f_extreme ≥ extreme_threshold` AND `f_middle ≤ middle_threshold`

### Pre-locked outcomes

**O1_PLATFORM_THRESHOLD_ESTABLISHED**: Convergence between TCGA-KIRC and TCGA-PRAD (Mann-Whitney p > 0.05); thresholds derived as specified; thresholds within reasonable bounds (extreme ≥ 18%, middle ≤ 11% — the empirical HM450K range observed across multiple existing data points)

**O2_PLATFORM_DIVERGENCE_DOCUMENTED**: TCGA-KIRC and TCGA-PRAD produce significantly different f_extreme distributions (Mann-Whitney p ≤ 0.05). Outcome: report both per-tissue thresholds; defer single-platform-threshold establishment; flag as cookbook lesson that HM450K dye-bias correction has tissue-dependent behavior.

**O3_CALIBRATION_DEGENERATE**: Either TCGA-KIRC or TCGA-PRAD produces f_extreme distributions outside the empirical HM450K range (extreme < 18% or > 35%, or middle > 15%) — flag as data integrity issue with the calibration cohort itself; do not establish platform threshold from divergent data.

**O4_CALIBRATION_DATA_UNAVAILABLE**: TCGA-KIRC or TCGA-PRAD adjacent-normal samples cannot be acquired through public NIH GDC during this session.

### Pre-locked retroactive contextualization

After the threshold is established (O1) or per-tissue thresholds are reported (O2), the existing VAL outcomes that ran on TCGA HM450K sesame Level 3 are retroactively contextualized **without changing their sealed status**:
- VAL-062 (TCGA-COAD primary), VAL-098 (TCGA-READ), VAL-099 (TCGA-COAD revisit), VAL-101 (TCGA-LIHC) — outcomes remain as sealed; cookbook documentation updated to indicate which VALs would have passed/failed the now-established platform threshold
- VAL-101 specifically: extreme 26.6% will be evaluated against the established threshold; outcome label O5_DATA_INTEGRITY_FLAG remains sealed regardless

### Pre-locked cardio-epic unblock criterion

Upon outcome O1_PLATFORM_THRESHOLD_ESTABLISHED:
- VAL-103 (GSE69138), VAL-104 (GSE84395), VAL-105 (GSE84274) preregs become draftable using the established threshold
- Each prereg seals independently with the platform-specific threshold from VAL-106 cited

Upon outcome O2_PLATFORM_DIVERGENCE_DOCUMENTED:
- Cardio-epic VALs use whichever per-tissue threshold has the closer biological match (vascular tissue → use the more permissive of the two; whole blood → use the average); each prereg states the choice and rationale before sealing

Upon outcome O3 or O4:
- Cardio-epic VALs remain blocked; new calibration cohort selection required

## Pre-locked execution constraints

- Single Python script `val_106.py` using only stdlib + standard scientific libraries
- Bootstrap not required (this is a distribution-shape calibration, not an effect-size test)
- All decisions made by methodology specified above; no human-in-loop adjustments after seal
- RNG seed 20260428 for any stochastic step
- Source data acquisition via NIH GDC public API; all sample IDs and SHA-256 of downloaded β-matrices recorded in cohort_manifest.json

## What does NOT propagate from this VAL

- This is a CALIBRATION VAL. It does not produce biological findings about kidney or prostate cancer.
- The KIRC and PRAD samples are used purely as healthy-adjacent-normal HM450K substrate calibration anchors.
- No disease claims are made about KIRC or PRAD from this VAL.
- The thresholds established are platform-specific calibration values for downstream CHK-3.1 application only.

## Reproducibility triple per CHK-7.6

After execution, deliver:
1. **Inputs**: cohort_manifest.json with all sample IDs, GDC URLs, file SHA-256, file sizes
2. **Environment**: Python version, pandas/numpy versions, runtime, memory
3. **Expected headline output**: established thresholds in results.json + per-sample CHK-3.1 distribution data in per_sample.csv

## Pre-registration seal

This prereg is sealed via SHA-256 of its content prior to any β-value access from either calibration cohort. The seal commits the methodology, threshold derivation rule, structural separation criteria, and outcome categories before the calibration data are observed.

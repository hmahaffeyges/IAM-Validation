# VAL-107 — Cardio-epic CHK-3.1B calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal HM450K sesame Level 3

**Status:** PRE-REGISTERED, sealed before β-value access at the cardio-epic subset CpGs  
**Purpose:** Establish the CHK-3.1B (card-specific marker subset bimodality) threshold for cardio-epic on TCGA HM450K sesame Level 3 substrate, per the CHK-3.1A/B split convention adopted 2026-04-28  
**Calibration class:** PLATFORM_CALIBRATION (companion to VAL-106 CHK-3.1A calibration)  
**Date:** 2026-04-28  
**RNG seed:** 20260428

## Context

CHK-3.1 was split into CHK-3.1A (full-genome bimodality) and CHK-3.1B (card-specific marker subset bimodality) on 2026-04-28 per Heath signoff. VAL-106 established the CHK-3.1A pattern for TCGA HM450K sesame Level 3 substrate (KIRC f_extreme 56.32%, PRAD 54.58%, combined 55.87%). VAL-107 establishes the CHK-3.1B threshold for cardio-epic specifically.

## The cardio-epic CHK-3.1B marker subset (frozen before β-access)

The cardio-epic CHK-3.1B marker subset is the union of all card-acquisition-time-evaluable CpG IDs the cardio-epic scoring will use:

- **Loyfer 25-tile reference** (run-everything architecture per Heath signoff 2026-04-26): 6,105 unique CpGs
- **UniLIFE 19-cell Stage 3 immune** (Guo 2025): 1,906 CpGs
- **Salas Blood.EPIC IDOL 450K** (Stage 3 immune baseline): 350 CpGs
- **Total unique CpGs**: 8,100

**Excluded from acquisition-time CHK-3.1B (evaluated at scoring time per cookbook decision):**
- EpiSCORE HeartRef (207 integer marker IDs require Illumina manifest mapping; cg-ID intersection with input file is not acquisition-time-evaluable)
- Caggiano CelFiE TIM heart_meth + endothelial_meth (WGBS chrom/start/end region format, requires region-to-CpG mapping per array)
- Xu-538 immune panel (patent-protected, CpG list not in vault; HM450K coverage 484/538 = 90.0% per VAL-100 documentation)

The 8,100-CpG subset is the acquisition-time portion of CHK-3.1B for cardio-epic. The deferred components are gated separately at scoring time, not at acquisition.

**Subset frozen file:** `cardio_epic_chk31b_subset.txt` (sorted CpG IDs, one per line, SHA-256 computed below)

## Calibration cohorts

**Cohort 1 — TCGA-KIRC adjacent-normal**  
n=160 candidate, n=144 QC-pass on full-genome (per VAL-106 manifest). All NIH GDC public, sesame Level 3.

**Cohort 2 — TCGA-PRAD adjacent-normal**  
n=50 candidate, n=50 QC-pass on full-genome (per VAL-106 manifest). All NIH GDC public, sesame Level 3.

**Manifest:** Reuses VAL-106 cohort_manifest.json (210 samples, all SHA-256-tracked). No new downloads.

## Methodology (frozen by sealed prereg)

For each sample in the manifest:

1. Read the full sesame Level 3 β file
2. Subset to the 8,100 cardio-epic marker CpGs
3. Compute on the subset:
   - n_subset_valid: CpGs in subset with non-NA β value in this sample
   - f_extreme_subset: fraction of subset CpGs with β<0.10 or β>0.90
   - f_middle_subset: fraction of subset CpGs with 0.40 ≤ β ≤ 0.60
   - median β across subset
4. QC threshold for CHK-3.1B subset coverage: ≥7,000 valid CpGs in subset (≥86% of 8,100). Below this → CHK-3.1B fails on coverage grounds, regardless of bimodality (this catches lift-over dropouts and panel-coverage damage).

## Pre-locked threshold derivation rule

After per-sample CHK-3.1B distributions are computed:

- Per-cohort statistics: mean, median, SD, min, max of f_extreme_subset and f_middle_subset
- Cross-cohort statistics on KIRC + PRAD combined
- Mann-Whitney U test for KIRC vs PRAD f_extreme_subset convergence

**CHK-3.1B threshold derivation rule** (analogous to CHK-3.1A but adapted for subset measurements):
- `extreme_threshold_B = max(8.0, mean_subset(f_extreme_subset) - 2*SD_subset(f_extreme_subset))` rounded down to nearest 0.5%
- `middle_threshold_B = min(15.0, mean_subset(f_middle_subset) + 2*SD_subset(f_middle_subset))` rounded up to nearest 0.5%
- The 8.0% extreme floor and 15.0% middle ceiling are set wider than CHK-3.1A's bounds to accommodate the inherently-different distribution shape on a curated marker subset (which preferentially selects bimodal CpGs by atlas design)
- A test sample passes CHK-3.1B for cardio-epic iff (n_subset_valid ≥ 7000) AND (f_extreme_subset ≥ extreme_threshold_B) AND (f_middle_subset ≤ middle_threshold_B)

## Pre-locked outcomes

**O1_CHK_3_1B_THRESHOLD_ESTABLISHED**: Convergence between TCGA-KIRC and TCGA-PRAD on f_extreme_subset (Mann-Whitney p > 0.05); thresholds derived as specified; thresholds within reasonable bounds (extreme ≥ 10%, middle ≤ 25% — wider envelope than CHK-3.1A because subset distributions on curated markers are inherently more variable).

**O2_PLATFORM_DIVERGENCE_DOCUMENTED**: TCGA-KIRC and TCGA-PRAD produce significantly different f_extreme_subset distributions (Mann-Whitney p ≤ 0.05). Outcome: report both per-tissue thresholds; cardio-epic uses the wider (more permissive) of the two for HM450K, with rationale logged.

**O3_CALIBRATION_DEGENERATE**: Either KIRC or PRAD f_extreme_subset is outside [10%, 80%] OR f_middle_subset > 30%. Flag as data integrity issue with the calibration cohort itself or with the cardio-epic subset definition.

**O4_SUBSET_COVERAGE_FAILURE**: ≥10% of calibration samples fail the n_subset_valid ≥ 7000 coverage threshold. Flag as evidence the cardio-epic subset has structural lift-over damage on TCGA HM450K, even on healthy samples — would require redefining the subset before any cardio-epic VAL.

## Pre-locked unblock criterion

Upon outcome O1_CHK_3_1B_THRESHOLD_ESTABLISHED:
- VAL-108 (GSE69138 stroke 3-subtype), VAL-109 (GSE84395 PAH endothelial cells), VAL-110 (GSE84274 aortic dissection / BAV) preregs become draftable
- Each prereg seals with both CHK-3.1A threshold (from VAL-106) and CHK-3.1B threshold (from VAL-107) pre-locked
- Each cardio VAL prereg states the threshold values explicitly and references VAL-106 + VAL-107 SHA-256 seals

Upon outcome O2_PLATFORM_DIVERGENCE_DOCUMENTED:
- Cardio-epic uses the more-permissive of the two per-tissue thresholds, with rationale logged in each VAL prereg

Upon outcome O3 or O4:
- Cardio-epic VALs remain blocked; subset definition or calibration cohort reconsidered

## What does NOT propagate from this VAL

- This is a CALIBRATION VAL. No biological findings about kidney or prostate.
- The KIRC and PRAD samples are calibration anchors only.
- The CHK-3.1B threshold established applies only to cardio-epic on TCGA HM450K sesame Level 3 substrate. Other cards need their own CHK-3.1B calibration on the same cohort. Other substrates need separate CHK-3.1A and CHK-3.1B calibrations.

## Reproducibility per CHK-7.6

After execution:
1. Inputs: cohort_manifest.json (reused from VAL-106), cardio_epic_chk31b_subset.txt with SHA-256
2. Environment: Python 3 stdlib only; runtime ~1-2 minutes
3. Output: results.json + per_sample.csv

## Pre-registration seal

This prereg is sealed via SHA-256 of its content prior to any subset β-value access from either calibration cohort. The seal commits the methodology, subset definition, threshold derivation rule, and outcome categories before the calibration data are observed at the subset level.

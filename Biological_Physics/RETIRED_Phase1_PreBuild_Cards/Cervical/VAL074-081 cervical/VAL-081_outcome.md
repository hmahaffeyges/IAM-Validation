# VAL-081 — GSE68339 Lando 2015 cervical SCC OUTCOME

**Card:** cervical-epic v0.1
**Date:** 2026-04-25
**Cohort:** GSE68339 Lando 2015 cervical SCC HM450 (270 tumor biopsies, all FIGO stage II/III SCC, no normals)
**Specimen:** Cervical tumor biopsy
**Outcome:** O5_NEGATIVE_DIRECTION — tumors read BELOW VAL-073 normals at large n; confirms tissue-arm cohort heterogeneity established by VAL-074

## Summary

VAL-081 is a cancer-only cervical SCC cohort (Norwegian Radium Hospital, Lando et al. 2015) with 270 tumor biopsies on HM450. Pre-locked decision criterion uses VAL-073 GSE99511 normal cervical tissue (n=28, mean A=0.6811 ± 0.0222) as external comparator: cervical SCC tumors should read above normal at d ≥ +0.5 with lower CI > 0 = O1_PASS_CANCER_CONFIRMATION.

Result: VAL-081 tumors mean A = **0.6640 ± 0.0411**, BELOW VAL-073 normals at d = **−0.4306** [−0.8213, −0.0399]. Only 6.7% of the 270 tumors fall above the VAL-073 normal 95th percentile.

This is the SAME negative-direction pattern as VAL-074 GSE46306 (Farkas/Swedish cohort, n=43). Two independent European tissue cohorts at large total n=313 read cervical disease BELOW the Verlaat (Amsterdam) normal baseline.

## Numerical results

| Statistic | VAL-073 normal (anchor) | VAL-073 SCC (anchor) | VAL-081 tumors (this cohort) |
|---|---|---|---|
| n | 28 | 4 | 270 |
| Mean A | 0.6811 | 0.7083 | 0.6640 |
| SD A | 0.0222 | 0.0109 | 0.0411 |
| Cohen's d vs VAL-073 normal | (reference) | +1.27 | **−0.43** |
| 95% CI | — | [+0.18, +2.37] | [−0.82, −0.04] |

VAL-081 tumor distribution:
- Range: [0.5697, 0.7969]
- 6.7% above VAL-073 normal p95 threshold (A > 0.7246)
- 3.7% above VAL-073 normal p99 threshold (A > 0.7384)
- 93.3% of VAL-081 tumors score WITHIN OR BELOW the VAL-073 normal range

Per CHK-3.5: no saturation flagging required — A-scores at 50-67% of immune ceiling, well below runtime flag.

## Data integrity status: VERIFIED

- File format: GenomeStudio AVG_Beta output, raw β.
- β distribution sanity check: bimodal, appropriate for tissue (~30% at extremes).
- Panel coverage: full 538/538 Xu-538 CpGs present on HM450.
- Sample assignment: all 270 samples are cervical SCC tumor biopsies per Sample_title verification.
- QC pass: 270/270 (no sample dropped).

The data is real raw β. The reading is real. The interpretation depends on which cohort baseline you compare to.

## What this means for cervical-epic v0.1

VAL-081 is the third tissue cohort and the second to read negative-direction:

| VAL | Cohort | Country | n | Direction vs cohort-internal normal |
|---|---|---|---|---|
| VAL-072 | TCGA-CESC | mixed | 3 paired | exploratory only (n=3, CI straddles zero) |
| **VAL-073** | **GSE99511 Verlaat (Amsterdam)** | **NL** | **68** | **POSITIVE: Normal vs CIN3 d=+0.73, monotonic Normal<CIN3<SCC** |
| VAL-074 | GSE46306 Farkas (Stockholm) | SE | 43 | NEGATIVE: Normal vs CIN3 d=−0.61 (cohort uses 20 HPV-negative healthy) |
| VAL-081 | GSE68339 Lando (Oslo) | NO | 270 | NEGATIVE: tumors d=−0.43 vs VAL-073 normals (no internal normals) |

**The Verlaat cohort is the outlier**, not VAL-074 / VAL-081. At total tissue-arm n=311 across the three cohorts with normals or external comparators, the dominant signal direction is NEGATIVE.

This is a real framework finding that requires cervical-epic v0.1 to NOT make a clinical claim on tissue-arm Stage 1 immune scoring at this evidence level. The single-cohort Verlaat anchor (VAL-073) cannot carry the card's clinical claim against the weight of VAL-074 + VAL-081 disagreeing.

## Most likely explanations (not yet distinguished)

1. **Cohort-design heterogeneity.** VAL-073 (Verlaat) used population-normal cervical tissue from women without CIN history attending colposcopy with normal histology. VAL-074 (Farkas) used HPV-negative healthy cervical samples — a stricter normal selection that may itself shift baseline. VAL-081 (Lando) has no internal normals; using VAL-073 normals as external comparator confounds with the Verlaat-specific baseline.

2. **HPV genotype distribution.** VAL-074 normals are HPV-negative by selection. VAL-073's HPV status of normals was not specified at parse time. VAL-081 tumors are mostly HPV-positive. HPV-driven cervical lesions may shift the immune compartment differently than general inflammation, and the Xu-538 panel may not capture the directional shift in HPV-driven cervical immunology.

3. **Stage / progression heterogeneity.** VAL-081 tumors are FIGO stage II/III SCC — invasive but mostly locoregional. The Xu-538 immune-class signal may shift with disease stage in non-monotonic ways.

4. **Population/regional heterogeneity.** Three Northern European populations (NL, SE, NO) but different recruitment criteria. Cohort-specific selection biases not yet enumerated.

## Cancellation hypothesis status (per CCL-031)

Test 1 pooled is NEGATIVE-direction in VAL-081 (and VAL-074), POSITIVE in VAL-073. This is NOT the AD-instance bidirectional cancellation pattern (pooled-null + directional-pass). It is a cohort-direction-flip, similar to CCL-019 (CRC compartment-flip). Per CCL-031, this is NOT bidirectional cancellation; it's its own pattern of cohort heterogeneity that requires Test 2 (lymphoid vs myeloid sub-panel split) to dissect — and Test 2 is blocked on OQ-2026-01.

## Card consequences

VAL-081 is the load-bearing finding for cervical-epic v0.1's tier downgrade. Combined with VAL-074, the tissue arm cannot be claimed at single_cohort_validated tier; the VAL-073 anchor stands alone against two cohorts at total n=313 reading the opposite direction.

cervical-epic v0.1 final tier: `exploratory_with_cohort_heterogeneity`. Path forward documented in v0.2+ engineering plan: build cervical-LBC-specific Stage 1 panel, or substitute published cervical methylation panels (FAM19A4/miR124-2, ZNF671, EPB41L3, PAX1/NREP-AS1) with dedicated H_min calibration, OR run Test 2 once OQ-2026-01 operationalizes to determine whether the cohort-direction-flip is lymphoid-myeloid driven.

## Reproduction
- Pre-reg SHA: (sealed at runtime)
- Results JSON SHA: 4200b4f87242281d91475163842d0b9c
- RNG seed: 20260425
- Panel: Xu-538 SHA ada672960...
- Source: GSE68339 GEO public access (Lando et al. 2015, Genes Chromosomes Cancer)
- External comparator: VAL-073 GSE99511 normal cervical tissue (n=28, mean A=0.6811 ± 0.0222)

## Lessons cited
- cerv-LL-010 (cohort-baseline shifts are diagnostic)
- cerv-LL-014 (common sense biology check)
- cerv-LL-016 (diagnostic order)
- CHK-3.2 (cross-cohort healthy baseline check)
- CCL-029 (cohort-completeness — VAL-081 surfaces VAL-073 outlier status)
- CCL-031 (NOT bidirectional cancellation — cohort-direction-flip, not AD-instance)

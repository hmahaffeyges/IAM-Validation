# CPG-VAL-020 — Pre-Registration

**VAL ID:** CPG-VAL-020
**Title:** Hannum aging anchor reproduction — full SOP chain on IAMAtlas (Walther + 115-cell A-scoring + n=601 Mahalanobis + IAMCellularAge + 6-tier breakpoints)
**Date sealed:** 2026-06-06
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Heath's directive:** "I think we should download and reproduce it again for the meeting, to show our diligence. We may also learn something new with the full IAMAtlas and two deconvolvers, etc."

## Cohort

- **Source:** GSE40279 (Hannum 2013, Mol Cell 49(2):359-367), n=656 healthy whole blood
- **Demographics:** Age 19-101 (broad span), M=318 F=338, ethnicity Caucasian-European (n=426) + Hispanic-Mexican (n=230), 9 plates
- **Platform:** Illumina HumanMethylation450
- **Acquisition:** GEO series_matrix.txt.gz downloaded + parsed 2026-06-06; 473,034 CpGs × 656 samples saved as float32 npz at `/tmp/geo_downloads/GSE40279_beta_matrix.npz`

## Chain modules used (ALL canonical, no shortcuts)

| Stage | Module | Artifact | Notes |
|---|---|---|---|
| 2 | `WaltherIAMDeconvolver.deconvolve(refine_celltypes=True)` | IAMAtlas REBUILD (483,092 CpGs × 8 classes × 115 cell-types) | Primary deconvolver |
| 4 | `score_per_celltype` + `score_per_class` | `iamatlas_celltype_markers_v0_2.json` (115 celltypes × 100 markers, H_min_by_class frozen 2026-04-06) | Canonical markers |
| 5 | `MahalanobisHealthyHull.score()` | `mahalanobis_healthy_reference_v0_1.json` (n=601 HC, 112 features in 115-cell A-score feature space) | Pooled HC from GSE51057+GSE51032 |
| 6 | `IAMCellularAge.score_patient` | `age_reference_matrix.json` (80-cell baseline, decade-binned β_mean curves per architectural class) | β_mean inversion |
| 7 | 6-tier breakpoints v1.2 | `tier_breakpoints.json` (SUPPRESSED <0.95 / NORMAL [0.95,1.04) / ELEVATED [1.04,1.07) / WARBURG_TRANSITION [1.07,1.10) / SIGNIFICANTLY_ELEVATED [1.10,1.12) / BREACH ≥1.12) | Universal physics-derived |

## Primary signal

`Pearson r(immune_cellular_age, chronological_age)` — the canonical anchor pre-build VAL-006 found r=0.9999 with the Hannum 71-CpG clock.

## Decision rule

This is a REPRODUCTION test, NOT a pass/fail test. The locked-scope verbiage is "show our diligence." We expect that the new chain may produce different results because:
- Pre-build VAL-006 used a regression-trained predictor (Hannum 71-CpG clock fit to chrono_age by construction)
- The new IAMCellularAge is a physics-based β_mean inversion against a fixed 80-cell baseline (no training, no fit)
- The 80-cell baseline was built from foundation cohort GSE51057+GSE51032 (EPIC-Italy women, 40-65)
- When applied to Hannum (mixed-sex, 19-101 US/Mexican), inversion may saturate

Reporting requirements regardless of outcome:
1. `r(immune_cellular_age, chrono_age)` Pearson + Spearman + MAE
2. `r(A_immune, chrono_age)` — the raw architectural-aging signal at the H(β_mean)/H_min layer
3. `r(A_class, chrono_age)` for each of 8 classes
4. Mahalanobis distance distribution vs n=601 HC reference
5. Tier distribution per class
6. Walther status distribution (chain integrity)
7. Cellular age saturation rate
8. Cosmic Methylome PNG for one example HC sample

## Pre-specified analysis plan

- Headline: `r(immune_cellular_age, chrono_age)` — report whatever it is
- Diagnostic: if cellular age saturates, report raw A_immune slope vs age as the genuine architectural signal
- Cross-cohort context: report Mahalanobis distribution as evidence of n=601 HC reference cohort-boundedness

## What constitutes a "successful" VAL

A VAL is successful when:
- Chain runs end-to-end without errors (Walther status OK for all samples)
- All canonical modules produce real numeric outputs (no stubs)
- The honest finding is documented, whether it confirms or contradicts pre-build

## Frozen inputs (SHAs to be recorded at seal)

- IAMAtlas REBUILD CSV: 605,124,914 bytes
- celltype markers artifact: from `iamatlas_celltype_markers_v0_2.json` source_sha256 field
- Mahalanobis HC reference: `mahalanobis_healthy_reference_v0_1`
- Age reference matrix: 80-cell baseline
- H_min(immune): 0.838889 (G-003b MCMC frozen 2026-04-06)

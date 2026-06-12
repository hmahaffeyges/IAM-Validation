# CPG-VAL-017 — Inflammaging in pooled hull HC (Pre-Registration)

**Card:** Immune universal v1.0
**Date pre-registered:** 2026-06-07
**Status:** PREREG

## Question
Does the immune class A-score show a quantifiable "inflammaging" trajectory — accelerating decline in late adult life — when pooled across the v0_5 Mahalanobis HC hull cohorts where age metadata is available?

## Background
VAL-015 established the linear-trend signal in Hannum alone (r=-0.20, decade-median monotonicity Spearman ρ=-0.85). VAL-017 extends this to the pooled-cohort scale (multi-population, multi-platform) and tests for the "acceleration after age 70" phenomenon (inflammaging) that is the established gerontology signal.

## Cohorts (pooled where age metadata available)
- GSE40279 Hannum (HM450, US, ages 19-101) — primary aging cohort, n=656
- GSE50660 Tsaprouni (HM450, UK, ages 40-65) — moderate age range
- GSE141682 Han Chinese (EPIC, ages 18-62) — n=42, 18-62
- Optional: any other cohort with age metadata pulled from existing CSVs

Note: foundation EPIC-Italy (GSE51057/51032) cohorts age 40-65 fixed range; AddNeuroMed/AIBL/GIFT have age but range narrower. Use what's available with age annotated.

## Pre-specified pass conditions

**Primary (HARD):**
1. Pooled A_immune-vs-age regression: r < -0.15, p < 0.001 (cross-cohort signal preserved)
2. Per-cohort: A_immune-vs-age direction SAME (negative slope) in ≥3 contributing cohorts
3. Late-life acceleration: per-decade slope after age 70 STEEPER than slope before age 50 (compares decade-medians)

**Secondary (DIAGNOSTIC):**
4. Standardized per-cohort effect: |z-score of A_immune at age 80| vs |z at age 30| differ by ≥0.5 z-units

## Method
1. Pool age + A_immune from all cohorts with available metadata
2. Cohort-fixed-effects regression: A_immune ~ age + cohort
3. Per-decade medians (pooled)
4. Compare slopes pre-50 vs post-70 (piecewise linear)
5. Per-cohort breakdowns

## Deliverables
val_017.py, CPG_VAL_017_per_sample.csv, results.json, stratified_results.json,
null_results.json, cohort_manifest.json, OUTCOME.md

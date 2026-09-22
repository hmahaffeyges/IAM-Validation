# CPG-VAL-015 — Immune class aging trajectory (Pre-Registration)

**Card:** Immune universal v1.0
**Date pre-registered:** 2026-06-07
**Status:** PREREG (pass conditions stated BEFORE execution)

## Question
Does the immune class's information-loss (A-score decline with age) follow a quantifiable aging trajectory in the Hannum cohort, distinct from a simple linear-age-clock fit?

## Background
CPG-VAL-020 (Hannum anchor reproduction) found r(A_immune, chronological_age) = -0.184 (p=1.97e-6) and r(A_stem_pluri, age) = -0.184. These are CROSS-COHORT physics-layer correlations — not Hannum-71-CpG-clock tautological r=0.99. VAL-015 converts this into a quantitative trajectory readout in A-score space.

## Cohort
- GSE40279 Hannum 2013, n=656 whole blood healthy
- Age range 19-101, mixed sex
- HM450 platform
- Source: VAL-020 sealed 115-cell A-scores CSV

## Pre-specified pass conditions

**Primary (HARD):**
1. A_immune slope vs chronological age is significantly negative (Pearson r < -0.10, p < 0.001)
2. Slope SURVIVES sex stratification (significant in BOTH male and female subgroups, same sign)
3. Slope SURVIVES random 50/50 split (cross-validation: both halves yield consistent slope ± 0.05)

**Secondary (DIAGNOSTIC, no fail condition):**
4. Per-decade median A_immune monotone decline (Spearman across 9 age decades)
5. Stratification by sex reveals quantitative ΔA per decade
6. Non-immune class controls (e.g., A_stromal_average) do NOT show same magnitude decline (specificity check)

## Pre-specified fail / null conditions
- Pass condition 1: |r| < 0.10 OR p > 0.001 → FAIL
- Pass condition 2: opposite signs across sex strata → FAIL
- Pass condition 3: half-vs-half slope difference > 0.05 → FAIL

## Method
1. Load existing 115-cell A-score long-format CSV from VAL-020
2. Pivot to wide (gsm × celltype A-score)
3. Join with sample metadata (age, sex)
4. Compute A_immune class average per sample (mean over immune-classed celltypes)
5. Per-decade median + slope regression
6. Sex-stratified analyses
7. 50/50 split null

## Deliverables
- val_015.py (sealed runner)
- CPG_VAL_015_per_sample.csv (per-sample A_immune, A_stem_pluri, A_stromal, age, sex)
- results.json (primary slopes, p-values, decade medians)
- stratified_results.json (sex strata slopes)
- null_results.json (50/50 split test)
- cohort_manifest.json (provenance)
- OUTCOME.md (PASS/FAIL declaration + interpretation)

## Outcome codes
- PASS: all 3 hard conditions met
- DIRECTIONAL: r < 0 with p<0.001 but fails stratification or CV
- NULL: |r| < 0.10 or p > 0.001

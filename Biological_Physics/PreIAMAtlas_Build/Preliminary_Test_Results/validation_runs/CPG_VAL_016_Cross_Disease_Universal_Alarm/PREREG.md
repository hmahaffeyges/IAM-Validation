# CPG-VAL-016 — Cross-disease universal alarm (Pre-Registration)

**Card:** Immune universal v1.0
**Date pre-registered:** 2026-06-07
**Status:** PREREG (pass conditions stated BEFORE execution)

## Question
Does the immune class A-score fire as a "something is off" universal alarm across diseases of different etiologies — oncology (breast pre-dx) AND neurodegeneration (AD)?

## Background
The immune universal v1.0 card claims the immune-class architectural information score (A_immune, mean over 33 immune celltypes in the 115-cell atlas) is a CROSS-DISEASE alarm — informationally a "patient deviation from healthy immune posture" signal that does NOT require disease-specific training.

VAL-019 (just completed) showed that the bidirectional decomposition reveals disease-specific FIRING PATTERNS (AD activates both up + down; breast activates only down). VAL-016 tests the complementary claim: the POOLED A_immune (direction-naïve) shifts case-vs-HC in BOTH diseases — i.e., disease-agnostic "alarm" component.

## Cohorts
- **AIBL GSE153712**: AD vs HC (EPIC 850K) — 161 AD, 471 HC
- **GSE51057**: Breast pre-dx case vs HC (HM450) — 11 case, 177 HC
- **GSE51032**: Breast pre-dx case vs HC (HM450) — 36 case, 424 HC
- All canonical 115-cell A-scores already computed.

## Pre-specified pass conditions

**Primary (HARD):**
1. A_immune Cohen's d significantly NON-ZERO in BOTH AD and breast cohorts (|d| ≥ 0.20)
2. Disease-specific direction allowed (sign may differ) — universality means SHIFTS, not same-direction shifts
3. Pooled-cohort meta-analysis: combined d significantly non-zero (|d_meta| ≥ 0.20, p < 0.01)

**Secondary (DIAGNOSTIC):**
4. Class-specificity: A_immune effect stronger than at least 4 of the 7 other classes in at least one cohort

## Method
1. Load canonical 115-cell A-scores for AIBL + GSE51057 + GSE51032
2. Compute A_immune class average per sample (mean over immune celltypes)
3. Cohen's d case vs HC per cohort + p-value via Welch t-test
4. Per-class Cohen's d across all 8 classes per cohort
5. Meta-d via inverse-variance weighting

## Deliverables
val_016.py, CPG_VAL_016_per_sample.csv, results.json, stratified_results.json,
null_results.json, cohort_manifest.json, OUTCOME.md

## Outcome codes
- PASS: all 3 hard conditions met
- DIRECTIONAL: 2/3 met
- NULL: <2 met

## Note re Crohn's
VAL-128 Crohn's blood cohort uses a PRE-CANONICAL pipeline (Xu-538 + Loyfer
reference panels), not the canonical 115-cell markers. Direct cross-cohort
comparison with breast/AD canonical A-scores would be apples-to-oranges.
For this VAL we use the 2 cohorts already canonically scored (breast +
AD). Crohn's re-scoring with canonical markers is queued as a separate
work item (would require β data + Walther pipeline run).

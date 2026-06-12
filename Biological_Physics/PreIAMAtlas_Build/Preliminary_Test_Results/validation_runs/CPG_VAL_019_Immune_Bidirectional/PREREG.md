# CPG-VAL-019 — Bidirectional immune direction discrimination (Pre-Registration)

**Card:** Immune universal v1.0
**Date pre-registered:** 2026-06-07
**Status:** PREREG (pass conditions stated BEFORE execution)

## Question
Does the bidirectional decomposition of the VAL-051 sealed 7-CpG immune panel (2 up + 5 down in AD-anchored direction) provide MEANINGFUL discrimination over the direction-naïve pooled signal, AND does it reveal disease-specific bidirectional firing patterns across breast pre-dx and AD?

## Background
- VAL-050 found pooled 18-CpG immune panel signal NULL (pooled d=+0.08 in AIBL)
- VAL-051 found Rule A 7-CpG directional panel (2 up + 5 down in AD-direction): AIBL holdout d=+0.62 AUC=0.84
- The bidirectional doctrine: pooled A discards directional information; signal can be recovered by treating up-CpGs and down-CpGs as separate scores.

## Cohorts
- **GSE51057** breast pre-dx cases vs HC (foundation EPIC-Italy, HM450)
- **AIBL GSE153712** AD vs HC (EPIC 850K) — VAL-051 training/holdout reference
- VAL-051 sealed 7-CpG Rule A panel (canonical, frozen 2026-04-23)

## Pre-specified pass conditions

**Primary (HARD):**
1. In AIBL (the AD-anchored cohort), the directional signal d ≥ +0.30 in BOTH up-CpGs and down-CpGs subpanels (each direction contributes)
2. Pooled (direction-naïve mean) d is SMALLER than directional combined d in absolute value — demonstrates information loss from pooling
3. In GSE51057 (breast pre-dx, NOT AD-anchored), the bidirectional firing pattern is DIFFERENT from AIBL — i.e., disease-specificity is preserved

**Secondary (DIAGNOSTIC):**
4. Per-CpG sign concordance with panel direction in AIBL cases vs HC

## Method
1. Extract β values for the 7 panel CpGs from each cohort's existing β CSV
2. For each sample: compute pooled mean β (direction-naïve), up-CpG mean β, down-CpG mean β
3. For each cohort: Cohen's d cases vs HC for each metric
4. Compare per-direction d vs pooled d
5. Cross-cohort comparison: AIBL vs GSE51057 firing patterns

## Deliverables
- val_019.py
- CPG_VAL_019_per_sample.csv (per-cohort, per-sample beta values + scores)
- results.json
- stratified_results.json (per-cohort)
- null_results.json (random direction shuffle null)
- cohort_manifest.json
- OUTCOME.md

## Outcome codes
- PASS: all 3 hard conditions met
- DIRECTIONAL: only condition 1 met (directional signal present but interpretation requires nuance)
- NULL: condition 1 fails

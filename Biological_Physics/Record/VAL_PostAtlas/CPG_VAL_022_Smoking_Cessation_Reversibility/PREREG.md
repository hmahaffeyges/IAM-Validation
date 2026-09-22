# CPG-VAL-022 — Lifestyle reversibility: smoking cessation trajectory (Pre-Registration)

**Card:** Immune universal v1.0
**Date pre-registered:** 2026-06-07
**Status:** PREREG (pass conditions stated BEFORE execution)

## Question
Does the immune-class A-score in FORMER smokers (who made a documented lifestyle change) sit intermediate between never-smokers (baseline) and current smokers (active insult), and closer to never than to current — i.e., does the framework's architectural readout show that lifestyle improvement reverses trajectory toward healthy baseline?

## Why this matters for the Escobedo conversation
Clinical patients want to know whether changes they make CAN move them back toward health, not just whether the framework can detect departure from health. Smoking cessation is the cleanest natural-experiment for "lifestyle improvement" available in the GSE50660 Tsaprouni cohort — 263 individuals who made a documented quit. If A_immune(former) is intermediate and closer to never, that's quantitative evidence that the architectural state IS movable.

## Cohort
- **GSE50660 Tsaprouni 2014** (UK, HM450, whole blood, mean age 55, n=464)
- Smoking field encoded: 0 = never (n=179), 1 = former (n=263), 2 = current (n=22)
- Already scored canonically (115-cell A-scores in v0_5 hull build)
- Already in Mahalanobis HC hull v0_5

## Pre-specified pass conditions

**Primary (HARD):**
1. A_immune(current) significantly differs from A_immune(never) (Welch t-test p < 0.05, |d| ≥ 0.30) — confirms smoking insult IS detectable in A_immune
2. A_immune(former) is INTERMEDIATE: between never and current means (not outside the bracket)
3. Reversibility ratio: |A_immune(former) - A_immune(never)| / |A_immune(current) - A_immune(never)| ≤ 0.7 (former is at least 30% of the way back from current toward never)

**Secondary (DIAGNOSTIC):**
4. Effect survives age adjustment (linear model A_immune ~ smoking_status + age + gender)
5. Per-class specificity: A_immune shows reversibility more strongly than at least 4 of 7 other class A-scores

## Method
1. Load GSE50660 115-cell A-scores (already on disk from hull build)
2. Join with metadata (smoking, age, gender)
3. Compute A_immune class mean per sample (mean over immune celltypes in canonical class)
4. Three-group analysis: never vs former vs current
5. Cohen's d pairwise + reversibility ratio
6. Age-adjusted linear regression (covariate)
7. Per-class scan

## Outcome codes
- PASS: all 3 hard conditions met
- DIRECTIONAL: conditions 1+2 met but condition 3 fails (former trajectory present but not strongly reversible)
- NULL: condition 1 fails (smoking effect not detected) OR condition 2 fails (former is outside the bracket — anomalous)

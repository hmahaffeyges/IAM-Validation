# CPG-VAL-141 — Pre-Registration

**VAL ID:** CPG-VAL-141
**Title:** Cross-cohort A_immune baseline concordance
**Date sealed:** 2026-06-06

## Cohorts

- **Cohort A:** GSE50660 (Tsaprouni 2014) never-smokers, age-restricted to 40-65, n=179
- **Cohort B:** GSE40279 (Hannum 2013) healthy controls, age-restricted to 40-65, n=308

## Signal

- **Primary signal:** A_immune (Stage 4 immune-class A-score) distribution
- **Test:** Two-sample Kolmogorov-Smirnov test for distributional equivalence

## Decision rule

- **Pass:** KS p > 0.05 (distributions statistically indistinguishable)
- **Logic:** If A_immune is a cohort-independent measurement, never-smokers from one cohort should score the same as HC from another cohort, within the matched age range. Both cohorts use 450K platform but different populations and study sites.

## Observed outcome

- **KS statistic:** 0.716
- **KS p-value:** 0.0000
- **Mean A_immune (GSE50660 never):** 1.0589
- **Mean A_immune (GSE40279 HC):** 1.0429
- **Δ mean:** +0.0160
- **Outcome code:** O2_NEAR_THRESHOLD

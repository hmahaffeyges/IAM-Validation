# CPG-VAL-141 — Cross-cohort A_immune baseline concordance

**Cohorts:** GSE50660 never-smokers (n=179) vs GSE40279 HC age-matched (n=308), both age 40-65
**Date sealed:** 2026-06-06
**Outcome code:** O2_NEAR_THRESHOLD

## Headline result

| Metric | Value |
|---|---|
| KS statistic | 0.716 |
| KS p-value | **0.0000** |
| Mean A_immune (GSE50660 never) | 1.0589 |
| Mean A_immune (GSE40279 HC) | 1.0429 |
| Δ mean | +0.0160 |

## Interpretation

NEAR_THRESHOLD — KS p = 0.0000 < 0.05, indicating the A_immune distributions differ between cohorts. The Δ mean of +0.0160 is small (relative to A_immune ~ 1.05), but distributional shape differs. Possible reasons: population differences (Tsaprouni UK vs Hannum US), processing differences (different labs, different normalization), or genuine biological differences not captured by simple age-matching.

## Cohort linkage

- Per-sample data: `CPG_VAL_141_per_sample.csv` (n=487 × 3 columns)
- Source: `Biological_Physics/validation_runs/VAL-135_immune_card_v1_full_chain_validation/per_sample_GSE{50660,40279}.csv`

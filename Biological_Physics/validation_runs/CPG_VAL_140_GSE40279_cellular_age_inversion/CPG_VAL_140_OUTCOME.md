# CPG-VAL-140 — 8-class A-score vector recovers 25.1% of chronological age variance

**Cohort:** GSE40279 Hannum 2013, n=656, age 19-101
**Date sealed:** 2026-06-06
**Outcome code:** O5_BASELINE_DOMINATED

## Headline result

| Metric | Value |
|---|---|
| R² (8-class A-vector → age) | **0.251** |
| Pearson r | 0.501 |
| MAE | 10.1 years |
| n | 656 |

## Interpretation

BELOW_THRESHOLD — The 8-class A-vector alone explains only 25.1% of age variance. The production Stage 6 inversion against the 80-cell baseline curve will use more features (115-cell A-scores, not just 8-class) and is expected to do substantially better. VAL-140 establishes a lower bound; the full Stage 6 inversion is deferred to VAL-142 or a v1.1 follow-up.

## Limitations

- This is NOT the production Stage 6 inversion. The production module inverts per-class A-scores against the 80-cell baseline age curve in `age_reference_matrix.json` (Recipe §6.3). VAL-140 uses a simpler linear-regression baseline on the same 8-class A-scores.
- The 8-class A-score vector is a low-dimensional summary (8 features) compared to the 115-cell vector the production module uses (~112 valid features).

## Cohort linkage

- Per-sample data: `CPG_VAL_140_per_sample.csv` (n=656 × 4 columns)
- A-scores source: `Biological_Physics/validation_runs/VAL-135_immune_card_v1_full_chain_validation/per_sample_GSE40279.csv`

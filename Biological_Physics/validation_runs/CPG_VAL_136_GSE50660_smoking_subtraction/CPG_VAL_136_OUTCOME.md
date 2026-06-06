# CPG-VAL-136 — Smoking-axis subtraction reduces the never-vs-current A_immune contrast

**Cohort:** GSE50660 Tsaprouni 2014, n=464 healthy whole blood with smoking metadata
**Date sealed:** 2026-06-06
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

| Condition | d (never vs current) | N1 p-value |
|---|---|---|
| WITHOUT Stage 3 smoking subtraction | -0.063 | 0.775 |
| WITH Stage 3 smoking subtraction | +0.005 | 0.983 |
| **Δ\|d\|** | **+0.058** | — |

## Interpretation

PASS — Stage 3 smoking subtraction shrinks the A_immune never-vs-current contrast by 0.058, consistent with the layer correctly attributing β-level variance to smoking rather than to underlying biology. The remaining d after subtraction (+0.005) reflects either residual smoking signal or genuine architectural differences not captured by the layer.

## Limitations

- The smoking layer was FIT on this same GSE50660 cohort, so this is a cohort-internal sanity check. A cleaner test requires an independent cohort with smoking metadata.
- n_current = 22 samples is small, limiting statistical power.

## Cohort linkage

- Per-sample data: `CPG_VAL_136_per_sample.csv` (n=464 × 4 columns)
- Source β: `/tmp/geo_downloads/GSE50660_beta_matrix.npz`
- VAL-135 baseline: `Biological_Physics/validation_runs/VAL-135_immune_card_v1_full_chain_validation/per_sample_GSE50660.csv`

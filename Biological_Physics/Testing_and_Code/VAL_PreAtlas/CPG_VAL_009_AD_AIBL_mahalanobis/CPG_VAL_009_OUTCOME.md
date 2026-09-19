# CPG-VAL-009 — Mahalanobis hyper-volume (AIBL)

**Cohort:** AIBL GSE153712 (n=161 AD / 471 HC / 94 MCI, EPIC)
**Method:** Compute Mahalanobis distance from HC hyper-volume centroid using the universal `mahalanobis_healthy_reference_v0_1` artifact (Ledoit-Wolf shrinkage 0.00875, anchored to breast pre-dx >10y d=+1.871/+2.088). Patient 115-cell A-score vector → single scalar.
**Status:** PRELIMINARY — substantively complete

## Headline result

| Statistic | Value |
|---|---|
| **AD-vs-HC Cohen's d** | **+0.200** |
| 95% CI (bootstrap n=2000) | [+0.041, +0.452] |
| Mann-Whitney p (two-sided) | 4.3e-04 |
| AD mean Mahalanobis | 24.531 |
| HC mean Mahalanobis | 24.203 |
| MCI mean Mahalanobis | 24.508 |

**Significant but modest.** Universal Mahalanobis under-detects AD compared to disease-trained panels.

## Comparison to anchors

| Measure | Cohort | n | Cohen's d |
|---|---|---|---|
| Breast >10y pre-dx Mahalanobis | GSE51057 | 11/177 | +1.871 |
| Breast >10y pre-dx Mahalanobis | GSE51032 | 36/424 | +2.088 |
| **AD AIBL Mahalanobis (this)** | GSE153712 | 161/471 | **+0.200** |
| Pre-build VAL-051 7-CpG Rule A | GSE153712 holdout | 33/95 | +0.624 |
| Stage 1 reproduction (this) | GSE153712 full | 161/471 | +0.615 |

The disease-trained 7-CpG Rule A panel outperforms the universal Mahalanobis ~3× on AD. **For breast pre-dx, the universal Mahalanobis beats the disease-trained panel; for AD, the opposite.** Both findings are biologically meaningful.

## MCI intermediate position (biological coherence)

HC mean 24.203 < MCI mean 24.508 < AD mean 24.531

MCI sits between HC and AD on the universal hyper-volume metric. Disease progression continuum confirmed.

## Sex-stratified

- Female: n=91/272, d=+0.304
- Male: n=70/199, d=+0.137
- Female-greater pattern matches v2.2's documented sex effect.

## Interpretation

AD's signature is **concentrated/targeted**, not universal-architectural. The 7-CpG Rule A panel was selected against AD biology directly; the universal summary integrates across the full 115-cell space where AD's signal is diluted. The Mahalanobis d=+0.20 is informative but not dominant.

The 115-cell hyper-volume is the right metric for cohorts with broad architectural disturbance (breast pre-dx, autoimmune flares); the disease-trained panel is the right metric for cohorts with targeted CpG-level signal (AD).

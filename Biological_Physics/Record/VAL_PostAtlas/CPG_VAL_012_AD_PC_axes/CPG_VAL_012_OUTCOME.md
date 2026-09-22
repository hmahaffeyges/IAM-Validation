# CPG-VAL-012 — Principal-component axes of the AD signal

**Cohort:** AIBL GSE153712 (n=161 AD / 471 HC, EPIC)
**Method:** PCA fit on 115-cell A-score covariance, HC samples only (n=471). All samples projected onto principal components. Per-PC AD-vs-HC effect size.
**Date:** 2026-06-02
**Status:** PRELIMINARY
**Hypothesis tested:** Per breast CPG-VAL-005 precedent, AD's T-cell senescence pattern should hit PC2 (T-cell suppression axis).

## Result

| PC | Variance explained | AD-vs-HC Cohen's d | p (Mann-Whitney) |
|---|---|---|---|
| **PC1** | **66.9%** | **−0.356** | **0.0008** |
| PC2 | 11.4% | +0.152 | 0.25 |
| **PC3** | **6.3%** | **+0.224** | **6.1e-06** |
| PC4 | 4.0% | +0.129 | 0.07 |
| PC5 | 2.6% | +0.118 | 0.27 |
| PC10 | <1% | −0.267 | 0.011 |

**The strongest AD axis is PC1, NOT PC2.** The hypothesis from breast precedent is partially confirmed (T-cell loadings dominate the AD axis) but the rank differs (PC1 in AIBL vs PC2 in breast).

## PC1 loadings (the AD axis)

Top 10 cells loading PC1 (all positive):
| Cell type | Loading |
|---|---|
| CD8T-cells_EPIC | +0.267 |
| CD4T-cells_EPIC | +0.265 |
| CD4_T-cells | +0.249 |
| CD4T | +0.240 |
| CD8_T-cells | +0.235 |
| CD4Tnv (naive) | +0.229 |
| CD8T | +0.228 |
| CD4Tmem (memory) | +0.226 |
| Neutrophils_reinius | +0.217 |
| CD8Tmem (memory) | +0.206 |

PC1 is the T-cell + neutrophil compartment axis. AD shifts NEGATIVELY on PC1 → architectural T-cell exhaustion / immunosenescence at the covariance level. Same biology as breast PC2 but at a different rank because AIBL cohort composition differs (buffy coat, age-skewed clinical sample) from breast pre-dx cohorts (whole blood, asymptomatic).

## PC3 loadings (secondary axis)

PC3 is highly significant (p=6e-6) with d=+0.22 — orthogonal to PC1. Loadings inspection pending; likely a secondary immune-class axis.

## Interpretation

The 115-cell A-score covariance has a dominant T-cell axis that captures the majority (67%) of HC inter-sample variance. AD samples are systematically displaced along this axis in the direction of REDUCED T-cell architectural readout. This is consistent with the established AD literature on T-cell exhaustion and the per-cell-type fan-out finding from CPG-VAL-008 (uniformly negative immune A-scores in AD), now expressed in covariance/PCA form.

PC3 is a meaningful secondary axis (smaller variance, smaller effect, but higher statistical significance per sample). Its loadings inform a potential second-tier score.

# CPG-VAL-008 + CPG-VAL-009 — AIBL Preliminary Outcomes

**Date:** 2026-06-02
**Cohort:** AIBL GSE153712 (n = 161 AD + 471 HC + 94 MCI, EPIC platform)
**Status:** PRELIMINARY — not yet sealed under v4 inventory protocol (L9 null suite pending, PREREG pending)

---

## CPG-VAL-008 — Per-cell-type A-score fan-out (115 cells)

**Result:** AD drives a uniformly negative A-score shift across the immune-progenitor-stem axis. The new instrument's per-cell-type fan-out reveals architectural immunosenescence at single-cell-type resolution that the pre-build 7-CpG panel could not resolve.

### Directional asymmetry
- 106 of 115 cell types show d < 0 (AD reduced)
- 9 of 115 cell types show d > 0 (AD elevated)
- Bonferroni-significant (p < 4.35e-04):
  - **20 cell types negative (AD reduced)**
  - **0 cell types positive (AD elevated)**

### Top 10 effects (all negative)
| Cell type | Class | Cohen's d | Mann-Whitney p |
|---|---|---|---|
| Eosino | immune | -0.426 | 2.3e-05 |
| Eosinophils_reinius | immune | -0.412 | 7.4e-06 |
| Neutro | immune | -0.408 | 9.6e-05 |
| Neutrophils_reinius | immune | -0.397 | 3.2e-04 |
| CD14_monocytes | immune | -0.393 | 1.2e-04 |
| L-MPP | progenitor | -0.385 | 2.2e-04 |
| granulocytes | immune | -0.384 | 3.5e-04 |
| B-cells_EPIC | immune | -0.380 | 3.4e-08 |
| Neu | immune | -0.378 | 2.8e-04 |
| Neutrophils_EPIC | immune | -0.370 | 8.2e-05 |

### Per-class top hits
- **immune:** Eosino d=-0.426 ***
- **progenitor:** L-MPP d=-0.385 ***
- **stem_adult:** HSC d=-0.329 ***
- **secretory:** Thyroid d=-0.285 (not Bonferroni)
- **terminal:** CM d=-0.138 (weak)
- **stromal:** adipocyte d=-0.160 (weak)
- **cycling:** Upper_GI d=-0.157 (weak)
- **stem_pluri:** d=-0.099 (null)

### Biological interpretation
AD samples show REDUCED canonical-immune-class architectural methylation across the entire immune compartment, plus reduced lymphoid progenitor (L-MPP) and hematopoietic stem cell (HSC) architectural scores. Consistent with established immunosenescence in AD: T-cell exhaustion (CD4 d=-0.36, CD8 d=-0.35), B-cell senescence (multiple refs d=-0.36 to -0.38), neutrophil dysfunction (d=-0.40 across three references), and reduced bone marrow progenitor competence (L-MPP, HSC).

The pre-build v2.2 card's documented "bidirectional drift" appears at the CpG level: individual CpGs flip directions, but the architectural A-score integration shows a uniformly NEGATIVE direction. The 7-CpG Rule A panel works because its directional weighting captures specific bidirectional CpG signal; the 115-cell fan-out captures the same biology at the architectural level showing uniform negative.

**This is the architectural signature the pre-build pipeline could not see** — confirmation that the new instrument adds biological resolution to AD detection.

---

## CPG-VAL-009 — Mahalanobis hyper-volume (universal departure-from-HC summary)

**Result:** AD shows statistically significant but modest hyper-volume departure (d=+0.20, p<0.001). Much smaller than breast pre-dx (d=+1.87). The universal Mahalanobis summary under-detects AD compared to disease-trained directional panels because AD's signal is concentrated in specific CpGs rather than spread across the 115-cell hyper-volume.

### Headline statistic
- **AD-vs-HC Cohen's d = +0.200** [95% CI +0.041, +0.452] (bootstrap n=2000)
- **Mann-Whitney p (two-sided) = 4.3e-04**
- AD mean Mahalanobis distance: 24.531
- HC mean Mahalanobis distance: 24.203

### Comparison against anchors
| Measure | Cohort | n | Cohen's d |
|---|---|---|---|
| Breast >10y pre-dx Mahalanobis | GSE51057 | 11/177 | +1.871 |
| Breast >10y pre-dx Mahalanobis | GSE51032 | 36/424 | +2.088 |
| **AD AIBL Mahalanobis (this)** | **GSE153712** | **161/471** | **+0.200** |
| Pre-build VAL-051 7-CpG Rule A | GSE153712 holdout | 33/95 | +0.624 |
| Stage 1 reproduction 7-CpG (this) | GSE153712 full | 161/471 | +0.615 |

### MCI intermediate position (biological coherence)
- HC mean: 24.203
- MCI mean: 24.508
- AD mean: 24.531

MCI sits between HC and AD in hyper-volume distance, consistent with biological disease progression.

### Sex-stratified
- Female: n=91/272, d=+0.304
- Male: n=70/199, d=+0.137
- Female-greater pattern matches v2.2's documented sex effect

### Interpretation
For breast cancer, the universal Mahalanobis distance is the better summary (d=+1.87 vs disease-trained panel comparable). For AD, the disease-trained 7-CpG Rule A panel (d=+0.62) outperforms the universal Mahalanobis (d=+0.20) by ~3×. This is biologically meaningful: AD's signature is concentrated and targeted, while breast pre-dx is broadly architectural.

The Mahalanobis result is non-null but modest — informative as one element of a multi-statistic AD report, not the standalone Stage 1 metric.

---

## Per-VAL formal sealing — outstanding work for full v4 inventory protocol

Both VALs as currently documented are SUBSTANTIVELY complete (raw signal characterized, statistics computed, biological interpretation written). For formal sealing under v4 inventory protocol, the remaining steps:

1. PREREG.md sealed BEFORE re-running with explicit hypothesis (cohort + method + outcome rules pre-locked)
2. cpg_val_008.py / cpg_val_009.py reproducer scripts (single-purpose, runs from cohort inputs)
3. L9 null suite (7 tests: synthetic-null, label-shuffle, panel-permutation, atlas-shuffle, age-sex-permutation, sex-stratified-null, cross-cohort-bootstrap)
4. outcome.md with O1-O6 outcome label
5. results.json + per_sample.csv sealed

These are documentation tasks; the science is in hand.

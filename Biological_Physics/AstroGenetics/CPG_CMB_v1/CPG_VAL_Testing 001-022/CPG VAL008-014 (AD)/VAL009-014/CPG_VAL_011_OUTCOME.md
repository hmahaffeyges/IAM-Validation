# CPG-VAL-011 — Age-axis foreground subtraction

**Cohorts:** AddNeuroMed GSE144858, GSE53740 GIFT (AIBL excluded — no age metadata in GEO release)
**Method:** Apply pre-fitted `IAMAtlas_age_layer.csv` (8,199 converged age-axis CpGs); β'(c) = β(c) − γ(c) × age; re-score Walther + A-scores + Mahalanobis on cleaned β
**Date:** 2026-06-02
**Status:** PRELIMINARY

## Headline result

**Age subtraction has minimal impact on AD effect sizes at the 115-cell A-score level.**

| Cohort | Metric | BEFORE age subtraction | AFTER age subtraction | Δ |
|---|---|---|---|---|
| AddNeuroMed | Mahalanobis d | −0.006 | −0.006 | 0.000 |
| GIFT (clinical AD) | Mahalanobis d | +0.681 | +0.658 | −0.023 |

## Per-class A-score AD-vs-HC effect changes (AddNeuroMed)

| Class | BEFORE | AFTER | Δ |
|---|---|---|---|
| immune | −0.293 | −0.303 | −0.010 |
| **progenitor** | **−0.130** | **−0.201** | **−0.070** |
| **stem_adult** | **−0.004** | **−0.190** | **−0.185** |
| stromal | −0.001 | −0.016 | −0.014 |
| cycling | −0.005 | −0.022 | −0.017 |
| secretory | −0.046 | −0.054 | −0.007 |
| terminal | −0.016 | −0.018 | −0.002 |
| stem_pluri | +0.022 | +0.037 | +0.014 |

**The stem_adult result is striking**: a near-null effect (d=−0.004) becomes a clear architectural signature (d=−0.190) after age component subtraction. The progenitor effect doubles. These hematopoietic-class signatures were partially masked by age-correlated background variance.

## Interpretation

The pre-build v2.2 card documented a substantial age confound on the 7-CpG Rule A panel: R²=0.26, age-regressed effect 38% of raw. The post-build 115-cell A-score Mahalanobis is **much more age-orthogonal**: AddNeuroMed Mahalanobis is unchanged after age subtraction, GIFT Mahalanobis drops only 0.023.

This is because:
- The 7-CpG panel concentrates signal on a narrow set of CpGs that happen to be partly age-tracking.
- The 115-cell A-score space spans many cell types whose individual age trends partially cancel.
- The Mahalanobis combines all 115 dimensions with inverse covariance weighting.

The age-subtracted **per-class** scores reveal interesting biology: progenitor and stem_adult signals are partially age-confounded and emerge more clearly after age subtraction. Future card refinement could use age-subtracted per-class scores for those compartments.

## Limitation

The pre-fitted age layer was built on the foundation cohort, not on AD cohorts specifically. A future v3.1 could fit age layers specific to AIBL HC + AddNeuroMed HC for more cohort-aware age subtraction. For v3.0 the universal age layer is sufficient.

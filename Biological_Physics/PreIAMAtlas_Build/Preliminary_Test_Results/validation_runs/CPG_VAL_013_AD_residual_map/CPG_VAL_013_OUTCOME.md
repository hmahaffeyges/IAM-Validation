# CPG-VAL-013 — Per-CpG residual map → CPG_ad_panel_v1 candidate

**Cohorts:** AIBL GSE153712 (training), AddNeuroMed GSE144858 (cross-cohort concordance)
**Method:** For each Walther class marker CpG c, compute predicted β as Σ_class f_class × class_ref[c][class]. Residual = observed β − predicted β. Per-CpG AD-vs-HC effect size on the residual.
**Date:** 2026-06-02
**Status:** PRELIMINARY — candidate panel staged, not yet validated

## Headline result

**6,724 class marker CpGs scored on AIBL; 6,350 on AddNeuroMed; 6,018 in both.**

| Metric | AIBL |
|---|---|
| Median per-CpG |d| | 0.096 |
| 95th percentile |d| | 0.268 |
| 99th percentile |d| | 0.333 |
| Maximum |d| | 0.493 |
| CpGs with d < −0.3 | 135 |
| CpGs with d > +0.3 | 28 |
| Negative-to-positive ratio | 4.8 : 1 |

**Strong directional asymmetry:** 4.8× more CpGs with AD-residual REDUCED than ELEVATED. Same biological direction as the 115-cell A-score fan-out — AD samples are systematically below class-prediction expectation.

## Cross-cohort concordance (AIBL vs AddNeuroMed)

- 6,018 CpGs in both cohorts
- Spearman ρ = 0.231 (p = 10⁻⁷⁴)
- 271 CpGs strong (|d|>0.2) in both
- **241 of those 271 concordant in sign (88.9%)**

The cross-cohort concordance rate of 88.9% on strong CpGs supports the biological reality of the per-CpG residual signature.

## Comparison to breast CPG-VAL-003

| Metric | Breast (GSE51057 vs GSE51032) | AD (AIBL vs AddNeuroMed) |
|---|---|---|
| Concordant strong CpGs | 1,392 | 241 |
| Spearman ρ | (similar magnitude) | 0.231 |

AD has roughly 17% as many concordant CpGs as breast. The signal is real but more diffuse, consistent with AD's documented bidirectional drift and the fact that AIBL is at-diagnosis (not 10-year pre-dx like the breast cohorts).

## CPG_ad_panel_v1 candidate panel

A 200-CpG candidate panel was emitted from the AIBL residual map (top by |d|):
- 40 positive-direction CpGs
- 160 negative-direction CpGs
- Min |d|: 0.314, Max |d|: 0.493
- Atlas anchor: IAMAtlas REBUILD class_ref
- **Status: CANDIDATE — requires cross-cohort holdout validation before operational use**
- File: `CPG_ad_panel_v1_candidate.json`

The panel is bidirectional (consistent with v2.2's Directional-Score Principle), atlas-anchored (no reliance on external panel sources), and operationally compatible with the 7-CpG Rule A panel architecture but at 30× resolution.

## Next steps

1. Apply CPG_ad_panel_v1 candidate to AddNeuroMed as a HOLDOUT (no refinement on AddNeuroMed data)
2. If AddNeuroMed holdout reproduces an effect size in the d=+0.5 to +0.8 range, the panel is a viable competitor to the 7-CpG Rule A panel
3. If AddNeuroMed holdout drops to d<+0.3, the panel overfit AIBL and needs refinement

This work belongs in a future CPG-VAL-013b or directly in a card v3.1 operational scoring bump.

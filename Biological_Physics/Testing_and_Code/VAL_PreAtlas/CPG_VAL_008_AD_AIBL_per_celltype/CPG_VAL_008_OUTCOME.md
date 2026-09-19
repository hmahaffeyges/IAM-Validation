# CPG-VAL-008 — Per-cell-type A-score fan-out (AIBL)

**Cohort:** AIBL GSE153712 (n=161 AD / 471 HC, EPIC platform)
**Method:** Score all 115 cell-type A-scores per patient (via `iamatlas_a_scoring.score_per_celltype` using v0_2 markers + H_min frozen by class). Compute Cohen's d for each cell type AD-vs-HC.
**Status:** PRELIMINARY — substantively complete

## Headline result

AD drives a **uniformly negative A-score shift** across the immune-progenitor-stem axis.

- 106 of 115 cell types show d < 0 (AD reduced)
- 9 of 115 cell types show d > 0 (AD elevated)
- Bonferroni-significant (p < 4.35e-04, 115 comparisons):
  - **20 cell types negative**
  - **0 cell types positive**

## Top 10 effects (all negative)

| Cell type | Class | Cohen's d | Mann-Whitney p |
|---|---|---|---|
| Eosino | immune | −0.426 | 2.3e-05 *** |
| Eosinophils_reinius | immune | −0.412 | 7.4e-06 *** |
| Neutro | immune | −0.408 | 9.6e-05 *** |
| Neutrophils_reinius | immune | −0.397 | 3.2e-04 *** |
| CD14_monocytes | immune | −0.393 | 1.2e-04 *** |
| L-MPP | progenitor | −0.385 | 2.2e-04 *** |
| granulocytes | immune | −0.384 | 3.5e-04 *** |
| B-cells_EPIC | immune | −0.380 | 3.4e-08 *** |
| Neu | immune | −0.378 | 2.8e-04 *** |
| Neutrophils_EPIC | immune | −0.370 | 8.2e-05 *** |

## Per-class top hits

| Class | Top cell type | d |
|---|---|---|
| immune | Eosino | −0.426 *** |
| progenitor | L-MPP | −0.385 *** |
| stem_adult | HSC | −0.329 *** |
| secretory | Thyroid | −0.285 |
| terminal | CM | −0.138 |
| stromal | adipocyte | −0.160 |
| cycling | Upper_GI | −0.157 |
| stem_pluri | (stem_pluri) | −0.099 |

## Interpretation

AD shows REDUCED canonical-immune-class architectural readout across the entire immune compartment plus lymphoid progenitor (L-MPP) and hematopoietic stem cell (HSC) A-score reduction. Established AD biology: T-cell exhaustion (CD4 d=−0.36, CD8 d=−0.35), B-cell senescence (multiple refs d=−0.36 to −0.38), neutrophil dysfunction (d=−0.40 across three references), bone-marrow progenitor decline (L-MPP, HSC).

The pre-build v2.2 card's "bidirectional drift" framing is resolved at the **architectural** level: individual CpGs flip, but the cell-type A-score integration shows a uniformly NEGATIVE direction. The 7-CpG Rule A panel captures the bidirectional CpG-level signal via directional weighting; the 115-cell fan-out captures the same biology at architectural resolution showing uniform negative.

This is the architectural signature the pre-build pipeline could not see.

## Full table

`per_celltype_AD_vs_HC.csv` — all 115 cells × {d, p_mwu, AD_mean, HC_mean, n_ad, n_hc}.

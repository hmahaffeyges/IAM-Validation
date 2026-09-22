# VAL-110 — Outcome Record

**Outcome:** `O2_AORTIC_ANY_TILE_DIFFERENTIATING` (per sealed prereg)

**Sealed prereg SHA-256:** `1041738ccc8bcdd45a4754d599a28ad80fde3a7b37b6c18b4d528f4fe0271bc8`  
**Cohort:** GSE84274 ascending aorta tissue, n=24 (normal 6 + dissection 12 + BAV+dilation 6)  
**Substrate:** GenomeStudio V2011.1 raw output, HM450K

## Headline numbers

**CHK-3.1A self-calibration (within-cohort substrate anchor):**
- 23 / 24 samples pass (95.8%); fail rate 4.2%
- Cohort f_extreme mean 33.95% ± 2.21% (close to GSE69138's 31.81% — both GenomeStudio substrates produce similar bimodality on different tissue types)

**QC-pass groups:** normal 6, dissection 11, BAV_dilation 6

**Stage 1 immune A-score (Salas IDOL proxy):**
- Normal vs dissection: d = **+0.56**
- Normal vs BAV+dilation: d = **+1.08**
- Dissection vs BAV: d = +0.05 (no etiology differentiation)

**Stage 2 cardio-relevant tile A-scores:**

| Tile | Normal vs Dissection | Normal vs BAV | Dissection vs BAV |
|---|---|---|---|
| Vascular_endothelial_cells | −0.04 | −0.14 | −0.15 |
| Left_atrium | **−0.60** | **−0.81** | −0.42 |
| Adipocytes | **−0.71** | **−0.88** | −0.37 |

Top |d| Stage 2 normal vs BAV: Hepatocytes −3.64, Lung_cells −3.25, Colon_epithelial_cells −2.74 (massive non-vascular tile divergence — peri-aortic adipose contamination affects many tile signatures).

**Stage 3:**
- UniLIFE normal vs dissection: d = +0.55
- UniLIFE normal vs BAV: d = +0.55
- Dissection vs BAV: d = +0.02 (etiology-equivalent)

## Outcome interpretation

The framework discriminates aortic pathology from normal aorta on multiple stages. Stage 1 immune A-score shows progressively stronger discrimination from normal: dissection d = 0.56, BAV d = 1.08. Stage 2 tile-level differentiation is dominated by non-cardiovascular tiles (Hepatocytes, Lung_cells, Pancreatic), which is not directly cardio-relevant — likely reflects peri-aortic adipose tissue contamination in the bulk aortic sample.

The Vascular_endothelial_cells tile specifically does NOT differentiate aortic pathology (|d| ≤ 0.15 on all contrasts). This is an important finding for cardio-epic: ascending aortic bulk tissue methylation is dominated by smooth muscle cells and fibroblasts, not endothelial cells, so the Vascular_endothelial tile reference doesn't capture aortic pathology signal. The Left_atrium and Adipocytes tiles produce moderate discrimination (|d| = 0.6-0.9 normal vs BAV).

The two pathological etiologies (dissection vs BAV+dilation) are framework-equivalent — no contrast reaches |d| ≥ 0.5. This is consistent with both pathologies sharing chronic structural stress / medial dysregulation as their final common methylation signature.

## Biological interpretation

Aortic dissection and BAV+dilation share a common methylation signature in ascending aorta tissue, framework-distinguishable from normal aorta. The strongest signal is in Stage 1 immune (suggesting infiltrating inflammatory cells in pathological aorta) and in non-vascular Stage 2 tiles (likely peri-aortic adipose contamination differing between normal and pathological samples).

The VAL-110 result complements VAL-109: PAH on cultured pure PECs shows direct vascular-tile discrimination (d=0.79); aortic bulk tissue does not (d=0.04). The framework reads "what's actually in the sample" — pure endothelial cell substrate gives endothelial signal, mixed aortic tissue gives mixed signal dominated by other cell types.

## What propagates to cardio-epic v0.1

1. **Stage 1 immune A-score discriminates aortic pathology from normal** with strong d (>0.5) and is framework-operational for cardio-epic.
2. **Stage 2 Left_atrium and Adipocytes tiles** show moderate discrimination — useful for combined-tile cardio scoring.
3. **Stage 2 Vascular_endothelial_cells tile does NOT discriminate aortic pathology** — important caveat for cardio-epic. Sample-substrate matters: pure cell type vs mixed bulk tissue.
4. **Aortic dissection vs BAV+dilation are framework-equivalent.** Cardio-epic should not claim etiology stratification for aortic pathology at v0.1.
5. **Substrate-specific CHK-3.1A baseline for GenomeStudio aortic tissue:** ~34% f_extreme (close to GSE69138 whole blood ~32%, suggesting GenomeStudio substrate has stable bimodality across tissue types).

## What does NOT propagate

- No claim that bulk aortic methylation is the right substrate for vascular-cell-of-origin scoring — the data are clear that bulk aortic tissue is dominated by non-endothelial cell types.
- No etiology stratification.
- No Xu-538 production-panel results (Stage 1 proxy).

## Outcome status

`O2_AORTIC_ANY_TILE_DIFFERENTIATING` — sealed.

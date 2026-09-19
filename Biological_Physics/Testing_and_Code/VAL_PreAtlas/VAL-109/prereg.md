# VAL-109 — Cardio-epic Stage 1+2+3 on GSE84395 PAH pulmonary endothelial cells

**Status:** PRE-REGISTERED, sealed before framework scoring  
**Card:** cardio-epic v0.1  
**Date:** 2026-04-28  
**RNG seed:** 20260428

## Cohort

GSE84395, n=39 cultured pulmonary endothelial cells (PECs):
- 18 control PECs
- 11 idiopathic PAH (iPAH)
- 10 heritable PAH (hPAH)

Substrate: HM450K + GPL16304, β values are minfi `preprocessFunnorm` output (functional normalization). This is a third distinct substrate category (different from TCGA sesame Level 3 and from GSE69138 GenomeStudio AVG_Beta). Treated under within-cohort self-substrate-anchor approach per the same approach as VAL-108.

## Hypothesis

Pulmonary arterial hypertension involves vascular endothelial cell dysfunction. The framework should detect:
- Reduced Stage 2 Loyfer Vascular_endothelial_cells tile A-score in PAH samples (since PEC methylation should be characteristic for the EC class) — OR converse: increased deviation from healthy EC reference reflecting PAH-specific dysregulation
- Possible Stage 1 immune signal if PAH carries inflammatory dysregulation

This is a within-cohort case-control test of cardio-epic on the actual cell-of-origin substrate (cultured PECs) — direct test of Stage 2 cell-of-origin tiles on their corresponding cell type.

## Methodology (frozen)

CHK-3.1A self-calibration for fn-normalized substrate (computed at execution from cohort's own distribution; pass criterion: f_extreme within ±2*SD of cohort mean, n_valid >= 400000).

Stage 1 (Salas IDOL 350-CpG immune entropy as Xu-538 proxy)  
Stage 2 (Loyfer 25-tile run-everything per-tile A-score, all 25 tiles)  
Stage 3 (UniLIFE 19-cell + Salas 6-cell pooled entropy)

## Outcomes (frozen)

**O1_PAH_FRAMEWORK_DIFFERENTIATING**: Any pair (control vs iPAH, control vs hPAH, iPAH vs hPAH) shows |d| ≥ 0.5 with CI excluding zero on Stage 1, Stage 2, or Stage 3.

**O2_PAH_VASCULAR_TILE_DIFFERENTIATING**: Specifically on Vascular_endothelial_cells or Left_atrium tiles (the cardio-relevant Stage 2 tiles), |d| ≥ 0.5 across at least one disease vs control contrast — direct framework discrimination on the assayed cell type.

**O3_PAH_FRAMEWORK_UNDIFFERENTIATING**: No |d| ≥ 0.5 across any contrast at any stage — PAH is not framework-distinguishable from healthy at the assayed substrates.

**O4_HPAH_VS_IPAH_DIFFERENTIATING**: hPAH vs iPAH specifically shows |d| ≥ 0.5 — heritable vs idiopathic etiologies are framework-distinguishable.

**O5_DATA_INTEGRITY_FLAG**: ≥10% CHK-3.1A self-cal failures.

## Pre-registration seal

Sealed via SHA-256 of this content prior to framework scoring on GSE84395.

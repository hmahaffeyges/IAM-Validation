# VAL-110 — Cardio-epic Stage 1+2+3 on GSE84274 ascending aorta dissection / BAV / normal

**Status:** PRE-REGISTERED, sealed before framework scoring  
**Card:** cardio-epic v0.1  
**Date:** 2026-04-28  
**RNG seed:** 20260428

## Cohort

GSE84274, n=24 ascending aorta tissue samples:
- 6 normal aorta
- 12 aortic dissection
- 6 bicuspid aortic valve with dilation (BAV)

Substrate: HM450K, GenomeStudio V2011.1 raw output (similar family to GSE69138 but on aortic tissue not whole blood). Within-cohort case-control + 3-way design. Self-substrate-anchor approach.

## Hypothesis

Ascending aortic tissue contains vascular smooth muscle cells, endothelial cells, and adventitial fibroblasts. Aortic dissection involves medial layer disintegration; BAV+dilation involves chronic structural stress. The framework should detect:
- Stage 2 Vascular_endothelial_cells tile differentiation between normal aorta and pathological aorta
- Adipocytes / Left_atrium tile signals (peri-aortic adipose tissue and adjacent cardiac tissue contamination)

## Methodology (frozen)

CHK-3.1A self-calibration (cohort distribution ±2*SD envelope, n_valid ≥ 400000)  
Stage 1 (Salas IDOL 350-CpG entropy as Xu-538 proxy)  
Stage 2 (Loyfer 25-tile run-everything)  
Stage 3 (UniLIFE 19-cell + Salas 6-cell)

## Outcomes (frozen)

**O1_AORTIC_VASCULAR_DIFFERENTIATING**: Vascular_endothelial_cells tile shows |d| ≥ 0.5 across normal vs (dissection or BAV) — direct framework discrimination on vascular tissue.

**O2_AORTIC_ANY_TILE_DIFFERENTIATING**: Stage 2 any tile shows |d| ≥ 0.5 across at least one disease vs normal contrast.

**O3_AORTIC_FRAMEWORK_UNDIFFERENTIATING**: No |d| ≥ 0.5 across any contrast at any stage.

**O4_DISSECTION_VS_BAV_DIFFERENTIATING**: Dissection vs BAV shows |d| ≥ 0.5 — the two pathological etiologies are framework-distinguishable.

**O5_DATA_INTEGRITY_FLAG**: ≥10% CHK-3.1A failures.

## Pre-registration seal

Sealed via SHA-256 prior to framework scoring on GSE84274.

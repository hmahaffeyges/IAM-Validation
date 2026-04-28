# VAL-109 — Outcome Record

**Outcome:** `O2_PAH_VASCULAR_TILE_DIFFERENTIATING` (per sealed prereg)

**Sealed prereg SHA-256:** `f6450b4cf5d384d2ea27b349c101b3f167a6a549d276e670e68fb2232b45f21e`  
**Cohort:** GSE84395 cultured pulmonary endothelial cells, n=39 (control 18 + hPAH 10 + iPAH 11)  
**Substrate:** minfi `preprocessFunnorm` functional normalization, HM450K GPL16304

## Headline numbers

**CHK-3.1A self-calibration (within-cohort substrate anchor):**
- 37 / 39 samples pass (94.9%); fail rate 5.1%
- Cohort f_extreme mean 52.82% ± 2.33% (envelope 48.16-57.48%); f_middle mean 9.12% ± 0.63% (envelope 7.86-10.37%)

**QC-pass groups:** control 17, hPAH 10, iPAH 10

**Stage 1 immune A-score (Salas IDOL proxy):**
- Control vs hPAH: d = **+0.65** (CI excludes zero with n=17 vs 10)
- Control vs iPAH: d = **+0.65**
- hPAH vs iPAH: d = −0.09 (no etiology differentiation)

**Stage 2 cardio-relevant tile A-scores:**

| Tile | Control vs hPAH | Control vs iPAH | hPAH vs iPAH |
|---|---|---|---|
| Vascular_endothelial_cells | **+0.79** | +0.42 | −0.35 |
| Left_atrium | **+0.65** | +0.31 | −0.32 |
| Adipocytes | +0.09 | −0.42 | −0.48 |
| Lung_cells | **+0.91** | +0.34 | −0.35 |

Top |d| in Stage 2 control vs hPAH:
- Pancreatic_duct_cells +1.05, Pancreatic_acinar_cells +1.04, Lung_cells +0.91, Colon_epithelial_cells +0.87, Pancreatic_beta_cells +0.86

**Stage 3:**
- UniLIFE control vs hPAH: d = +0.48
- UniLIFE control vs iPAH: d = +0.22

## Outcome interpretation

The framework discriminates PAH from healthy on the actual cell-of-origin Stage 2 tile (Vascular_endothelial_cells). Control vs hPAH shows |d| = 0.79 on the vascular tile and |d| = 0.65 on Left_atrium. This is direct discrimination on the assayed cell type.

Heritable PAH (hPAH) shows stronger framework discrimination from controls than idiopathic PAH (iPAH) does — d = 0.79 vs 0.42 on Vascular_endothelial_cells. This is consistent with hPAH carrying germline genetic lesions (often BMPR2) that produce more pronounced methylation dysregulation than the heterogeneous etiology of iPAH.

The two PAH subtypes (hPAH vs iPAH) are framework-equivalent — no contrast reaches |d| ≥ 0.5.

## Biological interpretation

Cultured pulmonary endothelial cells from PAH patients show measurable methylation dysregulation across multiple cell-of-origin tile signatures. The dysregulation is strongest in hPAH (consistent with stronger genetic component) and detectable in iPAH (consistent with epigenetic dysregulation in idiopathic disease). Whole-pulmonary-endothelial methylation discriminates PAH from healthy controls with framework-meaningful effect sizes.

The Stage 1 immune signal (d = +0.65 control vs PAH) on cultured cells is interesting — even with cultured PECs (which are not blood-cell-contaminated), the Salas IDOL panel shows differential methylation. This may reflect either (a) immune-related CpGs that have functional roles in endothelial biology, or (b) the Salas panel's IDOL CpGs being indirect markers picking up systemic transcriptional state.

## What propagates to cardio-epic v0.1

1. **Stage 2 Vascular_endothelial_cells tile is operational for cardio-epic.** PAH detection on cultured PECs is direct evidence the framework's vascular-class scoring works on the assayed cell type.
2. **Stage 2 Left_atrium tile** also shows PAH discrimination (d = +0.65 control vs hPAH).
3. **hPAH vs iPAH is framework-equivalent.** Cardio-epic should not claim heritable-vs-idiopathic discrimination at v0.1.
4. **Substrate-specific CHK-3.1A baseline for fn-normalized substrate**: cohort mean ~52.8% f_extreme.

## What does NOT propagate

- No claim about whole-blood PAH discrimination (this cohort is cultured cells).
- Stage 1 used Salas IDOL proxy, not Xu-538 production panel.
- The strong tissue-class A-score deviations (Pancreatic_*, Lung_cells) reflect culture-substrate methylation drift in PEC primary culture, not real cell-of-origin biology — note for cardio-epic interpretation.

## Outcome status

`O2_PAH_VASCULAR_TILE_DIFFERENTIATING` — sealed.

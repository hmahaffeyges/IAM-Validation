# Per-Card Disease Residual Maps

**Layer 3 base maps** — per-CpG residual signatures per card cohort. Where the disease signal lives at sub-cellular resolution after cellular composition has been factored out.

## What a residual map is

For each card, every patient runs through Stages 1+2 of the EDEAR pipeline:
1. **Stage 1 (Walther IAM Deconvolver)** produces per-class fractions for that patient.
2. **Stage 2 reconstruction**: at each marker CpG, predicted β = Σ (class_fraction × class_reference_β).
3. **Per-CpG residual** = observed β − reconstructed β. This isolates the disease-specific signal from cellular-composition variation.
4. **Per-CpG case-vs-HC Cohen's d** on the residuals identifies the loci where the card's disease signature lives — orthogonal to whatever cellular composition shifts have already been captured at the class-A-score level.

## Files in this folder

| File | Card | Cohort | Tool | N (case/HC) | Concordant CpGs |
|---|---|---|---|---|---|
| `breast_epic_residual_map_v0_1.csv` | breast-epic | GSE51057 + GSE51032 >10yr breast pre-dx | production Walther IAM Deconvolver | 47 / 601 | 1,392 (|d|>0.3 both cohorts, same sign) |

Each CSV has columns: `cpg`, `d_GSE51057`, `d_GSE51032`, `concordant_strong`, `mean_abs_d`, optionally `in_xu538`.

## Breast-epic v0.1 — key finding (2026-05-29)

- **1,173 hypomethylated CpGs (case β < reconstructed β)** vs only 219 hypermethylated — 5.4-to-1 ratio.
- **Dominant signature is loss of methylation** below cellular-composition expectation at 10+ years pre-diagnosis. Classic field-effect cancerization signature, quantified.
- **1,389 NEW candidate CpGs** not in the Xu-538 disease-trained panel. Candidates for an expanded breast-epic panel that complements Xu-538.
- Top concordant loci: cg20124336 (d=−2.17/−1.89), cg16188349 (d=−1.67/−1.67), cg27467249 (d=−2.17/−1.17). All hypomethylated, all replicating across cohorts.

## How the card consumes these maps

For each customer through the breast-epic card:
1. Production deconvolver → class fractions
2. Reconstructed β at the 7,114 deconvolver class markers
3. Per-CpG observed − reconstructed = patient residual vector
4. Compare patient residual at the concordant 1,392 CpGs to the cohort residuals here
5. Per-CpG z-score → card-specific layer 3 evidence


---

## Post-build CPG-VAL sealed anchors (added 2026-06-02)

The three maps in this folder are the operational artifacts of the post-IAMAtlas-build foundation VALs (Family A, CPG-VAL-001 through CPG-VAL-007, run 2026-05-29 against GSE51057 + GSE51032 pre-dx >10y). All three maps were generated with the production runtime stack: IAMAtlas REBUILD + Walther IAM Deconvolver + iamatlas_celltype_markers_v0_2.

| Map | Source VAL | Headline | Null suite |
|---|---|---|---|
| `breast_epic_residual_map_chr_annotated.csv` | **CPG-VAL-003** | 1,392 concordant CpGs (`concordant_strong=True`); 1,173 hypomethylated vs 219 hypermethylated (5.4:1 field-effect hypomethylation signature); top: cg20124336 d=−2.17/−1.89, cg16188349 d=−1.67/−1.67. These 1,392 CpGs are the SEED for CPG_breast_panel_v1. | 7/7 PASS Sealed |
| `breast_epic_bimodality_map.csv` | **CPG-VAL-004** (RESTATED) | 1,492 CpGs show case-vs-HC bimodality asymmetry; 1,096 GAIN bimodality (73%), 396 LOSE bimodality (27%). Original framing focused on the 396 losses; restated framing notes the gain direction dominates 2.77:1. | RESTATE per N_bimo_001 |
| `breast_epic_pca_projections.csv` | **CPG-VAL-005** | PC1 (70.7% var, 8-class): broad cellular drift d=+1.07/+0.57. **PC2 (115-cell): T-cell SUPPRESSION axis d=−0.67/−0.58 replicating across cohorts** — immunosurveillance failure signature 10+ years pre-diagnosis. | 7/7 PASS Sealed |

**Cohort source for all three maps:** EPIC-Italy GSE51057 + GSE51032 pre-dx >10y filter (47 cases + 601 HC pooled). Foundation cohort per-cell-type A-scores at `Biological_Physics/validation_runs/foundation_cohort/`. Cohort source paper: Severi G et al. *Carcinogenesis* 2014;35(10):2349-2357 (DOI: 10.1093/carcin/bgu138).

**Disease matrix companion:** The breast_cancer / long_pre_dx row of `DISEASE_MATRIX/disease_cell_signature_matrix_v1_5.csv` carries CPG-VAL-001/002/003/005/007 citation aliases in its `evidence_anchors` field alongside the original TODO 1.1/1.2/1.3/1.5 + pre-build VAL-046/047/049/093/094/095/096 references.

**Card consumption:** `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` references all three of these maps via its `cpg_native_post_build_addendum.operational_data_files_in_this_card_folder.residual_maps` block.

**Null-suite artifacts:** `Biological_Physics/chain_of_custody/L9_null_suite/test_runs/CPG_VAL_00{3,4,5}_*` (null_results.json + per_sample.csv where applicable).

**Full narrative:** `post_build_evidence/v2_CPG_IAMAtlas_Evidence_Report.html`.

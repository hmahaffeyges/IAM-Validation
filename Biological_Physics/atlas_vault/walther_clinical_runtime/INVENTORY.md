# walther_clinical_runtime — INVENTORY

**Folder:** `Biological_Physics/atlas_vault/walther_clinical_runtime/`
**Generated:** 2026-06-02 by Walther from the folder Heath uploaded as `All pipeline Files.zip`
**Supersedes:** the older `SYSTEM_INVENTORY.md` that documented the pre-consolidation `pipeline_runtime_matrices/` + `chain_of_custody/L4_component_separation/` + `components/` layout. All those folders' contents are now consolidated here under one runtime root.

Status tags:
- **CANONICAL** = the live production version
- **CURRENT** = preferred over alternates, but a non-canonical variant exists alongside
- **SUPERSEDED** = older version kept for audit; do not load at runtime
- **ARCHIVAL** = read-only validation output, not loaded at runtime

---

## Top-level files

| File | Size | Status | Purpose |
|---|---|---|---|
| `README.md` | 12 KB | CANONICAL | Folder map + SOP-stage cross-reference |
| `INVENTORY.md` | (this file) | CANONICAL | Flat per-file inventory with sizes + SHAs |
| `walther_clinical_BUILD_SPEC_v1_1.md` | 64 KB | CANONICAL | Build spec for the future orchestrator |
| `CPG_Chain_of_Custody_SOP_v1_2.md` (+ 6 part files) | 340 KB + 348 KB | CANONICAL | The operational SOP (Stages 0–10, L1–L9 chain) |

---

## A_Scoring_Module/ — Stage 4 entropy scoring

| File | Size | Status | Purpose |
|---|---|---|---|
| `iamatlas_a_scoring.py` | 8 KB | CANONICAL | `score_per_class()`, `score_per_celltype()`, `load_artifact()`. The Recipe §2.3 A = H(β_mean)/H_min formula. |

---

## Age_Reference_Matrix_80_cells/ — Stage 6 baseline

| File | Size | Status | Purpose |
|---|---|---|---|
| `age_reference_matrix.json` | 28 KB | CANONICAL | 80-cell baseline (8 classes × 10 decadal bins, ages 4–95). SHA: `54232a5d1e269acb...0c5ef56a` |
| `age_reference_matrix.csv` | 8 KB | CANONICAL | Same content, flat CSV for human inspection |
| `age_reference_matrix.py` | 12 KB | CANONICAL | Python helpers `AGE_REFERENCE` dict + `age_ref_A()`, `age_ref_sd()`, `age_ref_percentiles()` |

---

## Cancer_prior/ — Stage 9 risk context

| File | Size | Status | Purpose |
|---|---|---|---|
| `cancer_prior.json` | 4 KB | CANONICAL | US lifetime cancer incidence per architectural class (cycling 0.055, secretory 0.140, immune 0.020, ...). SHA: `8ae529be32fb7d20...ece827805` |

---

## Celltype_Marker/ — Stage 4 marker artifact

| File | Size | Status | Purpose |
|---|---|---|---|
| `iamatlas_celltype_markers_v0_2.json` | 232 KB | **CANONICAL** | 115 cell types × top-100 one-vs-rest markers + `celltype_to_class` + `H_min_by_class`. SHA: `46ea5be1db377f2b...d19c47bd` |
| `iamatlas_celltype_markers_v0_2.sha256` | 4 KB | CANONICAL | Sidecar SHA-256 |
| `iamatlas_celltype_markers_v0_1_SUPERSEDED.md` | 4 KB | NOTE | Explains why v0_1 is superseded |
| `OLD/iamatlas_celltype_markers_v0_1.json` | 232 KB | SUPERSEDED | Prior version, archived |

---

## Cfdna_weight_nonderived_placeholder/ — Stage 7 cfDNA branch

| File | Size | Status | Purpose |
|---|---|---|---|
| `cfdna_weight.json` | 4 KB | CANONICAL | Healthy-blood cfDNA tissue-of-origin weights (immune 0.70, cycling 0.12, secretory 0.08, ...). Snyder 2016 + Moss 2018. Conditional consumption — Stage 7 only loads when `substrate == "plasma_cfdna"`. SHA: `ca632d039a2590fd...298d7cd61` |

---

## CPG_Null_Runner/ — L9 chain-of-custody scaffolding

| File | Size | Status | Purpose |
|---|---|---|---|
| `cpg_null_runner.py` | 36 KB | CANONICAL | Unified 8-null framework (N1–N8). Every VAL passes through this before sealing. NOT invoked per-patient at runtime. |

---

## DISEASE_MAPS_CARDS/ — Stage 8 per-card matching (Path A)

### Breast_EPIC/ — V1 first card (only currently re-VAL'd against the new atlas)

**`breast_epic_card_json/`**

| File | Size | Status | Purpose |
|---|---|---|---|
| `breast-epic_card_v3_0.json` | 66 KB | **CANONICAL** | Card definition v3.0 (2026-06-02). Strict-additive bump of v2.3: same Stage 1/2/3 operational logic + same H_min anchors + same tier thresholds, plus 5 appended post-build CPG-VAL entries (001/002/003/005/007). |
| `breast-epic_card_v2_3.json` | 56 KB | SUPERSEDED → `OLD/` | Pre-build vintage (2026-04-26). Operational logic preserved verbatim in v3.0; this file archived for audit. SHA: `f4ea6d2b301dd8ed...8c0b43cc913` |
| `breast-epic_README.md` | 36 KB | CANONICAL | Card documentation (v2.3 operational logic — unchanged in v3.0 bump) |
| `breast-epic_v3_0_release_notes.md` | 5 KB | CANONICAL | v3.0 bump release notes — what changed, what didn't, where v2.3 lives |

**`breast_epic_residual_maps/`** — Layer-3 base maps for this card

| File | Size | Status | Purpose |
|---|---|---|---|
| `breast_epic_residual_map_chr_annotated.csv` | 616 KB | CANONICAL | 1,392 concordant CpGs (`|d|>0.3` both cohorts, same sign) with chr/MAPINFO. Primary matched-filter layer. SHA: `ff186abe737faa52...51947666b7de3` |
| `breast_epic_bimodality_map.csv` | 1.6 MB | CANONICAL | Bimodality loss detection layer. SHA: `491385849b8fbc63...8cc6f4d1f538bb8` |
| `breast_epic_pca_projections.csv` | 116 KB | CANONICAL | PCA projections layer (cross-cohort signature). SHA: `2ecd5d85a6ef1102...15b8bf0117fec0` |
| `README_Breast_residual_maps.md` | 4 KB | CANONICAL | Layer-3 base-maps documentation |
| `OLD/breast_epic_residual_map_v0_1.csv` | 532 KB | SUPERSEDED | Prior single-map version, archived |

### OLD/ — pre-consolidation archives

| File | Size | Status | Purpose |
|---|---|---|---|
| `01_README_card_residual_maps.md` | 4 KB | SUPERSEDED | Old residual-maps README before per-card split |
| `breast-epic_README.md` | 36 KB | SUPERSEDED | Old breast-epic README before per-card split |
| `breast-epic_card_v2_3.json` | 56 KB | SUPERSEDED | Old breast-epic card JSON before per-card split |

---

## DISEASE_MATRIX/ — Stage 8 cross-disease pattern matching (Path B)

| File | Size | Status | Purpose |
|---|---|---|---|
| `disease_cell_signature_matrix_v1_4.csv` | 36 KB | **CANONICAL** | 77 rows × 131 cols (8 metadata + 123 cell-type). Bumped 2026-05-29 with TODO 1.1/1.2/1.3/1.5 findings. SHA: `8600d3e7f5449722...58213ce32` |
| `disease_cell_signature_matrix_engine_schema_v1_2.md` | 8 KB | CANONICAL | The contract — value encoding, match-magnitude function (Mahalanobis-style sign-aligned product weighted by √n), tier-mapping function |
| `README_disease_signature_matrix_folder.md` | 8 KB | CANONICAL | Folder orientation + version log + push policy |
| `OLD/disease_cell_signature_matrix_v1_3.csv` | 36 KB | SUPERSEDED | Prior version, archived |

---

## Family_history_multiplier/ — Stage 9 risk context

| File | Size | Status | Purpose |
|---|---|---|---|
| `family_history_multiplier.json` | 4 KB | CANONICAL | Per-class first-degree-relative RR multipliers (immune 2.5, cycling 2.2, secretory 2.0, ...). Conditional consumption — Stage 9 only applies when intake supplied family history. SHA: `86468854acda627e...b7534967c` |

---

## IAMAtlas_REBUILD/ — the calibrated instrument

| File | Size | Status | Purpose |
|---|---|---|---|
| `IAMAtlasREBUILD.csv.xz` | 97 MB (compressed) | **CANONICAL** | 483,092 CpGs × 8 architecture classes × 115 cell types × {mean, sd, ci_lo, ci_hi}. Uncompresses to ~577 MB. SHA: `41b7c16f043bce96...8646e9fb94c32ee`. LFS-tracked. |
| `IAMAtlasREBUILD_provenance.json` | 4 KB | CANONICAL | Build pipeline, H_min values frozen 2026-04-06 (terminal 0.7728, immune 0.838889, secretory 0.8433, progenitor 0.8522, cycling 0.8561, stromal 0.8630, stem_adult 0.8737, stem_pluri 0.9822), class list. SHA: `91b688f173b68c6c...9403c66` |
| `IAMAtlasREBUILD_celltype_to_class.json` | 4 KB | CANONICAL | 115-cell → 8-class flat dict. Single source of truth for class assignment. SHA: `06f89ea339e2e30d...0139a22e9249` |

---

## IAM_Cellular_Age/ — Stage 3 foreground + Stage 6 scoring

| File | Size | Status | Purpose |
|---|---|---|---|
| `iam_cellular_age_scoring.py` | 20 KB | CANONICAL | `IAMCellularAge` class. Recipe §6.3 canonical inversion (replaces rejected Horvath-style clock). Stage 6 scorer. |
| `age_axis_foreground.py` | 16 KB | CANONICAL | `AgeAxisForeground` class. Per-CpG age regression for foreground subtraction. Stage 3 cleaner. |
| `IAMAtlas_age_layer.csv` | 632 KB | CANONICAL | Per-CpG (α, γ, R², n_samples), 8,199 CpGs at 100% convergence. SHA: `56017c5241ecc126...4383803` |
| `age_layer_diagnostics.json` | 4 KB | CANONICAL | Per-CpG fit diagnostics for the age layer |
| `age_clock_diagnostics.json` | 4 KB | ARCHIVAL | Diagnostics from the rejected Horvath-style clock (kept for audit) |
| `cellular_ages_v4_epic_italy_validation.csv` | 508 KB | ARCHIVAL | Per-patient cellular ages on 1,174 EPIC-Italy cohort. NOT loaded at runtime. |

---

## Literature_anchors_Report_building/ — Stage 9

| File | Size | Status | Purpose |
|---|---|---|---|
| `literature_anchors.json` | 8 KB | CANONICAL | Published per-class A-score anchors (Lister 2013, De Jager 2014, Shireby 2022, etc.) for evidence-report context. SHA: `b78e7d19096192fd...0ce2011d55c22b002` |

---

## Mahalanobis_healthy_reference/ — Stage 5 hyper-volume scoring (L6 covariance metric)

| File | Size | Status | Purpose |
|---|---|---|---|
| `iamatlas_mahalanobis_scoring.py` | 8 KB | CANONICAL | `MahalanobisHealthyHull` class. Single-number patient summary + top-10 axis decomposition. |
| `mahalanobis_healthy_reference_v0_1.json` | 368 KB | CANONICAL | Pooled-HC centroid (n_hc=601) + Ledoit-Wolf covariance (shrinkage=0.0088). Validation anchor: d=+1.871 GSE51057 / +2.088 GSE51032 on >10yr breast pre-dx. SHA: `fae063012ff7542a...13014a95b2b` |
| `mahalanobis_per_patient.csv` | 28 KB | ARCHIVAL | Per-patient distances from breast pre-dx validation cohort. NOT loaded at runtime. |

---

## NILC_Deconvolver/ — Stage 2 cross-method (L4 component separation)

| File | Size | Status | Purpose |
|---|---|---|---|
| `nilc_deconvolver-2.py` | 28 KB | **CURRENT** | NILC v2, Phase B2.1 (2026-05-30) — departure-from-consensus reformulation. The production cross-method check. |
| `nilc_deconvolver.py` | 28 KB | ARCHIVAL | NILC v1, prior algorithm. Kept for audit / regression-comparison. |
| `nilc_fractions_v2_departure.csv` | 128 KB | ARCHIVAL | NILC v2 per-patient class fractions on 1,174 EPIC-Italy |
| `nilc_fractions_all.csv` | 128 KB | ARCHIVAL | NILC v1 per-patient class fractions on 1,174 EPIC-Italy |
| `nilc_walther_crosscheck_v2.json` | 20 KB | ARCHIVAL | v2 cross-method gate report (current) |
| `nilc_walther_crosscheck.json` | 20 KB | ARCHIVAL | v1 cross-method gate report |

---

## Synthetic_Patient_Generator/ — L9 testing scaffolding

| File | Size | Status | Purpose |
|---|---|---|---|
| `synthetic_patient_generator.py` | 24 KB | CANONICAL | FFP10/NPIPE-analog synthetic patient builder for N6/N7 null calibration and end-to-end pipeline testing. NOT invoked per-patient at runtime. |

---

## Tier_breakpoints/ — Stage 7 thresholding

| File | Size | Status | Purpose |
|---|---|---|---|
| `tier_breakpoints.json` | 4 KB | CANONICAL | Engine breakpoints (A_NORMAL_MAX=1.05, A_MARGINAL_MAX=1.07, A_DETECTABLE_MAX=1.10), Warburg threshold (1.07), saturation thresholds (STRUCTURAL=1.10, RUNTIME_MARGIN=0.005), customer-facing vocabulary translation (BELOW_NORMAL→SUPPRESSED, MARGINAL→ELEVATED, etc.). SHA: `1e98640d58b61aec...d49fa7815cf45d8d2f` |

---

## Walther_iam_deconvolver/ — Stage 2 primary deconvolver (L4)

| File | Size | Status | Purpose |
|---|---|---|---|
| `walther_iam_deconvolver.py` | 20 KB | CANONICAL | Production deconvolver. `WaltherIAMDeconvolver` class. Class-level NNLS with optional cell-type refinement. Streaming. 60% / 80% confidence gates. |
| `walther_iam_deconvolver_README.md` | 12 KB | CANONICAL | Deconvolver-specific documentation |

---

## Pipeline-stage cross-reference (quick lookup)

| SOP stage | Walkthrough stage | Owning folder(s) | Chain link |
|---|---|---|---|
| Stage 0 (intake §11–§19) | Stage 0 part 1 | — (engine-level QC) | L1 |
| Stage 1 (β computation §20–§27) | Stage 0 part 2 | — (engine-level calibration) | L2 + L3 |
| Stage 2 (deconvolution §28–§34) | Stage 1 | `Walther_iam_deconvolver/`, `NILC_Deconvolver/` | L4 |
| Stage 3 (foreground §35–§40) | Stage 3 part 1 | `IAM_Cellular_Age/age_axis_foreground.py` + `IAMAtlas_age_layer.csv` | L4 cont. |
| Stage 4 (A-score §41–§46) | Stage 2 | `A_Scoring_Module/`, `Celltype_Marker/` | (scoring) |
| Stage 5 (Mahalanobis §47–§51) | Stage 2.5 | `Mahalanobis_healthy_reference/` | **L6** |
| Stage 6 (cellular age §52–§58) | Stage 3 part 2 | `IAM_Cellular_Age/iam_cellular_age_scoring.py` + `Age_Reference_Matrix_80_cells/` | (scoring) |
| Stage 7 (tier §59–§64) | Stage 4 | `Tier_breakpoints/` + `Cfdna_weight_nonderived_placeholder/` (conditional) | (thresholding) |
| Stage 8 (dual matching §65–§69) | Stage 5 | `DISEASE_MAPS_CARDS/` (Path A) + `DISEASE_MATRIX/` (Path B) | (rule-based + L6 metric) |
| Stage 9 (report §70–§76) | Stage 6 | `Literature_anchors_Report_building/`, `Cancer_prior/`, `Family_history_multiplier/` (conditional) | (report assembly) |
| Stage 10 (delivery §77–§79) | Stage 7 | — (engine-level delivery) | L1 closes loop |
| L9 audit (§80–§91, above runtime) | n/a | `CPG_Null_Runner/`, `Synthetic_Patient_Generator/` | L9 |

EMPTY chain links in V1: **L5** (correlation structure — Phase C), **L7** (likelihood — Phase E), **L8** (parameter inference — Phase E). Declared empty rather than faked. See SOP v1.2 §2.

---

## Total

- **49 data + module files** + **9 docs** (README + INVENTORY + 7-file SOP + build spec) = **58 files**
- **~102 MB on disk** (97 MB is the LFS-tracked compressed atlas; everything else is ~5 MB)
- **17 canonical artifacts with SHA-256 anchored above** for integrity verification

---

*Generated 2026-06-02 by Walther (Claude) on Heath's instruction. To regenerate after additions, walk the folder, compute SHA-256 of canonical artifacts, update this file. Convention: SHA-256 truncated to first 16 + last 9 chars for readability; full SHAs in each artifact's sidecar or provenance JSON.*

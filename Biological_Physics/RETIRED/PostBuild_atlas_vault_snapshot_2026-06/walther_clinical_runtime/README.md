# walther_clinical_runtime/

**The complete set of runtime files the future `walther_clinical.py` orchestrator will consume to run the CPG Chain-of-Custody SOP end-to-end on a patient IDAT.**

Uploaded by Heath W. Mahaffey on 2026-06-02 as `All pipeline Files.zip`, staged here under atlas_vault for repo-level reference and reproducibility. This folder is the authoritative answer to "what does the orchestrator need in its run folder?"

---

## Folder map

| Folder | Contents | Used at SOP stage |
|---|---|---|
| `IAMAtlas_REBUILD/` | `IAMAtlasREBUILD.csv.xz` (100MB compressed, 577MB uncompressed; 483,092 CpGs × 8 classes × 115 cell types), `IAMAtlasREBUILD_provenance.json` (H_min values frozen 2026-04-06), `IAMAtlasREBUILD_celltype_to_class.json` (115-cell → 8-class assignments) | §28 + all atlas-consulting steps |
| `Walther_iam_deconvolver/` | `walther_iam_deconvolver.py` (~440 lines, `WaltherIAMDeconvolver` class — class-level NNLS with optional cell-type refinement, streaming, 60%/80% gates) + README | Stage 2 (SOP §30–§31) |
| `NILC_Deconvolver/` | `nilc_deconvolver.py` (v1, 593 lines) + `nilc_deconvolver-2.py` (v2 current, 603 lines — Phase B2.1 departure-from-consensus reformulation), `NILCDeconvolver` class, `cross_method_comparison()` function, prior cross-check JSONs and fractions CSVs | Stage 2 cross-method (SOP §32–§33) |
| `IAM_Cellular_Age/` | `iam_cellular_age_scoring.py` (`IAMCellularAge` class — canonical Recipe §6.3 inversion), `age_axis_foreground.py` (`AgeAxisForeground` class — per-CpG age regression for foreground subtraction), `IAMAtlas_age_layer.csv` (per-CpG α/γ/R²/n, 8,199 CpGs at 100% convergence), `cellular_ages_v4_epic_italy_validation.csv` (1,174 EPIC-Italy validation output), diagnostics JSONs | Stage 3 foreground (SOP §35) + Stage 6 cellular age (SOP §52–§58) |
| `A_Scoring_Module/` | `iamatlas_a_scoring.py` — `score_per_class()` + `score_per_celltype()` + `load_artifact()` functions | Stage 4 (SOP §41–§45) |
| `Celltype_Marker/` | `iamatlas_celltype_markers_v0_2.json` (237 KB — 115 cell types × top-100 one-vs-rest markers, CURRENT canonical version), `.sha256` anchor, `OLD/iamatlas_celltype_markers_v0_1.json` (SUPERSEDED), supersession note | Stage 4 (SOP §29, §44) |
| `Mahalanobis_healthy_reference/` | `iamatlas_mahalanobis_scoring.py` (`MahalanobisHealthyHull` class, top-10 axis decomposition), `mahalanobis_healthy_reference_v0_1.json` (373 KB — pooled HC centroid + Ledoit-Wolf covariance, n_hc=601), per-patient validation CSV | Stage 5 (SOP §47–§51) |
| `Age_Reference_Matrix_80_cells/` | `age_reference_matrix.{json, csv, py}` — 80-cell baseline (8 classes × 10 decadal bins, ages 4–95) with A_mean/A_sd/β_mean/β_sd/n_samples/percentiles per cell, plus Python helpers `age_ref_A()`, `age_ref_sd()`, `age_ref_percentiles()` | Stage 6 cellular age (SOP §53) |
| `Tier_breakpoints/` | `tier_breakpoints.json` — engine breakpoints (1.05 / 1.07 / 1.10), customer-facing label translation (BELOW_NORMAL→SUPPRESSED, etc.), Warburg threshold, saturation thresholds | Stage 7 (SOP §59–§63) |
| `Cfdna_weight_nonderived_placeholder/` | `cfdna_weight.json` — healthy-blood cfDNA tissue-of-origin weights (immune 0.70, cycling 0.12, secretory 0.08, ...), Snyder 2016 + Moss 2018 | Stage 7 cfDNA branch (SOP §61, conditional consumption — only when substrate is plasma cfDNA) |
| `DISEASE_MATRIX/` | `disease_cell_signature_matrix_v1_4.csv` (77 rows × 131 cols — current canonical, bumped 2026-05-29 with TODO 1.1/1.2/1.3/1.5 findings), `disease_cell_signature_matrix_engine_schema_v1_2.md` (the contract — value encoding, match-magnitude algorithm, tier-mapping function), README, `OLD/` archive with v1.3 | Stage 8 (SOP §65–§69) |
| `DISEASE_MAPS_CARDS/` | Per-card folders. Each card has: card_json subfolder (card JSON + README) + residual_maps subfolder (chr-annotated residual map, bimodality map, PCA projections, residual maps README). `Breast_EPIC/` is the first card pair currently. `OLD/` archives prior versions. | Stage 8 per-card matching (SOP §66–§69) |
| `Literature_anchors_Report_building/` | `literature_anchors.json` (5 KB) — published per-class A-score anchors (healthy + disease + cancer) with source citations | Stage 9 (SOP §71) |
| `Cancer_prior/` | `cancer_prior.json` — US lifetime cancer incidence per class | Stage 9 (SOP §72) |
| `Family_history_multiplier/` | `family_history_multiplier.json` — per-class first-degree-relative RR | Stage 9 (SOP §73, conditional consumption — only when intake supplied family history) |
| `CPG_Null_Runner/` | `cpg_null_runner.py` (~1,100 lines) — L9 8-null orchestration framework | L9 audit (SOP §80–§88, §91) |
| `Synthetic_Patient_Generator/` | `synthetic_patient_generator.py` — FFP10/NPIPE-analog synthetic patient builder for N6/N7 nulls and end-to-end pipeline testing | L9 audit (SOP §86, §87, §89) |

---

## Per-card structure — confirmed pattern from breast-epic

A production card has TWO subfolders:

**`{card_name}_card_json/`** — the card definition:
- `{card}_card_v{version}.json` (card rules, panel CpGs, H_min anchor, covariate thresholds, validation anchors, expected direction)
- `{card}_README.md` (card-specific documentation)

**`{card_name}_residual_maps/`** — the per-card signal layers (Layer 3 base maps):
- `{card}_residual_map_chr_annotated.csv` (concordant CpGs with chr/MAPINFO — the main matched-filter layer, e.g. 1,392 concordant CpGs for breast-epic at >10yr pre-dx)
- `{card}_bimodality_map.csv` (bimodality loss detection layer — additional signal channel)
- `{card}_pca_projections.csv` (PCA projections layer — cross-cohort signature)
- `README_{card}_residual_maps.md`

Total: 4 active files per card across two subfolders, plus their readmes. (Earlier SOP language about "5 files per card" was approximate — the real count varies by card and includes the chr-annotated residual map as the primary layer plus bimodality and PCA as secondary layers.)

The orchestrator at Stage 8 will load each card's JSON + the three signal-layer CSVs, match the patient's data against each in turn, and produce one card verdict per loaded card.

---

## Version state at upload time

| Artifact | Version | Status |
|---|---|---|
| Cell-type markers | **v0_2** | Current canonical. v0_1 in `Celltype_Marker/OLD/` is SUPERSEDED. |
| Disease signature matrix | **v1.4** | Current canonical. v1.3 in `DISEASE_MATRIX/OLD/` is superseded. |
| Schema | **v1_2** | Stable (only bumps on structural changes). |
| IAMAtlas | **REBUILD** | Current (482K CpGs × 8 classes × 115 cell types; 577MB uncompressed). The OLD `IAMAtlas.csv.xz` (collapsed flatness-bug version) is NOT here — only REBUILD is canonical. |
| H_min values | **frozen 2026-04-06** | terminal 0.7728, immune 0.838889, secretory 0.8433, progenitor 0.8522, cycling 0.8561, stromal 0.8630, stem_adult 0.8737, stem_pluri 0.9822 |
| NILC | v1 + **v2 (current — departure-from-consensus)** | v2 is the production version per Phase B2.1 2026-05-30. |
| Cellular age scorer | **v4** | The canonical Recipe §6.3 inversion. Replaces the rejected Horvath-style `iam_cellular_age_clock.py` (not here). |

---

## What this folder does NOT contain

These are referenced by the orchestrator but live elsewhere or need building before orchestrator build:

- `walther_clinical.py` itself (the orchestrator — not yet built; see `walther_clinical_BUILD_SPEC_v1_1.md` in this repo)
- `WALTHER_CLINICAL_MANIFEST.json` (per-file SHA-256 manifest, generated at build time after all dependencies are finalized)
- Cards other than `Breast_EPIC` (per-card VAL re-runs against the new atlas + deconvolver are pending; the orchestrator build is blocked on those)
- `methylprep` / `minfi` for IDAT → β conversion (Python and R packages installed via pip/conda at orchestrator deploy time)

---

## How a future AI / operator uses this folder

1. Read `walther_clinical_BUILD_SPEC_v1_1.md` first (same repo).
2. Verify each module's API by reading the actual `.py` file before writing any call to it. Do not infer signatures from this README.
3. Verify each JSON's schema by reading the actual `.json` file before writing any access to its keys. Same rule.
4. For the disease matrix specifically, read both the matrix CSV AND the schema MD — the matrix is uninterpretable without the schema's value-encoding rules and match-magnitude algorithm.
5. For cards specifically, expect per-card data in the per-card subfolders (one card JSON + multiple residual map CSVs).
6. The atlas is `IAMAtlasREBUILD.csv.xz` — decompress at orchestrator startup if not already cached. Never load a bare `IAMAtlas.csv` that has lost the REBUILD marker.

---

*Staged 2026-06-02 by Walther (Claude) on Heath's instruction. Source: `All pipeline Files.zip` upload.*

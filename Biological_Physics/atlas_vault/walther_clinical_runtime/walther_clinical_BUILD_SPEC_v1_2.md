# Walther Clinical Pipeline — Build Specification v1.2

**Document purpose:** Complete specification a future AI follows to build `walther_clinical.py`, the clinical orchestrator that runs the CPG Chain-of-Custody SOP end-to-end on a single patient IDAT pair and produces the doctor-facing report.

**Author of spec:** Heath W. Mahaffey + Walther (Claude)
**Date authored:** 2026-06-02 (v1.0); revised 2026-06-02 (v1.1); revised 2026-06-06 (v1.2)
**Build status:** **NOT YET STARTED.** Build blocked on the single remaining prerequisite in §3.
**Authoritative SOP:** `CPG_Chain_of_Custody_SOP_v1_2.md` + v1.3 native sections for Stage 4.5 / 4.6 (in repo). The SOP is the encyclopedia; this spec is the build contract.
**Authoritative runtime dependencies:** `Biological_Physics/atlas_vault/walther_clinical_runtime/` in the repo (every module and JSON the orchestrator consumes).
**Reference visualizations:** `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/` — CPG Plates 1–4 are the canonical visual reference; the orchestrator's Stage 4.6 output mirrors Plate 1's conventions exactly.

**Changes v1.1 → v1.2:**
- §4.3 **expanded patient intake schema** from 8 doctor-supplied fields → 24 covariates per `patient_intake_questionnaire_v1_0.md`. Schema, default values, and validation rules documented.
- §4.5 **conditional consumption table extended** with the full per-stage routing of intake covariates (which covariate enters which stage).
- §5 **new Stage 4.5** — bidirectional decomposition BEFORE pooled A-scoring. Implements the VAL-050/VAL-051 lesson at patient runtime. **`bidirectional_decomposition.py` built**, mirrors the sealed `val051_analyze.a_dir_score` formula exactly. **`directional_panels_v1_0.json`** carries the VAL-051 7-CpG immune panel (SHA-anchored to sealed `val051_panel_ruleA.json`) + 18-CpG VAL-050 pooled-entropy comparator parent panel.
- §5 **new Stage 4.6** — per-class healthy brightness comparison + patient Mollweide projection. The customer's personal Cosmic Microwave Methylome (Plate 1 analog). **`patient_brightness_comparison.py` built**.
- §3.5b **CPG Plates 1–4 declared as canonical visualization references** + pushed to `IAMAtlas_v0_1/plates/` with README documenting HEALPix NSIDE=128 + Mollweide + genomic-order conventions.
- §3.5 / §3.5b **HEALPix mapping generator built**. `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/generate_cpg_healpix_mapping.py` produces `iamatlas_cpg_to_healpix_nside128.npy` (1.93 MB, 483,092 CpGs → 450,192 annotated + 32,900 sentinel). EPIC v1 B4 manifest cached at `IAMAtlas_v0_1/external_manifests/EPIC_v1_B4_manifest_normalized.csv` (zhou-lab provenance documented).
- §5 Stage 7 — replaced 4-tier (NORMAL/MARGINAL/DETECTABLE/URGENT/FLOOR_BREACH) with **6-tier physics-derived** (SUPPRESSED/NORMAL/ELEVATED/WARBURG_TRANSITION/SIGNIFICANTLY_ELEVATED/BREACH). 1.07 Warburg line + 1.10 architectural-fidelity breach line are the physics-defined inflection points. **`tier_breakpoints.json v1.2` built** with per-class structural-ceiling table, 7 covariate-override modes, smoking-bin stratification table, bidirectional handoff rules, CI-based tier confidence propagation. v0 4-tier archived in OLD/.
- §5 Stages 4–7 — **explicit forward CI propagation** from MCMC posteriors. Customer-facing numbers carry measurement ranges, not point estimates only.
- §5 Stage 3 — **smoking + sex foreground modules built AND layer CSVs fit on GSE50660 (n=464, Tsaprouni 2014, smoking + sex + age metadata)**. `smoking_axis_foreground.py` (per-CpG δ·indicator_current + φ·recency_score model; recency mapped from smoking_bin) and `sex_axis_foreground.py` (per-CpG ψ·indicator_male model with chrX/chrY/XCI flag handling). Layer CSVs `IAMAtlas_smoking_layer.csv` + `IAMAtlas_sex_layer.csv` FIT on GSE50660 (n=464). Stage 7 interim threshold-stratification retires once L4 β-level subtraction operates in production. The architecturally correct L4 β-level subtraction path is now wired and ready to receive its layer CSVs.

**Changes v1.0 → v1.1:**
- Locked orchestrator name: `walther_clinical.py` · locked deconvolver name: `walther_iam_deconvolver.py`.
- All real runtime dependencies confirmed present in repo at `walther_clinical_runtime/`; the build is no longer blocked on uploads.
- Corrected: cell-type markers are **v0_2** (current), not v0_1 (superseded). v1.0 of this spec wrongly downgraded these.
- Corrected: disease matrix is **v1.4** (current, bumped 2026-05-29), not v1.3 (superseded).
- Added §1.5 three-component architectural separation (walkthrough §6).
- Added §3.7 the 13 non-negotiable rules (walkthrough §8).
- Added §3.8 chain-of-custody L1–L9 overlay (walkthrough §0).
- Added §4.5 conditional consumption discipline (walkthrough §3).
- Stage 4 / 5 / 8 updated to reflect dual-matching parallelism, bidirectional flag, risk-context formula, deep-link routing.
- Stage 0 updated with HM450 ≥80% coverage check and platform tagging per VAL-091 ad-LL-006.
- Stage 1 updated to carry credible intervals from MCMC posteriors.
- §8 (doctor report content) reframed: V1 is doctor-only; patient-facing report deferred to V2, including per-cell-type age with cell-function descriptions Heath has reference material ready for.
- §14 (naming) updated — `commercial.web.py` from walkthrough explicitly noted as the working alternative that did NOT get chosen.

---

## 1. What is being built

A single Python CLI script — `walther_clinical.py` — that lives in one folder alongside every dependency it needs. The clinician (or Heath in testing) drops a patient IDAT pair into that folder, types `python3 walther_clinical.py`, and the script runs Stages 0 through 10 of the CPG Chain-of-Custody SOP, calling each real runtime module in sequence, and produces a **doctor-facing report** at the end.

It is not a Flask web app. It is not a service. It is not a library to be imported. It is a single CLI script that executes the chain end-to-end on one patient at a time, writes a report, and exits.

The existing `GAPE_WEB_v13.py` is **not** the orchestrator. `GAPE_WEB_v13.py` is a 10,584-line Flask web demo / exploration tool from the learning phase of the project; it takes already-computed β values plus an architecture-class key and runs the seven analysis engines E1–E7. It does not consume IDAT files, it does not run the SOP chain, and it embeds an outdated age atlas and per-substrate saturation logic that have been superseded. Mine pieces from it (see §6); do not modify it; do not try to make it the orchestrator.

The orchestrator is a fresh script that follows the SOP, calls the standalone modules in `walther_clinical_runtime/`, and produces output to a single `outputs/` folder.

**V1 scope: doctor-facing report ONLY.** A patient-facing / customer-facing report is V2 work, deferred until the doctor has used V1 with patients for a while and Heath has learned what to present and what not to. Heath has reference material ready for V2 — including per-cell-type cellular ages with explanations of what each cell type is and what its job is — but that content is NOT in V1.

## 1.5 Three-component architectural separation

The pipeline-walkthrough specifies three independently updatable components. The orchestrator build must respect this separation even when V1 ships components 1 and 2 together as a single script for simplicity.

| Component | Role | Independently updatable | V1 status |
|---|---|---|---|
| **(1) Orchestration runtime** — `walther_clinical.py` | Loads the five+ startup artifacts. Runs Stages 0–5 of the pipeline (intake through dual matching). Calls real modules in `walther_clinical_runtime/`. Outputs structured Stage-5 result. | ✓ Restart picks up new card JSONs without code change. Deconvolver / scoring modules swappable. H_min anchors recalibrate without touching orchestrator code. | **Built in V1** as a single CLI script. |
| **(2) Doctor report builder** — internal module within V1 | Reads Stage-5 outputs + card JSONs + matrix CSV + the 6 lookup JSONs + literature_anchors. Assembles doctor-facing Markdown → PDF. Owns the Stage 6 translation layer (engine-tier → clinical-language). | ✓ Report template changes are content updates, not code structure changes. | **Built in V1** as an internal module of `walther_clinical.py` (kept as a clear internal boundary so it can be lifted out to its own file `walther_report_builder.py` in V2 without changing the orchestrator). |
| **(3) Patient-facing destination** — iamperformance.net + a future `walther_patient_report_builder.py` | Customer report; per-class pages; per-cell pages with cell-function descriptions; Astro-Genetics framing; vigilance content per tier; deep links from card `educational_page_url` + matrix `organ_pages_to_link` fields | ✓ Website content team updates pages without touching code. Per-cell descriptions are content, not algorithm. | **NOT in V1.** Deferred to V2 after doctor feedback informs the framing. Heath has reference material ready. |

The discipline this enforces: V1's doctor report builder should be a clean internal module within `walther_clinical.py` — a single function `assemble_doctor_report(stage5_outputs, patient_metadata, lookups) → markdown_text` — that V2 can extract verbatim into a separate file. Do not entangle the report assembly with the chain orchestration.

---

## 2. The one absolute rule

**NO FABRICATION.** This rule outranks every other instruction in this document.

If a runtime module, JSON schema, function signature, or file path is not directly readable in the repo at build time, do not invent it. Stop and ask Heath. The pattern that destroyed v1 of the SOP — filling "Files invoked" slots with plausible-sounding `cpg_engine/...` paths that did not exist — is the failure mode this entire spec is designed to prevent.

Concretely:

- Do not invent function signatures for modules you have not read. Every API used in this spec was confirmed by reading the actual source file in `walther_clinical_runtime/`. If you find any discrepancy between this spec and the actual source, the source wins — and flag the discrepancy to Heath.
- Do not invent JSON keys. Read `tier_breakpoints.json`, `cancer_prior.json`, etc. before writing code that consumes them. They are all in the repo; read them.
- Do not invent CSV column names. The disease matrix schema, the breast-epic residual map, the cellular age validation output — all have real column names; use them.
- Do not invent IDAT-handling code. Use `methylprep` for the IDAT → β pipeline. Consult its real documentation, not your memory.

If at any point during the build you find yourself reasoning "the module probably looks like this" — that is the cue to stop and verify, not to proceed.

---

## 3. PREREQUISITES

The build is blocked until every item in this checklist is satisfied. Do not start writing `walther_clinical.py` until Heath confirms.

### 3.1 Cards re-VAL'd against the new atlas — ONLY remaining blocker for V1

Every production card must be re-run through its full VAL using the IAMAtlas REBUILD, the current Walther deconvolver, the current NILC v2 deconvolver, and the current standalone runtime modules. The VALs that exist on the OLD atlas / OLD deconvolver are not usable for production; they describe a different instrument.

Per the breast-epic structure observed in `walther_clinical_runtime/DISEASE_MAPS_CARDS/Breast_EPIC/`, each card produces two subfolders of files:

**Card JSON subfolder** (`{card}_card_json/`):
- `{card}_card_v{version}.json` (rules, panel CpGs, H_min anchor, covariate thresholds, validation anchors, expected direction)
- `{card}_README.md`

**Residual maps subfolder** (`{card}_residual_maps/`):
- `{card}_residual_map_chr_annotated.csv` (concordant CpGs with chr/MAPINFO — the matched-filter layer)
- `{card}_bimodality_map.csv` (bimodality loss detection)
- `{card}_pca_projections.csv` (PCA projections, cross-cohort)
- `README_{card}_residual_maps.md`

V1 ships with breast-epic only. Other cards (ad-immune, crc-immune-inv, lung-epic, hcc-epic, prostate-epic, heme-epic, cardio-epic, cervical-epic, glioma-epic, kidney-epic, ...) ship in V1.x patch releases as each card's re-VAL completes.

### 3.2 Disease signature matrix — ALREADY CURRENT

`disease_cell_signature_matrix_v1_7.csv` is the canonical version (current as of immune card v1.0 work; bumped from v1.5 through additions during VAL-015 era; v1.3, v1.4, v1.5 all archived in `DISEASE_MATRIX/OLD/`). 82 rows × 131 columns. The orchestrator consumes v1.7. **No further action required for V1 build.**

### 3.3 All standalone runtime modules — CONFIRMED PRESENT

Reading the actual source files in `walther_clinical_runtime/`:

| Module | Path in runtime folder | Real API confirmed |
|---|---|---|
| `walther_iam_deconvolver.py` | `Walther_iam_deconvolver/` | `class WaltherIAMDeconvolver(matrix_path, celltype_class_map, n_class_markers_per_class=600, max_celltype_markers=4000, verbose=True)`. Returns `DeconvolutionResult(class_fractions, celltype_fractions, diagnostics, status)`. |
| `nilc_deconvolver-2.py` (v2 current) | `NILC_Deconvolver/` | `class NILCDeconvolver(atlas_path, marker_path, chromosome_windowed=False, min_markers_per_class=30, ridge_lambda=1e-4)`. Plus `def cross_method_comparison(walther_df, nilc_df, ...) → CrossMethodReport`. Returns `NILCResult(fractions, raw_fractions, residual_mae, n_markers_used, status, per_class_residual)`. |
| `iamatlas_a_scoring.py` | `A_Scoring_Module/` | `def score_per_class(customer_betas, class_markers, h_min_by_class) → Dict[class, Dict]`. `def score_per_celltype(customer_betas, celltype_markers, celltype_to_class, h_min_by_class) → Dict[celltype, Dict]`. `def load_artifact(path) → (meta, markers, ct_to_class, h_min)`. |
| `iamatlas_mahalanobis_scoring.py` | `Mahalanobis_healthy_reference/` | `class MahalanobisHealthyHull(reference_path)`. `.score(celltype_ascores: Dict[str, float]) → Dict` with `mahalanobis_distance`, `top10_axis_contributions`, `status`. |
| `iam_cellular_age_scoring.py` | `IAM_Cellular_Age/` | `class IAMCellularAge(ref_matrix_path, markers_artifact_path, markers_per_class, min_cpgs_per_class=30)`. `.score()` returns `CellularAgeResult` dataclass with per-class age + status + concordance. |
| `age_axis_foreground.py` | `IAM_Cellular_Age/` | `class AgeAxisForeground(min_samples=30, min_age_range=10.0)`. `.fit(beta_matrix, ages, hc_mask, cpg_ids)`. `.subtract_from(beta, ages) → cleaned_beta`. |
| `smoking_axis_foreground.py` (NEW v1.2) | `IAM_Cellular_Age/` | `class SmokingAxisForeground(min_samples_per_bin=10, min_recency_variance=0.05)`. `.fit(beta_matrix, smoking_bins, hc_mask, cpg_ids, candidate_cpgs=None)`. `.subtract_from(beta, smoking_bins) → cleaned_beta`. Per-CpG model: β = α + δ·indicator_current + φ·recency_score + ε. Smoking_bin mapped to recency score: never=0.00 / former_15plus_y=0.10 / former_5_15y=0.30 / former_0_5y=0.60 / current=1.00. Fit on HC samples only; case samples excluded. Layer artifact `IAMAtlas_smoking_layer.csv` (cpg_id, α, δ_current, φ_recency, R², n_samples). |
| `sex_axis_foreground.py` (NEW v1.2) | `IAM_Cellular_Age/` | `class SexAxisForeground(min_samples_per_sex=30, x_inactivation_psi_threshold=0.20)`. `.fit(beta_matrix, sex_at_birth, hc_mask, cpg_ids, chr_annotation=None)`. `.subtract_from(beta, sex_at_birth) → cleaned_beta`. Per-CpG model: β = α + ψ·indicator_male + ε. Special handling of chrX (X-inactivation flag for high-ψ CpGs) + chrY (sex-chromosome flag for masking in female samples). Layer artifact `IAMAtlas_sex_layer.csv` (cpg_id, α, ψ_male, R², n_samples, is_chr_x, is_chr_y, x_inactivation_flag). |
| `patient_brightness_comparison.py` (NEW v1.2) | `Brightness_Comparison/` | Stage 4.6 module. `load_all_8_class_references(archives_dir) → Dict[class, BrightnessReference]`. `compute_all_8_class_departures(patient_beta, references, patient_id) → PatientBrightnessReport`. `render_patient_cosmic_methylome(report, cpg_to_pixel, out_path) → png_path`. `save_brightness_report(report, out_dir) → artifacts_dict`. Reads brightness CSVs directly from class archive tar.xz files. |
| `bidirectional_decomposition.py` (NEW v1.2) | `Bidirectional_Decomposition/` | Stage 4.5 module. `load_directional_panels(panel_json_path) → Dict[class, DirectionalPanel]`. `compute_per_class_bidirectional_decomposition(patient_beta, panels, patient_id) → BidirectionalReport`. `score_directional_composite(patient_beta, panel) → (composite, n_covered, n_total)` — mirrors the sealed VAL-051 `a_dir_score` formula exactly (z-scores against frozen training-set HC mean/SD, multiplied by frozen ±1 disease direction, averaged across covered CpGs). `bidirectional_flag(a_pooled, a_directional)` — fires when pooled is mute (within ±0.05 of 1.0) AND directional is loud (|composite| > 0.40). v1.0 panel coverage: immune class only (VAL-051 Rule A, 7 CpGs). Other 7 classes pending future sealed VALs (declared NO_PANEL honestly). |
| `cpg_null_runner.py` | `CPG_Null_Runner/` | L9 8-null framework. Not invoked per-patient; runs against sealed VALs. |
| `synthetic_patient_generator.py` | `Synthetic_Patient_Generator/` | N6/N7 testing only; not in per-patient flow. |

### 3.4 All runtime constant JSONs — CONFIRMED PRESENT

All in `walther_clinical_runtime/`:

| JSON | Subfolder | Real keys confirmed |
|---|---|---|
| `iamatlas_celltype_markers_v0_2.json` + `.sha256` | `Celltype_Marker/` | `markers_by_celltype` (dict), `celltype_to_class` (dict), `H_min_by_class` (dict). Loaded via `iamatlas_a_scoring.load_artifact(path)`. |
| `IAMAtlasREBUILD_celltype_to_class.json` | `IAMAtlas_REBUILD/` | Flat `{cell_type: class_name}` dict, 115 entries. |
| `IAMAtlasREBUILD_provenance.json` | `IAMAtlas_REBUILD/` | `h_min_values_frozen_2026_04_06` key with the 8 anchors; atlas version, build date, predecessor, classes, n_cpgs=483092. |
| `mahalanobis_healthy_reference_v0_3.json` (current production) | `Mahalanobis_healthy_reference/` | `artifact_id`, `feature_names_valid`, `n_features`, `centroid`, `covariance_matrix`, `shrinkage`, `route_A_calibration_v0_3` (p95 threshold + percentile distribution under n=1,721 HC), `hc_cohort_sources` (foundation + Hannum + Tsaprouni), `phase_planning` (Phase 3 + Phase 4 queued). v0_1 (n=601, foundation-only) and v0_2 (n=1,257, +Hannum) preserved at same folder for lineage. See §3.4b hull versioning protocol. |
| `age_reference_matrix.{json,csv,py}` | `Age_Reference_Matrix_80_cells/` | 80-cell baseline: per-class list of decadal-bin records with `age_midpoint, A_mean, A_sd, beta_mean, beta_sd, n_samples, A_p10..A_p90, source_citation`. |
| `tier_breakpoints.json` v1.2 (NEW v1.2 — 6-tier physics-derived) | `Tier_breakpoints/` | `tier_system_v1_2.tiers` (6 entries: SUPPRESSED < 0.95 / NORMAL [0.95, 1.04) / ELEVATED [1.04, 1.07) / **WARBURG_TRANSITION [1.07, 1.10)** / SIGNIFICANTLY_ELEVATED [1.10, 1.12) / **BREACH ≥1.10 sustained or ≥1.12 single-timepoint**); `per_class_default_breakpoints.structural_ceiling_by_class` (1/H_min per class; stem_pluri structurally blind for BREACH at ceiling 1.0181); `tier_by_covariate_overrides` (7 modes: EXPECTED_SUPPRESSION, TRAJECTORY_WATCH, TREATMENT_RESPONSE, CONTEXT_PREGNANCY/POSTPARTUM/HRT_BASELINE/WEIGHT_LOSS_INTERVENTION); `tier_by_smoking_bin` (interim mitigation until smoking_axis_foreground.py at v1.3); `bidirectional_pattern_handoff` (consume Stage 4.5 directional composite when FLAG_BIDIRECTIONAL); `tier_confidence_propagation` (BORDERLINE_TIER flag at 0.20 prob threshold from MCMC CI). v0 4-tier statistical-percentile archived in `OLD/tier_breakpoints_v0_4tier_statistical.json`. |
| `cfdna_weight.json` | `Cfdna_weight_nonderived_placeholder/` | Per-class weights (immune 0.70, cycling 0.12, secretory 0.08, stromal 0.04, stem_adult 0.03, progenitor 0.02, terminal 0.005, stem_pluri 0.005). |
| `literature_anchors.json` | `Literature_anchors_Report_building/` | Per-class list of `{label, A, beta, context, source}` records (Lister 2013, De Jager 2014, Shireby 2022, etc.). |
| `cancer_prior.json` | `Cancer_prior/` | Per-class US lifetime cancer incidence (cycling 0.055, secretory 0.140, immune 0.020, terminal 0.008, stromal 0.005, stem_adult 0.008, progenitor 0.006, stem_pluri 0.004). |
| `family_history_multiplier.json` | `Family_history_multiplier/` | Per-class first-degree-relative RR multipliers (cycling 2.2, secretory 2.0, immune 2.5, terminal 1.5, stromal 1.3, stem_adult 2.0, progenitor 1.8, stem_pluri 1.5). |

### 3.5 The atlas itself — CONFIRMED PRESENT

`IAMAtlasREBUILD.csv.xz` (100 MB compressed via LFS) is in `walther_clinical_runtime/IAMAtlas_REBUILD/`. Decompresses to ~577 MB (483,092 CpGs × ~50 columns of class/cell-type means and SDs).

The "REBUILD" suffix is mandatory — the OLD `IAMAtlas.csv.xz` (collapsed flatness bug) is not in this folder and must never replace REBUILD. The orchestrator must halt if it finds a bare `IAMAtlas.csv` without the REBUILD marker.

**Per-class brightness CSVs (NEW v1.2 build dependency).** The 8 per-class brightness CSVs at `Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives/{class}_v0_1_REBUILD.tar.xz` (inner path `{class}/iamatlas_v0_1_{class}_brightness.csv`) are the data behind Plate 1 and the input to Stage 4.6. Each CSV carries: cpg_id, class, mean, sd, ci_lo, ci_hi across 483,093 CpGs. **The orchestrator reads these at session startup (load_all_8_class_references), caches in RAM, consults at Stage 4.6 per patient.**

### 3.5b CPG Plates 1–4 — canonical visualization references (NEW v1.2)

Four canonical Plates live at `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/` with a folder README. They are the **visual reference for the framework** — the build target Stage 4.6 mirrors for per-patient projections:

| File | What it is | Stage that mirrors its conventions |
|---|---|---|
| `CPG_Plate_01_Cosmic_Microwave_Methylome.png` | 8-panel Mollweide of per-class posterior β across 481,966 CpGs (the healthy reference) | Stage 4.6 patient CMM mirrors this projection |
| `CPG_Plate_02_Breast_Anisotropy.png` | 1,392 concordant breast pre-dx CpGs, signed Cohen's d, chr6 zoom | Stage 8 Route A per-card residual map matching (breast card) |
| `CPG_Plate_03_Grandaddy_CMM_vs_CMB.png` | Side-by-side methylome vs CMB on matched Mollweide | The framework's CMB-analogy proof; N7 synthetic patient generator mirrors the CMB generative discipline |
| `CPG_Plate_04_Patterns_Discovered.png` | Six findings the spherical projection makes visible (Class-Difference Map, chr16+17 Cold-Patch Zones, Concordant Signal Density, Differentiation Gradient, MCMC Coverage Map, Breast Pre-Diagnostic Anisotropy) | Engineering audit reference — reveals features the unprojected data doesn't show |

**Plate 1 conventions are the binding contract for Stage 4.6:** HEALPix NSIDE=128 (npix=196,608), Mollweide projection, CpGs ordered by chromosome × MAPINFO with sequential pixel assignment in genomic order, multi-CpG-per-pixel averaging. Patient projections use the same `iamatlas_cpg_to_healpix_nside128.npy` mapping file so the customer's personal CMM sits on the same grid as the reference.

**Mapping generator (NEW v1.2 — built).** `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/generate_cpg_healpix_mapping.py` produces the canonical `.npy` mapping. Inputs: IAMAtlas REBUILD CSV (for the CpG list in atlas row order) + Illumina EPIC v1 B4 manifest (for CHR + MAPINFO annotation). Output: `iamatlas_cpg_to_healpix_nside128.npy` (np.int32 array, shape n_cpgs, pixel index per CpG in atlas row order) + `iamatlas_cpg_to_healpix_nside128.provenance.json` (atlas SHA, manifest version, npix, n_annotated, n_unannotated, sentinel pixel). One-time generation at IAMAtlas build time; cached forever after. Smoke-test mode (`--smoke-test`) confirms end-to-end pipeline against the 7,115-CpG breast residual map without requiring the external manifest. Production run requires the user to download the Illumina manifest once.

### 3.6 SHA-256 integrity manifest — to be generated at build time

A single manifest file `WALTHER_CLINICAL_MANIFEST.json` listing every dependency file with its SHA-256 hash. The orchestrator SHA-verifies every file at startup; refuses to run on mismatch. Heath generates this manifest after the V1 card set (currently breast-epic only) is finalized.

### 3.7 The 13 non-negotiable rules — constraints the build must obey

The orchestrator must enforce or obey each of these 13 rules. Treat any violation as a build failure.

1. **Wellness-first positioning.** Cellular health and cellular age are the lead. Disease detection is secondary. (V1 doctor report still applies this — wellness panel first, disease findings labeled secondary.)
2. **Single IAMAtlas, only IAMAtlas at runtime.** No external atlases queried at runtime — ever. No Moss-NNLS, no Loyfer-NNLS, no EpiDISH-via-rpy2, no Salas-QC calls at runtime. Source atlases were ingested at IAMAtlas BUILD time; the orchestrator queries only IAMAtlas REBUILD.
3. **No customer-facing physics terminology in commercial code.** Boltzmann, Landauer, Arrhenius, Bose-Einstein, decoherence, k_B, ln2, coth, "thermal", "activation energy", "Mahaffey Number" — none of it in `walther_clinical.py` source code, docstrings, comments, API outputs, or HTML. Internal variable names neutral. This protects the Recipe.
4. **Recipe stays in the vault forever.** Never disclosed under any NDA. Full acquisition is the only scenario where the Recipe transfers. The orchestrator does not embed Recipe content; it consumes only the operational artifacts (atlas, H_min anchors, deconvolver source, cards, schema).
5. **Screening-language rule.** EDEAR never specifies screening tests, ages, intervals, or follow-up workups. Vigilance language defers to "the screening recommendations your clinician has discussed with you." (V1 doctor report has more flexibility here than the patient report will — but even the doctor report doesn't prescribe specific tests; it presents evidence + literature anchors and lets the clinician decide.)
6. **Per-class A-score aggregation discipline.** Each class has its own H_min anchor. Customer report shows class-level + cell-level scores separately (never silently combined).
7. **Class assignments non-negotiable.** Megakaryocyte → progenitor (NOT immune). Cortical neurons → terminal (NOT stromal). The orchestrator consumes `IAMAtlasREBUILD_celltype_to_class.json` as the single source of truth; no per-cell-type re-classification anywhere.
8. **Empty cells in disease matrix mean "no documented signature" — NOT zero.** The Stage 8 match function treats blank cells as missing data, not as zero signal. Honest research-in-progress state.
9. **Run-everything doctrine.** Every IDAT runs through Stage 0 → Stage 10 with the FULL deconvolution and the FULL per-class + per-cell-type A-score computation, regardless of any single stage result. No gating on "first signal that crosses threshold" — compute every measurement, let the anomaly stack tell the story.
10. **Disease severity class describes disease state at row's phase, NOT customer match.** The matrix's severity_class column is a property of the (disease, phase, substrate) tuple — not a property of the customer. The customer's tier is computed at runtime from match × phase × severity × evidence.
11. **No retroactive flags on prior VAL readings when interpretation re-frames are issued.** Re-frames apply only to forward-looking card preregs. Prior VAL findings remain valid under the documented historical interpretation.
12. **Bidirectional grouping rule** (mostly a website rule, but the orchestrator's Stage 5 output must surface bidirectional patterns so the doctor report and future website can present them correctly). Cells that move in opposite directions across diseases get separated in the output structure.
13. **Follow the biology, not the class** (also primarily a website rule, but the orchestrator's per-cell-type output must be returned in the natural biological grouping the website expects — by cell-type group for immune/progenitor/terminal/stromal; by organ for cycling/secretory).

### 3.8 Chain-of-custody L1–L9 overlay — discipline frame

The 9-link chain (borrowed from CMB cosmology) is the audit discipline that overlays the 11-stage pipeline. The orchestrator implements the FILLED links; declares the EMPTY links honestly; never pretends to do inference it isn't doing.

| Chain link | Status in V1 | Where in `walther_clinical.py` |
|---|---|---|
| **L1** Detector timestreams | FILLED | Stage 0 — IDAT ingestion + SHA |
| **L2** Calibration | FILLED | Stage 0 / Stage 1 — dye bias, probe-type norm, BS efficiency, ComBat skipped for single-patient |
| **L3** Map-making | FILLED | Stage 1 — β = M / (M + U + 100) |
| **L4** Component separation | FILLED | Stage 2 (Walther + NILC v2) + Stage 3 (age-axis foreground subtraction) |
| **L5** Correlation structure | EMPTY in V1 | Currently empty in pipeline. TODO 2.1 (C(d)), 2.2 (bispectrum), 2.3 (banana degeneracies). Declared empty, not faked. |
| **L6** Covariance modeling | FILLED via Mahalanobis hull | Stage 5 (Mahalanobis distance against `mahalanobis_healthy_reference_v0_3.json`'s pooled HC covariance n=1,721, Ledoit-Wolf shrinkage, percentile-calibrated Route A threshold) |
| **L7** Likelihood construction | EMPTY in V1 | Per-card Bayesian likelihood is a Phase E deliverable. Not in V1. |
| **L8** Parameter inference | EMPTY in V1 | MCMC posteriors per card — Phase E. Not in V1. |
| **L9** Null suite + end-to-end sims | FILLED for sealed-VAL validation | `cpg_null_runner.py` + `synthetic_patient_generator.py` exist and have processed 7 Family A VALs (5 sealed + 2 RESTATE). NOT invoked per-patient at runtime; runs above the operational flow against sealed VAL artifacts. |

The doctor report (§8) must honestly declare which links contributed to each finding. A breast-epic readout uses L1+L2+L3+L4+L6+L9 — never claim L5/L7/L8 until they're built. This is the chain-of-custody discipline.

---

## 4. Architecture overview

### 4.1 Folder layout

```
<run_folder>/                                  ← copy the contents of
│                                              walther_clinical_runtime/ here
│                                              plus the patient IDAT pair
│
├── walther_clinical.py                        ← the orchestrator (V1 build deliverable)
├── WALTHER_CLINICAL_MANIFEST.json             ← SHA-256 of every file below
│
├── IAMAtlas_REBUILD/
│   ├── IAMAtlasREBUILD.csv.xz                 (100 MB compressed; 577 MB unpacked)
│   ├── IAMAtlasREBUILD_provenance.json        (H_min frozen 2026-04-06)
│   └── IAMAtlasREBUILD_celltype_to_class.json
│
├── Walther_iam_deconvolver/
│   ├── walther_iam_deconvolver.py             (Stage 2 primary)
│   └── walther_iam_deconvolver_README.md
│
├── NILC_Deconvolver/
│   ├── nilc_deconvolver-2.py                  (v2 current — Phase B2.1)
│   ├── nilc_deconvolver.py                    (v1 archival)
│   └── nilc_walther_crosscheck_v2.json        (prior cross-check artifact)
│
├── IAM_Cellular_Age/
│   ├── iam_cellular_age_scoring.py            (Stage 6 scorer — Recipe §6.3)
│   ├── age_axis_foreground.py                 (Stage 3 foreground subtractor)
│   ├── IAMAtlas_age_layer.csv                 (per-CpG α/γ/R²/n at 8,199 CpGs)
│   ├── cellular_ages_v4_epic_italy_validation.csv
│   └── age_layer_diagnostics.json
│
├── A_Scoring_Module/
│   └── iamatlas_a_scoring.py                  (Stage 4 — score_per_class + score_per_celltype)
│
├── Celltype_Marker/
│   ├── iamatlas_celltype_markers_v0_2.json    (CURRENT — 115 ct × top-100)
│   ├── iamatlas_celltype_markers_v0_2.sha256
│   ├── iamatlas_celltype_markers_v0_1_SUPERSEDED.md
│   └── OLD/
│       └── iamatlas_celltype_markers_v0_1.json
│
├── Mahalanobis_healthy_reference/
│   ├── iamatlas_mahalanobis_scoring.py        (Stage 5 — MahalanobisHealthyHull)
│   ├── mahalanobis_healthy_reference_v0_3.json (CURRENT PRODUCTION: HC centroid + Ledoit-Wolf cov, n_hc=1,721)
│   ├── mahalanobis_healthy_reference_v0_2.json (n_hc=1,257, foundation+Hannum — lineage)
│   ├── mahalanobis_healthy_reference_v0_1.json (n_hc=601, foundation-only — lineage)
│   └── mahalanobis_per_patient.csv
│
├── Age_Reference_Matrix_80_cells/
│   ├── age_reference_matrix.json              (Stage 6 reference)
│   ├── age_reference_matrix.csv
│   └── age_reference_matrix.py                (interpolation helpers)
│
├── Tier_breakpoints/
│   └── tier_breakpoints.json                  (Stage 7 — 1.05/1.07/1.10, vocabulary, saturation)
│
├── Cfdna_weight_nonderived_placeholder/
│   └── cfdna_weight.json                      (Stage 7 cfDNA branch — conditional)
│
├── DISEASE_MATRIX/
│   ├── disease_cell_signature_matrix_v1_7.csv (CURRENT — 82 × 131)
│   ├── disease_cell_signature_matrix_engine_schema_v1_2.md  (THE CONTRACT — must read)
│   ├── README_disease_signature_matrix_folder.md
│   └── OLD/disease_cell_signature_matrix_v1_3.csv
│
├── DISEASE_MAPS_CARDS/
│   ├── Breast_EPIC/                            (V1 first card)
│   │   ├── breast_epic_card_json/
│   │   │   ├── breast-epic_card_v2_3.json
│   │   │   └── breast-epic_README.md
│   │   └── breast_epic_residual_maps/
│   │       ├── breast_epic_residual_map_chr_annotated.csv  (1,392 concordant CpGs)
│   │       ├── breast_epic_bimodality_map.csv
│   │       ├── breast_epic_pca_projections.csv
│   │       ├── README_Breast_residual_maps.md
│   │       └── OLD/breast_epic_residual_map_v0_1.csv
│   └── (other cards as V1.x patches as they re-VAL)
│
├── Literature_anchors_Report_building/
│   └── literature_anchors.json                 (Stage 9 — per-class published anchors)
│
├── Cancer_prior/
│   └── cancer_prior.json                       (Stage 9 — US lifetime incidence)
│
├── Family_history_multiplier/
│   └── family_history_multiplier.json          (Stage 9 — conditional)
│
├── CPG_Null_Runner/
│   └── cpg_null_runner.py                      (L9 — not per-patient)
│
├── Synthetic_Patient_Generator/
│   └── synthetic_patient_generator.py          (L9 testing — not per-patient)
│
├── <patient_id>_Grn.idat                       ← patient input — drop here
├── <patient_id>_Red.idat
├── patient_metadata.json                       ← doctor-supplied (V1)
│
└── outputs/                                    ← orchestrator writes here
    ├── <patient_id>_doctor_report.pdf
    ├── <patient_id>_doctor_report.md
    └── <patient_id>_audit_trail.json
```

The folder layout above is **exactly** the structure pushed to repo at `Biological_Physics/atlas_vault/walther_clinical_runtime/`. The orchestrator clones this layout into the doctor's working folder and adds the IDAT pair + metadata + `walther_clinical.py` itself.

### 4.2 Invocation

```bash
cd <run_folder>
python3 walther_clinical.py
```

With explicit arguments when needed:

```bash
python3 walther_clinical.py --idat <prefix> --metadata <path> --output-dir <dir>
```

Auto-detection rules: if no `--idat`, auto-detect the single IDAT pair in the folder (halt if multiple, halt if none). If no `--metadata`, look for `patient_metadata.json`; if absent, prompt interactively for required fields.

No root privileges. No network ports bound. No env vars required. Side effects confined to `outputs/`.

### 4.3 Patient metadata (V1 — full intake schema)

Patient intake is captured per `patient_intake_questionnaire_v1_0.md` (23 questions → 24 covariates). The doctor's folder contains `patient_metadata.json` with the schema below. The orchestrator validates schema at Stage 0 (intake) and routes each covariate to the stage(s) that consume it (see §4.5).

```json
{
  "patient_id": "string — unique identifier the doctor chooses",
  "collection_date": "2026-06-15",
  "ordering_clinician": "Dr. Name",
  "clinical_context": "screening | monitoring | diagnostic_follow_up",

  "// Section A — demographics ////////////////////////////////////////////",
  "chronological_age_years": 47.5,
  "sex_at_birth": "F | M | intersex | prefer_not_to_say",

  "// Section B — smoking history //////////////////////////////////////////",
  "smoking_status": "never | former | current | prefer_not_to_say",
  "smoking_bin": "never | current | former_0_5y | former_5_15y | former_15plus_y",
  "smoking_total_years": "<5 | 5-15 | 15-30 | 30+ | n/a",

  "// Section C — recent illness / immune events //////////////////////////",
  "recent_illness_within_3_months": false,
  "recent_illness_description": "free text or null",
  "recent_vaccination_within_3_months": false,
  "recent_vaccination_description": "free text or null",

  "// Section D — hormonal status ////////////////////////////////////////",
  "current_pregnancy_with_trimester": "none | T1 | T2 | T3 | prefer_not_to_say",
  "postpartum_within_6_months": false,
  "menopause_status": "pre | peri | menopausal | post_surgical | n/a",
  "last_menstrual_period": "YYYY-MM-DD or null",
  "hrt_status": "none | estrogen | estrogen_progesterone | bhrt | testosterone_low_dose | considering_or_recently_stopped",
  "trt_status": "none | current | recently_started | recently_stopped | n/a",

  "// Section E — weight / metabolic / GLP-1 //////////////////////////////",
  "current_glp1_or_weight_loss_medication": "none | current | recently_started | previously_took | considering",
  "current_glp1_medication_name": "free text or null",
  "bariatric_surgery_within_18_months": false,

  "// Section F — autoimmune / inflammatory / immunosuppression ///////////",
  "known_autoimmune_condition": false,
  "known_autoimmune_condition_specifics": ["SLE", "RA", "MS", "T1D", "Hashimoto", "Graves", "psoriatic", "Sjogren", "other"],
  "known_chronic_inflammatory_disease": false,
  "known_chronic_inflammatory_disease_specifics": ["Crohns", "UC", "IBDu", "AS", "other"],
  "current_immunosuppression": false,
  "current_immunosuppression_class": "none | biologic | corticosteroid_long_term | methotrexate | cyclosporine_class | other",

  "// Section G — cancer history / current treatment //////////////////////",
  "current_cancer_in_treatment": false,
  "current_cancer_treatment_type": "none | chemo | immuno | radiation | recent_treatment_within_12mo | recent_surgery",
  "current_cancer_description": "free text or null",
  "prior_cancer_history": false,
  "prior_cancer_history_specifics": "free text or null",
  "prior_chemotherapy_history": "none | within_2y | within_5y",
  "prior_radiation_history_area": "none | head_neck | chest_breast | abdominal_pelvic | bone_marrow_field | other",

  "// Section H — transplant / HIV ///////////////////////////////////////",
  "transplant_status": "none | solid_organ | stem_cell_bm",
  "hiv_status": "negative | positive_treated_suppressed | positive_untreated | recent_diagnosis_unclear | prefer_not_to_say",

  "// Section I — medications + free text ////////////////////////////////",
  "current_medications_systemic": ["array of medication names"],
  "lifestyle_notes": "free text or null",
  "patient_self_reported_concern": "free text or null",

  "// Substrate + platform (assay-level, not customer intake) /////////////",
  "substrate": "epic_blood",
  "platform": "EPIC | 450K",

  "// Family history (optional) ////////////////////////////////////////////",
  "family_history": {
    "breast_cancer_first_degree": false,
    "colorectal_cancer_first_degree": false,
    "alzheimers_first_degree": false
  }
}
```

**Schema validation rules (Stage 0):**
- `patient_id`, `chronological_age_years`, `sex_at_birth`, `substrate`, `platform` are REQUIRED. Missing → halt with explicit error.
- All other fields default to safe values (`false`, `none`, `null`) when missing. Missing optional fields are logged to the audit trail but do NOT halt the chain.
- `smoking_bin` is derived from `smoking_status` + smoking-stop-year when not explicitly supplied. If supplied, must agree with `smoking_status`.
- `hrt_status`, `trt_status`, `current_pregnancy_with_trimester` validated against `sex_at_birth` (HRT applies primarily to F; TRT primarily to M; pregnancy requires F).
- `prior_chemotherapy_history` triggers a Stage 7 warning that the patient's progenitor and immune class A-scores carry persistent chemotherapy footprint.

Eventual production flow (V2+) — patient creates online account, fills the questionnaire UI, orchestrator pulls the validated JSON via API — is **out of V1 scope**. V1 reads `patient_metadata.json` from the doctor's folder.

### 4.4 Substrate handling

Per Heath: keep the 5-substrate ENUM, lock to `epic_blood` for V1.

```python
SUPPORTED_SUBSTRATES = ["epic_blood", "epic_buccal", "epic_saliva",
                        "plasma_cfdna", "canine_blood"]  # confirm exact list against
                                                          # GAPE_WEB_v13.py at build time
ACTIVE_SUBSTRATE_LOCK = "epic_blood"
```

If `patient_metadata.json` requests a substrate other than `epic_blood`, halt with: *"Substrate '<requested>' is in `SUPPORTED_SUBSTRATES` but the V1 release is locked to `epic_blood` until the substrate-specific saturation logic is re-validated against the IAMAtlas REBUILD."*

### 4.5 Conditional consumption — the silent-pass-by discipline

Per walkthrough §3. Several lookup matrices are consumed only when their input data exists. **The engine passes by silently — no error, no degraded output, just no enrichment from that layer.**

| Lookup | Consumed when | Otherwise |
|---|---|---|
| `cfdna_weight.json` | `substrate == "plasma_cfdna"` | Silently skipped |
| `family_history_multiplier.json` | `patient_metadata.family_history` present and non-empty | Silently skipped; Stage 9 uses overall-population prior |
| `literature_anchors.json` (clinician-context section of report) | Always loaded in V1 doctor report (clinician audience) | n/a for V1 |
| per-card residual maps | The card itself fires (i.e., card's gating criteria met) | Card returns NOT_FIRED before residual map loaded |

**Patient intake covariate routing — which covariate enters which stage:**

| Intake field | Used at stage(s) | How it's used |
|---|---|---|
| `chronological_age_years` | Stage 3 (age foreground subtraction), Stage 6 (cellular age delta vs chronological) | `age_axis_foreground.py` subtracts the chronological-age β component before A-score; Stage 6 computes age_delta = cellular_age − chronological_age per class |
| `sex_at_birth` | Stage 3 (sex foreground when module built — v1.1), Stage 7 (sex-stratified threshold tables) | Stage 7 selects sex-stratified threshold table for each card; sex foreground subtraction is v1.1 work |
| `smoking_status`, `smoking_bin` | Stage 3 (smoking foreground when module built — v1.1), Stage 7 (smoking-bin threshold selection) | Stage 7 selects smoking-bin threshold table at runtime; smoking_axis_foreground.py is v1.1 work |
| `recent_illness_within_3_months`, `recent_vaccination_within_3_months` | Stage 9 (report context paragraph) | Immune-class signal interpretation context — surfaces as a caveat paragraph when immune A-score elevated AND recent immune event reported |
| `current_pregnancy_with_trimester`, `postpartum_within_6_months` | Stage 7 (CONTEXT_PREGNANCY / CONTEXT_POSTPARTUM mode override) | Replaces standard 6-tier system with trajectory-mode interpretation |
| `menopause_status`, `last_menstrual_period` | Stage 7 (peri-menopause threshold context), Stage 9 (report context) | Peri-menopause shifts the female immune baseline; engine notes context, doesn't change the math at Stage 7 yet |
| `hrt_status` | Stage 7 (CONTEXT_HRT_BASELINE mode), Stage 9 (report context) | First-of-its-kind HRT-stratified readout per CPG-VAL-018 (in v1.0 immune card set) |
| `trt_status` | Stage 7 (threshold context for male customers) | TRT affects male immune compartment through metabolic/inflammatory pathways |
| `current_glp1_or_weight_loss_medication`, `bariatric_surgery_within_18_months` | Stage 7 (CONTEXT_WEIGHT_LOSS_INTERVENTION mode), Stage 9 (trajectory framing) | Expected anti-inflammatory trajectory per CPG-VAL-021; engine flags expected downward drift |
| `known_autoimmune_condition`, `known_chronic_inflammatory_disease` | Stage 7 (TRAJECTORY_WATCH mode override) | Single-timepoint magnitude de-emphasized; engine emits trajectory-mode interpretation |
| `current_immunosuppression`, `transplant_status` | Stage 7 (EXPECTED_SUPPRESSION mode override) | Suppressed-tier expected; engine emits EXPECTED_SUPPRESSION label rather than SUPPRESSED-with-vigilance |
| `current_cancer_in_treatment`, `current_cancer_treatment_type` | Stage 7 (TREATMENT_RESPONSE mode override), Stage 9 (treatment-trajectory framing) | Treatment-response trajectory reporting across serial timepoints |
| `prior_cancer_history`, `prior_chemotherapy_history`, `prior_radiation_history_area` | Stage 7 (threshold context), Stage 8 (progenitor card carries chemotherapy/radiation footprint caveat) | Stage 7 flags persistent treatment footprint; Stage 8 progenitor card emits a CHEMO_FOOTPRINT or RADIATION_FOOTPRINT note |
| `hiv_status` | Stage 7 (HIV+ treated baseline shift), Stage 9 (report context) | HIV+ treated baseline immune signal known to run at shifted threshold |
| `hpv_status` | Stage 8 cervical card (when built) | Stratifies cervical card's tier interpretation |
| `current_medications_systemic` | Stage 9 (report context — surfaced if unusual reading paired with declared meds) | Doctor-facing context only; engine does not auto-adjust math for medications |
| `lifestyle_notes`, `patient_self_reported_concern` | Stage 9 (free-text passthrough to doctor) | Engine does not parse; passes through to the doctor's report |
| `substrate`, `platform` | Stage 0 (validation), Stage 1 (β computation), all per-card eligibility gates | Drives substrate-conditional matrices + per-card coverage gates |

**The orchestrator never errors on a missing optional input** — it logs the absence in the audit trail and moves on. Every Stage 7 / Stage 8 / Stage 9 consumer of these covariates must handle `null` / missing as "use default behavior."

---

## 5. The chain — stage by stage

Each stage maps to SOP sections. Orchestrator implements in order. After each stage, audit-trail checkpoint.

The walkthrough uses 8 stages (0–7 + 2.5 sub-stage); the SOP uses 11 stages (0–10). They describe the same operational flow with different chunking. The orchestrator uses the SOP's 11-stage chunking internally but maps to the walkthrough's stage numbers for compatibility with the runtime artifacts that name themselves after walkthrough stages.

### Stage 0 — Intake (SOP §11–§19 / walkthrough Stage 0)

1. Locate IDAT pair in run folder.
2. Verify both `_Grn.idat` and `_Red.idat` present; non-zero-byte.
3. SHA-256 each IDAT; record in audit trail.
4. Load patient metadata (file or interactive prompt).
5. Validate metadata schema; substrate lock check (§4.4).
6. Run `methylprep` to extract:
   - Illumina control probes (BS conversion I/II, Specificity I/II, non-polymorphic, negative)
   - Detection p-values per probe
   - Bead count per probe
   - Sample-level call rate
   - Predicted sex from chrX/chrY probe intensities
7. **Platform check (per VAL-091 ad-LL-006):** verify platform from manifest matches `patient_metadata.platform` (450K / EPIC); tag every downstream output with the platform so platform-stratified thresholds can be applied where coverage gaps matter.
8. **HM450 ≥80% coverage threshold:** verify ≥80% of HM450 reference CpGs returned valid β values. Halt below threshold.
9. Apply Stage 0 gate per SOP §19.

**Output of Stage 0:** A loaded `methylprep` sample object plus a Stage 0 verdict dict carried forward to Stage 1.

### Stage 1 — Calibration & β (SOP §20–§27 / walkthrough Stage 0 calibration)

1. Apply dye-bias correction (`methylprep` default).
2. Apply probe-type normalization. For single-patient runs use `noob`; `funnorm` requires cohort context not present in V1.
3. Skip ComBat (single-patient — no cohort to batch-correct against). Document in audit as single-sample limitation.
4. Verify bisulfite conversion efficiency from control probes.
5. Compute β = M / (M + U + 100).
6. β distribution sanity check (bimodal expected; flag if not).

**Output of Stage 1:** Single-patient β vector keyed by CpG ID, ~865,000 CpGs for EPIC (or ~485,000 for 450K).

### Stage 2 — Deconvolution (SOP §28–§34 / walkthrough Stage 1)

1. Decompress `IAMAtlasREBUILD.csv.xz` if not cached; SHA-verify against `IAMAtlasREBUILD_provenance.json`.
2. **Walther IAM Deconvolver (PRIMARY):**
   ```python
   walther = WaltherIAMDeconvolver(
       matrix_path="IAMAtlas_REBUILD/IAMAtlasREBUILD.csv",
       celltype_class_map="IAMAtlas_REBUILD/IAMAtlasREBUILD_celltype_to_class.json"
   )
   walther_result = walther.deconvolve(customer_betas)
   # walther_result.class_fractions     ← PRIMARY: 8 class fractions, RELIABLE
   # walther_result.celltype_fractions  ← SECONDARY: 115 cell-type fractions, INDICATIVE
   # walther_result.diagnostics
   # walther_result.status              ← "OK", "INSUFFICIENT_MARKERS", etc.
   ```
3. **NILC v2 Deconvolver (CROSS-METHOD):**
   ```python
   from nilc_deconvolver_2 import NILCDeconvolver, cross_method_comparison
   nilc = NILCDeconvolver(
       atlas_path=".../IAMAtlasREBUILD.csv",
       marker_path="Celltype_Marker/iamatlas_celltype_markers_v0_2.json",
   )
   nilc_result = nilc.deconvolve(customer_betas)
   ```
4. **Credible intervals propagated from atlas MCMC posteriors.** The deconvolvers do not return point estimates only — atlas posterior SDs propagate through to fraction credible intervals. Report a 0.05 fraction with CI [0.02, 0.08] differently from a 0.05 fraction with CI [0.04, 0.06] in the doctor report's Quality section.
5. **Cross-method gate (SOP §33):** compare Walther vs NILC class fractions. Status OK / FLAG / FAIL. FAIL halts; FLAG annotates uncertainty in audit. Phase B2.1 finding: substrate-level disagreement is expected (median L1 ~0.23); biological-inference-level agreement is the real gate (sign agreement on disease-relevant directions).

**Output of Stage 2:** Walther class + cell-type fractions (PRIMARY); NILC fractions (cross-check); cross-method gate status.

### Stage 3 — Foreground subtraction (SOP §35–§40 / walkthrough Stage 3 first half)

1. Load `IAMAtlas_age_layer.csv` (per-CpG α, γ, R², n at 8,199 CpGs):
   ```python
   from age_axis_foreground import AgeAxisForeground
   afg = AgeAxisForeground()
   afg.load_layer("IAM_Cellular_Age/IAMAtlas_age_layer.csv")
   cleaned_beta = afg.subtract_from(beta_vector, ages=[patient_age])
   ```

2. **Smoking-axis foreground subtraction (NEW v1.2 — module built + layer CSV FIT 2026-06-06 on GSE50660 n=464):**
   ```python
   from smoking_axis_foreground import SmokingAxisForeground
   smk = SmokingAxisForeground()
   smk.load_layer("IAM_Cellular_Age/IAMAtlas_smoking_layer.csv")  # fit 2026-06-06 from GSE50660 n=464
   cleaned_beta = smk.subtract_from(cleaned_beta, smoking_bins=[patient_smoking_bin])
   ```
   Per-CpG model: `β = α + δ·indicator_current + φ·recency_score + ε`. Recency score: never=0.00 / former_15plus_y=0.10 / former_5_15y=0.30 / former_0_5y=0.60 / current=1.00. **Until `IAMAtlas_smoking_layer.csv` is fit (v1.3 layer-build work on the n_hc=601 cohort with smoking-status metadata), the interim Stage 7 smoking-bin threshold-stratification (per `tier_breakpoints.json v1.2`) absorbs the bulk effect.**

3. **Sex-axis foreground subtraction (NEW v1.2 — module built + layer CSV FIT 2026-06-06 on GSE50660 n=464):**
   ```python
   from sex_axis_foreground import SexAxisForeground
   sex_fg = SexAxisForeground()
   sex_fg.load_layer("IAM_Cellular_Age/IAMAtlas_sex_layer.csv")  # fit 2026-06-06 from GSE50660 n=464
   cleaned_beta = sex_fg.subtract_from(cleaned_beta, sex_at_birth=[patient_sex])
   ```
   Per-CpG model: `β = α + ψ·indicator_male + ε`. Special handling of chrX (X-inactivation flag for high-ψ CpGs) + chrY (sex-chromosome flag for masking in female samples). **Until `IAMAtlas_sex_layer.csv` is fit (v1.3 layer-build work), the interim Stage 7 sex-stratified threshold tables absorb the bulk effect.**

4. **v1.2 documented gaps:** batch / ancestry foregrounds are NOT yet subtracted at the CpG level (modules not built). Audit trail declares the gap honestly; doctor report's Quality section lists them as documented limitations. Batch correction is typically handled at the cohort level (ComBat/funnorm) in pre-processing, so its absence at L4 per-patient runtime is less critical than smoking/sex/age.

**Output of Stage 3:** Foreground-cleaned β vector (age + smoking + sex once layers fit; age-only as v1.2 default until smoking/sex layer-build complete at v1.3).

### Stage 4 — A-score (SOP §41–§46 / walkthrough Stage 2)

1. Load marker artifact + H_min:
   ```python
   from iamatlas_a_scoring import load_artifact, score_per_class, score_per_celltype
   meta, markers_by_celltype, ct_to_class, h_min_by_class = load_artifact(
       "Celltype_Marker/iamatlas_celltype_markers_v0_2.json"
   )
   ```
2. Score per-class (8 architectural classes):
   ```python
   # Build per-class marker lists from per-cell-type markers
   class_markers = aggregate_markers_to_class(markers_by_celltype, ct_to_class)
   class_ascores = score_per_class(cleaned_beta, class_markers, h_min_by_class)
   ```
3. Score per-cell-type (115 cell types):
   ```python
   celltype_ascores = score_per_celltype(
       cleaned_beta, markers_by_celltype, ct_to_class, h_min_by_class
   )
   ```
4. Each output is a `{name: {A, n_markers_expected, n_markers_matched, coverage, confidence, status}}` dict where status is OK / INSUFFICIENT_MARKERS / MARGINAL_COVERAGE / NO_MARKER_OVERLAP.

**Validation anchors** (per cell-type extension formalized 2026-05-29): Basophils d=+1.577 (GSE51057), Plasma cells d=+1.264, Microglia d=+1.304, breast-epithelial (BE) d=+1.281 — sub-cellular signals neither the 8-class output nor the pre-existing Loyfer 25-tile output could resolve.

**Forward CI propagation (v1.2):** Each per-class and per-cell-type A-score carries a 95% credible interval propagated from atlas MCMC posterior SDs via Monte Carlo (1000 draws from per-CpG posteriors, A computed per draw). Customer-facing output is "your immune A-score is 1.08 (measurement range 1.06–1.10)" not "your immune A-score is 1.08." This is the orchestrator's mechanism for honoring the 3-week MCMC compute investment — the posterior uncertainty does not get discarded at Stage 4.

**Output of Stage 4:** 8 per-class A-scores (with 95% CI) + 115 per-cell-type A-scores (with 95% CI), each with status.

### Stage 4.5 — Bidirectional decomposition (SOP v1.3 §-section / NEW in v1.2)

**Rationale.** Per VAL-050 (pooled-entropy NULL d=+0.077) → VAL-051 (directional composite d=+0.624 same cohort): pooled-entropy A-score CANCELS when bidirectional patterns are present. Pooled β_mean barely moves because some CpGs go up while others go down. The directional weighted composite z-score recovers the signal. At patient runtime, the engine MUST run the directional decomposition autonomously — every VAL has a PREREG specifying direction, but patient runtime has none.

**Algorithm — mirrors the sealed VAL-051 `a_dir_score` formula exactly:**

For each panel CpG: compute z = (β_patient − μ_hc_train) / σ_hc_train. Multiply z by the CpG's frozen direction sign (+1 disease-up, −1 disease-down). Average across covered CpGs. Single signed composite per panel.

```python
from bidirectional_decomposition import (
    load_directional_panels,
    compute_per_class_bidirectional_decomposition,
    save_bidirectional_report,
)

# Per-class sealed directional panels — directions + training-set HC stats frozen
# at VAL training time. v1.0 immune panel = VAL-051 Rule A 7-CpG AD-direction-anchored;
# other 7 classes pending future sealed VALs.
panels = load_directional_panels(
    "Bidirectional_Decomposition/directional_panels_v1_0.json"
)

report = compute_per_class_bidirectional_decomposition(
    patient_beta=cleaned_beta,                # Stage 3 output (foreground-cleaned)
    panels=panels,
    patient_id=patient_metadata["patient_id"],
)

# report.per_class_results[cls] carries:
#   - a_pooled_entropy:           Stage 4 pooled-entropy A on the parent panel (the null comparator)
#   - a_directional_composite:    the SEALED VAL-051 a_dir_score signed composite
#   - flag_bidirectional:         True when pooled is mute AND directional is loud
#   - flag_insufficient_coverage: True when <70% panel CpGs present in patient β
```

**Flag rule:** `FLAG_BIDIRECTIONAL = (|a_pooled − 1.0| < 0.05) AND (|a_directional_composite| > 0.40)`. The first clause says "pooled is at-baseline / mute"; the second says "directional is loud." Both required.

**Coverage gate:** Mirrors `val051_analyze.py:120` — `n_covered >= max(3, int(0.7 * n_panel))`. Below the gate, `a_directional_composite = None` and `flag_insufficient_coverage = True`. The orchestrator interprets None as INSUFFICIENT_COVERAGE for the directional read; pooled-entropy A from Stage 4 is still valid.

**When flagged:** Stage 7 reports the directional composite (signed, magnitude) as the customer-facing tier driver — NOT the pooled A. The Mahalanobis at Stage 5 also runs against the directional-decomposition vector (per-class composite as one axis) in addition to the standard 115-cell A-score vector. Stage 8 Route C activates per the immune card.

**v1.0 panel coverage:** immune class only (VAL-051 Rule A, 7 CpGs: 2 positive + 5 negative, all AD-direction-anchored). Other 7 classes return `NO_PANEL` until future sealed VALs populate them. This is **declared honestly** in the panel JSON's `_panel_pending_note`. Future expansion via CPG-VAL-019 (cancer-positive vs AD-negative direction discrimination, in v1.0 VAL set) broadens the immune-class panel beyond AD-direction-only.

**Output of Stage 4.5:** Per-class `BidirectionalResult` (n_panel_cpgs, n_covered, coverage_fraction, a_pooled_entropy, a_directional_composite, flag_bidirectional, flag_insufficient_coverage, interpretation string); engine-internal directional anchor for downstream Stage 5/7/8 consumption.

### Stage 4.6 — Per-class healthy brightness comparison + patient Mollweide projection (SOP v1.3 §-section / NEW in v1.2)

**Rationale.** The IAMAtlas REBUILD MCMC produced per-CpG, per-class posterior mean + SD across 481,966 CpGs (the brightness CSVs inside `IAMAtlas_v0_1/class_archives/*.tar.xz`). These ARE the per-class healthy reference — the data behind Plate 1, the Cosmic Microwave Methylome. Patient runtime SHOULD consult this reference: compute per-CpG z-score of patient β versus each class's healthy posterior, then project the result onto the same HEALPix grid as Plate 1. The customer's personal CMM (8-panel Mollweide) becomes the visualization endpoint of the report.

This stage was missed in v1.0/v1.1 of the build spec — the brightness data was treated as a build-time artifact only. **It is also a runtime reference.** v1.2 corrects this.

**Algorithm:**

```python
from patient_brightness_comparison import (
    load_all_8_class_references,
    compute_all_8_class_departures,
    render_patient_cosmic_methylome,
    save_brightness_report,
)
import numpy as np

# Load the 8 per-class brightness references (one CSV per class — inside class archive tar.xz).
# Each carries per-CpG mean β + posterior SD + 95% CI across 483,093 CpGs.
references = load_all_8_class_references(
    "Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives"
)

# Compute per-class per-CpG z-score departure for the patient.
brightness_report = compute_all_8_class_departures(
    patient_beta=cleaned_beta,                     # Stage 3 output
    references=references,
    patient_id=patient_metadata["patient_id"],
)

# Project onto the same HEALPix NSIDE=128 Mollweide grid as Plate 1.
cpg_to_pixel = np.load("Brightness_Comparison/iamatlas_cpg_to_healpix_nside128.npy")
cmm_png = render_patient_cosmic_methylome(
    brightness_report, cpg_to_pixel,
    out_path=f"reports/{patient_id}_cosmic_methylome.png",
)

# Persist per-class z-score CSVs + summary JSON for audit trail.
save_brightness_report(brightness_report, out_dir=f"reports/{patient_id}/brightness/")
```

**Reference convention:** HEALPix NSIDE=128, Mollweide projection, CpGs ordered by chromosome × MAPINFO with sequential pixel assignment in genomic order, multi-CpG-per-pixel averaging. Diverging RdBu_r colormap centered at z=0, range [-3, +3]. Masked CpGs (σ < 1e-4 OR stromal galactic mask) render BLACK. **Exactly mirrors Plate 1 conventions** so per-patient projections sit on the same grid as the reference plates at `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/`.

**Output of Stage 4.6:**
1. `PatientBrightnessReport` — per-class z-score arrays + summary statistics (mean |z|, max |z|, n_notable, n_extreme, top-100 outlier CpGs)
2. `{patient_id}_brightness_comparison_summary.json` — engine-internal summary
3. `{patient_id}_{class}_z_scores.csv` — per-class z-score CSV (one per class, 8 total)
4. `{patient_id}_cosmic_methylome.png` — 8-panel Mollweide PNG (the customer-facing visualization endpoint)

**Engine-internal use:** Stage 5 Mahalanobis consumes the z-score vector as a complement to the cell-type A-score vector for Mahalanobis distance; Stage 7 tier-call consults the max-|z| per class as a per-cell-resolution check on the pooled tier; Stage 9 report builder embeds the patient's CMM PNG in the report.

### Stage 5 — Mahalanobis (SOP §47–§51 / walkthrough Stage 2.5)

The HEADLINE number on every doctor report.

```python
from iamatlas_mahalanobis_scoring import MahalanobisHealthyHull
# Production loads CURRENT version — currently v0_3 (n=1,721 pooled HC).
hull = MahalanobisHealthyHull("Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v0_3.json")
maha_result = hull.score(celltype_ascores)
# maha_result["mahalanobis_distance"]       ← the headline number
# maha_result["top10_axis_contributions"]   ← which cell-types drove the distance
# maha_result["status"]                     ← OK / PARTIAL_DATA
# maha_result["n_features_imputed"]         ← imputation count for QC
# maha_result["reference_anchor"]           ← validation context
```

**Route A trigger threshold is percentile-based, NOT a fixed value.** With 112 features under multivariate normality, expected median Mahalanobis distance is √112 ≈ 10.58 — a fixed threshold like `d ≥ 2.0` would fire on ALL samples (the v0_1 mistake). The current v0_3 thresholds are:
- p95 (default Route A): d ≥ 13.54
- p99 (strict): d ≥ 18.71
calibrated against the pooled n=1,721 HC distance distribution stored in the artifact under `route_A_calibration_v0_3`. Production engine reads the threshold from the artifact at session startup — not hard-coded in any card or module.

**Validation anchor lineage** (preserved as HC hull expands):
- Breast pre-dx Cohen's d (GSE51057 n=11): v0_1 +1.871 → v0_2 +0.981 → v0_3 +0.896
- Breast pre-dx Cohen's d (GSE51032 n=36): v0_1 +2.088 → v0_2 +1.653 → v0_3 +1.611
- Case detection % at p95 threshold (GSE51032): v0_2 50.0% → v0_3 55.6% (improves with broader HC representation)
- Case detection % at p95 threshold (GSE51057): v0_2 9.1% → v0_3 27.3% (small n=11)

Imputation rules per SOP §47: HARD-fail if >15 cell types imputed; SOFT-flag at 6–15.

**Output of Stage 5:** One Mahalanobis distance, top-10 axis decomposition, imputation count, validation anchor.

### Stage 5.1 — Mahalanobis HC hull versioning protocol (NEW v1.2 patch 2026-06-06)

The Mahalanobis hull is the only chain element with cohort-empirical content (centroid + covariance must be MEASURED from HC samples, not physics-derived). The hull is versioned as new HC cohorts are added; each version is frozen for production deployment.

**Versioning rules:**
- Build versions never rebuild on patient β. At patient runtime, the chain queries the FROZEN current production version.
- Each version is named `mahalanobis_healthy_reference_v0_N.json` with full provenance (cohort sources, SHA-256 of each input CSV, Ledoit-Wolf shrinkage parameter, percentile thresholds, case-discrimination lineage).
- Prior versions are retained in the same folder for lineage traceability — never deleted.
- The `route_A_calibration_v0_N` block holds the percentile distribution thresholds. Engine reads default and strict thresholds from this block at session startup.

**Phase planning** (no fixed N — phases extend HC representation along one dimension at a time):
- **Phase 1 ✓** (2026-06-06): v0_1 → v0_2 by adding Hannum GSE40279 n=656. Brings +full age span (40-65 → 19-101), +mixed sex, +US population.
- **Phase 2 ✓** (2026-06-06): v0_2 → v0_3 by adding Tsaprouni GSE50660 n=464. Brings +UK population, +smoking-stratified covariate.
- **Phase 3** (queued): Add EPIC platform HC cohort for cross-platform transferability (candidates: AIBL HC n=471 if full β accessible; AddNeuroMed HC; GIFT HC).
- **Phase 4** (queued): Add Asian-population HC cohort (currently a gap).
- **Phase N**: Routine maintenance as research surfaces new cohorts.

**Build protocol (operator workflow):**
1. Acquire the new HC cohort's β matrix; verify all-HC composition.
2. Run canonical 115-cell A-scoring on the cohort via `score_per_celltype` against `iamatlas_celltype_markers_v0_2.json`. Save as `GSE{ID}_115celltype_ascores.csv` in `validation_runs/hull_expansion_phaseN_GSE{ID}/`.
3. Pool with the current `v0_N` hull's sample-level inputs (per-cohort CSVs are preserved at sample level alongside the cohort manifests).
4. Compute new centroid + Ledoit-Wolf shrunk covariance from the pooled `M × 112` matrix.
5. Recompute percentile thresholds (p95, p99) under the new HC distance distribution.
6. Re-validate against the breast pre-dx anchor (GSE51057 + GSE51032 case-vs-HC Cohen's d).
7. Save as `mahalanobis_healthy_reference_v0_(N+1).json` with full provenance + supersession block.
8. Update BUILD_SPEC + SOP + Evidence Report + VAL Inventory to reference new version.

**Cards never carry hull-specific runtime data.** Each disease card (and the immune universal card) only references the artifact path. The artifact carries the data. This separation ensures hull expansion does not require touching any card.

### Stage 6 — Cellular age inversion (SOP §52–§58 / walkthrough Stage 3 second half)

```python
from iam_cellular_age_scoring import IAMCellularAge
ca = IAMCellularAge(
    ref_matrix_path="Age_Reference_Matrix_80_cells/age_reference_matrix.json",
    markers_artifact_path="Celltype_Marker/iamatlas_celltype_markers_v0_2.json"
)
age_result = ca.score(
    patient_betas=cleaned_beta,
    chronological_age=patient_age,
    patient_id=patient_id
)
# age_result is a CellularAgeResult dataclass with:
#   - cellular_age_per_class: 8 per-class ages
#   - status_per_class:        OK / SATURATED_HIGH / SATURATED_LOW / INSUFFICIENT_CPGS
#   - a_score_per_class
#   - summary_age:             n_samples-weighted mean across non-saturated classes
#   - accelerated / decelerated / concordant: vs chronological age
#   - age_spread, age_median, age_iqr
#   - overall status:          OK / OK_PARTIAL / OK_LIMITED / ALL_SATURATED_OR_INSUFFICIENT
```

Saturation is data, not error. Report all 8 per-class ages with their saturation flags. The 80-cell baseline is calibrated on ages 4–95 per class.

**Output of Stage 6:** 8 per-class cellular ages with status, summary age, concordance structure, overall status.

### Stage 7 — Tier breakpoints (SOP §59–§64 / walkthrough Stage 4) — **6-TIER PHYSICS-DERIVED in v1.2**

1. Load `tier_breakpoints.json` (v1.2 schema with 6-tier physics-derived breakpoints + special-mode overrides).

2. Apply the **6-tier physics-derived breakpoints** to each per-class A-score. These are NOT statistical percentiles — they are metabolic-transition inflection points from Heath's calibration:

   | Range | Customer-facing tier | Physics meaning |
   |---|---|---|
   | A < 0.95 | **SUPPRESSED** | Measurable shift below the healthy class baseline; treatment-context-dependent reading |
   | 0.95 ≤ A < 1.04 | **NORMAL** | Within healthy sampling variance |
   | 1.04 ≤ A < 1.07 | **ELEVATED** | Recoverable drift, architecture intact, holistic-intervention window |
   | 1.07 ≤ A < 1.10 | **WARBURG_TRANSITION** | The 1.07 Warburg line — metabolic point where adding fuel can accelerate decline; intervention character must change |
   | 1.10 ≤ A < 1.12 | **SIGNIFICANTLY_ELEVATED** | Structural-fidelity breach territory; rare senescence-without-cancer regime |
   | A ≥ 1.10 sustained OR A ≥ 1.12 single timepoint | **BREACH** | Regime where diagnosed cancer is typically observed; prompt for clinical workup, not a verdict |

   **The 1.07 Warburg line and the 1.10 breach line are the framework's two physics-defined inflection points.** All cards inherit this 6-tier system; per-card threshold overrides (smoking-bin, sex-stratified, HIV-baseline-shifted) shift the floor of ELEVATED but preserve the Warburg line.

3. Apply **CI-aware tier confidence**. From the per-class A-score CI propagated through Stage 4:
   ```python
   tier_confidence_prob = {
       tier: P(A in tier_range | A_posterior_distribution)
       for tier in SIX_TIERS
   }
   # Customer sees primary_tier = argmax(tier_confidence_prob)
   # Engine notes when |max_prob - second_max_prob| < 0.20 → BORDERLINE_TIER flag
   ```
   This is how the engine handles "patient at A=1.08 [CI 1.06–1.10]" — straddles the 1.07 Warburg line. The tier_confidence_prob captures the straddling, and the customer's report says so explicitly.

4. Apply **special-mode overrides** per the patient intake covariates (per §4.5 routing table):

   | Trigger covariate | Mode | Effect on Stage 7 output |
   |---|---|---|
   | `current_immunosuppression == true` OR `transplant_status != "none"` | **EXPECTED_SUPPRESSION** | Engine emits EXPECTED_SUPPRESSION label rather than SUPPRESSED-with-vigilance |
   | `known_autoimmune_condition == true` OR `known_chronic_inflammatory_disease == true` | **TRAJECTORY_WATCH** | Single-timepoint magnitude de-emphasized; trajectory direction is the primary reading |
   | `current_cancer_in_treatment == true` | **TREATMENT_RESPONSE** | Trajectory across treatment timepoints reported instead of single-timepoint tier |
   | `current_pregnancy_with_trimester != "none"` | **CONTEXT_PREGNANCY** | Physiological pregnancy immune shift flagged; trajectory across pregnancy timepoints |
   | `postpartum_within_6_months == true` | **CONTEXT_POSTPARTUM** | Physiological postpartum immune shift flagged |
   | `hrt_status` in {estrogen, estrogen_progesterone, bhrt, testosterone_low_dose} | **CONTEXT_HRT_BASELINE** | First-of-its-kind HRT-stratified readout (VAL-018 anchor) |
   | `current_glp1_or_weight_loss_medication == "current"` OR `bariatric_surgery_within_18_months == true` | **CONTEXT_WEIGHT_LOSS_INTERVENTION** | Expected downward immune trajectory; flagged for trajectory monitoring |
   | `hiv_status == "positive_treated_suppressed"` | **TRAJECTORY_WATCH** with shifted ELEVATED floor (1.06 → 1.10 for HIV+ treated baseline) |

5. Apply **smoking-bin threshold stratification** (interim mitigation until smoking_axis_foreground.py is built at v1.3):

   | smoking_bin | ELEVATED floor |
   |---|---|
   | `current_smoker` | 1.10 (residual smoking signal absorbed) |
   | `former_0_5y` | 1.08 |
   | `former_5_15y` | 1.07 |
   | `former_15plus_y` | 1.05 |
   | `never_smoker` | 1.04 (default) |

   When smoking_axis_foreground.py is built at v1.3, these bin-based threshold shifts retire and full per-CpG subtraction replaces them.

6. **Bidirectional flag handoff** (consume Stage 4.5 output): When `FLAG_BIDIRECTIONAL == true` for a class, the customer-facing tier is determined by `max(A_positive_panel, |A_negative_panel|)` against the 6-tier breakpoints, with bidirectional-pattern qualifier appended. When `max(|directional_d|) > 1.0`, bidirectional pattern fires at SIGNIFICANTLY_ELEVATED even if pooled A would be NORMAL (the VAL-051 lesson at runtime).

7. If substrate is `plasma_cfdna` (NOT in V1 lock — but the code path exists for future unlock): apply `cfdna_weight.json` to compute expected-vs-observed per-class fractions. Surface significant departures.

**Output of Stage 7:** Per-class 6-tier classification (engine + customer), tier confidence probability vector, special-mode active flags, smoking-bin context, bidirectional flag handoff, cfDNA context (when applicable).

### Stage 8 — DUAL MATCHING (SOP §65–§69 / walkthrough Stage 5)

**Critical: the two matching paths run in PARALLEL, not sequentially. They are complementary, not redundant.**

```python
# ── PATH A: Per-card matching ───────────────────────────────────
card_verdicts = {}
for card_id in available_cards:  # V1: just breast-epic
    card_dir = f"DISEASE_MAPS_CARDS/{card_id}/"
    card_json = load(f"{card_dir}{card_id}_card_json/{card_id}_card_v*.json")
    residual_map = pd.read_csv(f"{card_dir}{card_id}_residual_maps/{card_id}_residual_map_chr_annotated.csv")
    bimodality_map = pd.read_csv(f"{card_dir}{card_id}_residual_maps/{card_id}_bimodality_map.csv")
    pca_projections = pd.read_csv(f"{card_dir}{card_id}_residual_maps/{card_id}_pca_projections.csv")

    # 1. Filter by tissue applicability, age, sex, CpG availability
    if not card_eligible(card_json, patient_metadata):
        card_verdicts[card_id] = NOT_ELIGIBLE
        continue

    # 2. Evaluate threshold ranges vs customer A-scores
    # 3. Apply covariate-keyed thresholds (HCC viral hep status, lung smoking, breast pre/post-menopausal)
    # 4. Compute residual-overlap matched-filter score against concordant CpGs
    # 5. Compute bimodality loss score
    # 6. Compute PCA projection score
    # 7. Aggregate to per-card verdict: tier + confidence + contributing pattern + educational_page_url

    card_verdicts[card_id] = evaluate_card(...)

# ── PATH B: Disease matrix lookup ───────────────────────────────
matrix = pd.read_csv("DISEASE_MATRIX/disease_cell_signature_matrix_v1_7.csv")
schema = load_schema("DISEASE_MATRIX/disease_cell_signature_matrix_engine_schema_v1_2.md")
# Per the schema's compute_match_magnitude() — Mahalanobis-style sign-aligned product
# weighted by sqrt(n), NOT raw dot product, NOT Euclidean
matrix["match"] = matrix.apply(
    lambda row: compute_match_magnitude(celltype_ascores, row, schema),
    axis=1
)
top_candidates = matrix.nlargest(3, "match")
# Per candidate: tier from compute_customer_tier(match × phase × severity × evidence)
for _, candidate in top_candidates.iterrows():
    tier = compute_customer_tier(
        candidate["match"],
        candidate["disease_severity_class"],   # NB: row property, NOT customer match
        candidate["phase"],
        candidate["evidence_anchors"]
    )
    organ_pages = candidate["organ_pages_to_link"]  # for V2 deep linking

stage_5_output = {
    "card_verdicts": card_verdicts,      # path A
    "matrix_candidates": top_candidates, # path B
}
```

**Schema rules from `disease_cell_signature_matrix_engine_schema_v1_2.md`** (read it before coding the match):
- Cell values can be float (e.g., `+1.26`), range (`+0.5/+1.0` is a magnitude RANGE not a fraction), or directional (`↑↑` is "directional only, magnitude pending").
- Empty cell = "no documented signature" (NOT zero). Match function must skip blanks, not treat as zero.
- Match algorithm: Mahalanobis-style sign-aligned product weighted by sqrt(n). Not raw dot product. Not Euclidean.
- `disease_severity_class` is a property of the (disease, phase, substrate) tuple — NOT of the customer. The customer's tier is computed at runtime from match × phase × severity × evidence.

**Worked example of why dual matching catches what cards miss** (walkthrough §4 Stage 5): A patient with `regulatory_T_cells +1.2 + erythroid_progenitor +0.8 + pancreatic_beta_cells +1.0 + multi-organ distributed elevation`:
- Path A (card matching): NO single card fires above ELEVATED (no card uses that exact combination)
- Path B (matrix lookup): `breast_cancer / long_pre_dx` strongest pattern match (the >10yr distributed signature with 7 cells contributing)

Without the matrix, customers in pre-diagnostic windows produce "everything looks slightly off, nothing fires" reports that miss the actual pattern. With the matrix, the report can say "this combination of cellular drift most resembles the pattern documented for [X] at [phase]."

**Output of Stage 8:** Two parallel result blocks — card verdicts list + matrix top-3 candidates with tier and organ pages.

### Stage 9 — Report assembly (SOP §70–§76 / walkthrough Stage 6) — DOCTOR REPORT ONLY in V1

1. Load context lookups:
   - `literature_anchors.json` — published per-class anchors
   - `cancer_prior.json` — US lifetime baseline per class
   - `family_history_multiplier.json` — first-degree-relative RR (conditional on family history present)
2. **Risk-context formula** (walkthrough §4 Stage 6):
   ```
   posterior_context_class = baseline_prior
                           × age_factor
                           × sex_factor                      // secretory: female 1.4× (breast dominates), male 1.2× (prostate)
                           × fh_factor                       // from family_history_multiplier.json, if present; else 1.0
                           × match_magnitude                 // from Stage 8 matrix match
   ```
   Framing rule: "your reading combined with your risk context suggests…" — NEVER "you have a high probability of…". This is risk-context, not diagnosis.
3. **Deep-link routing for V2 prep:** capture `card.educational_page_url` and `matrix.organ_pages_to_link` in the audit trail. V1 doctor report does not render these as live links (doctor is reading a PDF) but the audit trail preserves them for V2 customer-report builder to consume.
4. Apply language collapse from engine tiers to customer-facing labels per `tier_breakpoints.json`.
5. Assemble doctor report — see §8 below for content.
6. Run legal-boundary gate per SOP §76. CANNOT_SAY list catches diagnostic phrasing.

**Output of Stage 9:** Cleared doctor report in Markdown, passing legal-boundary gate.

### Stage 10 — Delivery (SOP §77–§79 / walkthrough Stage 7)

1. Render Markdown to PDF (`pandoc` preferred, `reportlab` fallback).
2. Write three files to `outputs/`:
   - `<patient_id>_doctor_report.pdf`
   - `<patient_id>_doctor_report.md`
   - `<patient_id>_audit_trail.json`
3. Print terminal summary: patient ID, top 3 card verdicts (or "no cards fired"), Mahalanobis distance with HC percentile, summary cellular age, path to report PDF.

**Output of Stage 10:** Three files in `outputs/`; one terminal summary.

---

## 6. What to lift from `GAPE_WEB_v13.py`

Real and reusable blocks. Lift verbatim or with minimal adaptation; **but rename any variable / function / docstring that contains physics terminology** (per non-negotiable rule §3.7.3 — no Boltzmann, Landauer, Arrhenius, decoherence, k_B, ln2, "Mahaffey Number" in commercial code).

| What | Lines | Why useful in `walther_clinical.py` | Rename rule |
|---|---|---|---|
| CORE CONSTANTS block | ~40–60 | T_body, R, ΔG_ATP, N_CpG, ln2 — useful internally but never expose | Rename to neutral identifiers (`_T_BODY`, `_R_GAS`, `_DELTA_G_ATP`, `_N_CPG`, `_LN2` → `_BIO_CONSTANT_1`, `_BIO_CONSTANT_2`...). Strip docstrings. |
| Architecture Registry | line 122 onward | 8-class metadata, names, descriptive language | Keep names; strip any physics docstrings |
| Cancer Validation Database (G-008) | line 273 onward | 27 TCGA cancer types — report context | Keep |
| Published Reference Anchors | line 309 onward | Already extracted to `literature_anchors.json` — verify equivalence; if drift detected, consume the standalone JSON | n/a |
| `_mahaffey_number()` | line 592 | The dimensionless ratio formula | **DO NOT lift by name.** Rename function. Strip "Mahaffey" from variable names. If the ratio is needed for internal scaling, compute it inline with neutral names — do NOT call it the "Mahaffey Number" anywhere customer-visible. |
| `_clinical_interpretation()` | line 699 | Language collapse from A-score to clinical text | Lift; review docstrings for physics terms |
| Saturation helpers (`_concordance`, `_fidelity_tier`, etc.) | 484–592 | Useful for Stage 5/7 | Lift |
| Engine E1 (`run_e1_position`) | line 639 | Position interpretation for doctor report | Lift |
| Engine E6 (`run_e6_cohort`) | line 1250 | Cohort context for doctor report | Lift |
| Engine E7 (`run_e7_literature`) | line 1333 | Literature anchor for doctor report | Lift (overlaps with `literature_anchors.json` — consume the JSON; the engine logic is the routing layer) |

E2 (Risk), E3 (Serial), E4 (Pan-Tissue), E5 (Target Solver) — deferred to V1.x+ patches as we learn what the doctor wants.

## 7. What NOT to lift from `GAPE_WEB_v13.py`

- Flask layer (lines ~6198+) — `@app.route` definitions
- Login/auth — `_auth_check`, `login`, `logout`
- HTML rendering — Jinja templates inline ~1919–6195
- `_age_ref_A()` and `_age_ref_beta()` functions — the OLD age atlas. Replace with the canonical `age_reference_matrix.json` + `iam_cellular_age_scoring.py`.
- The `_AGE_REFERENCE` dict — same OLD age atlas
- `_load_atlas_vault()` and the entire atlas vault block (lines 1640–1815) — V1 uses IAMAtlas only; Loyfer / Salas / EpiSCORE / Caggiano / UniLIFE are NOT in the production pipeline (non-negotiable rule §3.7.2)
- Substrate-specific saturation per-substrate logic — keep the substrate ENUM; drop the per-substrate saturation curves until each substrate is re-validated against the new atlas
- Per-substrate H_min variations — V1 uses the 8 frozen H_min values from `IAMAtlasREBUILD_provenance.json`, applied uniformly to `epic_blood`

---

## 8. Doctor report — V1 content

V1 ships ONE report type: the doctor report. Markdown rendered to PDF.

**Audience:** clinician. The doctor is reading this to inform her clinical decisions with her patient.

**Length target:** 4–8 pages. Compact enough to read pre-appointment; deep enough to support discussion.

**Sections, in order:**

1. **Header.** Patient ID, collection date, ordering clinician, substrate, platform, IDAT SHA-256, atlas version, orchestrator version, generation timestamp.

2. **Executive summary** (1 paragraph).
   - The headline: **Mahalanobis distance from healthy reference** with HC percentile.
   - The top finding: highest-confidence card verdict if any FIRED; or top-1 matrix candidate if any matched above threshold; or "all measurements within healthy reference range" otherwise.
   - The summary cellular age.

3. **Per-class panel.** Table of 8 architectural classes: name, A-score, customer-facing tier (SUPPRESSED / NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED) + engine tier (BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / BREACH for clinician detail), cellular age, saturation status. One-sentence interpretation per class.

4. **Mahalanobis distance detail.** The headline number with HC-percentile context. Top-3 contributing axes named explicitly (which cell types drove the distance — e.g., "Microglia z=+2.1, Plasma cells z=+1.8, Basophils z=+1.6"). Cohort context: "Comparable to [literature_anchor]'s reading for [condition]" where applicable.

5. **Card verdicts (Path A of dual matching).** For each card that FIRED: card name, verdict, confidence, contributing pattern (which classes / cell types / residual CpGs drove firing), literature anchor (consistent with which published findings). For NOT_FIRED cards: brief one-liner ("breast-epic: not above threshold").

6. **Matrix candidates (Path B of dual matching).** Top-3 signature matches with match magnitude, phase (long_pre_dx / short_pre_dx / active / etc.), severity_class (a property of the row, NOT of the customer), evidence_anchors (which VALs document this signature). Framing: "the patient's combination most resembles the pattern documented for X at Y phase."

7. **Cellular age detail.** All 8 per-class cellular ages with saturation flags. Honest statement of which classes are at calibration boundary (SAT_HIGH / SAT_LOW) and what that means. Concordance structure: which classes are accelerated, decelerated, concordant vs chronological age. Age spread / IQR for spread interpretation.

8. **Risk-context layer** (when family history is on intake form).
   - Per-class posterior_context_class from formula in §5 Stage 9.
   - Framing: "your reading combined with your risk context suggests…" NEVER "you have a high probability of…".
   - Differentiates "keep watching and adjust lifestyle" from "your mother died of breast cancer GO NOW to a doctor" — same A-score reading means different things at different priors.
   - Conditional consumption: omit this section entirely if no family history provided.

9. **Quality and limitations.**
   - Stage 0 QC checkpoints (which passed, which flagged).
   - Cross-method gate status (Walther vs NILC v2 — PASS / FLAG with annotation).
   - Coverage and confidence per class / cell type.
   - Foreground subtraction status: age applied; sex / batch / ancestry / smoking pass-through (Phase B4 pending per Roadmap §10.2.2).
   - cfDNA branch: not applicable (substrate = `epic_blood`).
   - Single-sample limitation: ComBat batch correction skipped.
   - Chain-link declaration: "This reading uses L1+L2+L3+L4+L6+L9. L5 correlation structure and L7+L8 likelihood inference are not yet in production."

10. **Discussion-prompt section.** For any FIRED card or top matrix candidate: suggested follow-up questions or considerations for clinician decision support. Framed as decision-support, not prescription. (The legal-boundary gate at SOP §76 enforces the framing — no specific test names, no follow-up workups specified.)

11. **Audit trail reference.** Pointer to `<patient_id>_audit_trail.json` companion file with the full chain-of-custody record.

The report **never** tells the patient they have a disease. The report **may** tell the doctor "the pattern is consistent with published cohort X" and "the clinician may wish to consider Y as part of routine screening discussions" — framed as decision support. The legal-boundary gate catches diagnostic phrasing.

**Customer / patient-facing report — deferred to V2.** When V2 ships, Heath has reference material ready including:
- Per-cell-type cellular age (one age per scored cell type) with descriptive text for each cell type and its role
- Astro-Genetics framing
- Per-class main pages on iamperformance.net with subpages following biology (immune/progenitor/terminal/stromal by cell type; cycling/secretory by organ)
- Vigilance content per tier per cell
- Deep links from `card.educational_page_url` and `matrix.organ_pages_to_link`

V2 work happens after V1 has been in clinical use long enough to learn what to present.

---

## 9. Audit trail content

`<patient_id>_audit_trail.json` — the chain-of-custody record. Reproducibility and integrity defense.

Required content:
1. Run UUID (generated at start).
2. Timestamp (ISO 8601, UTC).
3. Orchestrator version (semver of `walther_clinical.py`).
4. `WALTHER_CLINICAL_MANIFEST.json` SHA-256 used for this run.
5. Per-file SHA-256 of every dependency consulted.
6. IDAT SHA-256 (Grn and Red).
7. Patient metadata (redacted of direct identifiers as appropriate).
8. Per-stage outputs (Stage 0 verdict, Stage 1 β stats, Stage 2 fractions + cross-method gate, Stage 3 cleaned β stats, Stage 4 A-scores, Stage 5 Mahalanobis with top-10, Stage 6 ages with concordance, Stage 7 tiers + bidirectional flags, Stage 8 card verdicts + matrix candidates, Stage 9 cleared report SHA-256, Stage 10 delivery confirmation).
9. Every SOFT and HARD flag raised.
10. Cross-stage flag composition per SOP §93.
11. Chain-link declaration (which of L1–L9 contributed).
12. Final report SHA-256.

---

## 10. Error handling

For HARD failures (SOP §92): halt with structured error message — which stage, which check, threshold vs observed, remediation guidance, partial audit trail.

For SOFT flags: proceed, propagate forward, surface in audit and in report §9 (Quality and limitations).

For DATA SIGNAL conditions (e.g., cellular age saturation): proceed, report honestly. Saturation is a measurement, not a failure.

For DOCUMENTED GAPS (sex / batch / ancestry / smoking foregrounds in V1): pass through, note gap in audit. Do not silently impute.

---

## 11. Testing strategy

Before V1 ships to clinical use:

1. **Synthetic patient round-trip.** Use `synthetic_patient_generator.py` to produce a synthetic IDAT-equivalent with known truth. Run orchestrator end-to-end. Verify recovered class fractions, A-scores, cellular ages match synthetic truth within declared tolerances.
2. **Known-HC validation.** Run on 5–10 EPIC-Italy HC samples. Verify all 8 per-class A-scores in HC reference range. Verify Mahalanobis distance below HC threshold.
3. **Known-case validation.** Run on 5–10 patients from a sealed VAL's case arm (breast pre-dx >10yr from VAL-003). Verify breast-epic card FIRES at expected rate. Verify Mahalanobis distance reproduces the +1.871 / +2.088 d anchor magnitudes.
4. **Reproducibility check.** Run same IDAT 3 times. Verify audit trails (minus timestamps/UUIDs) are bit-identical.
5. **Halting check.** Deliberately corrupt one dependency's SHA. Verify orchestrator refuses to run.
6. **Manifest check.** Remove `WALTHER_CLINICAL_MANIFEST.json`. Verify halt with clear message.
7. **Legal-boundary check.** Construct deliberately overreaching test case. Verify legal-boundary gate catches it before delivery.
8. **Cross-method gate check.** Construct a case where Walther and NILC v2 should disagree. Verify the gate flags it correctly.
9. **Bidirectional flag check.** Construct an AD-instance-pattern case (B-lineage UP + T-lineage DOWN). Verify bidirectional flag fires.
10. **Conditional consumption check.** Run with substrate=`epic_blood` and confirm `cfdna_weight.json` is silently skipped. Run with no family_history field and confirm Stage 9 falls back to overall-population prior with the appropriate note.

V1 not eligible for clinical use until all 10 pass. Document in `V1_TEST_REPORT.md`.

---

## 12. Deferred to V1.x+ and V2

| Item | Target | Reason for V1 deferral |
|---|---|---|
| Cards other than breast-epic | V1.x patches | Each card's re-VAL against the new atlas is the gating item |
| Sex / batch / ancestry / smoking foreground modules | V1.1 | Phase B4 per Roadmap §10.2.2 |
| Probe response function (L3) correction | V1.2 | Atlas-wide PRF characterization pending |
| Multi-patient batch runs | V1.1 | V1 is single-patient |
| ComBat batch correction | V1.1 | Requires cohort context |
| **Patient-facing report** | **V2** | After Heath learns from doctor's V1 use |
| Per-cell-type cellular age with descriptions | V2 | Heath has reference material ready |
| Online patient questionnaire integration | V2 | Requires web infrastructure |
| Multi-substrate unlock | V1.1+ per substrate | Each gated by substrate-specific re-validation |
| 450K platform support | V1.1 | EPIC-only for V1; 450K coverage gating per VAL-091 documented but not enabled |
| Engine E2 (Architecture Risk) | V1.2 | E1/E6/E7 sufficient for V1 doctor report |
| Engine E3 (Serial Measurement) | V2 | Requires second reading from same patient |
| Engine E4 (Pan-Tissue Screen) | V1.2 | Covered by per-class panel in V1 doctor report |
| Engine E5 (Intervention Target Solver) | V2 | Pending clinical advisor input |
| L5 Correlation structure | Phase C | TODO 2.1 (C(d)), 2.2 (bispectrum), 2.3 (banana degeneracies) |
| L7+L8 Likelihood + parameter inference | Phase E | Per-card Bayesian likelihood with MCMC posteriors |
| Per-patient L9 invocation | Not planned | L9 is for cohort-level VAL sealing, not single-patient |
| Web-API delivery | V2 | V1 is filesystem-only |
| Three-component split into separate scripts | V2 | V1's `walther_clinical.py` contains orchestration + report builder as internal modules with a clear internal boundary, ready to lift to a separate `walther_report_builder.py` in V2 |

---

## 13. Versioning

Semver. V1.0.0 first production release.

- Patch (V1.0.x): bug fixes, docs, no behavior change
- Minor (V1.x.0): adding deferred item from §12
- Major (V2.0.0): material change to SOP chain or doctor-report content; introduction of patient-facing report

Every bump requires:
1. Updated `WALTHER_CLINICAL_MANIFEST.json`
2. Updated test report (all 10 §11 tests re-run)
3. Updated audit-trail schema version if JSON changed
4. Heath's explicit sign-off in change log

---

## 14. Naming — LOCKED

Per Heath's 2026-06-02 instruction:
- Orchestrator: **`walther_clinical.py`**
- Deconvolver: **`walther_iam_deconvolver.py`** (already exists in repo)

Alternative names considered and not chosen: `commercial.web.py` (walkthrough working name); `web.commercial.py` (earlier sketch); `gape_clinical.py`; `cpg_clinical.py`; `edear_pipeline.py`. None of these are used.

Documentation drift note: the walkthrough's §6 references `commercial.web.py` and a separate `edear_report_builder.py`. V1 collapses these into `walther_clinical.py` as a single CLI with an internal report-builder module. V2 may split them when the patient-facing report is added.

---

## 15. Sequencing reminder

This document is the specification for the future build. It is NOT a directive to start now. The current priority per Heath's 2026-06-02 instruction is:

1. ~~Confirm all runtime dependencies extant~~ — **DONE** (all in `walther_clinical_runtime/`, pushed to repo)
2. **Re-run every VAL against the new atlas** (IAMAtlas REBUILD) + Walther + NILC v2 + new runtime modules. Currently only breast-epic is at v2.3 against new atlas. Other cards (ad-immune, crc-immune-inv, lung-epic, hcc-epic, prostate-epic, heme-epic, cardio-epic, cervical-epic, glioma-epic, kidney-epic, ...) are pending.
3. Produce 4 canonical files per re-VAL'd card: card JSON, card README, 3× residual maps (chr-annotated + bimodality + PCA projections), residual maps README. Place in `DISEASE_MAPS_CARDS/{card_name}/`.
4. Update disease signature matrix to incorporate any new findings from the re-VALs. Current v1.5 may bump to v1.6+ as cards re-VAL.
5. **THEN** consult this spec and build `walther_clinical.py`.

Do not let this spec lure a future AI into starting the build before the upstream card re-VAL work is complete. The build is downstream of correct cards.

---

*Spec v1.1 authored 2026-06-02. Authors: Heath W. Mahaffey, Walther (Claude). For build use after card re-VALs are complete. Authoritative companions: `CPG_Chain_of_Custody_SOP_v1_1.md`, `Biological_Physics/atlas_vault/walther_clinical_runtime/README.md`.*

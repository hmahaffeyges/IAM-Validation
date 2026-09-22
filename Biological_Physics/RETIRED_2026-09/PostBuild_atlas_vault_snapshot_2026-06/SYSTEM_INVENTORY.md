# IAM Validation Repository — System Inventory

**Comprehensive inventory of all runtime artifacts, scoring modules, deconvolvers, chain-of-custody scaffolding, and validation outputs.**

**Last updated:** 2026-05-30 (Phase B3 v4 cellular age scorer added)
**Author:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

This README replaces and supersedes the standalone `pipeline_runtime_matrices/README.md` as the system-level inventory. As the engine has grown to include cross-method deconvolution (NILC), chain-of-custody scaffolding (L9 null suite), and Stage 2.5 multi-D departure scoring (Mahalanobis), one runtime-matrices README is no longer sufficient. This document indexes everything.

---

## The pipeline at a glance

The runtime pipeline runs in **eight stages** consuming **15 runtime artifacts** distributed across folders:

```
Stage 0 — QC                        (Walther runs internal QC)
Stage 1 — Bayesian deconvolution    [Walther IAM Deconvolver | NILC v2]
Stage 2 — A-score formula           [iamatlas_a_scoring.py + markers v0_1.json]
Stage 2.5 — Mahalanobis departure   [iamatlas_mahalanobis_scoring.py + ref v0_1.json]
Stage 3 — Cellular age + age axis   [iam_cellular_age_scoring.py + age_reference_matrix
                                     OR age_axis_foreground.py + IAMAtlas_age_layer.csv]
Stage 4 — Departure detection       [tier_breakpoints.json + cfdna_weight.json]
Stage 5 — Cell-fraction matching    [cards/]
Stage 6 — Report assembly           [literature_anchors.json + cancer_prior.json + family_history_multiplier.json]
Stage 7 — Delivery                  (engine code)
```

L9 null suite (`cpg_null_runner.py` + `synthetic_patient_generator.py`) runs above the pipeline as integrity scaffolding — every VAL passes through the null suite before being sealed.

---

## Section 1 — Pipeline runtime matrices and modules

**Location:** `Biological_Physics/atlas_vault/pipeline_runtime_matrices/`

The 12-file READ-ONLY mirror of the constants and modules `GAPE_WEB_v13.py` consumes between IDAT-in and report-out.

### Stage 2 — A-scores

| File | Type | Purpose |
|---|---|---|
| `iamatlas_a_scoring.py` | Module | `score_per_class()` + `score_per_celltype()`. Returns A-score + coverage + confidence per class / cell type. SIBLING to the deconvolver (different math, different CpG pool). |
| `iamatlas_celltype_markers_v0_1.json` | Lookup | Per-cell-type one-vs-rest marker CpGs, top-100 per cell type, 115 cell types. SHA-256: `a56576cd5a7b2219d22d9a7a6efccd141a43c6d5fe4f5eb1d81e7375e1061ddc`. |
| `iamatlas_celltype_markers_v0_1.sha256` | Audit | SHA-256 of the markers JSON. |

H_min anchors per class are NOT duplicated here — they live in `IAMAtlasREBUILD_provenance.json` under `h_min_values_frozen_2026_04_06`. One source of truth: terminal 0.7728, immune 0.838889, secretory 0.843264, cycling 0.856055, progenitor 0.852216, stromal 0.86295, stem_adult 0.873718, stem_pluri 0.982166.

### Stage 2.5 — Mahalanobis departure (universal headline number)

| File | Type | Purpose |
|---|---|---|
| `iamatlas_mahalanobis_scoring.py` | Module | `MahalanobisHealthyHull` class. Single distance number per patient with top-10 axis contributions. |
| `mahalanobis_healthy_reference_v0_1.json` | Lookup | Pooled-HC centroid + Ledoit-Wolf-regularized covariance in 115-cell-type A-score space. n_hc=601. Shrinkage=0.0088. SHA-256: `fae063012ff7542a56ae4f91a494bad087d714f944911d6ff289113014a95b2b`. |
| `mahalanobis_per_patient_breast_predx_validation.csv` | Audit | Per-patient Mahalanobis distances for the breast pre-dx validation cohort (n=648). NOT loaded at runtime. |

Validation anchor: >10yr breast pre-dx vs HC Cohen's d = +1.871 (GSE51057) and +2.088 (GSE51032). Beats Xu-538 by +0.752 on GSE51032 without being breast-trained.

### Stage 3 — Cellular age (Recipe §6.3 canonical)

| File | Type | Purpose |
|---|---|---|
| `iam_cellular_age_scoring.py` | **Module (NEW 2026-05-30)** | **`IAMCellularAge` class.** Per-class cellular age via canonical IAM inversion — `A = H(β_mean) / H_min` per class, inverted against the 80-cell baseline. Eight per-class ages per patient, never collapsed to a single number by default. The summary cellular age is the `n_samples`-weighted mean across non-saturated classes. No training set. No regression. The atlas is the calibrated instrument. |
| `age_reference_matrix.json` | Lookup | 80-cell baseline: 8 classes × 10 decadal age bins, each carrying `(age_midpoint, A_mean, A_sd, β_mean, β_sd, n_samples, A_p10, A_p25, A_p50, A_p75, A_p90, source_citation)`. The calibrated instrument the scorer inverts against. |
| `age_reference_matrix.csv` | Lookup | Flat CSV format for human / spreadsheet inspection. Same 80 rows. |
| `age_reference_matrix.py` | Lookup | Python drop-in with `AGE_REFERENCE` dict + `age_ref_A()`, `age_ref_sd()`, `age_ref_percentiles()` interpolation helpers. |
| `cellular_ages_v4_epic_italy_validation.csv` | Audit | **NEW 2026-05-30.** Per-patient cellular age outputs from running `iam_cellular_age_scoring.py` on the 1,174 EPIC-Italy cohort. Per-class A-scores, per-class cellular ages, per-class status (OK / SAT_HIGH / SAT_LOW / INSUFFICIENT_CPGS), saturation analysis. NOT loaded at runtime. |

**Note on saturation observed in EPIC-Italy validation.** On the 1,174 EPIC-Italy cohort, the IAMAtlas marker CpGs produce per-class A-scores that systematically saturate above or below the baseline A_mean range for 7 of 8 classes (cycling is the one in-range class for ~half the cohort). This is direct data on the cohort vs the calibration, not a bug. The bidirectional saturation pattern (three classes all-high, four all-low, one in-range) is the cohort's structural signature against the IAM reference.

### Stage 4 — Departure detection

| File | Type | Purpose |
|---|---|---|
| `tier_breakpoints.json` | Lookup | A-score breakpoints (1.05 / 1.07 / 1.10) and engine→customer label collapse. |
| `cfdna_weight.json` | Lookup | Healthy-blood cfDNA tissue-of-origin weights (Snyder 2016 + Moss 2018). Activates when substrate is plasma cfDNA. |

### Stage 6 — Report assembly

| File | Type | Purpose |
|---|---|---|
| `literature_anchors.json` | Lookup | Published per-class A-score anchors (healthy / disease cohorts). Lets the report position a reading against known biology. |
| `cancer_prior.json` | Lookup | US lifetime cancer incidence per class. Bayesian risk-prior context. |
| `family_history_multiplier.json` | Lookup | First-degree-relative RR per class. |

### Empty subdirectories (placeholders for future content)

| Path | Planned content |
|---|---|
| `card_residual_maps/` | Per-card residual structure maps; populated as VALs lock thresholds. |
| `disease_signature_matrix/` | Disease signature matrix v1.4 binary export; currently lives inside engine code at v1.4. |

---

## Section 2 — Deconvolvers (cross-method discipline)

**Walther IAM Deconvolver** is the production deconvolver. **NILC v2** is the sibling cross-method check that lives in chain-of-custody scaffolding.

### Walther IAM Deconvolver

**Location:** `Biological_Physics/atlas_vault/deconvolver/`

| File | Purpose |
|---|---|
| `walther_iam_deconvolver.py` | Production deconvolver. NNLS on IAMAtlas-marker β values. Streaming-capable. 60% / 80% confidence gates per the Recipe. |
| `README.md` | Deconvolver-specific README. |

### NILC v2 (cross-method sibling, L4 component separation)

**Location:** `Biological_Physics/chain_of_custody/L4_component_separation/`

| File | Purpose |
|---|---|
| `nilc_deconvolver.py` | Independent deconvolution via departure-from-consensus GLS, modeled on Planck NILC. v2 (2026-05-30) uses Planck-style frequency-channel-fluctuation reformulation. |
| `nilc_fractions_all.csv` | NILC v1 per-patient class fractions on 1,174 EPIC-Italy. |
| `nilc_fractions_v2_departure.csv` | NILC v2 (current) per-patient class fractions. |
| `nilc_walther_crosscheck.json` | v1 cross-method gate report. |
| `nilc_walther_crosscheck_v2.json` | v2 cross-method gate report. |
| `Phase_B2_FINDING.md` | Phase B2 finding (initial NILC + Walther cross-check). |
| `Phase_B2_1_FINDING.md` | **Phase B2.1 finding (current).** Strict fraction-level gate FAIL (median L1 0.23). Biological-inference gate PARTIAL PASS — sign agreement 4/5 on case-vs-HC effects, including agreement on the disease-relevant immune class direction. Same shape as Planck Commander/NILC/SMICA/SEVEM: methods disagree at substrate level, agree on cosmological inferences. |

---

## Section 3 — Chain-of-custody scaffolding

Modules and artifacts that validate the pipeline rather than run inside it.

### L9 null suite (Phase A — COMPLETE)

**Location:** `Biological_Physics/chain_of_custody/L9_null_suite/`

| File | Purpose |
|---|---|
| `cpg_null_runner.py` | Unified 8-null framework (N1–N8) every VAL passes through before sealing. |
| `synthetic_patient_generator.py` | Methylome FFP10/NPIPE analog — generates synthetic patients with controlled signal injection for null-suite calibration. |
| `test_runs/` | Per-VAL null suite outputs (CPG_VAL_001 through CPG_VAL_007 — 5 sealed, 2 RESTATE: VAL-004 gain dominates loss 2.77:1; VAL-006 chr6 LEC-corrected p=0.103). |

### L4 component separation (Phase B — IN PROGRESS)

See Section 2 above (NILC deconvolver). Phase B2.1 closed with partial inference-gate pass.

### Stage 3 component module — age-axis foreground (separate from cellular age scorer)

**Location:** `Biological_Physics/atlas_vault/components/`

| File | Purpose |
|---|---|
| `age_axis_foreground.py` | **Phase B3 L4 component module.** Per-CpG age regression layer for SUBTRACTING age component from raw β values before deconvolution. This is a foreground-removal module conforming to the future `foreground_registry.py` interface — NOT the cellular age scorer. Two different operations at two different pipeline stages. |
| `IAMAtlas_age_layer.csv` | Per-CpG (α, γ, R², n_samples) — 8,199 CpGs trained on 601 EPIC-Italy HC. 100% CpG convergence (gate target 80%). |
| `age_layer_diagnostics.json` | Per-CpG fit diagnostics. |

**Important distinction.** `age_axis_foreground.py` (this folder) and `iam_cellular_age_scoring.py` (pipeline_runtime_matrices) are two DIFFERENT things at two DIFFERENT stages:
- **B3 foreground module (this folder):** subtracts the age component from β at the CpG level, BEFORE downstream A-score computation. Outputs cleaned β values.
- **Cellular age scorer (pipeline_runtime_matrices):** computes the per-class cellular age FROM A-scores by inverting the 80-cell baseline. Outputs 8 cellular ages per patient.

Both run in Stage 3 of the pipeline. The first cleans the inputs; the second produces the customer-facing cellular age readout.

---

## Section 4 — Superseded artifacts

### age_clock/ folder — Horvath-style regression clock (REJECTED 2026-05-30)

**Location:** `Biological_Physics/atlas_vault/age_clock/`

| File | Status |
|---|---|
| `SUPERSEDED.md` | Note explaining the supersession. |
| `iam_cellular_age_clock.py` | Horvath-style elastic-net regression on 8-class A-scores. Rejected as "back of the hand thermometer." Superseded by `iam_cellular_age_scoring.py` in pipeline_runtime_matrices. |
| `8class_ascores_all.csv` | Per-patient 8-class A-scores computed with the old `mean(H(β_i))/H_min` formula. Superseded by `cellular_ages_v4_epic_italy_validation.csv`. |
| `age_clock_diagnostics.json` | Regression diagnostics from the rejected clock. |
| `Phase_B3_FINDING.md` | Phase B3 finding document — kept for historical record. Documents both the B3 module gate PASS (still valid) and the rejected clock prototype. |

These files are preserved for audit / historical reference and are NOT part of the production pipeline.

---

## Section 5 — Quick reference: "where does X live?"

| If you need... | Look in... |
|---|---|
| Production runtime constants + scoring modules | `pipeline_runtime_matrices/` |
| The Walther deconvolver source | `deconvolver/walther_iam_deconvolver.py` |
| The NILC cross-method deconvolver | `chain_of_custody/L4_component_separation/nilc_deconvolver.py` |
| L9 null-suite scaffolding | `chain_of_custody/L9_null_suite/` |
| Per-CpG age axis (for foreground subtraction) | `components/age_axis_foreground.py` |
| Per-class cellular age scoring (for customer report) | `pipeline_runtime_matrices/iam_cellular_age_scoring.py` |
| The 80-cell age calibration baseline | `pipeline_runtime_matrices/age_reference_matrix.{json,csv,py}` |
| Per-patient cellular age validation output (1174 EPIC-Italy) | `pipeline_runtime_matrices/cellular_ages_v4_epic_italy_validation.csv` |
| Phase findings (B2.1, B3) | `chain_of_custody/L4_component_separation/Phase_B2_1_FINDING.md`, `age_clock/Phase_B3_FINDING.md` |
| H_min values | `IAMAtlasREBUILD_provenance.json` (one source of truth) |

---

## Section 6 — Provenance and audit trail

Every JSON in `pipeline_runtime_matrices/` has a `_meta` block recording its origin in `GAPE_WEB_v13.py`, the purpose statement, and the extraction date. SHA-256 sealed where applicable (markers JSON, Mahalanobis reference JSON).

If a runtime constant is updated, the change should happen in `GAPE_WEB_v13.py` FIRST, then re-extracted to this folder. **These files are READ-ONLY mirrors of the engine source — they exist for inspection, citation, and version control, not for direct editing.**

The `iam_cellular_age_scoring.py` module is the exception to this rule — it's a STANDALONE module imported directly by the runtime; updates happen here first, then mirrored into the engine. Same pattern as `iamatlas_a_scoring.py` and `iamatlas_mahalanobis_scoring.py`.

Validation outputs (the per-patient CSVs) are reproducible from the scoring modules + the cohort β matrices. They are kept here for audit and for use as anchors when discussing the engine's behavior on the EPIC-Italy cohort.

---

## Section 7 — Change log (canonical pipeline events)

| Date | Event |
|---|---|
| 2026-04-06 | H_min values frozen across all 8 classes (post-MCMC convergence). |
| 2026-04-24 | Recipe v1.0 published. GAPE rebuild reference complete. |
| 2026-05-08 | Recipe v2.0 — EDEAR product layer added. IAMAtlas REBUILD canonicalized. |
| 2026-05-28 | `GAPE_WEB_v13.py` runtime matrices extracted to `pipeline_runtime_matrices/` (12 files). |
| 2026-05-29 | TODO 1.1 — per-cell-type A-scoring module + 115-cell-type markers added (Stage 2). TODO 1.2 — Mahalanobis Stage 2.5 added. Recipe v3.0 published. |
| 2026-05-30 | **Phase A COMPLETE** — L9 null suite + synthetic patient generator + 7 Family A VALs (5 sealed, 2 RESTATE). **Phase B2.1 COMPLETE** — NILC v2 cross-method gate documented (substrate-level disagreement, inference-level partial agreement). **Phase B3 COMPLETE** — age-axis foreground module (gate PASS 100% convergence). **Phase B3 cellular age scorer v4** — Horvath-style prototype rejected, canonical Recipe §6.3 inversion shipped to `pipeline_runtime_matrices/iam_cellular_age_scoring.py`. |

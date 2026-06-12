# Biological Physics — the Informational Actualization Model applied to the methylome

This track applies the Informational Actualization Model (IAM) — a first-principles thermodynamic framework — to DNA methylation. The premise the framework carries across scales is a single one: the Landauer cost of maintaining an irreversible information pattern at physiological temperature sets an architecture-class-specific entropy floor. For cells, that floor is the minimum entropy a healthy cell type can hold while keeping its identity. The clinical instrument built on it, the **Cellular Performance Gauge (CPG)**, reads how far a cell type's methylation pattern has departed from that derived floor.

The naming is literal, not decorative: the patient's per-CpG departure map is the **Cosmic Methylome Background (CMB)**, and the chain is constructed on the same data-processing pipeline used in CMB cosmology (see *Methodology* below).

## Two governing principles

These hold across the whole track and are stated first because they are easy to violate by habit:

1. **Derived, not comparison.** The score is `A = H(v) / H_min`, where `H` is binary Shannon entropy and `H_min` is a *derived* architectural floor (an MCMC posterior, frozen). It measures departure from a derived reference, not a statistical distance to a population. It is **not** deconvolution-against-a-reference-panel and it does **not** pool cohorts. Reference-atlas (Loyfer/Moss), pooled-cohort, and Mahalanobis-distance-to-a-population framing belong to a different paradigm and are not how this method works.
2. **No foregrounds subtracted.** The production chain subtracts no age / sex / smoking / batch foreground. Smoking-, age-, and sex-driven methylation change is part of the cellular departure the score is built to measure — removing it would remove signal. Intake facts are report annotations for the clinician, never operands in the score.

## Repository layout

- **`AstroGenetics/CPG_CMB_v1/`** — the current production chain: the CPG clinical pipeline and the derived reference atlas (the IAMAtlas), its provenance, and the per-class brightness archives. See that folder's README for the methodology and stage map.
- **`PreIAMAtlas_Build/`** — the pre-build development phase: the foundational test results and VAL runs (`Preliminary_Test_Results/`), the pre-build disease cards retained as a reference set for the new chain (`RETIRED_Phase1_PreBuild_Cards/`), and the atlas-build machinery. Kept as a distinct phase so the methods are not conflated.
- **`papers/`** — publications and figures.
- **`DETAILED_VALIDATION_RECORD.md`** — the full, granular validation record and per-study notes.

## Methodology — the CMB pipeline, applied to the methylome

The chain follows the Planck-style cosmic-microwave-background data-processing pipeline stage for stage: raw detector intensities (IDAT) → calibration → an all-sky map (the per-CpG β matrix) → component separation (deconvolution) → an information-theoretic statistic scored against a derived reference scale, with an end-to-end null-test suite for sealing results and Mollweide / HEALPix all-sky map rendering for the report. The full term-by-term mapping is documented in the current-chain README and in the chain-of-custody SOP. The one place the analogy is deliberately *not* followed is foreground subtraction (principle 2 above): in cosmology a galactic foreground is a separate physical source; in the methylome the "foreground" is often the patient's own biology, so it is annotated, not removed.

## Status and framing

This is a first-principles framework with **preliminary** results. The biological track has produced a record of validation runs whose outcomes are described throughout as *consistent with* or *tested against* the framework's predictions — never as resolved, confirmed, or proven. The pipeline has been run end-to-end on real patient methylation arrays. The necessary next step is **prospective patient validation**, which ongoing trial planning is intended to enable; absolute clinical claims are deferred until that data exists. Readers evaluating the evidence are pointed to `DETAILED_VALIDATION_RECORD.md` and the per-study notes rather than to summary claims here.

## Reproduce

The reference atlas, its provenance, the clinical orchestrator, and the runtime matrices are in `AstroGenetics/CPG_CMB_v1/`. The atlas floors (the frozen `H_min` values) and their derivation provenance are in `AstroGenetics/CPG_CMB_v1/IAM_Atlas/IAMAtlasREBUILD_provenance.json`. Per-study inputs, code, and environment are recorded with each VAL run under `PreIAMAtlas_Build/Preliminary_Test_Results/`.

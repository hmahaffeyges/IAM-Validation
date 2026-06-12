# Biological Physics — the Informational Actualization Model applied to the methylome

This track applies the Informational Actualization Model (IAM) — a first-principles thermodynamic framework — to DNA methylation. The premise the framework carries across scales is a single one: the Landauer cost of maintaining an irreversible information pattern at physiological temperature sets an architecture-class-specific entropy floor. For cells, that floor is the minimum entropy a healthy cell type can hold while keeping its identity. The clinical instrument built on it, the **Cellular Performance Gauge (CPG)**, reads how far a cell type's methylation pattern has departed from that derived floor.

The naming is literal, not decorative: the patient's per-CpG departure map is the **Cosmic Methylome Background (CMB)**, and the chain is constructed on the same data-processing pipeline used in CMB cosmology (see *Methodology* below).

<details>
<summary><strong>What is the Informational Actualization Model, and why does it apply to the methylome the way it does to the cosmos?</strong></summary>

<br>

The Informational Actualization Model treats every irreversible physical transition — cosmic structure forming, a qubit decohering, a transistor switching, a cell maintaining its methylation pattern — as an event whose information cost is paid at the nearest encoding surface. From one domain to the next the substrate changes, the noise mechanism changes, and the measurable inputs change; the underlying thermodynamic accounting does not. That is what lets a single framework describe cosmology and cellular thermodynamics without treating the systems as alike.

In every domain the same three quantities appear: a **physics floor** (the irreducible thermodynamic minimum, set by the Landauer cost of irreversible information maintenance at the operating temperature), an **architecture ceiling** (the most a given system can reach), and a **dimensionless ratio** for how far the system sits between them. For cells the floor is the minimum entropy a cell type can hold while keeping its identity — the architectural `H_min` — and the dimensionless ratio is the A-score.

Applied to the methylome the claim is concrete: maintaining a cell's DNA-methylation pattern is irreversible information maintenance, and at physiological temperature that maintenance carries a Landauer cost. That cost sets a class-specific entropy floor, and the A-score reads how far a cell type has drifted above it. A useful contrast for the clinical setting: most instruments in use are a form of steam detection — they fire only once disease is overt, the way a steam detector needs the water already boiling. Reading the floor departure directly is closer to a thermometer, reporting where each cell class currently sits on the scale rather than only whether a threshold has been crossed.

The same framework, on the cosmological side, produces parameter-free derivations that agree with observation to sub-percent levels (for example, a derived value for the cosmological constant within roughly 0.07% of the observed value). Those results belong to the separate IAM cosmology track and are mentioned here only to make the point that the methylome application rests on the same thermodynamic law — it is not a separate model fit to biology.

</details>

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

## Selected preliminary results

These are illustrative, stated as preliminary; each is from a dated, sealed validation run on the public repository, and the full per-study detail — with methodological caveats disclosed up front — is in `DETAILED_VALIDATION_RECORD.md`. All use the derived architectural floors with **no parameters fit to the outcome**.

- **A pre-clinical signal years before diagnosis (VAL-046).** Across seven published cohort/cancer combinations — including the Sister Study (n = 2,776), UK Biobank lung (n = 680), and a Nurses' Health colorectal cohort (n = 355) — participants who later developed cancer carried a small baseline architectural departure (mean ΔA ≈ +0.014) above matched cancer-free controls, detectable 2–5 years before clinical diagnosis. This is consistent with the framework's prediction of pre-clinical drift, and it is smaller than established-disease magnitudes, as a pre-clinical state should be.
- **Trajectories consistent with treatment response (VAL-044).** Across five published clinical-trial cohorts (glioblastoma, colorectal, breast, AML, melanoma), A-score trajectories tracked responders versus non-responders, with complete-response cases approaching the healthy floor (A ≈ 1.00).
- **An organ-wide field effect across TCGA (VAL-003, VAL-037).** Tested against the matched tumor / adjacent-normal pairs in TCGA (~4,000 analyzable pairs), adjacent-normal tissue showed a consistent architectural elevation in every cancer type examined, computed entirely within one normalization pipeline.
- **Substrate independence (VAL-021–024).** The same departure appears in four independent non-methylation substrates (fragmentomics), consistent with a thermodynamic rather than a methylation-specific origin.
- **An honest negative the framework predicted (VAL-038, VAL-041).** Tissue-level A-scores did not correlate with bulk-plasma cfDNA detectability (Spearman ρ ≈ −0.02) — expected, because plasma detection reflects tumor-shedding kinetics rather than tissue architecture. When plasma is first deconvolved to tissue of origin, per-tissue scoring recovers the signal. The limit was anticipated, not discovered after the fact.

Absolute cross-pipeline thresholds depend on normalization and are treated as such; the reported quantity throughout is the within-pipeline departure. The aim of the prospective trial work is to move these from preliminary to prospectively tested.

## Status and framing

This is a first-principles framework with **preliminary** results. The biological track has produced a record of validation runs whose outcomes are described throughout as *consistent with* or *tested against* the framework's predictions — never as resolved, confirmed, or proven. The pipeline has been run end-to-end on real patient methylation arrays. The necessary next step is **prospective patient validation**, which ongoing trial planning is intended to enable; absolute clinical claims are deferred until that data exists. Readers evaluating the evidence are pointed to `DETAILED_VALIDATION_RECORD.md` and the per-study notes rather than to summary claims here.

## Reproduce

The reference atlas, its provenance, the clinical orchestrator, and the runtime matrices are in `AstroGenetics/CPG_CMB_v1/`. The atlas floors (the frozen `H_min` values) and their derivation provenance are in `AstroGenetics/CPG_CMB_v1/IAM_Atlas/IAMAtlasREBUILD_provenance.json`. Per-study inputs, code, and environment are recorded with each VAL run under `PreIAMAtlas_Build/Preliminary_Test_Results/`.

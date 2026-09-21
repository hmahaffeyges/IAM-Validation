# Biological Physics — the Informational Actualization Model applied to the methylome

**What this field is called (2026-09-20): Physics of Methylation: Landauer Metrology** — measuring how far above the thermal noise quantum an information-writing process operates, against a fixed physical zero (H_min per cell class). Thermal noise is the unit (M = E_drive / k_B T), not the nuisance. Prior art: Sanchez & Mackenzie 2016 established that the methylome obeys Landauer's bound; Landauer metrology measures how far above it each cell class operates (Issue 003 §0b).


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

**New here? Read [`HANDOFF.md`](HANDOFF.md) first.**

## Two governing principles

These hold across the whole track and are stated first because they are easy to violate by habit:

1. **Derived, not comparison.** The score is `A = H(v) / H_min`, where `H` is binary Shannon entropy and `H_min` is a *derived* architectural floor (an MCMC posterior, frozen). It measures departure from a derived reference, not a statistical distance to a population. It is **not** deconvolution-against-a-reference-panel and it does **not** pool cohorts. Reference-atlas (Loyfer/Moss), pooled-cohort, and Mahalanobis-distance-to-a-population framing belong to a different paradigm and are not how this method works.
2. **No foregrounds subtracted.** The production chain subtracts no age / sex / smoking / batch foreground. Smoking-, age-, and sex-driven methylation change is part of the cellular departure the score is built to measure — removing it would remove signal. Intake facts are report annotations for the clinician, never operands in the score.

## Repository layout (reorganized 2026-09-19)

Five folders. A researcher starts at the first one.

| folder | what it is | start with |
|---|---|---|
| [`Physics_of_Methylation/`](Physics_of_Methylation/) | **Start here.** GAPE Issue 003 (the current report), the reproduction kit that verifies the measurement chain end to end, the chain-of-custody SOP (v2.0.0, matched to the engine at HEAD), the papers, and the methylome-vs-CMB plates | [`Physics_of_Methylation/README.md`](Physics_of_Methylation/README.md) |
| [`IAM_Atlas/`](IAM_Atlas/) | the derived reference atlas — 483,092 CpGs × 115 cell types, per-class MCMC posterior mean/sd, `H_min` provenance, the HEALPix sky mapping, and the scripts that built it | [`IAM_Atlas/README.md`](IAM_Atlas/README.md) |
| [`CPG_Engine/`](CPG_Engine/) | the running code: Stage 0–1 intake and calibration, the Walther deconvolver, the class gauge, the conductor, the patient-CMB module, runtime matrices, test data, disease cards, report builders | [`CPG_Engine/README.md`](CPG_Engine/README.md) |
| [`Testing_and_Code/`](Testing_and_Code/) | every validation run (VAL-001 … VAL-141) split into pre-Atlas and post-Atlas, the sealed foundation-cohort anchors, cohort manifests and extraction scripts, the full validation record and index | [`Testing_and_Code/README.md`](Testing_and_Code/README.md) |
| [`RETIRED/`](RETIRED/) | superseded material kept for the record: the pre-build phase (Phase-1 cards, early chain-of-custody, production data), the June-2026 `atlas_vault` snapshot, and the NILC deconvolver cut from the chain | [`RETIRED/README.md`](RETIRED/README.md) |

Nothing in `RETIRED/` is used by the current chain. Nothing outside `RETIRED/` is obsolete.

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

Start with [`Physics_of_Methylation/Reproduction_Kit/README_FIRST.md`](Physics_of_Methylation/Reproduction_Kit/README_FIRST.md). Five scripts verify the chain link by link against known answers — raw IDAT → β (bit-identical to the cached betas, 11/11), deconvolver vs documented outputs (MAE 0.0004), the sealed 115-cell anchors recomputed from raw GEO (r = 1.00000 on both cohorts), the two aggregation formulas on the same samples, and the deconvolver against known plasma mixtures. The atlas is `IAM_Atlas/IAMAtlasREBUILD.csv.xz` (decompress in place); the frozen `H_min` values and their derivation provenance are in `IAM_Atlas/IAMAtlasREBUILD_provenance.json` and printed in Issue 003.

*Research stage. Nothing in this tree is clinical validation.*


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Testing_and_Code/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.

**H_min CROSS-CHECK (PROC-HMIN-BOOT-01, 2026-09-20).** The eight methylation floors were calibrated by G-002 MCMC (37 reference cells, R-hat < 1.001). The April bootstrap cross-check (0.168 %, 24/32 in CI) covered the 32 non-methylation floors only; the methylation eight were bootstrapped for the first time on 2026-09-20 — 8/8 inside the 95 % CI, 0.060 % mean / 0.095 % max relative difference. Values unchanged. The calibration scripts are not at HEAD (removed 2026-04-19 as commercial) but are in public history at 22749f0 — a disclosure decision for the author. Record: `Testing_and_Code/PROC_data/PROC-HMIN-BOOT-01/`.

**LAB ZERO — PANEL SPECIFICATION (PROC-PANEL-01 → PROC-PANEL-03, 2026-09-20; supersedes the '20–30 arrays' wording above).** A laboratory's zero is measured once on **40** healthy arrays of any age mix through the same Stage 1 and map: z = median[A − c(decade)] − 1, where c is the reference healthy age curve (`reference_age_curve_v1.json`; healthy immune A rises ≈0.045 from the teens to the eighties within a lab, while between-lab offsets are parallel). A patient reads A″ = A − c(decade) − z. Tested leave-one-lab-out on four labs: a band built on three holds 75–84 % of the fourth's healthy donors. In code: `CPG_Engine/lab_zero.py` — panels under 40 are refused and `lab_zero=UNSET` is not reportable. Record: `Testing_and_Code/PROC_data/PROC-PANEL-01 → PROC-PANEL-03/`.

**THE VALIDATION COUNT, CORRECTED (PROC-HISTORY-01, 2026-09-21).** The record, not the tree: 3 G-series calibrations; **119 pre-Atlas VALs (VAL-001..128; 107 executed)** incl. the **T1–T15** cross-population series of VAL-049 (12 executed, 6 populations); **22 post-Atlas CPG-VALs (001..022; 21 executed)**; the Mahalanobis hull v0_1→v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; the September PROCs. The 2026-09-19 index said 103 — it keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders. `Testing_and_Code/VAL_INDEX.csv` is rebuilt (175 rows, unique keys by series); AD folders CPG-VAL-008..014 moved to `VAL_PostAtlas/`. Record: `Testing_and_Code/PROC_data/PROC-HISTORY-01/`.

**THE GAUGE SWITCH (PROC-SWITCH-01 → PROC-SWITCH-02, 2026-09-21; row B COMMISSIONED).** `cpg_conductor.run_full` now REPORTS the identity-loci gauge: A = H(β̄)/H_min on `iamatlas_gauge_identity_loci_v1_0.json`, on mapped β, minus c(decade) (`reference_age_curve_v1.json`), minus the laboratory zero (`lab_zero.py`), placed in `identity_band_v3.json` (four zeroed labs, n = 1,379, pooled p10–p90 0.9724–1.0248). The marker-union statistic is `diagnostic_marker_union` — never the reported A. Stages 5 and 6 carry `pending_recalibration=True`. Test: `Reproduction_Kit/test_gauge_switch.py`. **Finding:** the atlas posterior is a fifth laboratory (z = −0.0146) — SWITCH-01's S4 assumed zero and failed as sealed; every β source, including a simulator, is zeroed before it is read absolutely.

**STAGE 5 RE-BASED (PROC-MAHA-01, 2026-09-21; row 5 BUILT, not commissioned).** The departure now reads the identity gauge: z = (A″ − 1)/σ, σ = 0.0204 from `identity_band_v3`; on whole blood one banded axis, so the number is |z_immune| against 1.960 / 2.576; `bundle['mahalanobis']` carries the long keys the report builder reads plus the short aliases; UNSET → not reportable. The eight-class derived hull is `diagnostic_hull_marker_union`. **M2 failed as sealed:** Karolinska 9.8 % of healthy beyond p95 (bar 7 %). **Cause measured — the Sentrix chip:** per-chip median SD 0.020 there vs 0.012 elsewhere; chip-centring cuts every lab to 2–4 %. A laboratory constant cannot touch it; row 5b (chip term) is open and the acceptable false-alarm rate is the author's decision (PROC-MAHA-02). Record: `Testing_and_Code/PROC_data/PROC-MAHA-01/`.

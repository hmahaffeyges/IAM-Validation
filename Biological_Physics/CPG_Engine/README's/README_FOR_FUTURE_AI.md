**READ FIRST (2026-09-20):** **BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

---

# CPG CMB v2 — Engine snapshot for a future build/test session

This folder is a self-contained snapshot of the Cellular Performance Gauge (CPG) clinical
methylation chain, current as of this push. It is here so a future AI can run the chain,
reproduce the tests, or fix/extend it without re-deriving the architecture.

## What CPG does
From one patient methylation IDAT (450K or EPIC), the chain deconvolves the sample into the
8 architectural classes and 115 cell types, computes per-cell and per-class A-scores
(architectural state vs the derived healthy floor of 1.0), matches against the disease
signature matrix (Route B concordance + Mode 2 cell-of-origin presence), tiers the signal,
and writes one clinician-facing report. A second "confirmation" chain runs only when a flag
fires (it never mutates the primary): a derived-Mahalanobis global-departure adjudicator,
plus literature-anchor and residual-map qualifiers, appended as one integrated verdict.
From the second draw on, a per-cell trajectory tracks change against the patient's own prior.

## How to run
1. Python 3.11 with: methylprep 1.7.1, numpy, pandas (<2 for native methylprep), scipy,
   scikit-learn, matplotlib. (`conda create -n cpg python=3.11` then pip install.)
   - NOTE: in a Python 3.12 container, methylprep 1.7.1 needs the pandas-compat shim at
     `TEST_DATA/harness/pdshim.py` (it restores DataFrame.append). On 3.11 no shim is needed.
2. Per-patient run: `python run_batch.py --patients /path/to/patients` where each patient is
   `patients/<ID>/<YYYY-MM-DD>/` containing the IDAT pair + `questionnaire.json`.
   Single visit: `python walther_clinical.py --folder <visit_folder> --out <visit_folder>`.
3. The conductor auto-resolves the engine root by locating `IAM_Atlas/` and auto-decompresses
   `IAM_Atlas/IAMAtlasREBUILD.csv.xz` on first run.

## Atlas
The runtime atlas `IAM_Atlas/IAMAtlasREBUILD.csv.xz` is included here. The per-class brightness
archives are in `IAM_Atlas/iamatlas_class_archives/`. These are identical to CPG_CMB_v1 and to
`Biological_Physics/atlas_vault/`. The large decompressed CSV is NOT shipped (regenerated on run).

## What's current in v2 (vs v1)
- Mahalanobis adjudicator presence gate: a class counts only if genuinely present (abundance
  >= 3%) AND outside the NORMAL band [0.95, 1.04). Stops suppressed non-substrate classes
  (stem_pluri, terminal in blood) from inflating the distance. (stage_5_second_chain.py)
- Per-cell trajectory: per-cell deltas (not class scores) + rotation-toward-signature, led by
  deconvolver-resolved cells; bulk pseudo-cells excluded. (walther_clinical.py `_compute_trajectory`)
- Report: trajectory section, two-deconvolver explainer, Mahalanobis callout. (cpg_report_builder.py)
- Flowchart updated to match (flowchart_v4.html).

## Tests / reproduction
See `TEST_DATA/TEST_DATA_MANIFEST.md` for every sample (GEO accession, array, substrate) and the
result it established. Harness scripts are in `TEST_DATA/harness/`; demo reports in
`TEST_DATA/reports/`. Raw IDATs are public (URLs in the manifest); the calibrated betas cache
(`betas_cache.pkl`, ~100MB) is regenerable from them and is not shipped here.

## Key validated facts
- Whole blood DNA is ~96% immune (the immune class = all 51 leukocyte types incl. granulocytes);
  secretory/epithelial reads ~0 in healthy blood — correct, not a limitation.
- The chain reads secretory/cycling when epithelial DNA is present: CRC carcinoma (EPIC) stage 1
  secretory 12.9%, stage 4 secretory 24.8% (rising with stage). EPIC arrays are supported.
- Substrate matters: whole blood = immune-architecture / field-effect readout; plasma cfDNA =
  direct shed-tumour readout; tissue = positive control. A report must state its substrate.

## Standing rules (carried)
Source-doc before concluding; no fabrication; referee language ("consistent with", never
"confirms/validates/proves"); DERIVED-IAMAtlas-only (no cohort pooling); surgical edits with
before/after; set up tests fully then await go.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Testing_and_Code/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.

**H_min CROSS-CHECK (PROC-HMIN-BOOT-01, 2026-09-20).** The eight methylation floors were calibrated by G-002 MCMC (37 reference cells, R-hat < 1.001). The April bootstrap cross-check (0.168 %, 24/32 in CI) covered the 32 non-methylation floors only; the methylation eight were bootstrapped for the first time on 2026-09-20 — 8/8 inside the 95 % CI, 0.060 % mean / 0.095 % max relative difference. Values unchanged. The calibration scripts are not at HEAD (removed 2026-04-19 as commercial) but are in public history at 22749f0 — a disclosure decision for the author. Record: `Testing_and_Code/PROC_data/PROC-HMIN-BOOT-01/`.

**LAB ZERO — PANEL SPECIFICATION (PROC-PANEL-01 → PROC-PANEL-03, 2026-09-20; supersedes the '20–30 arrays' wording above).** A laboratory's zero is measured once on **40** healthy arrays of any age mix through the same Stage 1 and map: z = median[A − c(decade)] − 1, where c is the reference healthy age curve (`reference_age_curve_v1.json`; healthy immune A rises ≈0.045 from the teens to the eighties within a lab, while between-lab offsets are parallel). A patient reads A″ = A − c(decade) − z. Tested leave-one-lab-out on four labs: a band built on three holds 75–84 % of the fourth's healthy donors. In code: `CPG_Engine/lab_zero.py` — panels under 40 are refused and `lab_zero=UNSET` is not reportable. Record: `Testing_and_Code/PROC_data/PROC-PANEL-01 → PROC-PANEL-03/`.

**THE VALIDATION COUNT, CORRECTED (PROC-HISTORY-01, 2026-09-21).** The record, not the tree: 3 G-series calibrations; **119 pre-Atlas VALs (VAL-001..128; 107 executed)** incl. the **T1–T15** cross-population series of VAL-049 (12 executed, 6 populations); **22 post-Atlas CPG-VALs (001..022; 21 executed)**; the Mahalanobis hull v0_1→v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; the September PROCs. The 2026-09-19 index said 103 — it keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders. `Testing_and_Code/VAL_INDEX.csv` is rebuilt (175 rows, unique keys by series); AD folders CPG-VAL-008..014 moved to `VAL_PostAtlas/`. Record: `Testing_and_Code/PROC_data/PROC-HISTORY-01/`.

**THE GAUGE SWITCH (PROC-SWITCH-01 → PROC-SWITCH-02, 2026-09-21; row B COMMISSIONED).** `cpg_conductor.run_full` now REPORTS the identity-loci gauge: A = H(β̄)/H_min on `iamatlas_gauge_identity_loci_v1_0.json`, on mapped β, minus c(decade) (`reference_age_curve_v1.json`), minus the laboratory zero (`lab_zero.py`), placed in `identity_band_v3.json` (four zeroed labs, n = 1,379, pooled p10–p90 0.9724–1.0248). The marker-union statistic is `diagnostic_marker_union` — never the reported A. Stages 5 and 6 carry `pending_recalibration=True`. Test: `Reproduction_Kit/test_gauge_switch.py`. **Finding:** the atlas posterior is a fifth laboratory (z = −0.0146) — SWITCH-01's S4 assumed zero and failed as sealed; every β source, including a simulator, is zeroed before it is read absolutely.

**STAGE 5 RE-BASED (PROC-MAHA-01, 2026-09-21; row 5 BUILT, not commissioned).** The departure now reads the identity gauge: z = (A″ − 1)/σ, σ = 0.0204 from `identity_band_v3`; on whole blood one banded axis, so the number is |z_immune| against 1.960 / 2.576; `bundle['mahalanobis']` carries the long keys the report builder reads plus the short aliases; UNSET → not reportable. The eight-class derived hull is `diagnostic_hull_marker_union`. **M2 failed as sealed:** Karolinska 9.8 % of healthy beyond p95 (bar 7 %). **Cause measured — the Sentrix chip:** per-chip median SD 0.020 there vs 0.012 elsewhere; chip-centring cuts every lab to 2–4 %. A laboratory constant cannot touch it; row 5b (chip term) is open and the acceptable false-alarm rate is the author's decision (PROC-MAHA-02). Record: `Testing_and_Code/PROC_data/PROC-MAHA-01/`.

**ROW 6 CLOSED — CELLULAR AGE NOT REPORTABLE AT SINGLE-ARRAY RESOLUTION (PROC-AGE-01, 2026-09-21).** Inverting the healthy immune identity-gauge curve (`reference_age_curve_v1`) for one array resolves age to ~50 years: the curve moves 0.47 mA/yr and the within-laboratory spread is 0.0235; leave-one-lab-out on 1,379 healthy donors, 15.9 % within ±10 yr (bar 80 %), Spearman 0.27; a healthy 58-year-old inverts to 23. **The population aging trajectory stands and is reproduced** (0.47 mA/yr, monotone by decade, four labs = CPG-VAL-015's slope on Hannum; it is now the reference age curve). What is below resolution is one person's position on it. Sign differs by surface: marker-union A falls with age, identity-gauge A rises (RECON D2). `stage_6_cellular_age` returns `reportable=False` with the resolution sentence, which the report prints in place of an age; the marker-union inversion is `diagnostic_cellular_age`. With this, **no reported number in the chain reads the marker-union statistic.** Record: `Testing_and_Code/PROC_data/PROC-AGE-01/`.

- **Row 4.6 — the patient's sky — COMMISSIONED (PROC-CMB-04, 2026-09-21, four seals).** `cpg_conductor.run_full` bundle key `patient_sky`: z = (β − Σ f_c μ_c − m_lab)/s_lab on the mapped β, class panels gated by measured presence floors (`Runtime Matrices/Patient_CMB/`), HEALPix NSIDE 128 genomic order. NOT AVAILABLE without the laboratory's residual scale (built from the same 40-array healthy panel as the lab zero). Calibration constant stated on every sky: healthy held-out tail 2.6–3.2 %, not 5 % (C2′ failed as sealed by ≤ 0.004; recorded). The retired `patient_brightness_comparison.py` formula read 61 % of a healthy genome as anomalous (C1) and is closed. Kit test `test_patient_sky.py`.

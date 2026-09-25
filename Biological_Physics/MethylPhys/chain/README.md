# CPG Engine — the running code

The chain that scores a sample. Start at [`../doors/RUNBOOK.md`](../doors/RUNBOOK.md) to run one and [`../doors/CHAIN_SEQUENCE.md`](../doors/CHAIN_SEQUENCE.md) for the step order as the code calls it. When a run refuses, the cause and the fix are under the step that refused it in [`../sop/CPG_Chain_of_Custody_SOP_v2_0_0.md`](../sop/CPG_Chain_of_Custody_SOP_v2_0_0.md) (§11-§19), in Issue 003 section 3b, and on the report's own Troubleshooting tab.

| stage | file(s) |
|---|---|
| 0 intake — runs **before** calibration; a QUARANTINE stops the chain and nothing is scored | [`stage_0_intake.py`](stage_0_intake.py) (the SOP §11–§19 gates and the decision), [`stage_0_1_qc_handoff.py`](stage_0_1_qc_handoff.py) (decodes the control probes, negative controls, bead counts and chrX/chrY the QC gates read), `cpg_intake_form.html`, [`preflight.py`](preflight.py) |
| 1 calibration (raw IDAT → β, methylprep noob) | [`stage_1_idat_calibration.py`](stage_1_idat_calibration.py) ([`stage_1_calibration.py`](stage_1_calibration.py), [`idat_decoder_pure.py`](idat_decoder_pure.py), [`idat_parse.py`](idat_parse.py) are the pure-Python path) |
| 2 deconvolution (composition, presence) | `Walther_iam_deconvolver/walther_iam_deconvolver.py` — reads `../MethylPhys/atlas/IAMAtlasREBUILD.csv` |
| 4 class gauge + 7 tier | [`cpg_gauge_engine.py`](cpg_gauge_engine.py) (the 40-cell `H_MIN_TABLE`, age band, tiers), [`cpg_gauge.py`](cpg_gauge.py), `Runtime Matrices/` |
| 4.6 patient CMB | [`cpg_patient_cmb.py`](cpg_patient_cmb.py) — z-departure sky against the atlas posterior; uses `../MethylPhys/atlas/healpix_mapping/` |
| 5 second chain | [`stage_5_second_chain.py`](stage_5_second_chain.py) (Mahalanobis hull) |
| orchestrator | **[`cpg_conductor.py`](cpg_conductor.py)** |
| nulls (sealing) | `CPG_Null_Runner/cpg_null_runner.py` — the eight nulls N1–N8; a VAL is sealed only when its declared nulls pass |
| report | [`cpg_report_builder.py`](cpg_report_builder.py), `report_builders/` (strawman, patient wall, synthetic-patient harness), [`build_dashboard_v1.py`](build_dashboard_v1.py) |
| runtime constants | `Runtime Matrices/` — identity loci, discriminative markers (chrX-removed, canonical), age reference, tiers, Mahalanobis reference, directional panels |
| test data | `TEST_DATA/` — 11 IDAT pairs + [`TEST_DATA_MANIFEST.md`](TEST_DATA/TEST_DATA_MANIFEST.md) (documented expected outputs); `betas_cache.pkl` is not in git (see the Reproduction Kit) |
| disease side | `Disease Matrix/`, `Disease Cards : Residual Maps/`, `Crown Jewel and Patient Strawman/` |
| documents | `flowchart_vKISS.html` (stage map), `CPG_Doctor_Workflow_KISS.html`, `CPG_AstroGenetics_explainer_section.html`, `README's/README_FOR_FUTURE_AI.md`, [`CHANGELOG.md`](CHANGELOG.md), [`ROADMAP_TaskTracker.md`](ROADMAP_TaskTracker.md), [`RUN_MANIFEST_and_README.md`](RUN_MANIFEST_and_README.md), [`CPG_Lessons_Learned_2026-06-29.md`](CPG_Lessons_Learned_2026-06-29.md) |

**Two scoring surfaces, one rule each** (SOP §106): the class **gauge** on identity loci is `H(β̄)/H_min`; the 115-cell **separation** statistic on discriminative markers is the mean of per-CpG entropies. `Runtime Matrices/A_Scoring_Module/test_a_score_canonical.py` guards the separation surface; the gauge's Jensen-gap guard is in the Reproduction Kit's [`cpg_kit.py`](../kit/cpg_kit.py).

Verification of this code against known answers: [`../MethylPhys/kit/`](../MethylPhys/kit/).

Known: `report_builders/render_strawman_v2.py` and [`render_patient_wall.py`](report_builders/render_patient_wall.py) use Python ≥ 3.12 f-string syntax.


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Record/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; [[Issue 003](../manual/IAMPerformance_GAPEIssue003_RC1.pdf)](../manual/IAMPerformance_GAPEIssue003_RC1.pdf) RECON S1, §1.6.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Record/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.

**H_min CROSS-CHECK (PROC-HMIN-BOOT-01, 2026-09-20).** The eight methylation floors were calibrated by G-002 MCMC (37 reference cells, R-hat < 1.001). The April bootstrap cross-check (0.168 %, 24/32 in CI) covered the 32 non-methylation floors only; the methylation eight were bootstrapped for the first time on 2026-09-20 — 8/8 inside the 95 % CI, 0.060 % mean / 0.095 % max relative difference. Values unchanged. The calibration scripts are not at HEAD (removed 2026-04-19 as commercial) but are in public history at 22749f0 — a disclosure decision for the author. Record: `Record/PROC_data/PROC-HMIN-BOOT-01/`.

**LAB ZERO — PANEL SPECIFICATION (PROC-PANEL-01 → PROC-PANEL-03, 2026-09-20; supersedes the '20–30 arrays' wording above).** A laboratory's zero is measured once on **40** healthy arrays of any age mix through the same Stage 1 and map: z = median[A − c(decade)] − 1, where c is the reference healthy age curve ([`reference_age_curve_v1.json`](Runtime%20Matrices/A_Scoring_Module/reference_age_curve_v1.json); healthy immune A rises ≈0.045 from the teens to the eighties within a lab, while between-lab offsets are parallel). A patient reads A″ = A − c(decade) − z. Tested leave-one-lab-out on four labs: a band built on three holds 75–84 % of the fourth's healthy donors. In code: `MethylPhys/chain/lab_zero.py` — panels under 40 are refused and `lab_zero=UNSET` is not reportable. Record: `Record/PROC_data/PROC-PANEL-01 → PROC-PANEL-03/`.

**THE VALIDATION COUNT, CORRECTED (PROC-HISTORY-01, 2026-09-21).** The record, not the tree: 3 G-series calibrations; **119 pre-Atlas VALs (VAL-001..128; 107 executed)** incl. the **T1–T15** cross-population series of VAL-049 (12 executed, 6 populations); **22 post-Atlas CPG-VALs (001..022; 21 executed)**; the Mahalanobis hull v0_1→v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; the September PROCs. The 2026-09-19 index said 103 — it keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders. `Record/VAL_INDEX.csv` is rebuilt (175 rows, unique keys by series); AD folders CPG-VAL-008..014 moved to `VAL_PostAtlas/`. Record: `Record/PROC_data/PROC-HISTORY-01/`.

**THE GAUGE SWITCH (PROC-SWITCH-01 → PROC-SWITCH-02, 2026-09-21; row B COMMISSIONED).** `cpg_conductor.run_full` now REPORTS the identity-loci gauge: A = H(β̄)/H_min on [`iamatlas_gauge_identity_loci_v1_0.json`](Runtime%20Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json), on mapped β, minus c(decade) (`reference_age_curve_v1.json`), minus the laboratory zero ([`lab_zero.py`](lab_zero.py)), placed in [`identity_band_v3.json`](Runtime%20Matrices/A_Scoring_Module/identity_band_v3.json) (four zeroed labs, n = 1,379, pooled p10–p90 0.9724–1.0248). The marker-union statistic is `diagnostic_marker_union` — never the reported A. Stages 5 and 6 carry `pending_recalibration=True`. Test: `MethylPhys/kit/test_gauge_switch.py`. **Finding:** the atlas posterior is a fifth laboratory (z = −0.0146) — SWITCH-01's S4 assumed zero and failed as sealed; every β source, including a simulator, is zeroed before it is read absolutely.

**STAGE 5 RE-BASED (PROC-MAHA-01, 2026-09-21; row 5 BUILT, not commissioned).** The departure now reads the identity gauge: z = (A″ − 1)/σ, σ = 0.0204 from `identity_band_v3`; on whole blood one banded axis, so the number is |z_immune| against 1.960 / 2.576; `bundle['mahalanobis']` carries the long keys the report builder reads plus the short aliases; UNSET → not reportable. The eight-class derived hull is `diagnostic_hull_marker_union`. **M2 failed as sealed:** Karolinska 9.8 % of healthy beyond p95 (bar 7 %). **Cause measured — the Sentrix chip:** per-chip median SD 0.020 there vs 0.012 elsewhere; chip-centring cuts every lab to 2–4 %. A laboratory constant cannot touch it. Row 5b was then measured on 23 complete chips at 9–12 arrays each (PROC-MAHA-03): the chip term is real (ICC 0.197, F 3.862, permutation p 0.0005) and **no panel size reduces the false-alarm tail** — a single held-out reference per chip doubles it, because the offset being removed is smaller than the error in estimating it. Sealed NOT COMMISSIONED; the four laboratories' rates stand as published and print beside every departure. Record: `Record/PROC_data/PROC-MAHA-01/`.

**ROW 6 CLOSED — CELLULAR AGE NOT REPORTABLE AT SINGLE-ARRAY RESOLUTION (PROC-AGE-01, 2026-09-21).** Inverting the healthy immune identity-gauge curve (`reference_age_curve_v1`) for one array resolves age to ~50 years: the curve moves 0.47 mA/yr and the within-laboratory spread is 0.0235; leave-one-lab-out on 1,379 healthy donors, 15.9 % within ±10 yr (bar 80 %), Spearman 0.27; a healthy 58-year-old inverts to 23. **The population aging trajectory stands and is reproduced** (0.47 mA/yr, monotone by decade, four labs = CPG-VAL-015's slope on Hannum; it is now the reference age curve). What is below resolution is one person's position on it. Sign differs by surface: marker-union A falls with age, identity-gauge A rises (RECON D2). `stage_6_cellular_age` returns `reportable=False` with the resolution sentence, which the report prints in place of an age; the marker-union inversion is `diagnostic_cellular_age`. With this, **no reported number in the chain reads the marker-union statistic.** Record: `Record/PROC_data/PROC-AGE-01/`.

**RECORD (PROC-RECORD-03, 2026-09-21).** The 80-cell [`age_reference_matrix.json`](Runtime%20Matrices/A_Scoring_Module/age_reference_matrix.json) is the April HEALTHY_BASELINES table: typed β_mean per decade with literature labels, A by formula, Gaussian percentiles around a typed SD; no generating script. Its direction is confirmed by measurement (PROC-AGE-01), its slope is ~2× the measured 0.47 mA/yr, its level is pre-scale-offset; it is read by no reported path (`reference_age_curve_v1` supersedes it). The AD and breast 'cellular age in years' results were ΔA read through that typed slope: the ΔA (AD immune d = −0.56, *younger*, i.e. senescence) is the measurement; the years are withdrawn as a unit.

- **Row 4.6 — the patient's sky — COMMISSIONED (PROC-CMB-05, 2026-09-21, five seals; C2′ 4/4 on the restated bar [0.025, 0.08]).** `cpg_conductor.run_full` bundle key `patient_sky`: z = (β − Σ f_c μ_c − m_lab)/s_lab on the mapped β, class panels gated by measured presence floors (`Runtime Matrices/Patient_CMB/`), HEALPix NSIDE 128 genomic order. NOT AVAILABLE without the laboratory's residual scale (built from the same 40-array healthy panel as the lab zero). Calibration constant stated on every sky: healthy held-out tail 2.6–3.2 %, not 5 % (C2′ failed as sealed by ≤ 0.004; recorded). The retired formula read 61 % of a healthy genome as anomalous (C1) and is closed. Kit test [`test_patient_sky.py`](../kit/test_patient_sky.py).

- **Row 7 — tiers — COMMISSIONED (PROC-TIER-01, 2026-09-21).** One tier function, `MethylPhys/chain/cpg_tiers.py`, reads [`tier_breakpoints.json`](Runtime%20Matrices/Tier_breakpoints/tier_breakpoints.json); no tier word on a non-reportable gauge (§108 / UNMAPPED / lab_zero UNSET); A ≥ 1/H_min → AT_CEILING. Measured, not moved: under the July 1.01 onset 30 % of 1,379 healthy donors read ELEVATED on the identity gauge (1.07 admits 1; 1.10 none; healthy central 95 % = 0.954–1.041). PROC-TIER-02 set NORMAL to the healthy central 95 % → `tier_breakpoints.json` v1.4 [0.95, 1.04): 2.5 % of healthy read ELEVATED. Kit test [`test_tiers.py`](../kit/test_tiers.py).

- **Row 8 — disease matching — REMOVED FROM THE CHAIN (author, 2026-09-21).** The signature matrix and cards come from the preliminary VAL record; the report shows cells detected, fractions, A per cell and class, placement and flags, and names no disease. The matrix is record-side (see `Disease Matrix/DISEASE_MATRIX/README_STATUS.md`). PROC-MATCH-01's fixes (fail-closed origin gate, firewall, surface = seal) stand. **Sealing rule:** we seal a built tool against a bar; building it is exploration with a working note, not a seal.

- **Row 4.5 — bidirectional detector — COMMISSIONED (PROC-BIDIR-01, 2026-09-21).** VAL-050/051 reproduce from the kit; engine == sealed formula (2e-16); 726 AIBL samples × 18 CpGs re-extracted from the raw GEO file match the sealed betas exactly. **Row 9 — the report:** [`MethylPhys_Interface/build_methylphys.py`](MethylPhys_Interface/build_methylphys.py) renders the author's spec (cells, %, A per class with placement/tier, A per cell, departure + false-alarm rate, sky, flags; no condition named, no years; vocabulary guard); old `cpg_report_builder.py` is record-side.
## The order of steps

[`doors/CHAIN_SEQUENCE.md`](../doors/CHAIN_SEQUENCE.md) is generated from the code by
`chain/build_chain_sequence.py` and is the authority: it lists every call each path makes, in order.

Two interfaces exist and they do not run the same steps.

- **`MethylPhys_Interface/run_sample.py`** — one sample, 23 steps. Runs the ten Stage 0 steps (nine checks and the decision) on the IDAT
  pair first (arrival, manifest, integrity hash, control probes, detection p, bead count, call rate, platform
  coverage, sex, decision); stops at the first refusal and exits without scoring on QUARANTINE. Then
  [`stage_1_idat_calibration.py`](stage_1_idat_calibration.py), then `cpg_conductor.run_full` for the 11
  stages beginning at composition, then the report. Every number in the commissioning record and in Issue 003
  comes from this path.
  Intake needs two things from you that the files do not carry: `--sex` and `--age`. `--intake-log` and
  `--manifest-dir` keep the custody record; `--array-type` is read from the file's header unless you override
  it; `--no-intake` skips the gates and the report says so.
- **[`run_batch.py`](run_batch.py)** — a folder of patient visits. Drives [`chain/disease_matching.py` (the v1 conductor ``chain/disease_matching.py` (the v1 conductor [`walther_clinical.py`](walther_clinical.py) was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called)` was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called)](`chain/disease_matching.py` (the v1 conductor `walther_clinical.py` was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called)), which runs its own
  stage functions (`stage_2_deconvolution`, `stage_4_a_score`, `stage_7_tiers`, `stage_8_dual_matching`, `run_second_chain`), not the conductor.

**Named as chain, called by nothing** — 3 files carry `role=chain` in the inventory and are not called by
either path: [`idat_decoder_pure.py`](idat_decoder_pure.py), [`idat_parse.py`](idat_parse.py) (a pure-Python
IDAT decoder; the chain reads IDATs through methylprep instead) and
[`lineage_splitter.py`](Lineage_Splitter/lineage_splitter.py). [`build_chain_sequence.py`](build_chain_sequence.py) prints this list from
the code, so it cannot drift from the tree.

**Stage 0 is in the live path (PROC-STAGE0-02, 2026-09-23).** All ten steps run - nine checks and the decision: the intensity-dependent
ones read the array's own control probes, negative controls, bead counts and chrX/chrY through
[`stage_0_1_qc_handoff.py`](stage_0_1_qc_handoff.py). Measured on 732 healthy arrays — the sex call agrees
with the depositors' own labels on 729 of 731. The bisulfite threshold is reported and not applied until it
is calibrated on healthy data ([`../doors/PROC_STAGE0_04_PREREG.md`](../doors/PROC_STAGE0_04_PREREG.md));
every other gate applies. What each refusal means, and what to do about it: under the step that refused it in the SOP (§11-§19), in Issue 003 section 3b, and on the report's Troubleshooting tab.

## Recording a run so it can be pooled later

A reading is only half of what a run should leave behind. The other half is what the specimen was declared to
be and what produced the number, and neither can be recovered afterwards from an HTML file.

```
python3 chain/MethylPhys_Interface/run_sample.py \
  --grn SAMPLE_Grn.idat.gz --red SAMPLE_Red.idat.gz \
  --age 61 --sex M --lab GSE87571 --lab-zero -0.0117 --specimen "whole blood" \
  --covariate diagnosis=case --covariate stage=II --covariate cohort=YOUR_COHORT \
  --intake-log custody/intake.jsonl --out reports/SAMPLE.html --id SAMPLE
```

Every run now writes three things, not one:

| what | where | why it matters |
|---|---|---|
| the report | `--out` | the reading, for a human |
| the bundle | beside the report, `_bundle.json` (`--no-bundle` to suppress) | every stage's output: per-class A with the floor, age term, laboratory zero and scale map applied and its z against the band; all 115 per-cell readings with a credible interval and marker coverage each; the departure axes; the sky statistics per class; the whole Stage 0 record including both file hashes |
| one ledger row | `evidence_ledger.jsonl` beside the report (`--ledger` to place it) | one flat line per run, so a cross-sample matrix is a file read rather than a re-run. **The column set is not fixed** - it grows with the classes that are gauged and the covariates you pass - so read the keys, do not assume a count. The example row shipped in `chain/example_runs/` has **268 columns**: 22 run-level scalars, 115 per-cell A values with 115 coverages, 2 per-class A with their z and band terms, 3 composition percentages and 5 covariate fields |

**Covariates are recorded, not reported.** `--covariate key=value` (repeatable, or `--covariates file.json`)
goes into the custody record, the bundle and the ledger row. It never reaches report prose: the report states
the number of covariate fields captured and nothing more, because a reading states what was measured and the
phenotype a specimen was declared with is not a measurement. The vocabulary guard enforces this - it refused
the key name `diagnosis` on a page, which is the behaviour we want.

**Provenance is on the page.** The Run tab now opens with what produced this reading: the run timestamp, the
chain commit (and whether the working tree was clean), the decoder version, and a SHA-256 of all fourteen
inputs the chain read - atlas, identity loci, band, scale maps, age curve, tier table, and the chain modules
themselves. Two readings are comparable only if those hashes match, and a reader can now check that without
opening a bundle.

# READ ME FIRST — everything from the 2026-09-18/19 sessions, in one folder

Heath — this folder is the complete record of what was produced and what was run. Nothing has been pushed to
your GitHub repository; every file below is either (a) copied unchanged from the repo, (b) copied from your uploads,
or (c) written in these sessions. Column "origin" says which.

Read in this order: this file → `RUNBOOK.md` (how to run the tests) → `COMPONENT_MAP.md` (repo vs. this folder vs. your machine).

---

## 1. Files in this folder

| path | origin | what it is |
|---|---|---|
| `README_FIRST.md` | written | this index |
| `RUNBOOK.md` | written | the four rules, two Python environments, where to get the large files, the five tests with expected/observed, the two rulings, open items |
| `COMPONENT_MAP.md` | written | every component of the chain: where it lives, what is stale in the repo, the eight commits to make |
| `CHECKSUMS.sha256` | generated | sha256 of every file here except `data/` — `sha256sum -c CHECKSUMS.sha256` |
| `DATA_CHECKSUMS.sha256` | generated | sha256 of the two large files that were used (atlas CSV, betas cache) |
| `cpg_kit.py` | written | helper shared by the five tests: loads your JSONs, both A-score formulas, GEO streaming, the presence + Jensen-gap guard |
| `PROC_CAL_01.py` | written | **Stage 1**: raw IDAT → β through your `stage_1_idat_calibration.py`; compares to `betas_cache.pkl`. Observed 11/11 bit-identical |
| `PROC_DECON_01.py` | written | **deconvolver** vs your `TEST_DATA_MANIFEST.md` answer key; whole-blood composition; presence-gated immune gauge with age band. Observed MAE 0.0004 |
| `PROC_ANCHOR_01.py` | written | **sealed 115-cell anchors** recomputed from raw GEO (GSE51032, GSE51057). Observed r = 1.00000 both |
| `PROC_FORMULA_01.py` | written | the two aggregations × two loci sets on the 11 test samples — the measurement behind RULING A3 |
| `PROC_PLASMA_MIX_01.py` | written | deconvolver vs Moss 2018 known in-vitro mixtures (GSE122126). terminal PASS; hepatocyte, colon FAIL |
| `engine/walther_iam_deconvolver.py` | repo 66f37fe | Stage 2 deconvolver — **run** |
| `engine/stage_1_idat_calibration.py` | repo 66f37fe | Stage 1 — **run** |
| `engine/cpg_gauge_engine.py` | repo 66f37fe | the 40-cell H_MIN_TABLE, HEALTHY_BASELINE, tiers — **read; constants extracted** |
| `engine/cpg_gauge.py` | repo 66f37fe | gauge wrapper — read, not run |
| `engine/iamatlas_a_scoring.py`, `engine/test_a_score_canonical.py` | repo 66f37fe | separation-surface scorer and its guard — read; the formula was re-implemented in `cpg_kit.separation_A` and matched the seal |
| `engine/cpg_conductor.py` | **your upload** (not in repo) | the 2026-07 orchestrator — **read, not run**; must be committed (COMPONENT_MAP action 8) |
| `runtime/iamatlas_gauge_identity_loci_v1_0.json` | repo | 8 identity panels, H_min, H_min_β, band — run |
| `runtime/iamatlas_celltype_markers_v0_2.json` | **your upload** (chrX-removed) | canonical markers (RULING M1b) — run |
| `runtime/iamatlas_celltype_markers_v0_2_REPO_HEAD_prechrX.json` | repo | the pre-fix copy the 2026-05-29 seal used — run, for the v1 comparison only |
| `runtime/age_reference_matrix.json`, `tier_breakpoints.json` | repo / your upload (identical) | age band, tiers — run |
| `runtime/IAMAtlasREBUILD_celltype_to_class.json`, `IAMAtlasREBUILD_provenance.json` | repo | class map, atlas build record — run |
| `anchors_v1/*.csv`, `anchors_v1/cohort_manifest.json` | repo | the 2026-05-29 seal — SUPERSEDED, keep |
| `anchors_v2/*.csv`, `anchors_v2/RESEAL_REPORT.json` | written | re-sealed under the chrX-removed markers (RULING M1b) — commit |
| `results/PROC_*.json` | generated | each test's printed block, as data |
| `results/VAL_INDEX.csv`, `.json` | generated | all 175 validation records (G, VAL-001..128, T1..T15, CPG-VAL-001..022, hull, N7, September PROCs; unique keys by series) in the repo with title, date, cohorts, stated decision, record completeness, path (Issue 003 Appendix V) |
| `results/*.json` (formula_2x2, anchor_recompute, stage1_conformance, mix/cfdna/sepsis/wholeblood_decon, identity_shift) | generated | raw outputs of the day's runs |
| `issue003_build/IAMPerformance_GAPEIssue003_RC1.pdf` | written | **Issue 003 draft, 263 pages** |
| `issue003_build/build_gape_issue003.py`, `data003.py`, `gape002_lib.py`, `val_index.json`, `fig_four_skies.*` | written | regenerates the PDF: `CPG_TRIAL=../runtime python build_gape_issue003.py out.pdf` |
| `issue003_build/FullVersion_build_gape_issue002.py` | your upload | the Issue 002 script the 003 build reuses verbatim |
| `figures/fig_four_skies.png/.pdf` | written | Fig 5A-1: Planck CMB / atlas immune posterior mean / posterior sd / patient z — made with your `cpg_patient_cmb.py` |
| `figures/fig1_predictions`, `fig3_cosmology`, `fig4_ascore` | written | figures of `IAM_for_physicists` |
| `figures/cfdna_substrate`, `cpg_new_001_results` | written | plasma substrate result; the CPG-NEW-001 test (recorded as a domain violation — see docs) |
| `docs/IAM_for_physicists.md` / `.tex` | written | the physicist-facing reference (Overleaf-ready) |
| `docs/IAM_technical_assessment.md`, `IAM_physics_audit.md`, `IAM_simplification_plan.md`, `IAM_walkthrough_log.md`, `IAM_mirror_map.md` | written | day-1 review memos — historical; several positions in them were later corrected (e.g. the two-instrument split) |
| `docs/CPG_first_read.md`, `CPG_LEDGER.md`, `CPG_v2_strategy.md`, `CPG_which_results.md`, `CPG_CRC_strategy.md`, `CPG_method_audit.md`, `CPG_breast_CRC_learned.md`, `CPG_deconvolver_first_run.md`, `CPG_substrate_characterization.md` | written | the CPG reading and testing record, in order |
| `docs/PREREG_CPG-NEW-001.md`, `OUTCOME_CPG-NEW-001.md` | written | the one fresh cohort test; outcome carries a domain-violation addendum (whole-blood gauge applied to tissue) |
| `docs/coverage_gaps.md` | written | audit of what the eight source documents contained vs. what Issue 003 carried |
| `repo_staging/…/CPG_VAL_047_Breast_per_patient_prediagnostic/` | **your upload** (VAL047.zip) | VAL-047's prereg, phases 1–12 results and logs, arranged for the repo with a README — **not yet pushed** |

**Not in this folder, withdrawn:** `fig2_mahaffey.png` (showed M = 30.21, wrong), `iam_ascore_structural_check.png` (day-1 figure superseded by fig4).

---

## 2. Files YOU must place in `data/` (too large for the folder; never in git)

| file | put at | source | sha256 |
|---|---|---|---|
| IAM Atlas, decompressed | `data/IAMAtlasREBUILD.csv` (605 MB) | repo `Biological_Physics/IAM_Atlas/IAMAtlasREBUILD.csv.xz` → `xz -dk` | `52ff4ccb35752ba0337f45c9563d7309c6fe9c4bdb8720daa196ab30f4596985` |
| Stage-1 calibrated betas, 11 samples | `data/betas_cache.pkl` (140 MB) | your `10_TEST_DATA.zip` | `764a8731f0fb72fc690f42f155cf5facb55639855668205725b1a95c0b6a6860` |
| raw IDATs, 11 pairs | `data/idats/` | your `10_TEST_DATA.zip` (`idats/` incl. `CPG_test_IDATs/`) | — |
| GSE51032 series matrix | `data/GSE51032_series_matrix.txt.gz` (3.0 GB) | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE51nnn/GSE51032/matrix/` | prereg-locked value in VAL-047 |
| GSE51057 series matrix | `data/GSE51057_series_matrix.txt.gz` (1.2 GB) | same path, GSE51057 | `828059824b67af46fb022f872ff9f69395e2e99b5975b3101157731a04d98bb0` (VAL047_prereg.json) |
| GSE122126 EPIC matrix (plasma + Moss mixes) | `data/GSE122126-GPL21145_series_matrix.txt.gz` | `.../GSE122nnn/GSE122126/matrix/` | — |
| Illumina manifests (Stage 1, first run only) | `$HOME/.methylprep_manifest_files/` | `https://array-manifest-files.s3.amazonaws.com/` — `HumanMethylation450k_15017482_v3.csv.gz`, `HumanMethylationEPIC_manifest_v2.csv.gz` | — |

Everything in §1 plus everything in §2 is the complete set. Nothing else was used.

---

## 3. What was run vs. only read (so nobody has to ask again)

**Run, with a known-answer check:** Stage 1 (11/11), deconvolver (3 tissue + 7 blood + 33 plasma + 9 mixes), separation A-score on both anchors (648 samples), the two-formula comparison, `cpg_patient_cmb.py` for the figure.
**Read, not run:** `cpg_conductor.py`, `walther_clinical.py`, Mahalanobis hull (Stage 5), disease cards / signature matrix (Stage 8), NILC, every historical VAL (indexed only).

---

## 4. Repository actions — DONE 2026-09-19

All nine actions listed in earlier versions of this file were applied in commits 9a96834 (content) and the
reorganization commit that followed (layout). The repository tree is now:
`Physics_of_Methylation/` (start here) · `IAM_Atlas/` · `CPG_Engine/` · `Testing_and_Code/` · `RETIRED/`.
See `Biological_Physics/README.md` for the map.

| `test_gauge_switch.py` | generated 2026-09-21 | PROC-SWITCH-01 S1–S3 conformance on the kit's cached whole-blood betas: the reported A is the identity-loci gauge; UNSET refuses — run |

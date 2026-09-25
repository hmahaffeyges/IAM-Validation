# COMPONENT MAP — what lives where, and what a future test needs

Three places, three jobs. Nothing should exist in two of them without this file saying which copy is canonical.

| place | job | rule |
|---|---|---|
| **REPO** `github.com/hmahaffeyges/IAM-Validation` | canonical engine, atlas, runtime constants, sealed anchors, lessons | the only place code is edited; one commit per change; the kit records the commit hash it was cut from |
| **KIT** `the chain bundle, generated on demand by kit/build_chain_bundle.py` | frozen snapshot of exactly what [[Issue 003](../manual/IAMPerformance_GAPEIssue003_RC1.pdf)](../manual/IAMPerformance_GAPEIssue003_RC1.pdf) used + the PROC scripts + the runbook | regenerated from the repo at a named commit; never hand-edited; if it disagrees with the repo, the repo wins and the kit is re-cut |
| **YOUR FOLDER** (local, not in git) | large inputs and private material | test IDATs, `betas_cache.pkl`, GEO matrices, decompressed atlas CSV, `_gape_constants_private.py`, Recipe, patents, correspondence |

Kit path prefixes below are relative to the kit root. Repo paths are relative to the repo root; `MethylPhys/chain/` = `Biological_Physics/MethylPhys/chain/`, `VAULT/` = `Biological_Physics/RETIRED_2026-09/PostBuild_atlas_vault_snapshot_2026-06/`.

---

## A. The measurement core (every PROC needs these)

| component | canonical location | in kit as | status 2026-09-19 |
|---|---|---|---|
| IAM Atlas (483,092 CpGs × 8 classes + 115 cells) | REPO `MethylPhys/atlas/IAMAtlasREBUILD.csv.xz` | the compressed `.xz` IS in the repository; the 605 MB decompressed `MethylPhys/atlas/IAMAtlasREBUILD.csv` is not, and is produced locally (decompress; sha256 in `DATA_CHECKSUMS.sha256`) | canonical |
| atlas provenance + build | REPO `MethylPhys/atlas/IAMAtlasREBUILD_provenance.json` | `runtime/` | canonical |
| cell type → class map (115 → 8) | REPO `MethylPhys/atlas/IAMAtlasREBUILD_celltype_to_class.json` | `runtime/` | canonical |
| Walther deconvolver (Stage 2) | REPO `MethylPhys/chain/Walther_iam_deconvolver/walther_iam_deconvolver.py` | `MethylPhys/chain/` | canonical; PROC-DECON-01 PASS |
| NILC deconvolver (cross-method check) | REPO `MethylPhys/chain/NILC Deconvolver/` | not in kit | **CUT from chain 2026-07-02**; code retained; OPEN whether it returns |
| gauge identity loci (8 panels, H_min, H_min_β, band) | REPO `MethylPhys/chain/Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json` | `runtime/` | canonical |
| cell-type markers v0_2 (115 × 100) | **YOUR FOLDER** (chrX-removed copy) → **must be committed to REPO** `MethylPhys/chain/Runtime Matrices/Celltype_Marker/` | `MethylPhys/chain/Runtime Matrices/Celltype_Marker/iamatlas_celltype_markers_v0_2.json` (canonical, RULING M1b) + `..._REPO_HEAD_prechrX.json` (what the v1 seal used) | **repo is stale on this file** |
| 40-cell H_MIN_TABLE, HEALTHY_BASELINE, tiers | REPO `MethylPhys/chain/cpg_gauge_engine.py` | `MethylPhys/chain/` | canonical; byte-identical to Issue 002 |
| age reference band (8 classes × 10 decades) | REPO `MethylPhys/chain/` (trial bundle copy identical) [`age_reference_matrix.json`](../chain/Runtime%20Matrices/A_Scoring_Module/age_reference_matrix.json) | `runtime/` | canonical; compiled as H(β̄)/H_min |
| tier breakpoints v1.3 | REPO [`tier_breakpoints.json`](../chain/Runtime%20Matrices/Tier_breakpoints/tier_breakpoints.json) (last commit 66f37fe) | `runtime/` | canonical; two vocabularies remain (RECON T3) |
| Stage 1 calibrator | REPO `MethylPhys/chain/stage_1_idat_calibration.py` | `MethylPhys/chain/` | canonical; PROC-CAL-01 PASS 11/11 |
| conductor (presence-paired scoring, DETECT_FLOOR 0.01; 'replaces `chain/disease_matching.py` (the v1 conductor [`walther_clinical.py`](../../RETIRED_2026-09/v1_conductor_2026-09/walther_clinical.py) was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called)', 2026-07) | REPO `MethylPhys/chain/cpg_conductor.py` - committed 2026-09; this row said 'YOUR FOLDER only, must be committed' until 2026-09-25, which was true when written and is not now | `MethylPhys/chain/cpg_conductor.py` (289-line version; a 95-line stub also circulates — discard it) | repo is missing the file the chain is defined by (RECON D2 for the 3% second floor) |
| runtime A-scoring module + canonical test | REPO `MethylPhys/chain/Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py`, [`test_a_score_canonical.py`](../chain/Runtime%20Matrices/A_Scoring_Module/test_a_score_canonical.py) | `MethylPhys/chain/` | separation surface (mean-of-H); guard docstring to be amended per RULING A3 |
| kit scoring helpers (gauge_A with presence + Jensen guard, separation_A) | **KIT** [`cpg_kit.py`](../kit/cpg_kit.py) | `cpg_kit.py` | new 2026-09-19; **should be promoted into REPO** next to cpg_conductor.py |

## B. Sealed references and anchors

| component | canonical location | in kit as | status |
|---|---|---|---|
| foundation-cohort anchors v1 (GSE51032 n=460, GSE51057 n=188, 115 cells) | REPO `Biological_Physics/Record/VAL_PostAtlas/foundation_cohort/` | `anchors_v1/` | SUPERSEDED by v2 (RULING M1b); keep |
| anchors v2 (chrX-removed markers) | **KIT** `anchors_v2/` → **must be committed to REPO** beside v1 | `anchors_v2/` + [`RESEAL_REPORT.json`](../../Record/VAL_PostAtlas/foundation_cohort/anchors_v2_chrXremoved/RESEAL_REPORT.json) | new 2026-09-19 |
| Mahalanobis HC hull v0_5 (n=2,523; d≥13.62 / 18.43) | REPO `MethylPhys/chain/Runtime Matrices/` (`mahalanobis_healthy_reference_*`) | not in kit | specified in Issue 003 §5A.7; **PROC-HULL-01 not yet written** |
| disease-signature matrix v1.13 (81 rows) | REPO `MethylPhys/chain/Disease Matrix/` | not in kit | Stage 8; not exercised in Issue 003 |
| disease cards + residual maps (breast-epic, ad-immune, …) | REPO `MethylPhys/Record/disease_cards_residual_maps/` | not in kit | Stage 8; not exercised |
| brightness files (8 per-class per-CpG μ/σ/CI) | REPO `VAULT/IAMAtlas_v0_1/class_archives/{class}_v0_1_REBUILD.tar.xz` | not in kit | Stage 4.6 input |

## C. Visualization / CMB toolkit (the two folders you zipped)

| component | canonical location | your zip | in kit | status |
|---|---|---|---|---|
| Plates 1–4 + README | REPO `VAULT/IAMAtlas_v0_1/plates/` | `Mollweide & Brightness Comparison/Plates/` | no | canonical in repo; your zip has two extra plate variants (`_Cosmic_Methylome_Background`, `_Methylome_CMB_vs_Microwave_CMB`) **not in the repo** — commit or discard |
| HEALPix mapping (NSIDE 128, generator, provenance, .npy) | REPO `VAULT/IAMAtlas_v0_1/healpix_mapping/` | `cpg healpix mapping/` | no | canonical in repo (your zip lacks the .npy) |
| patient_brightness_comparison.py (Stage 4.6 module) | REPO `VAULT/walther_clinical (RETIRED 2026-09-25 -> RETIRED_2026-09/v1_conductor_2026-09/; its one live function is chain/disease_matching.py)_runtime/Brightness_Comparison/` | `Mollweide & Brightness Comparison/` | no | canonical in repo |
| cpg_patient_cmb.py (Stage 4.6, engine version, z-map with assessability mask) | REPO `MethylPhys/chain/cpg_patient_cmb.py` | — | no | canonical; supersedes the vault module for the running chain |
| brilliance_map.py, patient_brilliance_map_GSM1051533.png | REPO `MethylPhys/chain/cpg_conductor.py` - committed 2026-09; this row said 'YOUR FOLDER only, must be committed' until 2026-09-25, which was true when written and is not now | root of zip | no | **not in repo** — the PNG is a raw-β map, not the Stage-4.6 z-map; decide whether it ships |

## D. Documents

| component | canonical location | in kit | status |
|---|---|---|---|
| Issue 003 build chain (build, data003, gape002_lib, 002 script) | **KIT** `issue003_build/` → **should be committed to REPO** `MethylPhys/manual/` | yes | new |
| Issue 003 PDF | KIT `issue003_build/` | yes | v8 |
| SOP v2.0.0 (current) | **YOUR FOLDER** → **must be committed**; repo holds v1.3 under `VAULT/walther_clinical (RETIRED 2026-09-25 -> RETIRED_2026-09/v1_conductor_2026-09/; its one live function is chain/disease_matching.py)_runtime/` | no | repo is stale; §105 to be amended per RULING A3 |
| LESSONS_LEARNED.md, CPG_Lessons_Learned_2026-06-29.md, CHANGELOG.md, README_FOR_FUTURE_AI.md | REPO `MethylPhys/chain/` and `MethylPhys/Record/chain_readme_archive/` | no | canonical |
| Recipe, `_gape_constants_private.py`, patents, correspondence | **YOUR FOLDER** (vault IP) | no | never in repo or kit; `_gape_constants_private.py` still names the constant `n_bio` (retired name; the value 20.94 is the Mahaffey number) |
| GAPE_EDEAR_Reproduction_Paper_v3, IAM_for_physicists, Hubble2GAPE, Cellular Margin, Astro-Genetics | REPO `docs/papers/` or `Biological_Physics/papers/` (check each) | no | verify each is committed |

## E. Large inputs (your folder, never git)

| file | size | source | sha256 |
|---|---|---|---|
| `IAMAtlasREBUILD.csv` (decompressed) | 605 MB | repo `.xz` | `52ff4ccb…` (DATA_CHECKSUMS) |
| `betas_cache.pkl` | 140 MB | 10_TEST_DATA.zip | `764a8731…` |
| 11 IDAT pairs | ~200 MB | 10_TEST_DATA.zip / GEO suppl | — |
| GSE51032 / GSE51057 / GSE122126 series matrices | 3.0 / 1.2 / 0.9 GB | GEO FTP | — |

---

## F. To run any future test, a machine needs exactly

1. the **REPO** at a named commit (engine, atlas `.xz`, runtime JSONs, anchors, lessons);
2. the **KIT** cut from that commit (runbook, PROC scripts, [`cpg_kit.py`](../kit/cpg_kit.py), checksums) — or, once promoted, the same files inside the repo;
3. from **YOUR FOLDER**: the decompressed atlas, `betas_cache.pkl`, the IDATs, and whichever GEO matrices the test names;
4. two environments per `RUNBOOK.md §1`.

## G. Repo actions this map implies (in order)

1. commit the chrX-removed [`iamatlas_celltype_markers_v0_2.json`](../chain/Runtime%20Matrices/Celltype_Marker/iamatlas_celltype_markers_v0_2.json) (RULING M1b)
2. commit `anchors_v2/` beside `foundation_cohort/`, mark v1 SUPERSEDED in `cohort_manifest.json`
3. commit SOP v2.0.0 to `VAULT/walther_clinical (RETIRED 2026-09-25 -> RETIRED_2026-09/v1_conductor_2026-09/; its one live function is chain/disease_matching.py)_runtime/`, then amend §105 per RULING A3
4. commit the two extra plates from your zip (or delete them locally)
5. promote [`cpg_kit.py`](../kit/cpg_kit.py) and the five `PROC_*.py` into `MethylPhys/kit/`, and `issue003_build/` into `MethylPhys/manual/`
6. amend [`test_a_score_canonical.py`](../chain/Runtime%20Matrices/A_Scoring_Module/test_a_score_canonical.py)'s guard to name its surface and add a gauge-surface test
7. rename `n_bio` -> Mahaffey number (20.94) in [`cpg_gauge_engine.py`](../chain/cpg_gauge_engine.py), `_gape_constants_private.py`, and the 002 card text; reconcile IAM_Hubble2GAPE l.2192 (ln2 form, ~30 - wrong)
8. **commit [`cpg_conductor.py`](../chain/cpg_conductor.py) (289-line, 2026-07) to `MethylPhys/chain/`** — the running chain's orchestrator exists only in your folder


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Record/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

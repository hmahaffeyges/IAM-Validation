# CPG Engine — the running code

The chain that scores a sample. Formerly `AstroGenetics/CPG_KISS_Commercial_Engine/`; renamed 2026-09-19, contents unchanged except where noted in the CHANGELOG.

| stage | file(s) |
|---|---|
| 0 intake | `stage_0_intake.py`, `cpg_intake_form.html`, `preflight.py` |
| 1 calibration (raw IDAT → β, methylprep noob) | `stage_1_idat_calibration.py` (`stage_1_calibration.py`, `idat_decoder_pure.py`, `idat_parse.py` are the pure-Python path) |
| 2 deconvolution (composition, presence) | `Walther_iam_deconvolver/walther_iam_deconvolver.py` — reads `../IAM_Atlas/IAMAtlasREBUILD.csv` |
| 4 class gauge + 7 tier | `cpg_gauge_engine.py` (the 40-cell `H_MIN_TABLE`, age band, tiers), `cpg_gauge.py`, `Runtime Matrices/` |
| 4.6 patient CMB | `cpg_patient_cmb.py` — z-departure sky against the atlas posterior; uses `../IAM_Atlas/healpix_mapping/` |
| 5 second chain | `stage_5_second_chain.py` (Mahalanobis hull) |
| orchestrator | **`cpg_conductor.py`** (2026-07; replaces `walther_clinical.py`, kept for reference) |
| nulls (sealing) | `CPG_Null_Runner/cpg_null_runner.py` — the eight nulls N1–N8; a VAL is sealed only when its declared nulls pass |
| report | `cpg_report_builder.py`, `report_builders/` (strawman, patient wall, synthetic-patient harness), `build_dashboard_v1.py` |
| runtime constants | `Runtime Matrices/` — identity loci, discriminative markers (chrX-removed, canonical), age reference, tiers, Mahalanobis reference, directional panels |
| test data | `TEST_DATA/` — 11 IDAT pairs + `TEST_DATA_MANIFEST.md` (documented expected outputs); `betas_cache.pkl` is not in git (see the Reproduction Kit) |
| disease side | `Disease Matrix/`, `Disease Cards : Residual Maps/`, `Crown Jewel and Patient Strawman/` |
| documents | `flowchart_vKISS.html` (stage map), `CPG_Doctor_Workflow_KISS.html`, `CPG_AstroGenetics_explainer_section.html`, `README's/README_FOR_FUTURE_AI.md`, `CHANGELOG.md`, `ROADMAP_TaskTracker.md`, `RUN_MANIFEST_and_README.md`, `CPG_Lessons_Learned_2026-06-29.md` |

**Two scoring surfaces, one rule each** (SOP §106): the class **gauge** on identity loci is `H(β̄)/H_min`; the 115-cell **separation** statistic on discriminative markers is the mean of per-CpG entropies. `Runtime Matrices/A_Scoring_Module/test_a_score_canonical.py` guards the separation surface; the gauge's Jensen-gap guard is in the Reproduction Kit's `cpg_kit.py`.

Verification of this code against known answers: [`../Physics_of_Methylation/Reproduction_Kit/`](../Physics_of_Methylation/Reproduction_Kit/).

Known: `report_builders/render_strawman_v2.py` and `render_patient_wall.py` use Python ≥ 3.12 f-string syntax.


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

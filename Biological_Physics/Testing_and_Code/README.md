# Testing and Code — every validation run

Every pre-registered validation (VAL) in the cellular track, with its prereg, outcome and results where they exist, split by whether it ran before or after the IAM Atlas was built (2026-05-28; first Atlas-based runs 2026-06-03).

| folder | what |
|---|---|
| [`VAL_INDEX.csv`](VAL_INDEX.csv) | **the map** — all 103 VAL identifiers: title, date, era, cohorts, stated decision, record completeness, path |
| [`VAL_PreAtlas/`](VAL_PreAtlas/) | runs before the Atlas (VAL-037 … VAL-141 numbering, April–May 2026): the TCGA tumour/adjacent-normal work, the breast pre-diagnostic anchors' first form, CRC, AD, the cross-population runs. Restored VAL-047 is here |
| [`VAL_PostAtlas/`](VAL_PostAtlas/) | runs on the Atlas (CPG_VAL_001 … 022, June 2026): breast per-cell fan-out, Mahalanobis hull, AIBL, immune ageing, plus `foundation_cohort/` — the **sealed 115-cell anchors** (GSE51032, GSE51057; v1 of 2026-05-29 and `anchors_v2_chrXremoved/` of 2026-09-19) — and `post_build_evidence_reports/` |
| [`Cohorts_and_Manifests/`](Cohorts_and_Manifests/) | GEO/TCGA manifests, matched-pair lists, the `extract_*.py` scripts, `ad_immune_cohorts/`, `breast_epic_cohorts/`, `cross_population/`, `CASCADE_SUMMARY` |
| [`DETAILED_VALIDATION_RECORD.md`](DETAILED_VALIDATION_RECORD.md) | the narrative record, per study, caveats first |
| `README_validation_runs_original.md` | the original `validation_runs/` README, kept verbatim |

Outcome files use pre-registered codes (`O1_PRIMARY_VALIDATED`, `O2_…`, `O3_INVERTED`, `NULL`, `DIRECTIONAL`) and a `**Status:**` line; older files state the verdict in prose. The index does not paraphrase verdicts — where the file has no explicit code it says *see file*.

How these runs relate to the current instrument, and which of them were reproduced from raw data in September 2026 (the foundation-cohort anchors: r = 1.00000 on 648 samples), is in Issue 003 §10 and Appendix V.


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Testing_and_Code/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.

**H_min CROSS-CHECK (PROC-HMIN-BOOT-01, 2026-09-20).** The eight methylation floors were calibrated by G-002 MCMC (37 reference cells, R-hat < 1.001). The April bootstrap cross-check (0.168 %, 24/32 in CI) covered the 32 non-methylation floors only; the methylation eight were bootstrapped for the first time on 2026-09-20 — 8/8 inside the 95 % CI, 0.060 % mean / 0.095 % max relative difference. Values unchanged. The calibration scripts are not at HEAD (removed 2026-04-19 as commercial) but are in public history at 22749f0 — a disclosure decision for the author. Record: `Testing_and_Code/PROC_data/PROC-HMIN-BOOT-01/`.

**Where the evidence report's `Biological_Physics/evidence/` links now point.** The calibration scripts moved to [`../Physics_of_Methylation/Hmin_Calibration/`](../Physics_of_Methylation/Hmin_Calibration/) (restored 2026-09-20; also in Zenodo 10.5281/zenodo.19633499). The retired pre-Atlas evidence report itself — its twelve findings are cohort-level, summary-β, within-pipeline results, as its own caveats tab states 271 times — is the April snapshot; its first caveat anticipated the pipeline-scale offset measured in September (LESSON-SCALE-01).

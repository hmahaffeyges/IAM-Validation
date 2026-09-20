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

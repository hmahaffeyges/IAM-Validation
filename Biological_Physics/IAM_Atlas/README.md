# IAM Atlas — the derived reference

The canonical atlas: **483,092 CpGs × 115 cell types**, built 2026-05-28 by per-class MCMC (batch 5,000; immune 1,500), with a posterior **mean, sd, and 95 % CI per CpG per class** — 262 columns. This is the "brilliance" surface every patient is read against.

| file | what |
|---|---|
| `IAMAtlasREBUILD.csv.xz` (100 MB) | the atlas. Decompress in place: `xz -dk IAMAtlasREBUILD.csv.xz` → 605 MB CSV, sha256 `52ff4ccb…6985` |
| `IAMAtlasREBUILD_provenance.json` | build record — sources, sampler settings, date |
| `IAMAtlasREBUILD_celltype_to_class.json` | 115 cell types → 8 architecture classes |
| `iamatlas_class_archives/` | per-class MCMC archives (`*_v0_1_REBUILD.tar.xz`) |
| `healpix_mapping/` | CpG → HEALPix pixel (NSIDE 128, 196,608 pixels, atlas row order) used by every plate and by `CPG_Engine/cpg_patient_cmb.py` |
| `external_manifests/` | Illumina 450K/EPIC manifests used to annotate the atlas |
| `iamatlas_v0_1_mcmc_batched_FIXED.py`, `compact_atlas.py` | the build scripts |
| `IAMAtlas_FLATNESS_LESSON.md` | why the pre-build atlas was flat and what the rebuild changed |

The eight class floors `H_min` are derived from this atlas's healthy-reference posteriors (G-002/G-003b chains) and are printed in Issue 003 with their provenance. The identity loci the gauge reads (`CPG_Engine/Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json`) and the discriminative markers the separation surface reads (`…/Celltype_Marker/`) are both selected from this atlas.

Code that loads the atlas resolves it as `Biological_Physics/IAM_Atlas/IAMAtlasREBUILD.csv` relative to `CPG_Engine/`.


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

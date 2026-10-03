# IAM Atlas — the derived reference

The canonical atlas: **483,092 CpGs × 115 cell types**, built 2026-05-28 by per-class MCMC (batch 5,000; immune 1,500), with a posterior **mean, sd, and 95 % CI per CpG per class** — 262 columns. This is the surface every specimen is read againstt.

| file | what |
|---|---|
| [`IAMAtlasREBUILD.csv.xz`](IAMAtlasREBUILD.csv.xz) (100 MB) | the atlas. Decompress in place: `xz -dk IAMAtlasREBUILD.csv.xz` → 605 MB CSV, sha256 `52ff4ccb…6985` |
| [`IAMAtlasREBUILD_provenance.json`](IAMAtlasREBUILD_provenance.json) | build record — sources, sampler settings, date |
| [`IAMAtlasREBUILD_celltype_to_class.json`](IAMAtlasREBUILD_celltype_to_class.json) | 115 cell types → 8 architecture classes |
| `iamatlas_class_archives/` | per-class MCMC archives (`*_v0_1_REBUILD.tar.xz`) |
| `healpix_mapping/` | CpG → HEALPix pixel (NSIDE 128, 196,608 pixels, atlas row order) used by every plate and by [`stage_4_6_patient_cmb.py`](../chain/stage_4_6_patient_cmb.py) (the sky) |
| `external_manifests/` | Illumina 450K/EPIC manifests used to annotate the atlas |
| [`iamatlas_v0_1_mcmc_batched_FIXED.py`](iamatlas_v0_1_mcmc_batched_FIXED.py), [`compact_atlas.py`](compact_atlas.py) | the build scripts |
| [`IAMAtlas_FLATNESS_LESSON.md`](IAMAtlas_FLATNESS_LESSON.md) | why the pre-build atlas was flat and what the rebuild changed |

The eight class floors `H_min` were fitted by the G-002 MCMC on 37 published reference cell methylomes (April 2026) and are frozen; this atlas's per-locus posteriors are what a specimen's composition is predicted from and were printed with their provenance in the Edition 003 operations manual (retired with chain v2 on 2026-10-03, archived privately). The identity loci the gauge reads (MethylPhys/chain/Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json (archived privately)) and the discriminative markers the separation surface reads (`…/Celltype_Marker/`) are both selected from this atlas.

Code that loads the atlas resolves it as Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD.csv (decompressed from IAMAtlasREBUILD.csv.xz at run time) relative to `MethylPhys/chain/`.

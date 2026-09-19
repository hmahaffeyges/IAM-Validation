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

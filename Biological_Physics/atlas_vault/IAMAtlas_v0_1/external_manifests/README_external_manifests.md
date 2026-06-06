# External manifests — third-party CpG annotation data

These files are external dependencies (not produced by the framework) — public CpG annotation data needed by the framework's tools at IAMAtlas build time.

## EPIC_v1_B4_manifest_normalized.csv

**Source:** [zhou-lab/InfiniumAnnotationV1](https://github.com/zhou-lab/InfiniumAnnotationV1) (zhou-lab curated EPIC v1 hg19 annotation, derived from Illumina's official manifest).
**Direct download URL:** `https://raw.githubusercontent.com/zhou-lab/InfiniumAnnotationV1/main/Anno/EPIC/EPIC.hg19.manifest.tsv.gz`
**Date acquired:** 2026-06-06
**Row count:** 862,927 cg-prefixed CpGs (after dropping control / rs / ch / unmapped probes).
**Coverage of IAMAtlas:** 450,192 / 483,092 CpGs (93.2%); remaining 32,900 are HM450-only and absent from EPIC.

## HM450 manifest (HM450.hg19.manifest.tsv.gz, source)

**Source:** Same zhou-lab repo — `https://raw.githubusercontent.com/zhou-lab/InfiniumAnnotationV1/main/Anno/HM450/HM450.hg19.manifest.tsv.gz`
**Date acquired:** 2026-06-06
**Row count:** 482,421 cg-prefixed CpGs (HM450 array probes).

## EPIC_plus_HM450_combined_manifest_normalized.csv  **← CANONICAL FOR STAGE 4.6**

**Source:** Union of EPIC + HM450 manifests above (EPIC takes priority where both arrays carry a CpG).
**Date built:** 2026-06-06
**Row count:** 895,827 cg-prefixed CpGs.
**Coverage of IAMAtlas:** **483,092 / 483,092 CpGs (100%)** — full coverage of every atlas CpG. Zero sentinel pixels in patient Mollweide.

**Schema (all three CSVs):**
| Column | Type | Description |
|---|---|---|
| IlmnID | string | Illumina probe identifier (e.g. cg00000029) |
| CHR | string | Chromosome (e.g. "1", "X", "Y") with "chr" prefix stripped |
| MAPINFO | int | Genomic position (hg19) |
| platform | string | EPIC or HM450 (combined manifest only) |

**Used by:** `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/generate_cpg_healpix_mapping.py` — the one-time generator for `iamatlas_cpg_to_healpix_nside128.npy` (the Stage 4.6 patient Mollweide projection mapping).

**Versioning:** hg19 (the genome build the IAMAtlas was constructed against). If/when the IAMAtlas migrates to hg38 or to EPIC v2 (~937K probes), corresponding hg38/v2 manifests are acquired and the mapping regenerated.

**License:** Per zhou-lab/InfiniumAnnotationV1: redistribution permitted with attribution. Original Illumina manifest data is publicly distributed by Illumina.

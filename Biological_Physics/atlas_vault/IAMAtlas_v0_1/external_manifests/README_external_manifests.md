# External manifests — third-party CpG annotation data

These files are external dependencies (not produced by the framework) — public CpG annotation data needed by the framework's tools at IAMAtlas build time. Currently:

## EPIC_v1_B4_manifest_normalized.csv

**Source:** [zhou-lab/InfiniumAnnotationV1](https://github.com/zhou-lab/InfiniumAnnotationV1) (zhou-lab curated EPIC v1 hg19 annotation, derived from Illumina's official manifest).

**Direct download URL:** `https://raw.githubusercontent.com/zhou-lab/InfiniumAnnotationV1/main/Anno/EPIC/EPIC.hg19.manifest.tsv.gz`

**Date acquired:** 2026-06-06

**Normalization:** Original TSV columns renamed/filtered to match the Illumina manifest standard expected by the framework: `Probe_ID → IlmnID`, `CpG_chrm → CHR` (with "chr" prefix stripped), `CpG_beg → MAPINFO`. Only cg-prefixed probes retained (control probes, rs* SNP probes, and ch.* probes dropped).

**Schema:**
| Column | Type | Description |
|---|---|---|
| IlmnID | string | Illumina probe identifier (e.g. cg00000029) |
| CHR | string | Chromosome (e.g. "1", "X", "Y") with "chr" prefix stripped |
| MAPINFO | int | Genomic position (hg19) |

**Row count:** 862,927 cg-prefixed CpGs (after dropping control / rs / ch / unmapped probes from the parent 866,554-probe manifest).

**Used by:** `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/generate_cpg_healpix_mapping.py` — the one-time generator for `iamatlas_cpg_to_healpix_nside128.npy` (the Stage 4.6 patient Mollweide projection mapping). Covers the 483,093-CpG IAMAtlas REBUILD with margin.

**Versioning:** This manifest is `hg19` (the genome build the IAMAtlas was constructed against). If/when the IAMAtlas migrates to hg38 or to EPIC v2 (~937K probes), a corresponding hg38/v2 manifest is acquired and the mapping regenerated.

**License:** Per zhou-lab/InfiniumAnnotationV1: redistribution permitted with attribution. Original Illumina manifest data is publicly distributed by Illumina.

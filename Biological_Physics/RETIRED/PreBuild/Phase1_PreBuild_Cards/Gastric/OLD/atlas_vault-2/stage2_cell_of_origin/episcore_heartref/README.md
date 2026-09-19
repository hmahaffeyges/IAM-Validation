# EpiSCORE HeartRef — cardiac cell-type methylation reference (CpG-bridged)

## Source
- **Paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Communications 2022;13:3895. DOI: [10.1038/s41467-022-31805-3](https://doi.org/10.1038/s41467-022-31805-3)
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source file:** `data/HeartRef.rda` (commit master @ 2026-04-29)

## Bridging methodology
EpiSCORE distributes the heart reference matrix `mrefHeart.m` indexed by **Entrez Gene IDs** (207 markers × 5 cell types + weight column). The production scorer reads CpG-level reference matrices. Bridge methodology:

1. Load `mrefHeart.m` (207 Entrez × 6 columns) from `HeartRef.rda`.
2. Load `probeInfo450k.lv` (the EpiSCORE-distributed bridge from 450K CpG probes to Entrez Gene IDs).
3. For every probeInfo entry whose EID is in the 207 HeartRef Entrez IDs, emit a (probeID, EID, CM, EC, FB, MP, SMC, weight) row. The Entrez-level methylation profile is broadcast to every 450K CpG probe mapping to that gene.

This is the same bridging methodology used for VAL-094 (EpiSCORE BreastRef bridge).

## Final dimensions
- **3,727 unique 450K CpG probes × 5 cardiac cell types**
- 199 unique Entrez Gene IDs covered (8 of the 207 source Entrez IDs had no probeInfo450k mapping)
- Cell types: **CM** (cardiomyocyte), **EC** (endothelial), **FB** (fibroblast), **MP** (macrophage), **SMC** (smooth muscle)
- `weight` column preserved for EpiSCORE's weighted-NNLS deconvolution mode

## Files
- `episcore_heartref_cpg_bridged.csv` — production-ready CpG × cell-type matrix. **SHA-256:** `bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83`
- `episcore_heartref_entrez_matrix.csv` — original Entrez-keyed matrix (audit trail)
- `HeartRef_source.rda` — original R-data source from EpiSCORE GitHub

## License
GPL-2 per EpiSCORE repository.

## Used by
- VAL-111 (cardio-epic Stage 2 EpiSCORE HeartRef on GSE69138 + GSE84395 + GSE84274)
- cardio-epic v0.2 production deployment Stage 2 layered atlas (cardiac sub-cell-type resolution beyond Loyfer Vascular_endothelial_cells / Left_atrium tiles)

## Frozen
2026-04-29

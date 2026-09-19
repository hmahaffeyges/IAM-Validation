# EpiSCORE ProstateRef — prostate cell-type methylation reference (CpG-bridged)

## Source
- **Paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: [10.1038/s41592-022-01412-7](https://doi.org/10.1038/s41592-022-01412-7)
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source file:** `data/ProstateRef.rda` (commit master @ 2026-04-30)

## Bridging methodology
EpiSCORE distributes the prostate reference matrix `mrefProstate.m` indexed by **Entrez Gene IDs** (163 markers × 6 cell types + weight column). The production scorer reads CpG-level reference matrices. Bridge methodology:

1. Load `mrefProstate.m` (163 Entrez × 7 columns) from the EpiSCORE-distributed `ProstateRef.rda` (in atlas_vault as `ProstateRef__mrefProstate_m.csv`).
2. Load `probeInfo450k.lv` (the EpiSCORE-distributed bridge from 450K CpG probes to Entrez Gene IDs).
3. For every probeInfo entry whose EID is in the 163 ProstateRef Entrez IDs, emit a (probeID, EID, BE, EC, Fib, LE, Leu, SM, weight) row. The Entrez-level methylation profile is broadcast to every 450K CpG probe mapping to that gene.

This is the **same** bridging methodology used for VAL-094 (BreastRef bridge) and VAL-111 (HeartRef bridge).

## Final dimensions
- **2603 unique 450K CpG probes × 6 prostate cell types**
- 159 unique Entrez Gene IDs covered (of 163 source Entrez IDs)
- Cell types: **BE** (basal epithelial), **EC** (endothelial), **Fib** (fibroblast), **LE** (luminal epithelial — prostate adenocarcinoma cell of origin), **Leu** (leukocytes), **SM** (smooth muscle)
- `weight` column preserved for EpiSCORE's weighted-NNLS deconvolution mode

## Files
- `episcore_prostateref_cpg_bridged.csv` — production-ready CpG × cell-type matrix. **SHA-256:** `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2`
- `episcore_prostateref_entrez_matrix.csv` — original Entrez-keyed matrix (audit trail)
- (Source `ProstateRef__mrefProstate_m.csv` lives in the EpiSCORE pan-tissue folder and stays there as part of the broader pan-tissue MANIFEST.)

## License
GPL-2 per EpiSCORE repository.

## Used by
- VAL-114 — prostate-epic Stage 2 EpiSCORE ProstateRef calibration on TCGA-PRAD adjacent-normal n=50 (HM450 sesame Level 3)
- prostate-epic v0.3 production deployment Stage 2 layered atlas (prostate sub-cell-type resolution beyond Loyfer's single `prostate_epithelial` tile)

## Frozen
2026-04-30

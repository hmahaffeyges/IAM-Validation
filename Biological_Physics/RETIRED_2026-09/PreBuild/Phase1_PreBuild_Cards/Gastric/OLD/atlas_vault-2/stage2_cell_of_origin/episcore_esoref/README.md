# EpiSCORE EsoRef — esophageal cell-type methylation reference (CpG-bridged)

## Source
- **Paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: [10.1038/s41592-022-01412-7](https://doi.org/10.1038/s41592-022-01412-7)
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source file:** `data/EsoRef.rda` (commit master @ 2026-05-02, fetched via raw.githubusercontent.com)

## Bridging methodology
EpiSCORE distributes the esophageal reference matrix `mrefEso.m` indexed by **Entrez Gene IDs** (163 markers × 8 cell types + weight column). The production scorer reads CpG-level reference matrices. Bridge methodology:

1. Load `mrefEso.m` (163 Entrez × 9 columns: EC, Epi_basal, Epi_stratified, Epi_suprabasal, Epi_upper, Fib, Gland, IC, weight) from the EpiSCORE-distributed `EsoRef.rda` (in atlas_vault as `EsoRef__Eso_Mref_m.csv`).
2. Load `probeInfo450k.lv` (the EpiSCORE-distributed bridge from 450K CpG probes to Entrez Gene IDs; 485,577 array probes; 331,229 with EID; 19,357 unique EIDs).
3. For every probeInfo entry whose EID is in the 163 EsoRef Entrez IDs, emit a (probeID, EID, EC, Epi_basal, Epi_stratified, Epi_suprabasal, Epi_upper, Fib, Gland, IC, weight) row. The Entrez-level methylation profile is broadcast to every 450K CpG probe mapping to that gene.

This is the **same** bridging methodology used for VAL-094 (BreastRef bridge), VAL-111 (HeartRef bridge), VAL-117 (ProstateRef bridge), and VAL-119 (BladderRef bridge).

## Final dimensions
- **2,464 unique 450K CpG probes × 8 esophageal cell types**
- Cell types (8):
  - **EC** = Vascular Endothelial Cells
  - **Epi_basal** = Basal squamous epithelium (stem-cell layer at basement membrane)
  - **Epi_stratified** = Stratified squamous epithelium (layered above basal)
  - **Epi_suprabasal** = Suprabasal squamous epithelium
  - **Epi_upper** = Upper squamous epithelium (surface-most layer)
  - **Fib** = Fibroblasts (stromal)
  - **Gland** = Glandular cells (submucosal mucous-secreting esophageal glands)
  - **IC** = Immune Cells (intra-esophageal)
- `weight` column preserved for EpiSCORE's weighted-NNLS deconvolution mode

## Class assignments (H_min anchors per GAPE_WEB_v13.py lines 87-96)
- **Epi_basal** → secretory class      H_min = 0.843264 (basal squamous is a stem-progenitor secretory layer)
- **Epi_stratified** → secretory class H_min = 0.843264
- **Epi_suprabasal** → secretory class H_min = 0.843264
- **Epi_upper** → secretory class      H_min = 0.843264
- **Gland** → secretory class          H_min = 0.843264 (esophageal mucous gland)
- **EC** → stromal class               H_min = 0.862950
- **Fib** → stromal class              H_min = 0.862950
- **IC** → immune class                H_min = 0.838889

## Calibration anchor (VAL-124)
- **Source cohort:** TCGA HM450 sesame Level 3 adjacent-normal n=210 (TCGA-KIRC + TCGA-PRAD; same VAL-106 calibration cohort that anchored VAL-117/119/123)
- **Sealed prereg SHA:** `1bab7c99b35a3ebc680e93e6935a84f2b712fe7ec6663632d696a6a92433090f`
- **Outcome:** O1_CALIBRATION_SEALED
- **CHK gates:** CHK-3.1A 100%, CHK-3.1B 100% above ≥80% threshold, CHK-3.1C 0 duplicates
- **Cross-tile separation:** 0.0990 (largest observed across any EpiSCORE bridge calibrated to date — significantly above ProstateRef's own ~0.06 separation in VAL-117); flagged for kidney-card cross-card calibration follow-up to discriminate atlas overread from genuine cross-tissue gene-promoter biology
- **Most-elevated tile:** Epi_upper (mean 0.4698)
- **Least-elevated tile:** Gland (mean 0.3708)

## Files
- `episcore_esoref_cpg_bridged.csv` — production-ready CpG × cell-type matrix (10 columns: probeID, EID, EC, Epi_basal, Epi_stratified, Epi_suprabasal, Epi_upper, Fib, Gland, IC, weight). **SHA-256:** `6e650bd78ed2ee32d98ac4508b3b4295a1cd303a0fed2d850eeb9c35f897692b`
- `episcore_esoref_entrez_matrix.csv` — original Entrez-keyed matrix (audit trail). **SHA-256:** `6d27782d8a3fb40dc9d663ca534e2b2880b103213dfba7ca27110dca10031f0b`
- `bridge_esoref_to_array.py` — reproducibility script for the Entrez → 450K CpG broadcast

## License
GPL-2 per EpiSCORE repository.

## Used by
- VAL-124 — esophageal-epic Stage 2 EpiSCORE EsoRef calibration on TCGA HM450 adjacent-normal n=210
- VAL-126 — TCGA-STAD Phase C run-everything (cross-tissue overread test on gastric adenocarcinoma)
- VAL-127 — TCGA-ESCA Phase C run-everything (cell-of-origin retention test on esophageal squamous and adenocarcinoma subtypes; produced the cleanest EsoRef-on-target-tissue finding: Epi_stratified d=−0.99 in ESCC vs anchor)
- gastric+esophageal-epic v0.1 production deployment Stage 2 layered atlas

## Frozen
2026-05-02

# EpiSCORE OEref — oral epithelium cell-type methylation reference (CpG-bridged)

## Source
- **Paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: [10.1038/s41592-022-01412-7](https://doi.org/10.1038/s41592-022-01412-7)
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source file:** `data/OEref.rda` (commit master @ 2026-05-02, fetched via raw.githubusercontent.com)

## Bridging methodology
EpiSCORE distributes the oral epithelium reference matrix `mrefOE.m` indexed by **Entrez Gene IDs** (~340 markers × 9 cell types + weight column). The production scorer reads CpG-level reference matrices. Bridge methodology:

1. Load `mrefOE.m` from the EpiSCORE-distributed `OEref.rda` (in atlas_vault as `OEref__OE_Mref_m.csv`).
2. Load `probeInfo450k.lv` (the EpiSCORE-distributed bridge from 450K CpG probes to Entrez Gene IDs; 485,577 array probes; 331,229 with EID; 19,357 unique EIDs).
3. For every probeInfo entry whose EID is in the OEref Entrez IDs, emit a (probeID, EID, Basal, Fib, Gland, Macro, NeuIm, NeuMa, Peri, Plasma, Tcell, weight) row.

Same bridging methodology as VAL-094/111/117/119/124.

## Final dimensions
- **5,396 unique 450K CpG probes × 9 oral cell types**
- Cell types (9):
  - **Basal** = Basal squamous epithelium (oral mucosal stem layer)
  - **Fib** = Fibroblasts (stromal)
  - **Gland** = Glandular cells (oral submucosal mucous glands)
  - **Macro** = Macrophages
  - **NeuIm** = Immature neutrophils
  - **NeuMa** = Mature neutrophils
  - **Peri** = Pericytes (perivascular stromal)
  - **Plasma** = Plasma cells
  - **Tcell** = T cells
- `weight` column preserved for EpiSCORE's weighted-NNLS deconvolution mode

## Class assignments (H_min anchors per GAPE_WEB_v13.py lines 87-96)
- **Basal** → secretory class      H_min = 0.843264 (oral basal epithelium is a stem-progenitor secretory layer)
- **Gland** → secretory class      H_min = 0.843264
- **Fib** → stromal class          H_min = 0.862950
- **Peri** → stromal class         H_min = 0.862950
- **Macro** → immune class         H_min = 0.838889
- **NeuIm** → immune class         H_min = 0.838889
- **NeuMa** → immune class         H_min = 0.838889
- **Plasma** → immune class        H_min = 0.838889
- **Tcell** → immune class         H_min = 0.838889

## Calibration anchor (VAL-125)
- **Source cohort:** TCGA HM450 sesame Level 3 adjacent-normal n=210 (TCGA-KIRC + TCGA-PRAD; same VAL-106 calibration cohort)
- **Sealed prereg SHA:** `f7628a46c36f3d268b0eadbfe495e302a4373238c3cffeaf170ef152aa8b4c1c`
- **Outcome:** O2_PARTIAL_FLOORS (4/9 tiles cleared SD≥0.005 strict floor; 5 tiles tight 0.0037-0.0048; cross-tile separation 0.0407)
- **CHK gates:** CHK-3.1A 100%, CHK-3.1B 100% above ≥80% threshold, CHK-3.1C 0 duplicates
- All 9 per-tile q5 sealed for use as healthy-floor distributions

## Files
- `episcore_oeref_cpg_bridged.csv` — production-ready CpG × cell-type matrix (12 columns). **SHA-256:** `8f4e34ef63247b0ca09312fedb52abf5eee9ee1e8f09e35044ddceb8bdf3f651`
- `episcore_oeref_entrez_matrix.csv` — original Entrez-keyed matrix (audit trail). **SHA-256:** `12d44c311a28451771a3502382553ebdb704a8cedcedfc1e4d483e51b42c59e9`
- `bridge_oeref_to_array.py` — reproducibility script

## License
GPL-2 per EpiSCORE repository.

## Used by
- VAL-125 — gastric+esophageal-epic Stage 2 EpiSCORE OEref calibration on TCGA HM450 adjacent-normal n=210
- VAL-126 — TCGA-STAD Phase C run-everything (oral squamous-tile cross-tissue overread test on gastric adenocarcinoma)
- VAL-127 — TCGA-ESCA Phase C run-everything (consistent overread pattern with EsoRef in EAC; partial signal in ESCC)
- gastric+esophageal-epic v0.1 production deployment Stage 2 layered atlas

## Frozen
2026-05-02

# EpiSCORE BladderRef — bladder cell-type methylation reference (CpG-bridged)

## Source
- **Paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: [10.1038/s41592-022-01412-7](https://doi.org/10.1038/s41592-022-01412-7)
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source file:** `data/BladderRef.rda` (commit master @ 2026-04-30, fetched via raw.githubusercontent.com)

## Bridging methodology
EpiSCORE distributes the bladder reference matrix `mrefBladder.m` indexed by **Entrez Gene IDs** (163 markers × 4 cell types + weight column). The production scorer reads CpG-level reference matrices. Bridge methodology:

1. Load `mrefBladder.m` (163 Entrez × 5 columns: EC, Epi, Fib, IC, weight) from the EpiSCORE-distributed `BladderRef.rda` (in atlas_vault as `BladderRef__mrefBladder_m.csv`).
2. Load `probeInfo450k.lv` (the EpiSCORE-distributed bridge from 450K CpG probes to Entrez Gene IDs; 485,577 array probes; 331,229 with EID; 19,357 unique EIDs).
3. For every probeInfo entry whose EID is in the 163 BladderRef Entrez IDs, emit a (probeID, EID, EC, Epi, Fib, IC, weight) row. The Entrez-level methylation profile is broadcast to every 450K CpG probe mapping to that gene.

This is the **same** bridging methodology used for VAL-094 (BreastRef bridge), VAL-111 (HeartRef bridge), and VAL-117 (ProstateRef bridge).

## Final dimensions
- **2,696 unique 450K CpG probes × 4 bladder cell types**
- 158 unique Entrez Gene IDs covered (of 163 source Entrez IDs)
- 5 source EIDs without 450K CpG mapping (dropped): 1880, 2252, 26521, 51699, 54829
- Cell types:
  - **EC** = Vascular Endothelial Cells
  - **Epi** = Urothelial Epithelium — **the bladder cancer cell of origin**
  - **Fib** = Fibroblasts (stromal)
  - **IC** = Immune Cells (intra-bladder)
- `weight` column preserved for EpiSCORE's weighted-NNLS deconvolution mode

## Class assignments (H_min anchors per GAPE_WEB_v13.py lines 87-96)
- **Epi**  → secretory class      H_min = 0.843264 (urothelium is a barrier secretory epithelium)
- **EC**   → stromal class        H_min = 0.862950
- **Fib**  → stromal class        H_min = 0.862950
- **IC**   → immune class         H_min = 0.838889

## Files
- `episcore_bladderref_cpg_bridged.csv` — production-ready CpG × cell-type matrix. **SHA-256:** `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`
- `episcore_bladderref_entrez_matrix.csv` — original Entrez-keyed matrix (audit trail)
- `bridge_bladderref_to_array.py` — bridge script (reproducibility triple per CHK-7.6)
- (Source `BladderRef__mrefBladder_m.csv` lives in the EpiSCORE pan-tissue folder and stays there as part of the broader pan-tissue MANIFEST.)

## Source SHA-256 verification
- `BladderRef.rda` (EpiSCORE GitHub master): `a357383a492ebd6ec6262cb0bfba45f970c6a266ef2a1b83f813f31164a42135`
- `BladderRef__mrefBladder_m.csv` (atlas_vault Entrez-keyed source): `f73fbeab74dfbe5aec2829303757908df569bb969101180c2875a46505a3e758`
- `probeInfo450k.rda` (EpiSCORE GitHub master): `1b4d0bb8ebd0de3a5bd8b1c9cbf170599fce920da399076182070bdd93b57ca8`

## License
GPL-2 per EpiSCORE repository.

## Used by
- VAL-119 — bladder-epic Stage 2 EpiSCORE BladderRef calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (HM450 sesame Level 3)
- bladder-epic v0.1 production deployment Stage 2 layered atlas (bladder sub-cell-type resolution beyond Loyfer's single `Bladder` tile)

## Atlas family classification
**Gene-promoter atlas family** — same as ProstateRef, BreastRef, HeartRef. Per DISC-CARDIO-004 + prostate-LL-006, gene-promoter atlas family fitness depends on per-tissue cell-type distinctness. Bladder is the **third** per-tissue test of this rule (cardio HeartRef collapsed at VAL-111; prostate ProstateRef separated cleanly at VAL-117). VAL-119 will determine whether BladderRef separates or collapses on bladder biology.

## Frozen
2026-04-30

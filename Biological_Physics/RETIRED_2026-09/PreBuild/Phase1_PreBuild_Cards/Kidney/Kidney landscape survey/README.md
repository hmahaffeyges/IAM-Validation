# Jeong 2026 KidneyRef v2 — kidney cell-type methylation reference (HM450 hg19 CpG-bridged)

## Source

- **Paper:** Jeong H, Lake BB, Diep D, Li X, Yan Q, Gisch DL, Reinert S, Eadon MT, Gaut JP, Jain S, Zhang K. *A cross-species single-cell epigenome kidney atlas identifies epithelial cells as a driver of epigenetic aging.* bioRxiv 2026.01.22.700871; doi: [10.64898/2026.01.22.700871](https://doi.org/10.64898/2026.01.22.700871) (preprint, January 2026).
- **Affiliations:** Altos Labs (San Diego Institute of Science), Indiana University School of Medicine, Washington University School of Medicine. NIH Common Fund supported HuBMAP grant U54DK134301.
- **Substrate:** sciMETv2 single-cell DNA methylation, 64,203 high-quality nuclei across 12 human donors (7 CKD/disease + 5 healthy controls) and 6 mice. Cell-type pseudo-bulk methylomes 28.35-29.2M CpGs each.
- **Source files used (Supplementary Materials, media-2.xlsx):**
  - **Table S3** — consensus reference panel of 2,180 orthologous genomic regions defining 11 conserved kidney cell types with hg38 coordinates, per-cell-type vs rest methylation values, and significance scores. **This is the source of KidneyRef v2.**
  - **Table S6** — 8,996 altered-PT vs healthy-PT DMRs (kept here for v0.X+ disease-signature use).

## License notice

The bioRxiv preprint carries `cc_no` license: **"All rights reserved. No reuse allowed without permission"** (preprint footer).

This atlas-vault deposit is **for independent academic research and EDEAR validation only.** Production EDEAR commercial deployment requires written permission from the corresponding authors (Kun Zhang, Sanjay Jain) — **NOT YET OBTAINED.** Before any commercial roll-out: either obtain written license, build an independent atlas, or do not deploy.

## Bridging methodology

Table S3 from media-2.xlsx is in **wide target-vs-rest format**, NOT a full per-cell-type β matrix. Each row records:
- `GroupID` = cell type that "owns" the discriminating region
- `AvgMethyl_Human_Target` = β at this region in cells of GroupID
- `AvgMethyl_Human_Rest` = β at this region in cells of all OTHER 10 types (mean)

To assemble a CpG × cell-type matrix usable by the production scorer, the bridge applies the convention:
- For owning cell type `C` at its marker regions: β = `AvgMethyl_Human_Target`
- For all other cell types at C's marker regions: β = `AvgMethyl_Human_Rest`

Multi-region CpGs (CpGs falling in marker regions of more than one cell type) get averaged.

### Engineering steps

1. **Extract** Table S3 + S6 from `media-2.xlsx` as TSV (with SHA-256 stamps).
2. **LiftOver** hg38 → hg19 via pyliftover hg38ToHg19 chain (UCSC). Required because the cookbook substrate calibration anchor (TCGA HM450 sesame Level 3, VAL-106) is hg19-indexed.
3. **Load** HM450 hg19 manifest (`hm450_hg19_manifest.csv` — same manifest already in vault, used for Caggiano CelFiE TIM bridge).
4. **Intersect** each lifted-over hg19 region with HM450 CpGs whose pos lies in `[start, end]`.
5. **Assemble** matrix: cpg_id × 11 cell types, β values per the wide-format expansion convention.
6. **Deduplicate** by averaging when CpG falls in multiple regions (CHK-3.1C).

This is the **same general bridging methodology as Caggiano CelFiE TIM** (VAL-113 precedent), with added hg38→hg19 liftOver step.

## Final dimensions

- **5,587 unique HM450 CpG probes × 11 kidney cell types**
- 2,180 source regions (Table S3) → 2,178 successfully lifted to hg19 (99.9% retention)
- 1,459 of 2,178 hg19 regions intersect ≥1 HM450 CpG (67%; remainder lie in non-HM450-covered genomic regions)
- 6,966 total CpG-region hits → 5,587 unique CpGs after CHK-3.1C dedup

### Cell type taxonomy (11 cell types from Table S3)

| Code | Cell type | Source markers in S3 | Markers retained after bridge |
|---|---|---:|---:|
| **PT**     | Proximal Tubule | 99 | 395 |
| **POD**    | Podocyte | 259 | 700 |
| **DCT**    | Distal Convoluted Tubule | 58 | 175 |
| **TAL**    | Thick Ascending Limb | 18 | 98 |
| **PC**     | Principal Cell (collecting duct) | 55 | 327 |
| **CNT_IC** | Connecting Tubule + Intercalated Cell | 20 | 74 |
| **PEC**    | Parietal Epithelial Cell | 16 | 108 |
| **FIB**    | Fibroblast | 469 | 1,206 |
| **EC**     | Endothelial Cell | 324 | 1,056 |
| **Myeloid**| Myeloid lineage | 291 | 1,187 |
| **B**      | B-cell lineage | 571 | 1,640 |

(Markers retained after bridge > source marker count because the bridge expands one source region across multiple HM450 CpGs that fall within that region's [start, end] interval.)

## Class assignments (H_min anchors per GAPE_WEB_v13.py lines 87-95)

Mapped to GAPE engine's 8 architecture classes:

| Cell type | Architecture class | H_min (methyl) | Justification |
|---|---|---:|---|
| **PT**     | secretory | 0.843264 | Highly metabolic active transport (Na+/H+, glucose, amino acid reabsorption); analogous to hepatocyte secretory load |
| **DCT**    | secretory | 0.843264 | Active sodium/calcium reabsorption |
| **TAL**    | secretory | 0.843264 | Active sodium/chloride/potassium transport (NKCC2) |
| **PC**     | secretory | 0.843264 | Active aldosterone-regulated sodium reabsorption |
| **CNT_IC** | secretory | 0.843264 | Active acid/base secretion (intercalated cells) |
| **POD**    | terminal  | 0.772837 | Post-mitotic terminally differentiated; never divide; structural barrier of the glomerulus |
| **PEC**    | progenitor| 0.852216 | Parietal epithelial cells include the kidney's progenitor population (Bowman's capsule) |
| **FIB**    | stromal   | 0.862950 | Standard stromal architecture (matches HeartRef Fib, BladderRef Fib assignments) |
| **EC**     | stromal   | 0.862950 | Vascular endothelial; standard stromal class assignment |
| **Myeloid**| immune    | 0.838889 | Standard immune class |
| **B**      | immune    | 0.838889 | Standard immune class |

## Self-consistency QA

Cross-cell-type discrimination at marker rows (mean |β_target - β_rest| across 10 other cell types):

| Cell type | Markers | Mean \|spread\| | Median \|spread\| | Range |
|---|---:|---:|---:|---|
| PT       | 395   | 0.391 | 0.350 | [0.207, 0.785] |
| POD      | 700   | 0.392 | 0.385 | [0.199, 0.750] |
| DCT      | 175   | 0.331 | 0.341 | [0.188, 0.707] |
| TAL      | 98    | 0.295 | 0.298 | [0.149, 0.541] |
| PC       | 327   | 0.377 | 0.387 | [0.221, 0.724] |
| CNT_IC   | 74    | 0.331 | 0.362 | [0.143, 0.578] |
| PEC      | 108   | 0.289 | 0.306 | [0.174, 0.412] |
| FIB      | 1,206 | 0.358 | 0.319 | [0.198, 0.839] |
| EC       | 1,056 | 0.379 | 0.373 | [0.143, 0.775] |
| Myeloid  | 1,187 | 0.388 | 0.376 | [0.163, 0.671] |
| B        | 1,640 | 0.339 | 0.311 | [0.178, 0.817] |

All 11 cell types show substantial discriminating signal (mean |spread| 0.29-0.39).

For comparison, EpiSCORE BladderRef shows |spread| 0.55-0.67 — sharper because BladderRef is a 4-cell-type one-vs-rest reference. KidneyRef v2 is naturally softer (target-vs-rest-mean across 10 other cell types) but every cell type retains clean discriminating markers.

## CHK-3.1A bimodality

- f_extreme (<0.1 or >0.9): 0.076
- f_middle ([0.4, 0.6]): 0.235
- mean β: 0.527, median: 0.539

Substrate class flat-distribution reading: **self-cal class.** This reading is structural, not pathological — it reflects the wide target-vs-rest expansion convention (where non-owning cell types share the same β_rest mean). When stratified by marker vs non-marker rows, marker-bearing rows show clean discriminating spread (median 0.35) and **all 5,587 rows are marker-bearing for at least one cell type.**

## Pre-flight caveats (mandatory disclosure)

1. **Wide target-vs-rest format.** Source matrix is not a full per-cell-type β matrix. β values for non-owning cell types use the mean of OTHER cell types from that region (β_rest), not per-individual-cell-type values. Marker resolution is high but cross-cell-type discrimination at non-marker rows averages over 10 other types.
2. **5kb region granularity.** Table S3 regions are 5kb genomic windows (whole-genome bisulfite resolution). HM450 CpG positions are point coordinates. A single region typically spans 0-5+ HM450 CpGs.
3. **HG38 → HG19 liftOver.** Required for compatibility with TCGA HM450 sesame Level 3 substrate anchor (VAL-106). LiftOver dropped 2 of 2,180 source regions (no hg19 mapping); 0 split-chromosome cases.
4. **HM450 platform-bias.** HM450 array preferentially probes gene promoters and CpG islands; sciMETv2 covers genome-wide. 33% of lifted hg19 regions fall in non-HM450-covered intronic/intergenic spaces and are dropped at the intersection step.
5. **Commercial license not yet obtained.** Independent academic research use only. Production EDEAR commercial deployment requires written permission from Kun Zhang / Sanjay Jain.

## Files

- `jeong2026_kidneyref_v2_HM450_v1.csv` — production-ready CpG × 11 cell-type matrix
- `jeong2026_kidneyref_v2_INVENTORY.json` — atlas-vault inventory entry with SHA-256 + per-cell-type counts + CHK statuses
- `jeong2026_TableS3_consensus_panel_source.tsv` — original Table S3 (audit trail)
- `jeong2026_TableS6_alteredPT_DMRs_source.tsv` — Table S6 altered-PT DMRs (kept for v0.X+ disease signature)
- `bridge_jeong2026_to_HM450.py` — bridge script (reproducibility triple per CHK-7.6)
- `bridge_log.txt` — full execution log

## Source SHA-256 verification

- Table S3 source TSV: `c468ba9f37cd764df73f59a2200afb69bdfa4389a80b548bbc48073f82075440`
- Table S6 source TSV: `99fd4ec9b4d12d24f92834ed98e7b5e822e0d96e4fd07a0e88a27e08ed9a7b1a`
- HM450 hg19 manifest: `a4c987458d0128311c6b74ae2fa6db64c2ed6b5a1a789d07247af86cf116598a`
- **Bridged matrix SHA-256: `6fbe0b52e34fc4754e3faf5db1b16b8e960883a719934972ec97acaaf1d6908c`**

## Used by

- VAL-129 (planned) — kidney-epic Stage 2 KidneyRef v2 calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (HM450 sesame Level 3 anchor)
- kidney-epic v0.1 production deployment Stage 2 cell-of-origin atlas — replaces the rejected EpiSCORE KidneyRef (32-marker sparse one-vs-rest in episcore_zhu_teschendorff_2022/)

## Atlas family classification

**Single-cell pseudo-bulk WGBS atlas, marker-based reference.** Operationally similar to Caggiano CelFiE TIM (VAL-113) but with kidney-specific cell-type resolution and hg38→hg19 liftOver step. Per the atlas-fitness gradient:

- **Loyfer Moss 2018** (bulk-WGBS, 25 tiles including a single Kidney bulk tile): VAL-112 calibrated, single-tile bulk reference
- **EpiSCORE KidneyRef** (gene-promoter bridged, 32 markers × 4 cell types one-vs-rest sparse): structurally degenerate, expected DISC-KIDNEY-001 atlas-fitness null finding
- **Jeong 2026 KidneyRef v2** (this deposit; sciMET pseudo-bulk, 5,587 CpGs × 11 cell types target-vs-rest): the cell-type-resolved layer

## External validation (per Jeong 2026 paper, Supplementary Figure 2)

The paper's authors explicitly validated their sciMETv2 cell-type pseudo-bulk methylomes against **Loyfer 2023 flow-sorted WGBS data** (the same atlas already calibrated in our vault as `loyfer_moss_2018`, VAL-112). Their Supp Fig 2 shows clean co-clustering of sciMET PT/POD/EC/FIB profiles with Loyfer's flow-sorted equivalents on cis-regulatory region PCA. **This is direct external validation that our existing Loyfer Kidney bulk tile sits on the same methylation manifold as the cell-type-resolved sciMET reference.**

## Frozen

2026-05-03

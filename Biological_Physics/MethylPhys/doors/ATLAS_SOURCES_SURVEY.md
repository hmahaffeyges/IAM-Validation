# Atlas sources survey — what exists for the cells we are missing (2026-09-27)

Author: "do your diligence to locate any and all atlases that could help expand and improve our atlas … IF those [missing
terminal cells, the star cell and others] exist on another atlas we get them." And: a full tissue atlas, separate if it cannot
join the cell atlas.

**Order is unchanged** (ATLAS_V2_SPEC.md): v2 is built on the eleven v1 sources first and must pass A1–A9; new sources then enter
one at a time through the same script (scale map → coverage family → twin test → identity loci → reads 1.00 on its own profile).

## The target list — 44 cell columns the old disease matrix names that the 115-cell atlas cannot reach
Read from `chain/Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_13.csv` against
[`iamatlas_115_to_matrix_v0_2_mapping.json`](../chain/Disease%20Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json) (136 columns: 7 metadata, 8 class means, 121 cells; 77 reachable, 44 not).
The "star cells" are **astrocytes** (astro = star) and **stellate cells** (hepatic, pancreatic).

## Sources, verified
| source | platform | what it holds | accession | verified how |
|---|---|---|---|---|
| **Loyfer et al. 2023, Nature** — sorted-cell methylome atlas | WGBS, hg19, ~30×, beta at 28.2 M CpGs | 253 samples: 207 sorted cells, 46 cfDNA / WBC mixtures. Cell types listed below | GEO **GSE186458** | series matrix read 2026-09-27 ([`GSE186458_samples.json`](../atlas/sources/GSE186458_samples.json): GSM, tissue, cell type per sample) |
| **Tian et al. 2023, Science** — human brain single-cell methylome | snmC-seq3 / snm3C-seq; pseudobulk per cell type | 517k nuclei, 46 regions, **188 cell types incl. non-neurons (astrocytes, oligodendrocytes, OPC, microglia, vascular)** | GEO **GSE215353**; NeMO nemo:dat-jx4eu3g | Science data statement |
| **Zhou et al. 2026, Science** — human body single-cell atlas | snm3C-seq; pseudobulk per cell type | 86,689 nuclei, **16 tissues incl. brain (M1), placenta, pancreatic islets, blood**; 35 major / 206 subtypes | 4DN / browser; GEO id to confirm | Science / PMC abstract |
| HiBED reference sources (Zhang & Salas 2023) | 450K / EPIC — our platform | astrocytes (Weightman Potter 2021, primary, n=6), GABA/GLU neurons (Kozlenkov 2018), microglia (de Witte 2022, n=18), oligodendrocytes (Mendizabal 2019), endothelial (Lin 2018) | per-paper GEO ids to collect | HiBED paper, Table 1 |
| **Ontology-aware 450K compendium, Cell Reports Methods 2026** | 450K — our platform | **16,959 healthy primary samples, 86 tissue and cell types, 210 GEO studies** (55 with ≥2 studies); cell lines, diseased and treated excluded | sample lists in its supplementary tables → the source GSEs | PMC full text |
| Hepatic stellate (Oncotarget 2015) | **MeDIP promoter arrays** — enrichment, not beta | quiescent HSC, LSEC, hepatocytes, n = 2–3 | E-GEOD-66796 | ArrayExpress record — **not usable on the atlas scale** |

### Loyfer 2023, cell type × tissue (from the GEO record)
Cardiomyocyte (6) · Striated muscle (2) · Osteoblasts (1) · Podocyte (3) · Adipocytes (3) · Hepatocyte (6) · Oligodendrocytes (4) ·
Neuronal (10) · Pancreas acinar (4) / duct (4) / alpha (3) / beta (3) / delta (3) · Endocrine: colon, gastric, jejunum (6) ·
Breast basal (4) / luminal (3) · Keratinocyte (1) · Erythrocyte progenitors (3) · blood: B, memory B, naive/memory/effector CD4
and CD8, NK, monocytes, granulocytes · **Fibroblast**: dermal, colon, heart (7) · **Endothelium**: aorta, kidney glomerular,
kidney tubular, liver, lung alveolar, pancreas, saphenous vein (19) · **Macrophages**: colon, liver, lung alveolar, lung
interstitial (8) · **Smooth muscle**: aorta, bladder, bronchial, coronary artery, prostate (5) · **Epithelium** (63): bladder,
colon, endometrium, esophagus, fallopian tube, gallbladder, gastric antrum/body/fundus, jejunum, kidney glomerular and tubular,
larynx, lung alveolar, lung bronchus, lung pleural, ovary, pharynx, prostate, small intestine, thyroid, tongue, tonsil.

## Missing cell → where it exists
| missing (disease-matrix column) | found in | note |
|---|---|---|
| astrocytes, brain_astrocytes | **Tian 2023** pseudobulk; HiBED's primary astrocytes (cultured) | the star cell. Sorted in-vivo astrocytes on arrays are rare; the single-cell pseudobulk is the in-vivo source |
| hepatic_stellate, pancreatic_stellate | **no per-CpG source found**; check Zhou 2026's 206 subtypes (liver and pancreas are among its tissues only if listed — to confirm) | the MeDIP set cannot be put on the atlas scale |
| cardiomyocytes | atlas has `CM`; **Loyfer** (6) | mapping gap + second source |
| cardiac_fibroblasts | **Loyfer** heart fibroblast (4) | new |
| cardiac_smooth_muscle | **Loyfer** coronary-artery smooth muscle (1) | closest; one sample |
| cardiac_endothelial | not in Loyfer (aorta / saphenous only); Zhou 2026 heart if present | open |
| skeletal_muscle | **Loyfer** striated muscle (2) | new |
| osteoblasts | **Loyfer** (1) | new; one sample |
| BMSC | cultured MSC only (Roadmap-era) | culture changes methylation — record, don't adopt |
| pulmonary_macrophages | **Loyfer** lung alveolar + interstitial (5) | new |
| oral_macrophages | none found | open |
| lung_airway | **Loyfer** lung bronchus epithelium (3) | new |
| vascular_endothelial | **Loyfer** endothelium, 7 vascular beds (19) | new, and per-organ |
| breast_ductal / lobular / BE | **Loyfer** breast basal (4) + luminal (3) | second source for Breast / BE / LE |
| pancreatic_exocrine_pooled | **Loyfer** acinar (4), duct (4) | second source |
| esophageal_glandular, eso_basal, eso_epi_basal | **Loyfer** esophagus epithelium (2) | not split basal vs glandular |
| head_neck_larynx_secretory | **Loyfer** larynx, pharynx, tongue, tonsil epithelium (10) | second source |
| salivary | none found | open |
| rectal_epithelium | **Loyfer** colon epithelium (5) | colon, not rectum |
| liver_bile_duct | atlas has `Chol`; **Loyfer** gallbladder epithelium (1) | nearest, not the same cell |
| gastric differentiated / undifferentiated | atlas has; **Loyfer** antrum/body/fundus epithelium (9) | second source — the stomach cells the author asked about |
| bladder_loyfer_bulk | **Loyfer** bladder epithelium (5) | the name already says where it came from |
| prostate_LE / BE / SM | atlas has `Prostate`; **Loyfer** prostate epithelium (4), smooth muscle (1) | not split luminal vs basal |
| prostate_Fib / EC / Leu, mammary_stromal | none found | open |
| placenta | **Zhou 2026** placenta cell types | new; also a tissue-atlas entry |
| brain_pooled | tissue | tissue atlas |
| HSPC_pooled, *_cycling pools, epithelial_cycling_pooled | pools of cells the atlas already holds | not a source question |

**Beyond the list, new cells Loyfer adds:** podocytes, pancreatic alpha and delta cells, enteroendocrine (colon, gastric, jejunum),
keratinocytes, endometrium, fallopian tube, ovary, kidney glomerular epithelium, lung alveolar and pleural epithelium,
per-organ endothelium, naive/memory/effector T-cell subsets and memory B.
**From Tian 2023:** microglia, OPC and vascular cells as a second source, and neuronal subtypes the atlas has as one column.

**Still no source after this pass:** hepatic and pancreatic stellate (pending Zhou 2026's subtype list), oral macrophages, salivary
epithelium, prostate and mammary stromal cells, cardiac endothelium, in-vivo bone-marrow stromal cells.


## Why the atlas has thin cells — read from the author's v1 build vault (atlas_vault_OLD.zip, 2026-04/05)
v1 pooled **published reference matrices**, not the samples behind them. Most of those matrices are marker panels: each author kept
only the CpGs that separate their cell types. That is exactly the coverage families the v2 spec found:

| v1 input (vault inventory) | CpGs | cells | coverage family it made |
|---|---|---|---|
| Caggiano 2021 CelFiE TIM, bridged to array CpGs | 254 | 19 (dendritic, endothelial, eosinophil, erythroblast, macrophage, monocyte, neutrophil, placenta, tcell, adipose, brain, fibroblast, heart, hepatocyte, lung, mammary, megakaryocyte, skeletal, small_intestine) | the 252-locus family |
| Moss 2018 / Loyfer array atlas (`nloyfer/meth_atlas`) | 6,105 (7,890 before de-duplication) | 25 | the 6,105-locus family (Breast, Colon, Prostate, Kidney, Lung, Liver …) |
| EpiSCORE (Zhu & Teschendorff 2022) tissue references | 2k–4k per tissue | 42 across 13 tissues | several small families; note EpiSCORE's `mref` matrices are partly **imputed from single-cell RNA**, not all measured methylation |
| UniLIFE (Guo 2025) | 1,906 | 19 immune | an immune family |
| Salas IDOL | 450 | 6 immune | |
| EpiDISH companion panels | panel | 12 | |

**Consequence for v2:** the fix for a thin cell is not a better model of the panel — it is the **samples the panel was made from**, at
every locus:
- **Moss 2018 raw arrays are public**: GEO **GSE122126** (450K + EPIC IDATs; sorted adipocytes, cortical neurons, hepatocytes,
  pancreatic acinar/beta/duct, colon and lung epithelium, vascular endothelium, leukocytes). Through our own Stage 1 these become
  full-coverage cells on our own scale — no pipeline map at all.
- **Loyfer 2023 is public on GEO** (GSE186458, hg19 and hg38 beta files per sample). The vault's manifest listed it as EGA
  controlled access; that was wrong or has changed — the per-sample beta files are open.
- Caggiano's tissues came from public WGBS (Roadmap/ENCODE); EpiSCORE's imputed columns should be flagged or replaced by measured ones.

Downloading now on the AWS box: Loyfer hg19 beta for the 207 sorted samples (11.7 GB) and GSE122126 raw (1.7 GB).

## Platform work each needs (the reason they enter one at a time)
- **Loyfer (WGBS)**: beta files are per-CpG over the genome, hg19. Take the 450K/EPIC CpG coordinates from the atlas manifest,
  read those positions, keep coverage per locus as the precision term. Needs its own pipeline map onto the atlas scale.
- **Tian / Zhou (single-cell pseudobulk)**: per-cell-type methylation fraction, sparse per locus; coverage becomes the precision
  term the v2 model already has a slot for. Own pipeline map.
- **HiBED sources and the 450K compendium**: same platform as the chain — raw IDATs through our own Stage 1, the way every
  commissioned array goes, so no third-party normalization enters.

## Tissue atlas — separate, as the author suggested
A tissue is a mixture of cells; putting tissue profiles into the cell deconvolution would count the same cells twice. A separate
**tissue atlas** answers a different question — which tissues shed into this specimen — and is built the same way. Primary source:
the Cell Reports Methods 2026 compendium's GEO list (450K, healthy, primary, 86 types). Placenta, brain and the Loyfer tissues'
own bulk structure are cross-checks.

## Next
1. After SATSA, pull GSE186458 beta files (~253 × 28.2 M CpGs) to the AWS box's persistent disk; extract the atlas CpGs.
2. Pull the GSE215353 pseudobulk tables for non-neuronal types; confirm Zhou 2026's accession and its subtype list (the stellate check).
3. Collect the HiBED sources' GEO ids and the compendium's supplementary sample table.
4. None enters the atlas until v2 on the eleven v1 sources has passed A1–A9.

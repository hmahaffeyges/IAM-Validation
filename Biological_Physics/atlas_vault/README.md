# Atlas Vault — IAMPerformance / EDEAR Stage 2 + Stage 3 Reference Layer

**Purpose.** Durable, version-controlled storage of every methylation atlas and reference matrix EDEAR uses or plans to integrate. Files in this vault are committed to GitHub under the IAM-Validation repository, which provides redundancy that scratch container filesystems do not. The internet hosting these atlases (author GitHubs, Zenodo, EGA, bioRxiv, Bioconductor) is the *primary* source — this vault is the *backup* against the original sources disappearing.

**Last updated:** 2026-04-26 PM session  
**Total vault size:** 4.7 MB across 65 files (well within GitHub's 100 MB-per-file / 1 GB-recommended-repo limits)  
**Maintained by:** Heath W. Mahaffey / Walther

---

## Vault discipline — operational rules

1. **Every atlas EDEAR uses must live here.** Production atlases (Loyfer/Moss, Salas, UniLIFE) and Queue-1 atlases (EpiSCORE, Caggiano, Sabedot, MARLIN) are mirrored here as canonical durable copies. The container filesystem `/home/claude/atlases/` is **scratch** — it disappears between sessions. The vault is **durable** — it persists across sessions and across years.

2. **Every commit updates the INVENTORY.** `INVENTORY.json` lists every file with size and SHA-256. Any future session can verify integrity with: `sha256sum -c <(jq -r '.[] | "\(.sha256)  \(.path)"' INVENTORY.json)`.

3. **CSV formats preferred over R-data.** Where possible, atlases are stored both as their native format (`.rda` for R-pipeline compatibility) AND as CSV (for AI / Python / cross-language portability). The next instance of Walther in 6 months may not have R available — CSV doesn't care.

4. **Provenance is non-negotiable.** Every atlas directory has its own README documenting: original source URL, citation, license, download date, SHA-256 of source files, and any conversion steps performed.

5. **Surveillance feeds the vault.** Each monthly atlas surveillance sweep ends with: (a) update GAPE Reproduction Paper §7.17, §7.18, etc. with new findings; (b) acquire any newly-surfaced atlases that are tractable; (c) commit them to this vault; (d) update INVENTORY.json.

6. **License compliance.** Every atlas in this vault was downloaded from a publicly-licensed source. Original licenses (MIT, GPL-2/3, CC-BY, CC-BY-NC, ASL-2.0) are preserved in each atlas's subdirectory. EDEAR's commercial deployment must comply with each atlas's license terms — the atlases that are CC-BY-NC require attribution-only use; atlases that are GPL require derivative works to be GPL'd. License audit happens at v0.3 build-out.

---

## What the EDEAR pipeline runs through (per run, every IDAT)

Under run-everything architecture, every IDAT scored by EDEAR is processed through every atlas in the production layer. As of 2026-04-26 PM, the production layer is:

### Stage 2 — Cell-of-origin deconvolution (PRODUCTION)

| Atlas | Location in vault | Cell types | Status |
|---|---|---|---|
| **Loyfer/Moss array atlas** | `stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` | 25 cell types × 7,890 array CpGs | **PRODUCTION** |

### Stage 3 — Immune fraction estimation (PRODUCTION)

| Atlas | Location in vault | Cell types | Status |
|---|---|---|---|
| **Salas Blood.EPIC IDOL baseline** | `stage3_immune_fraction/salas_blood_epic_idol/` | 6 cell types × 450 EPIC CpGs (+ 350 CpG 450K legacy) | **PRODUCTION** |

### Stage 2/3 Queue-1 — APPROVED for v0.3, NOT YET in production scoring

These atlases are committed to the vault but the integration VAL has not landed — VAL-094 (EpiSCORE breast Stage 2), VAL-095 (UniLIFE Stage 3 head-to-head vs Salas), VAL-096 (closer-to-diagnosis windows on Loyfer) are the integration anchors.

| Atlas | Location in vault | Cell types | Integration VAL |
|---|---|---|---|
| **UniLIFE (Guo 2025)** | `stage3_immune_fraction/unilife_guo_2025/` | 19 immune cell types × 1,906 CpGs, lifespan-spanning | VAL-095 |
| **Salas IDOL-Ext** | `stage3_immune_fraction/salas_idol_ext/` | 12 cell types extended panel (data via ExperimentHub lazy fetch) | superseded by UniLIFE |
| **EpiSCORE pan-tissue** | `stage2_cell_of_origin/episcore_zhu_teschendorff_2022/` | 14 tissues × 4–10 cell types (28 reference matrices total) | VAL-094 |
| **Caggiano CelFiE** | `stage2_cell_of_origin/caggiano_celfie_2021/tim_matrix.txt` | 1,580 markers × 19 tissues (WGBS-region-based, **caveat**) | TBD |
| **Sabedot GeLB** | `stage2_cell_of_origin/sabedot_gelb_2021/GeLB.R` | EPIC glioma blood classifier (training script, requires GSE150289) | TBD |
| **Capper mnp_training** | `stage2_cell_of_origin/marlin_capper_training/` | brain tumor 450K/EPIC classifier training code | TBD |

### Stage 2/3 Queue-1 — APPROVED, NOT YET in vault (controlled access or 503 today)

Acquisition pending. These will be added to the vault at the next opportunity:

- **Tanaka 2025 6-cell neural cfDNA** — markers extractable from supplementary PDF without EGA access; primary nanopore data on EGA controlled access
- **Konigsberg/Cuadrat 2023 cardiac** — relies on Loyfer 2023 EGA-controlled atlas; cardiomyocyte markers in supplementary tables
- **Jacques 2025 17-tissue Ageing Atlas** — bioRxiv supp blocked from this container today; retry via direct browser
- **MethAgingDB** — Zenodo 503 today; retry next session
- **Ontology-aware Kim 2025-2026** — bioRxiv JS-gated; contact Yao lab (Rice) for code release status
- **Cuadrat 2026 Comm Bio guidelines** — Nature 503 today; methodology paper, not a downloadable atlas

### Queue-2 — longer engineering horizon

- **Liu 2023 brain scMCodes** — Allen Brain Cell Atlas, 188 brain cell types (single-cell to array projection)
- **Zhou 2025 Human Body Single-Cell Atlas** — 86,689 nuclei × 16 tissues, 206 cell subtypes
- **223-cell-type WGBS atlas** (arXiv 2506.00146) — source verification needed

---

## Per-atlas inventory and provenance

### `stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`

- **Atlas:** Loyfer/Moss 2018 + 2023 hybrid (Moss 2018 25-cell types using Loyfer 2023 marker selection methodology)
- **Source:** GitHub `nloyfer/meth_atlas` (2018) + Loyfer et al. *Nature* 613, 355–364 (2023) [DOI 10.1038/s41586-022-05580-6]
- **License:** MIT (per nloyfer/meth_atlas LICENSE)
- **Format:** CSV, 7,890 rows × 25 columns
- **Cell types:** B-cells, CD4T, CD8T, NK, Mono, Neu, Eos, granulocytes, hepatocytes, pancreatic acinar/beta/duct, vascular endothelial, cortical neurons, lung, head & neck larynx, kidney, breast, prostate, colon epithelial, upper GI, uterus/cervix, thyroid, bladder, adipocytes, left atrium, erythrocyte progenitors
- **Production status:** active — Stage 2 production scoring runs through this atlas every IDAT

### `stage2_cell_of_origin/episcore_zhu_teschendorff_2022/`

- **Atlas:** EpiSCORE pan-tissue methylation reference (Zhu, Teschendorff et al.)
- **Source:** GitHub `aet21/EpiSCORE` master branch
- **Citation:** Teschendorff AE, Zhu T, Breeze CE, Beck S. *Genome Biol* 21, 221 (2020). EPISCORE: cell type deconvolution of bulk tissue DNA methylomes from single-cell RNA-seq data.
- **License:** GPL-2 (per EpiSCORE DESCRIPTION)
- **Coverage:** 14 tissues — Bladder, Brain, Breast, Colon, Esophagus, Heart, Kidney, Liver, Lung, Olfactory Epithelium, Pancreas (+ extended 9-cell-type Pancreas), Prostate, Skin
- **Format:** 28 CSV files (14 expression-derived `expref*` × 14 DNAm-derived `mref*`), MANIFEST.json catalogs all 28
- **Conversion:** Original `.rda` → CSV via Python `rdata` library, 2026-04-26
- **Production status:** Queue-1 — VAL-094 will integrate breast tissue arm

### `stage3_immune_fraction/unilife_guo_2025/`

- **Atlas:** UniLIFE (Unified Lifecourse Immune Fraction Estimator)
- **Source:** GitHub `sjczheng/EpiDISH/data/centUniLIFE.m.rda`
- **Citation:** Guo X, Sulaiman M, Neumann A, Zheng SC, Cecil CAM, Teschendorff AE, Heijmans BT. *Genome Med* 17:63 (2025). DOI 10.1186/s13073-025-01489-7. "Unified high-resolution immune cell fraction estimation in blood tissue from birth to old age."
- **License:** GPL-2 (EpiDISH package)
- **Format:** Both .rda (281 KB) and CSV (712 KB); 1,906 marker CpGs × 19 immune cell types
- **Cell types:** 7 pan-lifespan (B, CD4T, CD8T, Mono, nRBC, Gran, NK) + 12 adult-specific (aCD4Tnv, aBaso, aCD4Tmem, aBmem, aBnv, aTreg, aCD8Tmem, aCD8Tnv, aEos, aNK, aNeu, aMono)
- **Operational use:** `EpiDISH::epidish(X, cent=centUniLIFE.m, method="RPC", maxit=500)$estF`
- **Compatibility:** 450K, EPICv1, EPICv2, WGBS
- **Production status:** Queue-1 #1 — VAL-095 will integrate as Stage 3 alongside or replacing Salas

### `stage3_immune_fraction/salas_blood_epic_idol/`

- **Atlas:** Salas / IDOL Optimized CpGs for adult whole blood EPIC + 450K
- **Source:** GitHub `immunomethylomics/FlowSorted.Blood.EPIC` master branch
- **Citation:** Salas LA, Koestler DC, Butler RA, Hansen HM, Wiencke JK, Kelsey KT, Christensen BC. *Genome Biol* 19:64 (2018). DOI 10.1186/s13059-018-1448-7.
- **License:** GPL-3 (Bioconductor package)
- **Format:** 5 .rda files + 2 CSV (450 EPIC CpGs × 6 cell types: CD8T, CD4T, NK, Bcell, Mono, Neu; plus 350 CpG × 6 cell type 450K legacy; plus cord blood reference)
- **Production status:** active — Stage 3 production baseline; will be compared against UniLIFE in VAL-095

### `stage3_immune_fraction/salas_idol_ext/`

- **Atlas:** Salas IDOL-Ext (extended 12-cell-type adult blood panel)
- **Source:** GitHub `immunomethylomics/FlowSorted.BloodExtended.EPIC` master branch
- **Citation:** Salas LA et al. (under review at time of package release, 2021); package on Bioconductor 3.13+
- **License:** see SoftwareLicense PDF in source repo
- **Format:** Pheno.csv + metadata.csv + R wrapper code; the actual RGChannelSet data (n=68 references, 450K + EPIC) loads lazily from Bioconductor ExperimentHub at GSE167998
- **Production status:** Queue-1 — superseded by UniLIFE for production scoring

### `stage2_cell_of_origin/caggiano_celfie_2021/tim_matrix.txt`

- **Atlas:** CelFiE TIM matrix (Caggiano et al.)
- **Source:** GitHub `christacaggiano/celfie` master branch
- **Citation:** Caggiano C, Celona B, Garton F, et al. *Nat Commun* 12, 2717 (2021). DOI 10.1038/s41467-021-22901-x.
- **License:** MIT
- **Format:** Tab-separated, 1,580 markers × 19 tissues
- **Marker format:** `chrom/start/end` genomic regions (NOT array CpG IDs) — **integration caveat**
- **Coverage:** dendritic, endothelial, eosinophil, erythroblast, macrophage, monocyte, neutrophil, placenta, T-cell, adipose, brain, fibroblast, heart, hepatocyte, lung, mammary, megakaryocyte, skeletal muscle, small intestine
- **Production status:** Queue-1 — requires region-to-CpG mapping per array platform before integration

### `stage2_cell_of_origin/sabedot_gelb_2021/GeLB.R`

- **Atlas:** GeLB — EPIC-array glioma blood classifier (Sabedot et al.)
- **Source:** GitHub `iSidneyTorresJr/GeLB` (extracted from supplementary; URL in script)
- **Citation:** Sabedot TS et al. *Neuro-Oncology* 23(9): 1494–1507 (2021). DOI 10.1093/neuonc/noab023.
- **License:** academic use per supplementary statement
- **Format:** R training script (6.5 KB) — script trains classifier from GSE150289 cohort data (Mendeley deposit cgrz6zztfg)
- **Production status:** Queue-1 — requires running the script to produce the trained classifier

### `stage2_cell_of_origin/marlin_capper_training/`

- **Atlas:** mnp_training — Capper et al. methylation-based brain tumor classifier (basis for MARLIN extension)
- **Source:** GitHub `mwsill/mnp_training` master branch
- **Citation:** Capper D, Jones DTW, Sill M, Hovestadt V, et al. *Nature* 555, 469–474 (2018). DOI 10.1038/nature26000.
- **License:** custom academic (LICENSE in repo)
- **Format:** R scripts (training, calibration, cross-validation, t-SNE, preprocessing) + filter probe lists (ambiguous, EPIC-V1B2, SNP, XY)
- **Production status:** Queue-1 — leukemia-specific MARLIN matrix is v0.3 build-out; this training scaffold is the foundation

---

## Disaster recovery — what to do if an original source disappears

Each atlas in this vault has been pulled from a GitHub repo, R package, or Zenodo deposit. If any of those disappears, the vault provides:

1. **The actual atlas data** (in CSV and/or .rda)
2. **The SHA-256 hash** for integrity verification
3. **The original URL and citation** for tracing provenance
4. **The license** for compliance auditing

To rebuild the cookbook from the vault alone (no internet), an operator needs:
- This `README.md` for context
- The atlas CSV/rda files
- The card pre-registrations and source code from `Biological_Physics/validation_runs/`
- The cookbook docs (delivered separately, not in repo, per IP discipline)

This is sufficient to re-run any VAL deterministically.

---

## What this vault is NOT

This vault is **not** a substitute for proper backup of the cohort IDAT files (which are large, gated, and live on GEO/EGA/dbGaP). EDEAR's input data (the IDATs from each customer or each validation cohort) is held separately and is governed by separate consent and access agreements. The vault is for the *atlas / reference-matrix layer only* — the part of the pipeline that does NOT contain personal genetic information.

The vault is **not** a substitute for the IAMPerformance H_min derivation chain, which is The Recipe and lives in the vault literally meaning the patent-protected unpublished framework. This atlas vault is the public reference layer; The Recipe is the private framework that scores against those references.

---

## Update log

- **2026-04-26 PM:** Vault created. Mirrored Loyfer/Moss (production), UniLIFE (Queue-1 #1), Salas Blood.EPIC IDOL (production baseline), Salas IDOL-Ext (metadata), EpiSCORE 14-tissue × 28 matrices (Queue-1), Caggiano CelFiE TIM (Queue-1), Sabedot GeLB R script (Queue-1), Capper mnp_training (Queue-1). Total 65 files, 4.7 MB. INVENTORY.json generated with SHA-256 per file.
- Future updates will append here.

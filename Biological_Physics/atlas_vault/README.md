# Atlas Vault — IAMPerformance / EDEAR Reference Layer

**Purpose.** Durable, version-controlled storage of every methylation atlas and reference matrix the EDEAR scoring engine uses to compute A-scores from a customer's IDAT. This is the **canonical source** for the production scoring server. The container filesystem `/home/claude/atlases/` is scratch and disappears between sessions — this vault is the durable backup against original-source disappearance (GitHub repos deleted, bioRxiv preprints withdrawn, EGA projects archived) and the operational source for the production server.

**Last updated:** 2026-04-26  
**Total vault size:** 6.0 MB across 79 files  
**Total reference matrices for scoring:** 8 distinct atlases / 42 reference matrices  
**Maintained by:** Heath W. Mahaffey

---

## What this vault contains, and what calls it

The EDEAR scoring engine (production server-side scoring of customer IDATs) loads every reference matrix in this vault and computes per-class A-scores under run-everything architecture. The flow is:

```
Customer IDAT (450K or 850K methylation array)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│   EDEAR scoring engine (commercial.web.py / production)    │
│                                                             │
│   1. β-extraction from IDAT (sesame/minfi)                  │
│   2. Stage 1 — pooled-entropy A-scores on disease panels   │
│   3. Stage 2 — cell-of-origin deconvolution                │
│      ├── Loyfer/Moss array atlas (production)              │
│      ├── EpiSCORE 14-tissue (Queue-1)                      │
│      ├── Caggiano CelFiE (Queue-1, WGBS-region)            │
│      └── Sabedot GeLB classifier (Queue-1, glioma)         │
│   4. Stage 3 — immune fraction estimation                  │
│      ├── Salas Blood.EPIC IDOL baseline (production)       │
│      ├── UniLIFE 19-cell lifespan (Queue-1 #1)             │
│      └── EpiDISH companion panels (5 reference panels)     │
│   5. Cellular age clock                                     │
│      └── (currently Horvath/Hannum; 17-tissue Ageing       │
│          Atlas pending acquisition for v0.3)               │
│   6. Per-class A-scores against H_min anchors              │
│   7. Tier classification + customer report                  │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
   Customer report:
   - Cellular age
   - Per-cell-class A-score
   - Per-tissue ΔA (departure from normal)
   - Immune-class signature
   - Tier (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH)
```

The H_min anchors and the framework that scores against atlas matrices (The Recipe) are NOT in this vault — they are patent-protected and live separately. This vault is the public reference layer.

---

## The 8 atlases / 42 reference matrices

### Stage 2 — Cell-of-origin deconvolution

| # | Atlas | Files | Cell types / tissues | Status | License |
|---|---|---|---|---|---|
| 1 | **Loyfer/Moss array atlas** | `stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` | 25 cell types × 7,890 array CpGs | **PRODUCTION** | MIT |
| 2 | **EpiSCORE pan-tissue** | `stage2_cell_of_origin/episcore_zhu_teschendorff_2022/` (28 CSV files) | 14 tissues × 4–10 cell types each (Bladder, Brain, Breast, Colon, Esophagus, Heart, Kidney, Liver, Lung, Olfactory Epithelium, Pancreas + 9-cell extended Pancreas, Prostate, Skin) | Queue-1 | GPL-2 |
| 3 | **Caggiano CelFiE TIM** | `stage2_cell_of_origin/caggiano_celfie_2021/tim_matrix.txt` | 1,580 markers × 19 tissues (WGBS-region-based, requires region-to-CpG mapping) | Queue-1 | MIT |
| 4 | **Sabedot GeLB** | `stage2_cell_of_origin/sabedot_gelb_2021/GeLB.R` | EPIC glioma blood classifier (R training script, requires GSE150289 cohort to train) | Queue-1 | academic |
| 5 | **Capper mnp_training** | `stage2_cell_of_origin/marlin_capper_training/` | Brain tumor 450K/EPIC classifier training scaffold (foundation for MARLIN leukemia matrix v0.3 build-out) | Queue-1 | custom academic |

### Stage 3 — Immune fraction estimation

| # | Atlas | Files | Cell types | Status | License |
|---|---|---|---|---|---|
| 6 | **Salas Blood.EPIC IDOL baseline** | `stage3_immune_fraction/salas_blood_epic_idol/` (5 .rda + 2 CSV) | 6 cell types (CD8T, CD4T, NK, Bcell, Mono, Neu) × 450 EPIC CpGs; 350 CpG × 6 cell type 450K legacy; cord blood reference | **PRODUCTION** | GPL-3 |
| 7 | **Salas IDOL-Ext** | `stage3_immune_fraction/salas_idol_ext/` (Pheno + metadata + R wrapper) | 12 cell types extended panel (data via Bioconductor ExperimentHub at GSE167998) | Queue-1 (superseded by UniLIFE) | custom academic |
| 8 | **UniLIFE Guo 2025** | `stage3_immune_fraction/unilife_guo_2025/centUniLIFE.m.rda` + `centUniLIFE_reference_matrix.csv` | 19 immune cell types × 1,906 CpGs lifespan-spanning birth → old age | Queue-1 #1 | GPL-2 |

### Stage 3 — EpiDISH companion panels (alternative immune deconvolution references)

| Companion | Files | Cell types |
|---|---|---|
| **cent12CT** | `cent12CT.m.rda` + `cent12CT_reference_matrix.csv` | 12 immune cell types × 600 CpGs (EPIC) |
| **cent12CT450k** | `cent12CT450k.m.rda` + `cent12CT450k_reference_matrix.csv` | 12 immune cell types × 600 CpGs (450K legacy) |
| **centBloodSub** | `centBloodSub.m.rda` + `centBloodSub_reference_matrix.csv` | 7 cell types × 188 CpGs (B, NK, CD4T, CD8T, Mono, Neutro, Eosino) |
| **centDHSbloodDMC** | `centDHSbloodDMC.m.rda` + `centDHSbloodDMC_reference_matrix.csv` | 7 cell types × 333 CpGs (DHS-prioritized) |
| **centEpiFibFatIC** | `centEpiFibFatIC.m.rda` + `centEpiFibFatIC_reference_matrix.csv` | 4 tissue compartments × 491 CpGs (Epi, Fib, Fat, Immune) |
| **centEpiFibIC** | `centEpiFibIC.m.rda` + `centEpiFibIC_reference_matrix.csv` | 3 tissue compartments × 716 CpGs (Epi, Fib, Immune) |

All 5 EpiDISH companion panels are at `stage3_immune_fraction/epidish_companion_panels/`.

---

## Vault discipline — operational rules

1. **Every atlas EDEAR uses must live here.** Production atlases (Loyfer/Moss, Salas, UniLIFE) and Queue-1 atlases (EpiSCORE, Caggiano, Sabedot, MARLIN) are mirrored here as canonical durable copies.

2. **Every commit updates the INVENTORY.** `INVENTORY.json` lists every file with size and SHA-256. Future sessions verify integrity:
   ```
   cd Biological_Physics/atlas_vault/
   jq -r '.[] | "\(.sha256)  \(.path)"' INVENTORY.json | sha256sum -c -
   ```

3. **CSV formats preferred over R-data.** Where possible, atlases are stored both as native `.rda` (for R-pipeline compatibility) AND as CSV (for Python / cross-language portability). The next instance of Walther in 6 months may not have R available — CSV doesn't care.

4. **Provenance non-negotiable.** Every atlas directory documents source URL, citation, license, download date, SHA-256, conversion steps.

5. **Surveillance feeds the vault.** Each monthly atlas surveillance sweep ends with: (a) update GAPE Reproduction Paper §7.17, §7.18, etc.; (b) acquire newly-surfaced atlases that are tractable; (c) commit to vault; (d) refresh INVENTORY.json.

6. **License compliance.** EDEAR's commercial deployment must comply with each atlas's license. CC-BY-NC requires attribution-only use; GPL requires derivative works to be GPL'd. License audit happens at v0.3 build-out.

---

## Per-atlas provenance and citations

### 1. Loyfer/Moss 2018 array atlas

- **Source:** `nloyfer/meth_atlas` GitHub (2018) + Loyfer et al. *Nature* 613, 355–364 (2023) DOI 10.1038/s41586-022-05580-6
- **License:** MIT
- **Format:** CSV, 7,890 rows × 25 columns
- **Cell types:** B-cells, CD4T, CD8T, NK, Mono, Neu, Eos, granulocytes, hepatocytes, pancreatic acinar/beta/duct, vascular endothelial, cortical neurons, lung, head & neck larynx, kidney, breast, prostate, colon epithelial, upper GI, uterus/cervix, thyroid, bladder, adipocytes, left atrium, erythrocyte progenitors

### 2. EpiSCORE pan-tissue atlas

- **Source:** `aet21/EpiSCORE` GitHub master branch
- **Citation:** Teschendorff AE, Zhu T, Breeze CE, Beck S. *Genome Biol* 21, 221 (2020)
- **License:** GPL-2
- **Coverage:** 14 tissues × 28 reference matrices (each tissue has both an `expref` mRNA-derived and an `mref` DNAm-derived reference)
- **Conversion:** Original `.rda` → CSV via Python `rdata` library, 2026-04-26

### 3. Caggiano CelFiE 2021 TIM matrix

- **Source:** `christacaggiano/celfie` GitHub master branch
- **Citation:** Caggiano C, Celona B, Garton F, et al. *Nat Commun* 12, 2717 (2021)
- **License:** MIT
- **Format:** Tab-separated, 1,580 markers × 19 tissues
- **Caveat:** Markers are `chrom/start/end` genomic regions, NOT array CpG IDs. Integration requires region-to-CpG mapping per array platform.

### 4. Sabedot GeLB 2021

- **Citation:** Sabedot TS et al. *Neuro-Oncology* 23(9): 1494–1507 (2021)
- **Format:** R training script (6.5 KB) + requires GSE150289 cohort data (Mendeley deposit cgrz6zztfg) to produce trained classifier
- **License:** academic per supplementary

### 5. Capper mnp_training (MARLIN building block)

- **Source:** `mwsill/mnp_training` GitHub master
- **Citation:** Capper D, Jones DTW, Sill M, Hovestadt V, et al. *Nature* 555, 469–474 (2018)
- **Format:** R scripts (training, calibration, cross-validation, t-SNE, preprocessing) + filter probe lists
- **License:** custom academic

### 6. Salas Blood.EPIC IDOL baseline

- **Source:** `immunomethylomics/FlowSorted.Blood.EPIC` GitHub master
- **Citation:** Salas LA, Koestler DC, Butler RA, Hansen HM, Wiencke JK, Kelsey KT, Christensen BC. *Genome Biol* 19:64 (2018)
- **License:** GPL-3
- **Format:** 5 .rda files + 2 CSV (450 EPIC CpGs × 6 cell types: CD8T, CD4T, NK, Bcell, Mono, Neu)

### 7. Salas IDOL-Ext (extended 12-cell panel)

- **Source:** `immunomethylomics/FlowSorted.BloodExtended.EPIC` GitHub master
- **Citation:** Salas LA et al. (under review at time of package release, 2021)
- **Format:** Pheno.csv + metadata.csv + R wrapper code; data lazy-loaded from Bioconductor ExperimentHub at GSE167998
- **License:** see SoftwareLicense PDF in source

### 8. UniLIFE (Guo 2025)

- **Source:** `sjczheng/EpiDISH` GitHub `data/centUniLIFE.m.rda`
- **Citation:** Guo X, Sulaiman M, Neumann A, Zheng SC, Cecil CAM, Teschendorff AE, Heijmans BT. *Genome Med* 17:63 (2025), DOI 10.1186/s13073-025-01489-7
- **License:** GPL-2 (EpiDISH package)
- **Format:** Both `.rda` (281 KB) and CSV (712 KB); 1,906 marker CpGs × 19 immune cell types
- **Cell types:** 7 pan-lifespan (B, CD4T, CD8T, Mono, nRBC, Gran, NK) + 12 adult-specific (aCD4Tnv, aBaso, aCD4Tmem, aBmem, aBnv, aTreg, aCD8Tmem, aCD8Tnv, aEos, aNK, aNeu, aMono)
- **Operational use:** `EpiDISH::epidish(X, cent=centUniLIFE.m, method="RPC", maxit=500)$estF`
- **Compatibility:** 450K, EPICv1, EPICv2, WGBS

---

## Atlases NOT in vault (acquisition pending)

These 8 additional atlases were identified in the 2026-04-26 surveillance sweep but were not acquired in this session due to network restrictions (bioRxiv 503, Zenodo 503) or controlled access (EGA). Each is documented with its acquisition path; they will be added to the vault on the next opportunity.

| Atlas | Why not on disk | Acquisition path |
|---|---|---|
| **Tanaka 2025 6-cell neural cfDNA** | EGA controlled access | Markers extractable from supplementary PDF; primary nanopore data via EGA application |
| **Konigsberg/Cuadrat 2023 cardiac** | Relies on EGA-controlled Loyfer 2023 | Cardiomyocyte markers in supplementary tables |
| **Jacques 2025 17-tissue Ageing Atlas** | bioRxiv supp blocked from container today | Retry via direct browser; bioRxiv DOI 10.1101/2025.07.21.665830 |
| **MethAgingDB** | Zenodo 503 today | Zenodo DOI 10.5281/zenodo.15714493 |
| **Ontology-aware Kim 2025-2026** | bioRxiv JS-gated; no public GitHub located | Contact Yao lab (Rice University) |
| **Cuadrat 2026 Comm Bio guidelines** | Nature 503 today | Methodology paper, not a downloadable atlas |
| **Liu 2023 brain scMCodes** | Allen Brain Cell Atlas, Queue-2 | Single-cell to array projection (engineering) |
| **Zhou 2025 Body Single-Cell Atlas** | bioRxiv supp, Queue-2 | 86,689 nuclei × 16 tissues, 206 subtypes |
| **223-cell-type WGBS atlas** | Source verification needed | arXiv 2506.00146 reference; locate primary publication |

---

## Disaster recovery

If any original atlas source disappears, this vault provides:

1. **The actual atlas data** (CSV + .rda)
2. **SHA-256 hash** for integrity verification (in INVENTORY.json)
3. **Original URL and citation** for tracing provenance
4. **License** for compliance auditing

To rebuild EDEAR's reference layer from the vault alone (no internet), an operator needs:
- This README.md for context
- The atlas CSV/rda files
- INVENTORY.json for integrity verification
- The card pre-registrations and source code from `Biological_Physics/validation_runs/`
- The H_min anchors and scoring framework (The Recipe — held separately)

This is sufficient to re-run any VAL deterministically.

---

## What this vault is NOT

This vault is **not** a substitute for IDAT files (cohort or customer methylation arrays). Those are gated separately and contain personal genetic information.

This vault is **not** the H_min derivation chain (The Recipe), which is patent-protected and lives separately. This vault is the **public reference layer**; The Recipe is the **private framework** that scores against those references.

This vault is **not** the EDEAR scoring engine itself. It is the data layer that the engine loads at startup. A reference scoring engine implementation pattern is provided alongside this vault as `commercial_web_scoring_engine_skeleton.py` for reference when building the production server.

---

## Update log

- **2026-04-26:** Vault initialized. Mirrored 8 atlases / 42 reference matrices (4.7 MB → 6.0 MB after EpiDISH companion panels added). Loyfer/Moss (production), UniLIFE (Queue-1 #1), Salas Blood.EPIC IDOL (production baseline), Salas IDOL-Ext (metadata), EpiSCORE 14-tissue × 28 matrices (Queue-1), Caggiano CelFiE TIM (Queue-1), Sabedot GeLB R script (Queue-1), Capper mnp_training (Queue-1), 5 EpiDISH companion panels. Total 79 files. INVENTORY.json generated with SHA-256 per file.

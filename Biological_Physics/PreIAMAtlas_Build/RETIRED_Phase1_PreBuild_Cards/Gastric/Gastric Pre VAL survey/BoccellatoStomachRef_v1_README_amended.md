# BoccellatoStomachRef v1 — Atlas Reference Documentation

**Atlas ID:** `boccellato_stomachref_v1`
**Build date:** 2026-05-02
**Status:** Built, CHK-3.1C dedupe gate PASS, awaiting Phase B calibration VAL on VAL-106 cohort
**Substrate:** EPIC 850K (GPL21145) SWAN-normalized + ChAMP-filtered
**Use case:** Stage 2 cell-of-origin reference for gastric-epic + esophageal-epic (cross-organ extension via shared columnar metaplasia signature) + secondary cell-of-origin reader for ANY EDEAR card under run-everything regime

---

## Source citation

Fritsche K, Boccellato F, Schlaermann P, Koeppel M, Denecke C, Link A, Malfertheiner P, Gut I, Meyer TF, Berger H. **DNA methylation in human gastric epithelial cells defines regional identity without restricting lineage plasticity.** *Clinical Epigenetics* 2022;14:193. DOI: [10.1186/s13148-022-01406-4](https://doi.org/10.1186/s13148-022-01406-4). PMID: 36585699.

GEO accession: **GSE141660** (GPL21145 sub-platform, EPIC 850K methylation arm)

License: Open Access (Creative Commons CC BY 4.0)

---

## Cohort and sample structure

**18 samples = 3 donors × 3 regions × 2 differentiation states.**

| GSM | Sample title | Region | State | Donor | Donor sex | Donor age |
|-----|--------------|--------|-------|-------|-----------|-----------|
| GSM4210705 | Antrum_undifferentiated_rep1 | Antrum | undiff (+W/R, stem-enriched) | hGAT23 | F | 55 |
| GSM4210706 | Antrum_undifferentiated_rep2 | Antrum | undiff (+W/R) | hGAT24 | M | 47 |
| GSM4210707 | Antrum_undifferentiated_rep3 | Antrum | undiff (+W/R) | hGAT26 | F | 69 |
| GSM4210708 | Antrum_differentiated_rep1 | Antrum | diff (−W/R, pit-cell-like) | hGAT23 | F | 55 |
| GSM4210709 | Antrum_differentiated_rep2 | Antrum | diff (−W/R) | hGAT24 | M | 47 |
| GSM4210710 | Antrum_differentiated_rep3 | Antrum | diff (−W/R) | hGAT26 | F | 69 |
| GSM4210711 | Corpus_undifferentiated_rep1 | Corpus | undiff (+W/R) | hGAT23 | F | 55 |
| GSM4210712 | Corpus_undifferentiated_rep2 | Corpus | undiff (+W/R) | hGAT24 | M | 47 |
| GSM4210713 | Corpus_undifferentiated_rep3 | Corpus | undiff (+W/R) | hGAT26 | F | 69 |
| GSM4210714 | Corpus_differentiated_rep1 | Corpus | diff (−W/R) | hGAT23 | F | 55 |
| GSM4210715 | Corpus_differentiated_rep2 | Corpus | diff (−W/R) | hGAT24 | M | 47 |
| GSM4210716 | Corpus_differentiated_rep3 | Corpus | diff (−W/R) | hGAT26 | F | 69 |
| GSM4210717 | Fundus_undifferentiated_rep1 | Fundus | undiff (+W/R) | hGAT23 | F | 55 |
| GSM4210718 | Fundus_undifferentiated_rep2 | Fundus | undiff (+W/R) | hGAT24 | M | 47 |
| GSM4210719 | Fundus_undifferentiated_rep3 | Fundus | undiff (+W/R) | hGAT26 | F | 69 |
| GSM4210720 | Fundus_differentiated_rep1 | Fundus | diff (−W/R) | hGAT23 | F | 55 |
| GSM4210721 | Fundus_differentiated_rep2 | Fundus | diff (−W/R) | hGAT24 | M | 47 |
| GSM4210722 | Fundus_differentiated_rep3 | Fundus | diff (−W/R) | hGAT26 | F | 69 |

**Cell type:** purified primary human gastric epithelial cells, isolated from sleeve resection tissue per Boccellato 2018 protocol (Boccellato F et al., *Gut* 2018), cultivated as plane mucosoids in air-liquid-interface inserts. Stem-cell-enriched (+W/R) maintained with WNT3A + R-spondin 1; differentiated (−W/R) achieved by 7-day W/R withdrawal which differentiates the population into pit-like cells.

**This is cell-type-pure DNA methylation, NOT bulk biopsy.** The published reference atlases that resemble this in construction approach are Loyfer 25-tile (sorted-cell WGBS) and Moss 2018 (sorted bulk peripheral-blood + tissues). EpiSCORE-style references (BladderRef, ProstateRef, etc.) are scRNA-seq-imputed, structurally distinct.

---

## Build methodology

### Source preprocessing (per Boccellato 2022 Methods, applied by authors before GEO deposit)

1. **Bisulfite conversion + Illumina MethylationEPIC standard protocol** at Life&Brain research center (Bonn, Germany) collaborating with Per Hoffmann, Institute of Human Genetics, University of Bonn.
2. **Subset-quantile Within Array Normalization (SWAN)** via R/Bioconductor `ChAMP` package.
3. **Filter:** detection p-value > 0.01 → drop; bead-count < 3 → drop; CpG falls near a SNP per Zhou W et al. 2016 → drop; probe aligns to multiple genomic locations per Nordlund J et al. 2013 → drop; non-cg probes → drop; probes located on sex chromosomes → drop.
4. **Output:** 738,115 CpGs survive filtering (compared to ~865K probes on raw EPIC 850K).
5. **Function `combineArrays()`** used internally to merge EPIC + 450K samples within the paper but our atlas extracts ONLY the EPIC sub-series (GPL21145).

### Type 1 atlas adaptation (this build)

**Adaptation operation:** for each of the 738,115 CpGs, compute the mean β-value across the 3 donor replicates within each (region, state) combination, producing 6 tile β-values per CpG.

**Output schema:**
```
CpG_ID, Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff
cg00000029, 0.055080, 0.054324, 0.056363, 0.062496, 0.033691, 0.067453
cg00000109, 0.814202, 0.751717, 0.808995, 0.821001, 0.840935, 0.805690
...
```

**File path:** `/home/claude/gastric_esophageal_sprint/atlas_acquisition/boccellato_stomachref_v1.csv`

**SHA-256:** `fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11`
**File size:** 48,715,676 bytes
**Total CpG rows:** 738,115
**Header row:** 1 (column names)
**Tiles:** 6 (Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff)

### Verification gates passed

| Gate | Status | Detail |
|------|--------|--------|
| **CHK-3.1A full-genome substrate gate** | PASS | 36.97% extreme (β<0.1 or β>0.9), 9.33% middle (β∈[0.4,0.6]) on raw 18-sample input — exceeds raw-EPIC threshold (extreme >30%, middle <10%) |
| **CHK-3.1C dedupe gate** | PASS | Zero duplicate CpG IDs across 738,115 rows |
| **Per-tile β distribution** | EXPECTED | Tile β medians 0.60-0.64, q5 ~0.04, q95 ~0.93 (bimodal as expected for stomach mucosa) |
| **Atlas-family-fitness** | DOCUMENTED | 5,977 CpGs (0.81% of total) have between-tile range >0.2 — consistent with Boccellato 2022's published 3,703 inter-regional FDR<5% DMs (we report broader pre-FDR superset; 1.6× ratio expected) |

---

## Atlas-family-fitness diagnostic (per-CpG between-tile β range)

| Statistic | Value |
|-----------|-------|
| Total CpGs with all 6 tiles populated | 738,115 |
| Median between-tile range | 0.0385 |
| 90th percentile | 0.0997 |
| 95th percentile | 0.1265 |
| 99th percentile | 0.1907 |
| Maximum | 0.9005 |
| Fraction with range >0.1 | 9.92% |
| Fraction with range >0.2 | 0.81% (5,977 CpGs) |
| Fraction with range >0.4 | 0.06% (459 CpGs) |

**Interpretation.** The atlas's discriminating power lives in the long tail. Most CpGs do not distinguish between gastric regions or differentiation states (the gastric mucosal lineage is a relatively homogeneous methylation program). The strong inter-regional discriminating CpGs (range >0.4, 459 CpGs) are the candidate operational tile-discrimination markers. Phase B calibration against VAL-106 cohort will measure how the per-tile A-scores separate when scoring is applied to non-stomach healthy tissue (kidney, prostate adjacent-normal): expectation is all 6 tiles read at similar "non-stomach" baselines.

**DISC-GASTRIC candidate finding (provisional, pre-Phase-B):** purified-cell-type atlases with 3 donor replicates per cell type produce tile-discrimination magnitudes substantially smaller than scRNA-seq-imputed gene-promoter atlases (which typically pre-select genes with dramatic cell-type expression differentials before bridging to CpGs). This is a **methodology-versus-methodology** observation, not an atlas-quality observation; the underlying biology of the gastric mucosa is genuinely homogeneous compared to e.g. luminal-vs-basal epithelial differentiation in prostate or breast. Phase B calibration is the empirical test.

---

## Coverage against Xu-538 panel

To be measured in Phase B calibration prereg's CHK-2.17 cohort-substrate-coverage pre-flight check.

Boccellato substrate is EPIC 850K SWAN+ChAMP filtered → 738,115 CpGs. Xu-538 panel is HM450-defined 538 CpGs. Expected EPIC-Xu538 overlap is ~434/538 (~80.7%) per cervical-epic precedent (CHK-1.2). Need to verify how many of those 434 survived ChAMP filtering on the Boccellato preprocessing pipeline.

---

## Use within EDEAR run-everything regime

Per Heath sign-off 2026-04-26 + the calibration-typology document: run-everything mandates Stage 1 + Stage 2 + Stage 3 with all panels + all atlases on every IDAT. BoccellatoStomachRef joins the atlas list as a Stage 2 cell-of-origin reader for foregut endoderm derivatives:

| Atlas | Stage | Cell types covered | Atlas mode |
|-------|-------|--------------------|-----------:|
| BoccellatoStomachRef v1 | Stage 2 | Antrum / Corpus / Fundus × undiff/diff (foregut endoderm gastric epithelium) | NEW (this build) |
| Layered Moss+Loyfer 25-tile | Stage 2 | 25 tissues including bulk `stomach`, `Upper_GI`, `esophagus`, `small intestine`, `Hepatocytes`, `Lung_cells`, `Bladder`, `Pancreas`, etc. | Existing |
| Caggiano CelFiE TIM | Stage 2 microenvironment | 19 tumor-infiltrating cell types | Existing (calibrated VAL-113) |
| EpiSCORE LiverRef (cross-check for HCC) | Stage 2 | 6 hepatic cell types incl. hepatocyte, cholangiocyte, hepatic stellate, Kupffer | On disk, bridge engineering future |
| Salas IDOL | Stage 3 | 6-cell immune fine-tune | Existing |
| UniLIFE Guo 2025 | Stage 3 | 19 immune cell types | Existing (calibrated VAL-082+) |
| Xu-538 panel | Stage 1 | 538-CpG immune-class architectural drift detector | Existing (panel SHA `ada6729605...`) |

---

## Subsequent VAL chain to seal the atlas calibration

1. **VAL-12X (Boccellato calibration on VAL-106 cohort, Phase B):** 6 BoccellatoStomachRef tiles × n=210 healthy adjacent-normal (TCGA-KIRC + TCGA-PRAD HM450 sesame Level 3). Outcomes: per-tile A-score mean / SD / q5 / q95 distributions, sealed as the healthy-floor thresholds for each tile (Type 2 calibration artifact). Operational floor is per-tile q5.

2. **VAL-12X+1 (atlas-family-fitness verification):** Confirm tile separation pattern on non-stomach tissue. Expectation per CCL-039: tile A-scores collapse on non-target tissue and separate on gastric tissue.

3. **VAL-12X+2/+3 (Phase C run-everything STAD + ESCA):** disease cohort scoring with all atlases simultaneously per the run-everything mandate.

---

## Files in this atlas

| File | Purpose | Persistence |
|------|---------|-------------|
| `GSE141660_EPIC_matrix.txt.gz` | Source SWAN+ChAMP β matrix from GEO (raw input, 18 samples × 738,115 CpGs) | Atlas vault — keep for reproducibility per CHK-7.6 |
| `GSE141660_HM450_matrix.txt.gz` | Companion HM450 sub-series (22 samples = 18 mucosoids + 4 5-aza controls) | Atlas vault — kept for cross-platform verification |
| `boccellato_stomachref_v1.csv` | Built atlas reference (738,115 CpGs × 6 tiles, mean across donor reps) — THE atlas | **Frozen artifact** — loaded at scoring time |
| `filelist.txt` | Provenance: GEO suppl directory listing showing IDAT availability | Atlas vault — provenance |
| `BUILD_LOG.md` | This file | Atlas vault README |

After GitHub push: `Biological_Physics/atlas_vault/stage2_cell_of_origin/boccellato_stomachref_v1/` + entry in `Biological_Physics/atlas_vault/INVENTORY.json`

---

## License + reproducibility

- **License:** CC BY 4.0 (from Boccellato 2022 source paper). Re-use permitted with citation.
- **Build script:** `build_boccellato_stomachref.py` (will be added at GitHub push time per CHK-7.6 reproducibility triple)
- **Build environment:** Python 3 standard library only (gzip, csv, hashlib, statistics, collections). No non-standard imports. ~6 minutes runtime, <1 GB memory.
- **Build inputs:** `GSE141660_EPIC_matrix.txt.gz` (71 MB compressed) at GEO FTP `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141660/matrix/GSE141660-GPL21145_series_matrix.txt.gz`
- **Expected output checksum:** SHA-256 `fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11`

---

## AMENDMENT — HM450-restricted derivative atlas (BoccellatoStomachRef_HM450 v1)

**Date:** 2026-05-02
**Reason for amendment:** CHK-2.17 cohort-substrate-coverage pre-flight check on TCGA HM450 sesame Level 3 substrate (5 random VAL-106 cohort samples, RNG seed 20260502) FAILED on the EPIC-built atlas at mean coverage 49.26% / min 48.77% — far below the CHK-2.8 substrate floor of 80%. Root cause: the EPIC 850K platform has ~865K CpG probes; HM450 has ~485K. Only 380,467 of the EPIC-built atlas's 738,115 CpGs (51.55%) exist on the HM450 platform. This is a known platform-mismatch failure mode, identical to VAL-117 ProstateRef amendment precedent. The atlas was restricted to the HM450-platform CpG subset; the original EPIC atlas is retained for provenance and for future EPIC-substrate scoring.

### Restricted atlas specification

- **File path:** `/home/claude/gastric_esophageal_sprint/atlas_acquisition/boccellato_stomachref_HM450_v1.csv`
- **SHA-256:** `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`
- **Size:** 25,110,908 bytes (380,467 CpG rows × 6 tile columns + header)
- **Tile structure:** identical to v1 — 6 tiles (Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff). Per-tile β values are unchanged from the EPIC build; this is a pure CpG-row-restriction operation, not a re-computation.

### Restriction methodology

1. Enumerate the HM450 CpG probe list from a representative TCGA HM450 sesame Level 3 sample file (`4d96e820-c934-452a-a734-c56adff2ee00.methylation_array.sesame.level3betas.txt`, file_id `217e1981-b406-4a0f-921a-93fd2979ad53`, sample TCGA-KIRC TCGA-BP-5183-11A from VAL-106 manifest). HM450 platform has 486,427 CpG probes per the sesame Level 3 standard layout.
2. Take the intersection of EPIC-build atlas CpGs (`boccellato_stomachref_v1.csv`, 738,115 CpGs after Boccellato authors' SWAN+ChAMP filtering) with the HM450 CpG list. Retained: 380,467 CpGs (51.55%).
3. Output the retained rows (CpG_ID + 6 tile β-values) to the restricted CSV. Tile β-values are unchanged; only the row set is restricted.
4. Compute SHA-256, verify CHK-3.1C dedupe gate (zero duplicates expected and observed), recompute per-tile β-distribution stats and atlas-family-fitness diagnostic.

### CHK-3.1C dedupe gate (HM450-restricted atlas)

PASS — zero duplicate CpG IDs across 380,467 rows.

### Per-tile β-distribution (HM450-restricted)

| Tile | n | mean | median | sd | q5 | q95 |
|------|---|------|--------|-----|-----|-----|
| Antrum_undiff | 380,467 | 0.4565 | 0.4316 | 0.3603 | 0.0352 | 0.9310 |
| Antrum_diff | 380,467 | 0.4509 | 0.4233 | 0.3555 | 0.0354 | 0.9258 |
| Corpus_undiff | 380,467 | 0.4453 | 0.4119 | 0.3491 | 0.0369 | 0.9202 |
| Corpus_diff | 380,467 | 0.4539 | 0.4238 | 0.3587 | 0.0343 | 0.9325 |
| Fundus_undiff | 380,467 | 0.4486 | 0.4147 | 0.3537 | 0.0353 | 0.9260 |
| Fundus_diff | 380,467 | 0.4490 | 0.4155 | 0.3551 | 0.0345 | 0.9274 |

Comparison to EPIC-build atlas (medians 0.60-0.64): the HM450-restricted atlas medians are slightly lower (0.41-0.43) because the EPIC-only CpGs that drop out are biased toward gene-body and intergenic regions (typically high methylation), while the HM450 platform was designed with promoter-CpG enrichment (typically variable methylation). This is a structural property of the platform difference, not a quality issue.

### Atlas-family-fitness diagnostic (HM450-restricted)

| Statistic | Value |
|-----------|-------|
| Total CpGs with all 6 tiles populated | 380,467 |
| Median between-tile range | 0.0333 |
| 90th percentile | 0.0900 |
| 95th percentile | 0.1156 |
| 99th percentile | 0.1808 |
| Fraction with range >0.1 | ~9.0% |
| Fraction with range >0.2 | 0.68% (2,593 CpGs) |
| Fraction with range >0.4 | 0.08% |

The HM450-restricted atlas retains 2,593 CpGs with between-tile range >0.2, compared to 5,977 in the EPIC build. This is approximately 43% retention of the discriminating-power CpGs — consistent with the ~52% probe-overlap and slightly biased against the most-discriminating CpGs because Boccellato 2022's published 3,703 inter-regional FDR-significant DMs include a substantial fraction at EPIC-only probes.

### CHK-2.17 verification (HM450-restricted atlas vs TCGA HM450 sesame Level 3 substrate)

| Sample | Probe-level coverage | Valid-β coverage | f_extreme within atlas | f_middle within atlas |
|--------|---------------------:|-----------------:|----------------------:|----------------------:|
| TCGA-KIRC TCGA-BP-5183-11A | 100.00% | 94.92% | 56.83% | 7.49% |
| TCGA-PRAD TCGA-G9-6353-11A | 100.00% | 96.17% | 53.13% | 8.49% |
| TCGA-KIRC TCGA-CZ-5451-11A | 100.00% | 96.08% | 55.64% | 7.80% |
| TCGA-KIRC TCGA-CZ-5468-11A | 100.00% | 96.02% | 57.00% | 7.16% |
| TCGA-KIRC TCGA-CJ-4903-11A | 100.00% | 94.62% | 56.09% | 7.49% |

**Mean valid-β coverage: 95.56% — PASS (≥90% target).**
**Min valid-β coverage: 94.62% — PASS (≥80% CHK-2.8 substrate floor).**
**CHK-2.17 PRE-FLIGHT GATE: PASS** on the HM450-restricted atlas.

The within-atlas-overlap bimodality (53-57% extreme, 7-8% middle) also exceeds the TCGA HM450K sesame Level 3 platform threshold (extreme ≥50.5%, middle ≤9.0%), confirming the restricted atlas inherits the substrate's CHK-3.1A bimodality property.

### Use within EDEAR run-everything regime

For TCGA HM450 sesame Level 3 substrate scoring (the standing EDEAR substrate per CCL-048), use the HM450-restricted atlas `boccellato_stomachref_HM450_v1.csv`. For native EPIC 850K substrate scoring (when EDEAR clinical pilot deploys EPIC arrays), use the original EPIC atlas `boccellato_stomachref_v1.csv`. Both are sealed in atlas_vault.

### Pre-flight artifacts

- `preflight_check/preflight_samples.json` — sample selection metadata
- `preflight_check/CHK-2.17_PREFLIGHT_RESULTS.json` — first run on EPIC atlas (FAIL at 49.26% coverage)
- `preflight_check/CHK-2.17_PREFLIGHT_RESULTS_HM450.json` — second run on HM450-restricted atlas (PASS at 95.56% coverage)

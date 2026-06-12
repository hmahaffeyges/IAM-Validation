# BoccellatoStomachRef v1 (EPIC source)

**Atlas ID:** `boccellato_stomachref_v1`
**Substrate:** Illumina MethylationEPIC (EPIC 850K), GPL21145
**Status:** Atlas vault provenance source. The HM450-restricted operational derivative is at `../boccellato_stomachref_HM450_v1/`.
**Calibration VAL:** none directly (HM450 derivative is calibrated as VAL-123)

---

## What this atlas is

EPIC 850K reference matrix for healthy human gastric mucosa, built from Boccellato 2022 GSE141660 (Fritsche K, Boccellato F et al., *Clinical Epigenetics* 2022;14:193, DOI 10.1186/s13148-022-01406-4, PMID 36585699).

Source GEO sample structure (18 samples): 3 donors × 3 stomach regions × 2 differentiation states.

| Region | Differentiation state | Donor | GSM IDs |
|--------|----------------------|-------|---------|
| Antrum | undiff (+W/R, stem-enriched) | hGAT23 (F-55), hGAT24 (M-47), hGAT26 (F-69) | GSM4210705, GSM4210706, GSM4210707 |
| Antrum | diff (−W/R, pit-cell-like)   | hGAT23, hGAT24, hGAT26 | GSM4210708, GSM4210709, GSM4210710 |
| Corpus | undiff (+W/R) | hGAT23, hGAT24, hGAT26 | GSM4210711, GSM4210712, GSM4210713 |
| Corpus | diff (−W/R)   | hGAT23, hGAT24, hGAT26 | GSM4210714, GSM4210715, GSM4210716 |
| Fundus | undiff (+W/R) | hGAT23, hGAT24, hGAT26 | GSM4210717, GSM4210718, GSM4210719 |
| Fundus | diff (−W/R)   | hGAT23, hGAT24, hGAT26 | GSM4210720, GSM4210721, GSM4210722 |

Cell-type-pure DNA methylation (purified primary gastric epithelial cells from sleeve resections, cultivated as plane mucosoids per Boccellato 2018 *Gut* protocol). NOT bulk biopsy.

Source preprocessing (per Boccellato 2022 Methods, applied by authors before GEO deposit): SWAN normalization + ChAMP filtering (detection p > 0.01, bead count < 3, SNP-overlapping CpGs per Zhou 2016, multi-mapping probes per Nordlund 2013, sex-chromosome probes excluded). 738,115 CpGs survive filtering.

---

## Construction (this atlas)

For each of the 738,115 surviving CpGs, compute the mean β-value across the 3 donor replicates within each (region, state) combination, producing 6 tile β-values per CpG.

Output schema:
```
CpG_ID, Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff
```

This standard reference-construction approach matches Loyfer 25-tile and EpiSCORE references.

---

## File specification

- **File:** `boccellato_stomachref_v1.csv`
- **Size:** 48,715,676 bytes
- **SHA-256:** `fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11`
- **Rows:** 738,116 (header + 738,115 CpGs)
- **Tile columns:** 6

### Verification gates passed

- **CHK-3.1A** (full-genome substrate gate): 36.97% extreme (β<0.1 or β>0.9), 9.33% middle (β∈[0.4,0.6]) on raw 18-sample input. Exceeds raw-EPIC threshold (extreme >30%, middle <10%). PASS.
- **CHK-3.1C** (atlas dedupe gate): zero duplicate CpG IDs across 738,115 rows. PASS.

### Atlas-family-fitness diagnostic

Per-CpG between-tile range distribution:

| Statistic | Value |
|-----------|-------|
| Median | 0.0385 |
| 95th percentile | 0.1265 |
| 99th percentile | 0.1907 |
| Maximum | 0.9005 |
| Fraction with range > 0.2 | 5,977 CpGs (0.81%) |
| Fraction with range > 0.4 | 459 CpGs (0.06%) |

Consistent with Boccellato 2022's reported 3,703 inter-regional FDR<5% DMs (we report broader pre-FDR superset; ratio 1.6× expected).

---

## Substrate compatibility

EPIC 850K substrate only. **For HM450 sesame Level 3 substrate scoring, use the HM450-restricted derivative** at `../boccellato_stomachref_HM450_v1/` (380,467 CpGs, SHA `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`).

The EPIC source is retained for provenance and for future EPIC-substrate scoring when EDEAR clinical pilot deploys EPIC arrays.

---

## License + reproducibility

- **License:** CC BY 4.0 (Boccellato 2022 source paper). Re-use permitted with citation.
- **Build script:** `build_boccellato_stomachref_v1.py`
- **Build environment:** Python 3 standard library only (gzip, csv, hashlib, statistics, collections). No non-standard imports.
- **Build inputs:** `GSE141660_EPIC_matrix.txt.gz` from GEO FTP `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141660/matrix/GSE141660-GPL21145_series_matrix.txt.gz` (71,243,285 bytes, SHA-256 `d43bd068645c9f9d2e63fb704d1f7caa4b02c137b0e007721d3f973738b25b04`)
- **Expected output checksum:** SHA-256 `fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11`
- **Build runtime:** ~6 minutes, <1 GB memory peak.

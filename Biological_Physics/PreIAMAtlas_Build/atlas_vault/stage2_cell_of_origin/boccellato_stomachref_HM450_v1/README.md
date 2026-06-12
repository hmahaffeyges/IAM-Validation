# BoccellatoStomachRef_HM450 v1

**Atlas ID:** `boccellato_stomachref_HM450_v1`
**Substrate:** Illumina HumanMethylation450 BeadChip (HM450)
**Status:** Operational atlas for TCGA HM450 sesame Level 3 substrate scoring
**Calibration VAL:** VAL-123 (sealed 2026-05-02, outcome O1_BOCCELLATO_CALIBRATION_SEALED)
**EPIC source:** `../boccellato_stomachref_v1/boccellato_stomachref_v1.csv` (SHA `fbe1dbfdec...`)

---

## What this atlas is

The HM450-platform-restricted derivative of BoccellatoStomachRef v1. Created because the EPIC source contains 738,115 CpGs but only 380,467 of those CpGs (51.55%) exist on the HM450 platform — applying the EPIC atlas directly to TCGA HM450 sesame Level 3 substrate would produce ~49% per-sample coverage, far below the CHK-2.8 substrate floor of 80%.

This atlas restricts the EPIC source to its intersection with the HM450 probe set. Tile β-values are unchanged from the EPIC source; only the CpG row-set is restricted.

This pattern mirrors **VAL-117 ProstateRef amendment precedent** — when an atlas built on a richer substrate needs to be used on a sparser substrate, restrict to the intersection rather than re-engineer the atlas.

---

## File specification

- **File:** `boccellato_stomachref_HM450_v1.csv`
- **Size:** 25,110,908 bytes
- **SHA-256:** `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`
- **Rows:** 380,468 (header + 380,467 CpGs)
- **Tile columns:** 6 (Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff)

### Verification gates passed

- **CHK-3.1C** (atlas dedupe gate): zero duplicate CpG IDs across 380,467 rows. PASS.
- **CHK-2.17 cohort-substrate-coverage pre-flight gate** (5 random TCGA-KIRC + TCGA-PRAD adjacent-normal samples from VAL-106 manifest, RNG seed 20260502):
  - First run on EPIC source: 49.26% mean coverage — FAIL
  - **Second run on this HM450-restricted atlas: 95.56% mean coverage, 94.62% min — PASS**

---

## Construction (this atlas)

1. Enumerate the HM450 CpG probe list from a representative TCGA HM450 sesame Level 3 sample file (file_id `217e1981-b406-4a0f-921a-93fd2979ad53`, sample `TCGA-KIRC TCGA-BP-5183-11A` from VAL-106 manifest). HM450 platform has 486,427 CpG probes.
2. Take the intersection of EPIC-build atlas CpGs (738,115) with the HM450 CpG list. Retained: 380,467 CpGs (51.55%).
3. Output the retained rows (CpG_ID + 6 tile β-values) to the restricted CSV. Tile β-values are unchanged; only the row set is restricted.

The build script `restrict_to_hm450.py` reproduces this build deterministically.

---

## Atlas-family-fitness diagnostic (HM450-restricted)

| Statistic | EPIC source v1 | HM450-restricted v1 | Notes |
|-----------|----------------:|---------------------:|-------|
| n CpGs | 738,115 | 380,467 | 51.55% retention |
| Median between-tile range | 0.0385 | 0.0333 | slightly lower (HM450 enriched for promoter CpGs) |
| Fraction with range > 0.2 | 5,977 (0.81%) | 2,593 (0.68%) | 43% retention of discriminating CpGs |
| Fraction with range > 0.4 | 459 (0.06%) | ~300 (0.08%) | comparable |

The discriminating-power retention (43%) is slightly lower than the probe-overlap retention (51.55%) because some of Boccellato 2022's published 3,703 inter-regional FDR-significant DMs are EPIC-only probes.

---

## Per-tile β-distribution (HM450-restricted)

| Tile | n | mean | median | sd | q5 | q95 |
|------|---|------|--------|-----|-----|-----|
| Antrum_undiff | 380,467 | 0.4565 | 0.4316 | 0.3603 | 0.0352 | 0.9310 |
| Antrum_diff | 380,467 | 0.4509 | 0.4233 | 0.3555 | 0.0354 | 0.9258 |
| Corpus_undiff | 380,467 | 0.4453 | 0.4119 | 0.3491 | 0.0369 | 0.9202 |
| Corpus_diff | 380,467 | 0.4539 | 0.4238 | 0.3587 | 0.0343 | 0.9325 |
| Fundus_undiff | 380,467 | 0.4486 | 0.4147 | 0.3537 | 0.0353 | 0.9260 |
| Fundus_diff | 380,467 | 0.4490 | 0.4155 | 0.3551 | 0.0345 | 0.9274 |

Comparison to EPIC-build atlas (per-tile medians 0.60-0.64): HM450-restricted medians are slightly lower (0.41-0.43) because the EPIC-only CpGs that drop out are biased toward gene-body and intergenic regions (typically high methylation), while the HM450 platform was designed with promoter-CpG enrichment (typically variable methylation). This is a structural property of the platform difference, not a quality issue.

---

## VAL-123 calibration outcome (sealed Type 2 healthy-floor thresholds)

Cohort: TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (203 QC-passed). Substrate: TCGA HM450K sesame Level 3 (VAL-106 standing healthy substrate).

| Tile | mean A | sd | q5 (operational floor) | q95 |
|------|-------:|----:|----------------------:|----:|
| Antrum_undiff | 0.1282 | 0.0051 | **0.1194** | 0.1355 |
| Antrum_diff   | 0.1324 | 0.0054 | **0.1236** | 0.1403 |
| Corpus_undiff | 0.1389 | 0.0055 | **0.1298** | 0.1471 |
| Corpus_diff   | 0.1314 | 0.0051 | **0.1222** | 0.1387 |
| Fundus_undiff | 0.1363 | 0.0053 | **0.1272** | 0.1440 |
| Fundus_diff   | 0.1354 | 0.0052 | **0.1264** | 0.1430 |

**Operational floor convention:** per-tile q5 is the healthy-floor threshold. A patient sample's tile A-score below this q5 (anomalously gastric-similar on this tile) flags an operational diagnostic event.

Maximum within-cohort tile range: **0.0394** (≥ 0.02 pre-locked floor — atlas does NOT collapse to substrate floor; produces meaningful per-tile separation).

These values are sealed in the `gastric_esophageal_epic_card_v0_1.json` `chk_3_1_thresholds_per_substrate.boccellato_stomachref_HM450_v1.tcga_hm450_sesame_level3` block and loaded at run-everything scoring time.

---

## Tile class assignment (H_min)

All 6 tiles are gastric epithelial cells classed as `secretory` (H_min = 0.843264) per G-003b MCMC frozen 2026-04-06.

---

## Use within EDEAR run-everything regime

For TCGA HM450 sesame Level 3 substrate scoring (the standing EDEAR substrate per CCL-048): use this atlas. For native EPIC 850K substrate scoring (when EDEAR clinical pilot deploys EPIC arrays): use the EPIC source atlas at `../boccellato_stomachref_v1/`.

---

## License + reproducibility

- **License:** CC BY 4.0 (Boccellato 2022 source paper). Re-use permitted with citation.
- **Restriction script:** `restrict_to_hm450.py`
- **Restriction inputs:**
  - EPIC source atlas: `../boccellato_stomachref_v1/boccellato_stomachref_v1.csv` (SHA `fbe1dbfdec...`)
  - HM450 probe-set source: any TCGA HM450 sesame Level 3 β file (probe list is identical across samples)
- **Build environment:** Python 3 standard library only. <30 seconds, <500 MB memory.
- **Expected output:** SHA-256 `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`, 25,110,908 bytes, 380,467 CpGs.

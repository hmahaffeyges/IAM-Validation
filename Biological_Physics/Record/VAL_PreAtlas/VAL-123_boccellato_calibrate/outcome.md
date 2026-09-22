# VAL-123 Outcome — BoccellatoStomachRef_HM450 Phase B Calibration

**VAL ID:** VAL-123
**Card target:** gastric+esophageal-epic v0.1
**Atlas:** BoccellatoStomachRef_HM450 v1 (HM450-restricted, SHA `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`)
**Cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (VAL-106 standing healthy substrate)
**Substrate:** TCGA HM450K sesame Level 3
**Executed:** 2026-05-02
**Prereg sealed:** 2026-05-02T18:45:54Z, SHA `91a5f06f984ce64b747bd2800fdde28f18dbc3105008f2008730a1ee659d71d7`

---

## Outcome class: O1_BOCCELLATO_CALIBRATION_SEALED

CHK-3.1A 96.7% (203/210, target ≥90%); CHK-3.1B 100.0% (210/210, target ≥95%); CHK-3.1C dedupe PASS (zero duplicates); max within-cohort tile range 0.0394 (≥0.02 floor — atlas does not collapse to substrate floor); cross-tile separation on means 0.0107.

The HM450-restricted Boccellato atlas calibrates cleanly on non-stomach healthy adjacent-normal tissue. Per-tile A-score distributions are sealed below as the Type 2 calibration artifact (frozen healthy-floor thresholds) for run-everything scoring per the atlas calibration typology document.

---

## Pre-flight CHK-2.17 history (BLOCKING gate, executed before prereg seal)

**First run** (against EPIC-source atlas `boccellato_stomachref_v1.csv`, 738,115 CpGs):
- Mean valid-β coverage: 49.26%
- Min: 48.77%
- **GATE FAIL** — root cause: EPIC vs HM450 platform mismatch. Only 380,467 of 738,115 atlas CpGs (51.55%) exist on the HM450 platform.
- Mirrors VAL-117 ProstateRef amendment precedent.

**Second run** (against HM450-restricted atlas `boccellato_stomachref_HM450_v1.csv`, 380,467 CpGs):
- Mean valid-β coverage: 95.56%
- Min: 94.62%
- **GATE PASS** — atlas restricted to HM450-platform CpG subset clears CHK-2.8 substrate floor (≥80%).

The atlas restriction is a pure CpG-row filter — no β values were re-computed. The original EPIC atlas remains in atlas_vault for provenance and for future EPIC-substrate scoring.

---

## CHK-3.1A full-genome substrate gate (TCGA HM450K sesame Level 3)

| Metric | Pre-locked threshold | Observed mean | Observed SD |
|--------|---------------------:|--------------:|------------:|
| f_extreme (β<0.1 or β>0.9) | ≥ 0.505 | 0.5512 | 0.0234 |
| f_middle (0.4≤β≤0.6) | ≤ 0.090 | 0.0750 | 0.0073 |

Pass rate: **203/210 (96.7%) — PASS** (gate threshold 90%).

Failed samples (n=7): primarily samples with elevated f_middle slightly above the 0.090 threshold or reduced f_extreme — substrate-marginal samples flagged for downstream review.

---

## CHK-3.1B per-sample atlas-CpG-coverage gate

| Metric | Pre-locked threshold | Observed |
|--------|---------------------:|---------:|
| Per-sample coverage | ≥ 0.80 (CHK-2.8 substrate floor) | mean 0.953, q5 0.946 |

Pass rate: **210/210 (100.0%) — PASS** (gate threshold 95%).

Coverage is universally high because the HM450-restricted atlas contains only CpGs that exist on the HM450 platform by construction.

---

## CHK-3.1C atlas dedupe gate

Total atlas rows: 380,467
Unique CpG IDs: 380,467
Duplicates: 0
**PASS**

---

## Per-tile healthy-floor distributions (Type 2 calibration artifact, SEALED)

| Tile | n | mean | SD | q2.5 | q5 (operational floor) | median | q95 | q97.5 | within-cohort range |
|------|---|------|-----|------|------------------------|--------|------|-------|--------------------|
| Antrum_undiff | 203 | 0.1282 | 0.0051 | 0.1170 | **0.1194** | 0.1280 | 0.1355 | 0.1369 | 0.0391 |
| Antrum_diff | 203 | 0.1324 | 0.0054 | 0.1216 | **0.1236** | 0.1322 | 0.1403 | 0.1419 | 0.0394 |
| Corpus_undiff | 203 | 0.1389 | 0.0055 | 0.1281 | **0.1298** | 0.1390 | 0.1471 | 0.1483 | 0.0379 |
| Corpus_diff | 203 | 0.1314 | 0.0051 | 0.1208 | **0.1222** | 0.1316 | 0.1387 | 0.1395 | 0.0374 |
| Fundus_undiff | 203 | 0.1363 | 0.0053 | 0.1255 | **0.1272** | 0.1364 | 0.1440 | 0.1450 | 0.0368 |
| Fundus_diff | 203 | 0.1354 | 0.0052 | 0.1247 | **0.1264** | 0.1356 | 0.1430 | 0.1437 | 0.0368 |

**Operational floor convention:** per-tile q5 is the healthy-floor threshold. A patient sample's tile A-score below this q5 (anomalously gastric-similar on this tile) flags an operational diagnostic event.

**These values are sealed in the gastric+esophageal-epic v0.1 card JSON's `chk_3_1_thresholds_per_substrate.boccellato_stomachref_HM450_v1.tcga_hm450_sesame_level3` block** and loaded at run-everything scoring time. EDEAR's official application loads these thresholds at startup and never re-calibrates.

---

## Cross-tile separation on means (atlas-family-fitness diagnostic)

| Statistic | Value |
|-----------|-------|
| Smallest tile mean | A_Antrum_undiff = 0.1282 |
| Largest tile mean | A_Corpus_undiff = 0.1389 |
| Cross-tile separation (max − min) | **0.0107** |
| Mean per-tile SD across cohort | ~0.0053 |
| Cross-tile separation in pooled-SD units | ~2.0 |

The atlas's 6 tiles separate from each other by approximately 2 within-tile SDs on non-stomach healthy tissue. This is small in absolute terms but substantially above the within-tile noise floor — the atlas is reading the cohort consistently and the tiles are not collapsing to a single substrate-floor value.

The maximum within-cohort tile range (0.0394) is **above the 0.02 tissue-floor-dominated threshold** pre-locked in the prereg, ruling out the O3_TISSUE_FLOOR_DOMINATED outcome class.

---

## Per-tissue stratification (KIRC vs PRAD)

| Tile | KIRC mean (n=159) | PRAD mean (n=44) | Δ (KIRC−PRAD) |
|------|-------------------:|------------------:|---------------:|
| Antrum_undiff | 0.1290 | 0.1253 | +0.0037 |
| Antrum_diff | 0.1332 | 0.1295 | +0.0037 |
| Corpus_undiff | 0.1398 | 0.1356 | +0.0042 |
| Corpus_diff | 0.1323 | 0.1281 | +0.0042 |
| Fundus_undiff | 0.1372 | 0.1331 | +0.0041 |
| Fundus_diff | 0.1363 | 0.1324 | +0.0039 |

PRAD reads systematically lower than KIRC on every tile, by approximately 0.004 (within 1 SD of within-tile noise). This is a sub-SD-level cross-tissue offset, not a CHK-3.2 baseline mismatch flag, but it is documented for forward reference. Interpretation: kidney parenchyma may share marginally more methylation features with gastric mucosoid stem-cell-enriched cultures than prostate epithelium does, possibly reflecting shared mesoderm vs endoderm origin patterns at a small subset of CpGs.

Sex stratification was declared underpowered at prereg time (PRAD all-male, KIRC ~70% male) and is not reported.

---

## CCL-039 atlas-family-fitness assessment

The Boccellato atlas is a **purified-cell-type atlas** (gastric epithelial mucosoids, 3 donors × 3 regions × 2 differentiation states). Compared to scRNA-seq-imputed gene-promoter atlases (EpiSCORE family), purified-cell-type atlases typically show smaller tile-discrimination magnitudes because they preserve the full methylation distribution of the source cells without pre-selecting genes with dramatic cell-type expression differentials.

The cross-tile separation observed here (0.0107 on means) is consistent with this expectation. The atlas's discriminating power is expected to manifest more strongly when applied to gastric-tissue samples, where tile-specific reference patterns can be matched against gastric-tissue β profiles. Phase C disease cohort scoring (VAL-124 TCGA-STAD, VAL-125 TCGA-ESCA) is the empirical test of that prediction.

---

## Run-everything mandate compliance

This calibration produces a Type 2 frozen artifact per the atlas calibration typology document Heath provided. EDEAR's official application loads the per-tile q5 thresholds at startup and applies them to every patient IDAT under the run-everything regime per Heath sign-off 2026-04-26. CHK-3.2 cross-cohort baseline checks at every subsequent VAL will reference the per-tile mean ± SD sealed here as the anchor distribution.

---

## Sealed numbers for downstream use

For card JSON `chk_3_1_thresholds_per_substrate.boccellato_stomachref_HM450_v1.tcga_hm450_sesame_level3`:

```json
{
  "anchor_cohort": "TCGA-KIRC + TCGA-PRAD adjacent-normal n=210",
  "anchor_val": "VAL-123",
  "atlas_sha256": "f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390",
  "tiles": {
    "Antrum_undiff": {"mean": 0.1282, "sd": 0.0051, "q5_floor": 0.1194, "q95": 0.1355},
    "Antrum_diff":   {"mean": 0.1324, "sd": 0.0054, "q5_floor": 0.1236, "q95": 0.1403},
    "Corpus_undiff": {"mean": 0.1389, "sd": 0.0055, "q5_floor": 0.1298, "q95": 0.1471},
    "Corpus_diff":   {"mean": 0.1314, "sd": 0.0051, "q5_floor": 0.1222, "q95": 0.1387},
    "Fundus_undiff": {"mean": 0.1363, "sd": 0.0053, "q5_floor": 0.1272, "q95": 0.1440},
    "Fundus_diff":   {"mean": 0.1354, "sd": 0.0052, "q5_floor": 0.1264, "q95": 0.1430}
  }
}
```

---

## CHK-7.6 reproducibility triple

**Source code:** `val123_boccellato_calibrate.py` (524 lines) + `val123_chunk_runner.py` (151 lines, used for chunked execution due to runtime constraints). Both Python 3 with NumPy + standard library. Inline source preserved at GitHub push time.

**Inputs:**
- `boccellato_stomachref_HM450_v1.csv` (SHA `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`, 25.1 MB, 380,467 CpGs)
- VAL-106 cohort 210 β files (TCGA-KIRC + TCGA-PRAD adjacent-normal HM450 sesame Level 3, manifest SHA `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`, total ~2.6 GB), each downloaded from GDC public-access API and SHA-256 verified.

**Environment:** Python 3.x, NumPy, csv, json, hashlib, math, time, pathlib, collections — standard scientific stack. Total runtime ~10 minutes wall-clock for n=210 cohort. Memory peak ~400 MB.

**Expected output:** `VAL-123_calibration_results.json` with outcome class O1_BOCCELLATO_CALIBRATION_SEALED + per-tile distributions matching the table above.

---

## Files in this VAL

| File | Push to GitHub? |
|------|-----------------|
| `prereg.md` | YES |
| `PREREG_SEAL.txt` | YES |
| `val123_boccellato_calibrate.py` | YES |
| `val123_chunk_runner.py` | YES |
| `cohort_manifest.json` | YES |
| `VAL-123_calibration_results.json` | YES |
| `VAL-123_per_sample_calibration.csv` | YES |
| `outcome.md` (this file) | YES |
| `VAL-123_execution_log.txt` | YES |
| `val123_per_sample_progress.ndjson` | NO (working file, supersedes by per_sample_calibration.csv) |

Atlas files pushed separately to `Biological_Physics/atlas_vault/stage2_cell_of_origin/boccellato_stomachref_HM450_v1/` with INVENTORY.json update.

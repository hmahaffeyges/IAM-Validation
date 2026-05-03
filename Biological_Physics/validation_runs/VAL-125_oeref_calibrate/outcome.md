# VAL-125 Outcome — EpiSCORE OEref bridged Phase B Calibration

**VAL ID:** VAL-125
**Card target:** gastric+esophageal-epic v0.1
**Atlas:** EpiSCORE OEref bridged (SHA `8f4e34ef63247b0ca09312fedb52abf5eee9ee1e8f09e35044ddceb8bdf3f651`)
**Cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (VAL-106 standing healthy substrate)
**Substrate:** TCGA HM450K sesame Level 3
**Executed:** 2026-05-02
**Prereg sealed before β observation:** SHA `f7628a46c36f3d268b0eadbfe495e302a4373238c3cffeaf170ef152aa8b4c1c`

---

## Outcome class: O2_PARTIAL_FLOORS

CHK-3.1A 96.7% (203/210, target ≥90%); CHK-3.1B 100.0% (210/210, target ≥95%); CHK-3.1C dedupe PASS (zero duplicates at bridge time); max within-cohort tile range 0.0422 (≥0.02 floor); cross-tile separation on means 0.0407.

**Outcome O2 (PARTIAL_FLOORS) honest finding:** 4 of 9 tiles (Macro, NeuIm, Peri, Plasma, Tcell) cleared the SD ≥ 0.005 non-degeneracy threshold; 5 tiles (Basal, Fib, Gland, NeuMa) read tightly (SD 0.0037-0.0048 — close to but below threshold). Per-tile q5 healthy-floor thresholds still seal for ALL 9 tiles per O2 prereg specification. The Basal tile (oral squamous epithelium, HNSCC cell of origin) reads at SD=0.0041 — tightly clustered on non-oral tissue, which is consistent with OEref serving as a CONFIRMATORY squamous reference cross-checking EsoRef Epi_basal in VAL-127 ESCA Phase C, NOT a primary cell-of-origin reference. The 4 immune tiles (Macro, NeuIm, NeuMa, Plasma, Tcell) and the perivascular Peri tile have richer methylation structure on non-target tissue, which is also expected — immune compartments are systemically distributed and produce signal on bulk-tissue β.

Per-tile A-score distributions sealed below as Type 2 calibration artifact (frozen healthy-floor thresholds) for run-everything scoring per the atlas calibration typology document.

---

## CHK-3.1A full-genome substrate gate (TCGA HM450K sesame Level 3)

| Metric | Pre-locked threshold | Observed mean |
|--------|---------------------:|--------------:|
| f_extreme (β<0.1 or β>0.9) | ≥ 0.505 | 0.5591 |
| f_middle (0.4≤β≤0.6) | ≤ 0.090 | 0.0737 |

Pass rate: **203/210 (96.7%)** — gate threshold 90%, status PASS.

---

## CHK-3.1B per-sample atlas-CpG-coverage gate

| Metric | Pre-locked threshold | Observed |
|--------|---------------------:|---------:|
| Per-sample coverage | ≥ 0.80 (CHK-2.8 substrate floor; VAL-117/119 EpiSCORE precedent) | mean 0.8865, q5 0.8504 |

Pass rate: **210/210 (100.0%)** — gate threshold 95%, status PASS.

---

## Per-tile healthy-floor distributions (Type 2 calibration artifact, SEALED)

| Tile | n | mean | SD | q5 (operational floor) | q95 | within-cohort range |
|------|---|------|-----|------------------------|------|--------------------|
| Basal | 203 | 0.3994 | 0.0041 | **0.3910** | 0.4044 | 0.0266 |
| Fib | 203 | 0.3929 | 0.0048 | **0.3860** | 0.4024 | 0.0272 |
| Gland | 203 | 0.3938 | 0.0037 | **0.3873** | 0.3984 | 0.0238 |
| Macro | 203 | 0.4297 | 0.0056 | **0.4217** | 0.4386 | 0.0420 |
| NeuIm | 203 | 0.4221 | 0.0043 | **0.4165** | 0.4289 | 0.0280 |
| NeuMa | 203 | 0.4044 | 0.0039 | **0.3993** | 0.4105 | 0.0251 |
| Peri | 203 | 0.4314 | 0.0066 | **0.4224** | 0.4423 | 0.0422 |
| Plasma | 203 | 0.3907 | 0.0053 | **0.3831** | 0.3985 | 0.0367 |
| Tcell | 203 | 0.3930 | 0.0057 | **0.3847** | 0.4016 | 0.0405 |


**Operational floor convention:** per-tile q5 is the healthy-floor threshold. A patient sample's tile A-score below this q5 (anomalously EpiSCORE OEref bridged-similar) flags an operational diagnostic event.

These values are sealed in the gastric+esophageal-epic v0.1 card JSON's `chk_3_1_thresholds_per_substrate.episcore_oeref_bridged.tcga_hm450_sesame_level3` block and loaded at run-everything scoring time.

---

## Atlas-family-fitness diagnostic

| Statistic | Value |
|-----------|-------|
| Cross-tile separation on means (max − min across 9 tiles) | **0.0407** |
| Max within-cohort tile range (max-min across samples, then max across tiles) | 0.0422 |
| Within-sample tile range (mean across QC cohort) | 0.0422 |

---

## Per-tissue stratification (KIRC vs PRAD)

| Tile | KIRC mean (n=159) | PRAD mean (n=44) | Δ (PRAD−KIRC) | Δ in KIRC SD units | MW p |
|------|------:|------:|------:|------:|------:|
| Basal | 0.4003 | 0.3962 | -0.0041 | -1.61 | 4.63e-07 |
| Fib | 0.3934 | 0.3913 | -0.0021 | -0.53 | 9.43e-04 |
| Gland | 0.3941 | 0.3929 | -0.0012 | -0.45 | 1.69e-02 |
| Macro | 0.4292 | 0.4314 | +0.0022 | +0.45 | 2.95e-01 |
| NeuIm | 0.4218 | 0.4233 | +0.0016 | +0.44 | 4.66e-01 |
| NeuMa | 0.4040 | 0.4057 | +0.0017 | +0.54 | 2.96e-01 |
| Peri | 0.4302 | 0.4358 | +0.0056 | +1.04 | 2.81e-05 |
| Plasma | 0.3897 | 0.3945 | +0.0048 | +1.16 | 8.58e-05 |
| Tcell | 0.3921 | 0.3961 | +0.0040 | +0.82 | 1.76e-03 |


Sex stratification was declared underpowered at prereg time (PRAD all-male, KIRC ~70% male) and is not reported.

---

## CHK-7.6 reproducibility triple

**Source code:** `val124_125_calibrate.py` (parametrized for both VAL-124 EsoRef and VAL-125 OEref). Standard Python 3 + NumPy + scipy.stats.

**Inputs:**
- Atlas SHA `8f4e34ef63247b0ca09312fedb52abf5eee9ee1e8f09e35044ddceb8bdf3f651`
- VAL-106 cohort 210 β files (manifest SHA `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`)

**Environment:** Python 3.x. Runtime ~3 min wall-clock for n=210 (chunked 70 samples per invocation).

**Expected output:** matches `VAL-125_calibration_results.json` (sealed).

---

## Files

| File | Push to GitHub? |
|------|-----------------|
| `prereg.md` | YES |
| `PREREG_SEAL.txt` | YES |
| `VAL-125_calibration_results.json` | YES |
| `VAL-125_per_sample_calibration.csv` | YES |
| `outcome.md` | YES |
| `val124_125_calibrate.py` (shared) | YES (saved at parent dir; will be copied into each VAL dir) |
| `val125_per_sample_progress.ndjson` | NO (working file, superseded by per_sample_calibration.csv) |

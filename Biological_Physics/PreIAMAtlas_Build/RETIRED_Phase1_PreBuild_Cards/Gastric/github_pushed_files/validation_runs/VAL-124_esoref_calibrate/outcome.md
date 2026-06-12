# VAL-124 Outcome — EpiSCORE EsoRef bridged Phase B Calibration

**VAL ID:** VAL-124
**Card target:** gastric+esophageal-epic v0.1
**Atlas:** EpiSCORE EsoRef bridged (SHA `6e650bd78ed2ee32d98ac4508b3b4295a1cd303a0fed2d850eeb9c35f897692b`)
**Cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (VAL-106 standing healthy substrate)
**Substrate:** TCGA HM450K sesame Level 3
**Executed:** 2026-05-02
**Prereg sealed before β observation:** SHA `1bab7c99b35a3ebc680e93e6935a84f2b712fe7ec6663632d696a6a92433090f`

---

## Outcome class: O1_CALIBRATION_SEALED

CHK-3.1A 96.7% (203/210, target ≥90%); CHK-3.1B 100.0% (210/210, target ≥95%); CHK-3.1C dedupe PASS (zero duplicates at bridge time); max within-cohort tile range 0.0597 (≥0.02 floor); cross-tile separation on means 0.0990.

**Special observation:** EsoRef cross-tile separation on means (0.0990) is the largest observed across any EpiSCORE-bridged atlas calibration (vs ProstateRef ~0.06, BladderRef 0.07, HeartRef 0.015). The four esophageal-epithelium differentiation-state tiles (Epi_basal → Epi_stratified → Epi_suprabasal → Epi_upper) span methylation programs distinct enough to register meaningful structure even on non-esophagus tissue. Most-elevated tile is Epi_upper (luminal/superficial squamous cells, mean 0.4698); least-elevated is Gland (submucosal glands, mean 0.3708). This unusually rich tile structure positions EsoRef as a strong cell-of-origin reference for ESCC subtype-stratified scoring in VAL-127 ESCA Phase C.

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
| Per-sample coverage | ≥ 0.80 (CHK-2.8 substrate floor; VAL-117/119 EpiSCORE precedent) | mean 0.8682, q5 0.8381 |

Pass rate: **210/210 (100.0%)** — gate threshold 95%, status PASS.

---

## Per-tile healthy-floor distributions (Type 2 calibration artifact, SEALED)

| Tile | n | mean | SD | q5 (operational floor) | q95 | within-cohort range |
|------|---|------|-----|------------------------|------|--------------------|
| EC | 203 | 0.3884 | 0.0119 | **0.3754** | 0.4117 | 0.0597 |
| Epi_basal | 203 | 0.3807 | 0.0072 | **0.3721** | 0.3929 | 0.0408 |
| Epi_stratified | 203 | 0.4331 | 0.0062 | **0.4202** | 0.4417 | 0.0343 |
| Epi_suprabasal | 203 | 0.4009 | 0.0057 | **0.3931** | 0.4114 | 0.0377 |
| Epi_upper | 203 | 0.4698 | 0.0084 | **0.4540** | 0.4829 | 0.0435 |
| Fib | 203 | 0.3871 | 0.0089 | **0.3774** | 0.4055 | 0.0475 |
| Gland | 203 | 0.3708 | 0.0076 | **0.3624** | 0.3862 | 0.0455 |
| IC | 203 | 0.3765 | 0.0101 | **0.3632** | 0.3965 | 0.0566 |


**Operational floor convention:** per-tile q5 is the healthy-floor threshold. A patient sample's tile A-score below this q5 (anomalously EpiSCORE EsoRef bridged-similar) flags an operational diagnostic event.

These values are sealed in the gastric+esophageal-epic v0.1 card JSON's `chk_3_1_thresholds_per_substrate.episcore_esoref_bridged.tcga_hm450_sesame_level3` block and loaded at run-everything scoring time.

---

## Atlas-family-fitness diagnostic

| Statistic | Value |
|-----------|-------|
| Cross-tile separation on means (max − min across 8 tiles) | **0.0990** |
| Max within-cohort tile range (max-min across samples, then max across tiles) | 0.0597 |
| Within-sample tile range (mean across QC cohort) | 0.0991 |

---

## Per-tissue stratification (KIRC vs PRAD)

| Tile | KIRC mean (n=159) | PRAD mean (n=44) | Δ (PRAD−KIRC) | Δ in KIRC SD units | MW p |
|------|------:|------:|------:|------:|------:|
| EC | 0.3837 | 0.4050 | +0.0213 | +3.34 | 6.69e-20 |
| Epi_basal | 0.3784 | 0.3890 | +0.0105 | +2.45 | 1.30e-12 |
| Epi_stratified | 0.4351 | 0.4260 | -0.0091 | -2.23 | 5.97e-15 |
| Epi_suprabasal | 0.4005 | 0.4024 | +0.0019 | +0.40 | 4.19e-01 |
| Epi_upper | 0.4724 | 0.4601 | -0.0123 | -2.12 | 3.97e-13 |
| Fib | 0.3841 | 0.3980 | +0.0139 | +2.81 | 3.06e-14 |
| Gland | 0.3680 | 0.3809 | +0.0129 | +3.31 | 1.17e-17 |
| IC | 0.3730 | 0.3893 | +0.0163 | +2.68 | 2.31e-15 |


Sex stratification was declared underpowered at prereg time (PRAD all-male, KIRC ~70% male) and is not reported.

---

## CHK-7.6 reproducibility triple

**Source code:** `val124_125_calibrate.py` (parametrized for both VAL-124 EsoRef and VAL-125 OEref). Standard Python 3 + NumPy + scipy.stats.

**Inputs:**
- Atlas SHA `6e650bd78ed2ee32d98ac4508b3b4295a1cd303a0fed2d850eeb9c35f897692b`
- VAL-106 cohort 210 β files (manifest SHA `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`)

**Environment:** Python 3.x. Runtime ~3 min wall-clock for n=210 (chunked 70 samples per invocation).

**Expected output:** matches `VAL-124_calibration_results.json` (sealed).

---

## Files

| File | Push to GitHub? |
|------|-----------------|
| `prereg.md` | YES |
| `PREREG_SEAL.txt` | YES |
| `VAL-124_calibration_results.json` | YES |
| `VAL-124_per_sample_calibration.csv` | YES |
| `outcome.md` | YES |
| `val124_125_calibrate.py` (shared) | YES (saved at parent dir; will be copied into each VAL dir) |
| `val124_per_sample_progress.ndjson` | NO (working file, superseded by per_sample_calibration.csv) |

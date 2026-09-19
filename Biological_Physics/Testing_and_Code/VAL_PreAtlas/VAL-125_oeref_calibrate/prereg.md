# VAL-125 — EpiSCORE OEref calibration on TCGA-KIRC+PRAD adjacent-normal n=210

**Sprint:** gastric+esophageal-epic v0.1 sprint, Phase B (atlas calibration)
**Card target:** gastric+esophageal-epic v0.1
**VAL ID:** VAL-125 (sequential after VAL-124 EsoRef calibration)
**Calibration cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (VAL-106 standing healthy substrate)
**Substrate:** TCGA HM450K sesame Level 3
**RNG seed:** 20260502

**SEALED BEFORE β DATA OBSERVED.** CCL-041 compliance.

---

## 1. Atlas under calibration

**EpiSCORE OEref (Oral Epithelium, CpG-bridged)** — `episcore_oeref_cpg_bridged.csv`
- **SHA-256:** `8f4e34ef63247b0ca09312fedb52abf5eee9ee1e8f09e35044ddceb8bdf3f651`
- **Source paper:** Zhu T, Liu J et al. "A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types." Nature Methods 2022;19:296. DOI 10.1038/s41592-022-01412-7. EpiSCORE GitHub `aet21/EpiSCORE`.
- **Source matrix:** `OEref__mrefOE_m.csv` (327 marker genes × 9 cell types + weight, SHA `c2a2e673f8b3ebd7c1bcbae60ae62571ecda77ab7a5ef06cf9333a23cfd33a67`)
- **Bridge methodology:** Entrez Gene ID → 450K probe via probeInfo450k.lv. Pattern mirrors VAL-117/119/124 bridge precedent.
- **Bridge result:** 327 source EIDs (314 mapped, 13 dropped) → **5,396 unique 450K CpGs × 9 cell types + weight**
- **Cell types:** Basal, Fib, Gland, Macro, NeuIm, NeuMa, Peri, Plasma, Tcell

**Tile class assignment** (G-003b MCMC posteriors frozen 2026-04-06):

| Tile | Description | Class | H_min |
|------|-------------|-------|-------|
| Basal | Oral squamous epithelium, basal layer (HNSCC cell of origin) | secretory | 0.843264 |
| Fib | Fibroblast / stromal | stromal | 0.862950 |
| Gland | Oral submucosal glands (minor salivary glands) | secretory | 0.843264 |
| Macro | Macrophages | immune | 0.838889 |
| NeuIm | Neutrophils, immature | immune | 0.838889 |
| NeuMa | Neutrophils, mature | immune | 0.838889 |
| Peri | Pericytes / smooth muscle around vessels | stromal | 0.862950 |
| Plasma | Plasma cells | immune | 0.838889 |
| Tcell | T lymphocytes | immune | 0.838889 |

**Use within sprint (carried from sprint Phase 0 cohort survey):** OEref Basal tile is the closest available squamous epithelial reference at the molecular level and serves as a CONFIRMATORY squamous reference for ESCC discrimination in VAL-127 ESCA Phase C. Oral cavity has different exposure history than esophagus (smokeless tobacco, betel, alcohol direct contact) so OEref is NOT the primary ESCC cell-of-origin reference; EsoRef Epi_basal is primary, OEref Basal cross-checks. This dual-reference test is the "first-of-kind multi-atlas subtype-discrimination" claim from the sprint Phase 0 cohort survey.

---

## 2. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING)

Same 5 random TCGA samples used for Boccellato + EsoRef preflight (RNG seed 20260502). Per-sample coverage threshold ≥ 80% per VAL-117/119 precedent.

**Pre-flight result (executed 2026-05-02):**
- Mean coverage: 89.38%
- Min coverage: 88.40%
- Max coverage: 90.03%
- All 5 samples ≥ 80% per-sample floor: **PASS**

**Decision: prereg seal proceeds.**

---

## 3. Pre-locked decision criteria

Each tile A-score on each sample: `A_tile = mean( |sample_β - tile_ref_β| ) / H_min_class`

### Outcome classes (one will fire after execution)

**O1_OEREF_CALIBRATION_SEALED** (target):
- CHK-3.1A pass rate ≥ 90%
- CHK-3.1B per-sample coverage pass rate ≥ 95% (threshold ≥ 80%)
- CHK-3.1C atlas dedupe PASS (already PASS at bridge time)
- Maximum within-cohort tile-range across the 9 tiles ≥ 0.02
- All 9 q5 healthy-floor thresholds sealed

**O2_PARTIAL_FLOORS:** 5-8 of 9 tiles non-degenerate; 1-4 collapse
**O3_TISSUE_FLOOR_DOMINATED:** all 9 tiles within-cohort range < 0.02
**O4_BRIDGE_FAILURE:** > 5% of cohort fails CHK-3.1B 80% gate
**O5_UNEXPECTED:** anything else

---

## 4-7. Same as VAL-124

CHK-3.1A pre-locked thresholds, CHK-3.1B per-sample coverage threshold, per-tissue stratification (KIRC vs PRAD reported separately, sex stratification underpowered), and Type 2 calibration artifact format identical to VAL-124. Sealed thresholds enter card JSON's `chk_3_1_thresholds_per_substrate.episcore_oeref.tcga_hm450_sesame_level3` block.

---

## 8. CHK-7.6 reproducibility triple

- **Source code:** `val125_oeref_calibrate.py` + `val125_chunk_runner.py`
- **Inputs:**
  - `episcore_oeref_cpg_bridged.csv` (SHA `8f4e34ef63247b0ca09312fedb52abf5eee9ee1e8f09e35044ddceb8bdf3f651`)
  - VAL-106 cohort 210 β files (manifest SHA `0330a3c6...`)
- **Environment:** Python 3.x. Expected ~10 min wall, ~400 MB memory.
- **Expected outputs:** `VAL-125_calibration_results.json`, `VAL-125_per_sample_calibration.csv`, `outcome.md`

---

## 9. Test 2 placeholder

CCL-030 / CHK-2.5: Test 2 BLOCKED on OQ-2026-01. N/A to calibration VAL.

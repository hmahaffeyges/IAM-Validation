# VAL-124 — EpiSCORE EsoRef calibration on TCGA-KIRC+PRAD adjacent-normal n=210

**Sprint:** gastric+esophageal-epic v0.1 sprint, Phase B (atlas calibration)
**Card target:** gastric+esophageal-epic v0.1
**VAL ID:** VAL-124 (sequential after VAL-123 BoccellatoStomachRef_HM450 calibration)
**Calibration cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (VAL-106 standing healthy substrate, manifest SHA `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`)
**Substrate:** TCGA HM450K sesame Level 3
**RNG seed:** 20260502

**SEALED BEFORE β DATA OBSERVED.** CCL-041 compliance.

---

## 1. Atlas under calibration

**EpiSCORE EsoRef (CpG-bridged)** — `episcore_esoref_cpg_bridged.csv`
- **SHA-256:** `6e650bd78ed2ee32d98ac4508b3b4295a1cd303a0fed2d850eeb9c35f897692b`
- **Source paper:** Zhu T, Liu J et al. "A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types." Nature Methods 2022;19:296. DOI 10.1038/s41592-022-01412-7. EpiSCORE GitHub `aet21/EpiSCORE`.
- **Source matrix:** `EsoRef__Eso_Mref_m.csv` (161 marker genes × 8 cell types + weight, SHA `975bbf80b07f31c802045fa152fb38917158e7dc4d9f363371c9173dca26c826`)
- **Bridge methodology:** Entrez Gene ID → 450K probe via probeInfo450k.lv (SHA `8fc1c08d5704afa8154d24894e3295e5f7cbe7d53c57a956c8ade194aca77092`). Pattern mirrors VAL-117 ProstateRef + VAL-119 BladderRef bridge precedent.
- **Bridge result:** 161 source EIDs (159 mapped, 2 dropped: EID 1410, 84290) → **2,464 unique 450K CpGs × 8 cell types + weight**
- **Cell types:** EC, Epi_basal, Epi_stratified, Epi_suprabasal, Epi_upper, Fib, Gland, IC

**Tile class assignment** (G-003b MCMC posteriors frozen 2026-04-06):

| Tile | Description | Class | H_min |
|------|-------------|-------|-------|
| EC | Vascular endothelial | stromal | 0.862950 |
| Epi_basal | Esophageal squamous epithelium, basal layer | secretory | 0.843264 |
| Epi_stratified | Esophageal squamous epithelium, stratified middle layer | secretory | 0.843264 |
| Epi_suprabasal | Esophageal squamous epithelium, suprabasal layer | secretory | 0.843264 |
| Epi_upper | Esophageal squamous epithelium, luminal/superficial layer | secretory | 0.843264 |
| Fib | Fibroblast / stromal | stromal | 0.862950 |
| Gland | Esophageal submucosal glands | secretory | 0.843264 |
| IC | Immune cells, intra-esophageal | immune | 0.838889 |

**Biology note (carried from sprint Phase 0 cohort survey):** ESCC arises from squamous epithelium (Epi_* tiles), EAC arises from columnar metaplasia (NOT in EsoRef — for EAC use BoccellatoStomachRef columnar-lineage reference). VAL-127 ESCA Phase C will use EsoRef + BoccellatoStomachRef + Loyfer to subtype-stratify ESCC vs EAC on cell-of-origin pattern.

---

## 2. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING)

Sample 5 random TCGA-KIRC + TCGA-PRAD adjacent-normal samples from VAL-106 manifest (RNG seed 20260502, samples reused from Boccellato preflight). Compute per-sample EsoRef coverage. Pass criteria per CHK-2.8 / VAL-117 / VAL-119 EpiSCORE-bridged precedent: **per-sample coverage ≥ 80%**.

**Pre-flight result (executed 2026-05-02):**
- Mean coverage: 87.57%
- Min coverage: 87.01%
- Max coverage: 87.95%
- All 5 samples ≥ 80% per-sample floor: **PASS**
- (Mean below 90% target but per-sample floor ≥ 80% — same precedent as VAL-117 ProstateRef q5=83.3% PASS and VAL-119 BladderRef q5=86.1% PASS)

**Decision: prereg seal proceeds.**

---

## 3. Pre-locked decision criteria

Each tile A-score on each sample: `A_tile = mean( |sample_β - tile_ref_β| ) / H_min_class`

### Outcome classes (one will fire after execution)

**O1_ESOREF_CALIBRATION_SEALED** (target):
- CHK-3.1A pass rate ≥ 90% (≥ 190/210)
- CHK-3.1B per-sample coverage pass rate ≥ 95% (≥ 200/210, threshold ≥ 80%)
- CHK-3.1C atlas dedupe PASS (already PASS at bridge time, 0 duplicates)
- Maximum within-cohort tile-range across the 8 tiles ≥ 0.02 (atlas does not collapse to substrate floor)
- All 8 q5 healthy-floor thresholds sealed in card JSON

**O2_PARTIAL_FLOORS:**
- 5-7 of 8 tiles produce non-degenerate distributions (per-tile SD ≥ 0.005); 1-3 tiles collapse
- Per-tile q5 still seals where calculable; collapsed tiles flagged in outcome.md

**O3_TISSUE_FLOOR_DOMINATED:**
- All 8 tiles read at within-cohort range < 0.02 (atlas collapses to substrate floor on non-target tissue)
- Cardio-LL-005 HeartRef precedent applies — atlas is esophagus-tissue-only detector
- Per-tile q5 still sealed; cookbook implication is EsoRef won't surface in non-esophagus runs

**O4_BRIDGE_FAILURE:**
- Per-sample CHK-3.1B coverage < 80% on > 5% of cohort
- Routes to bridge-repair pathway

**O5_UNEXPECTED:**
- Anything else (e.g. all tiles read NEGATIVE on healthy substrate)

**No bidirectional cancellation outcome applies** here per CCL-031 — calibration on healthy cohort produces a baseline distribution, not case-vs-control direction test.

---

## 4. CHK-3.1A pre-locked thresholds

Per VAL-106 sealed TCGA HM450K sesame Level 3 substrate calibration:
- f_extreme ≥ 0.505 (β < 0.1 or β > 0.9)
- f_middle ≤ 0.090 (0.4 ≤ β ≤ 0.6)
- Per-sample QC-pass requires both gates met
- Cohort pass rate ≥ 90% (≥ 190/210)

---

## 5. CHK-3.1B per-sample coverage threshold

Per CHK-2.8 substrate-floor for TCGA HM450K small EpiSCORE-bridged atlas subsets:
- Per-sample coverage ≥ 80% required
- Cohort pass rate ≥ 95% (≥ 200/210)
- Mirrors VAL-117 ProstateRef and VAL-119 BladderRef precedent

---

## 6. Per-tissue stratification

Cohort split:
- TCGA-KIRC adjacent-normal n=160 (kidney)
- TCGA-PRAD adjacent-normal n=50 (prostate; all-male)

KIRC and PRAD reported as separate strata. Sex stratification declared underpowered (PRAD all-male, KIRC ~70% male). Cross-tissue per-tile difference reported in CHK-3.2 anchor-SD units.

---

## 7. Sealed Type 2 calibration artifact

Output sealed at execution time:
- Per-tile A-score distributions (mean, SD, q2.5, q5, q25, median, q75, q95, q97.5, min, max, within-cohort range)
- Per-tile q5 = healthy-floor threshold (operational floor for run-everything scoring)
- Per-tissue stratification (KIRC vs PRAD per tile)
- CHK-3.2 cross-tissue difference in anchor-SD units

These thresholds are sealed in the gastric+esophageal-epic v0.1 card JSON's `chk_3_1_thresholds_per_substrate.episcore_esoref.tcga_hm450_sesame_level3` block and loaded at run-everything scoring time per the atlas calibration typology document.

---

## 8. CHK-7.6 reproducibility triple

- **Source code:** `val124_esoref_calibrate.py` + `val124_chunk_runner.py` (Python 3, NumPy, csv, hashlib, json — standard scientific stack)
- **Inputs:**
  - `episcore_esoref_cpg_bridged.csv` (SHA `6e650bd78ed2ee32d98ac4508b3b4295a1cd303a0fed2d850eeb9c35f897692b`)
  - VAL-106 cohort 210 β files (TCGA-KIRC + TCGA-PRAD adjacent-normal, manifest SHA `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`, ~2.6 GB local on disk)
- **Environment:** Python 3.x. Expected runtime ~10 min wall-clock for n=210, ~400 MB peak memory.
- **Expected outputs:**
  - `VAL-124_calibration_results.json`
  - `VAL-124_per_sample_calibration.csv`
  - `outcome.md`

---

## 9. Test 2 placeholder

Per CCL-030 / CHK-2.5: Test 2 (lymphoid vs myeloid sub-panel test) is BLOCKED on OQ-2026-01 immune-atlas staging. Not applicable to this calibration VAL.

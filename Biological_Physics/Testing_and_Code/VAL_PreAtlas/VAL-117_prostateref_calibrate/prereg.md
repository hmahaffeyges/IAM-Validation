# VAL-117 — Pre-Registration

**VAL ID:** VAL-117
**Card target:** prostate-epic v0.3 (Phase B calibration anchor)
**Atlas under calibration:** EpiSCORE ProstateRef (CpG-bridged), 2,603 CpGs × 6 prostate cell types
**Calibration cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (same cohort used for VAL-106/107 cardio calibration anchor)
**Substrate:** TCGA HM450K sesame Level 3
**Sealed:** [SEAL_TIMESTAMP at execution]
**Sealed BEFORE β access:** YES (this prereg.md is sealed before val117 script reads any β values)

---

## Question

Does the EpiSCORE ProstateRef CpG-bridged matrix produce per-tile A-score readings on a structurally-separated healthy substrate-matched cohort (TCGA HM450K sesame Level 3 adjacent-normal) that pass CHK-3.1A (full-genome substrate baseline) AND CHK-3.1B (atlas-subset coverage) AND CHK-3.1C (atlas dedup)? If yes, the per-tile healthy-floor distributions sealed here become the calibration anchor for ProstateRef in prostate-epic v0.3 production scoring on EPIC 850K and HM450K substrates.

This is the analog of VAL-111/VAL-112 cardio calibration: same cohort, same CHK-3.1A/B/C protocol, same substrate-matched-healthy-reference principle. Different atlas (ProstateRef vs HeartRef + Loyfer + Caggiano TIM).

---

## Why this calibration matters operationally

The v0.2 prostate-epic card scores Stage 2 against the layered Moss+Loyfer atlas, which has a single `prostate_epithelial` tile (secretory class, H_min = 0.843264, healthy reference β = 0.743). EpiSCORE ProstateRef adds prostate sub-cell-type resolution beyond that single tile:

- **BE** = Basal Epithelial
- **EC** = Endothelial Cells (vascular)
- **Fib** = Fibroblasts (stromal)
- **LE** = Luminal Epithelial — **the prostate adenocarcinoma cell of origin**
- **Leu** = Leukocytes (intra-prostatic immune compartment)
- **SM** = Smooth Muscle (peri-prostatic stromal)

For post-treatment monitoring trajectory (the clinical use case), separating LE from BE matters: prostate adenocarcinoma is a luminal-origin disease. A trajectory score that reads LE specifically — separated from BE, from intra-prostatic Leu, and from peri-prostatic stromal SM — is a higher-resolution trajectory than a single `prostate_epithelial` bulk-tissue tile.

This calibration VAL is what unblocks integration of ProstateRef into prostate-epic Phase C scoring.

---

## Atlas inventory + provenance

### EpiSCORE ProstateRef
- **Source paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: 10.1038/s41592-022-01412-7
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source matrix:** `ProstateRef.rda` → `mrefProstate.m` (163 Entrez Gene IDs × 6 cell types + weight)
  - SHA-256: `706a0910a6d284b268cd8d2bc6ffa131beadab9ee2d942a18178903e682a72a3`
- **Bridge methodology:** Same as VAL-094 BreastRef and VAL-111 HeartRef — Entrez Gene IDs broadcast to all 450K CpGs mapping to that gene via EpiSCORE's `probeInfo450k.lv`.
- **Bridged CpG matrix:** `episcore_prostateref_cpg_bridged.csv`
  - 2,603 unique 450K CpGs × 6 cell types
  - 159 of 163 source Entrez IDs covered (4 had no probeInfo450k mapping: 1880, 2252, 51510, 4695)
  - SHA-256: `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2`
- **License:** GPL-2 per EpiSCORE repository

### Atlas family classification
**Gene-promoter atlas family** (same as VAL-111 HeartRef, NOT tile-coverage-WGBS-derived). Per DISC-CARDIO-004 (atlas family matters at Stage 2), gene-promoter atlases are at risk of `O3_TISSUE_FLOOR_DOMINATED` outcome on heterogeneous β panels — A-scores cluster at gene-promoter average methylation (~0.5) regardless of substrate, with low within-cohort tissue discrimination (max range typically <0.02 vs the 0.10 threshold). **VAL-117 is also a smoke test for whether ProstateRef family-fits prostate substrates differently than HeartRef family-fit cardiac substrates.**

If smoke test produces healthy-floor A-scores all clustering 0.46-0.51 (HeartRef pattern), seal at `O3_TISSUE_FLOOR_DOMINATED`, atlas → atlases_deferred for next prostate-epic version with explicit unblock dependency. Otherwise, seal CHK-3.1B q5 threshold + per-tile healthy-floor distributions for use in Phase C.

### CHK-3.1C dedup pre-check
ProstateRef bridged matrix has 2,603 rows. If duplicate probeIDs exist, CHK-3.1C fires. Pre-execution check during val117 script run.

---

## Calibration cohort

- **TCGA-KIRC adjacent-normal:** n=160, sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/KIRC/`
- **TCGA-PRAD adjacent-normal:** n=50, sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/PRAD/`
- **Combined n:** 210 (same as VAL-106/107 cardio calibration anchor)
- **Substrate:** TCGA HM450K sesame Level 3 (canonical substrate baseline, f_extreme 55.87% ± 2.44% per VAL-106 sealed)

**Structural separation note (CCL-041).** TCGA-PRAD adjacent-normal is histologically normal tissue from prostate cancer patients. It is NOT population-normal prostate. For a calibration cohort intended to set healthy-floor thresholds on a prostate-specific atlas, this could be a confounder: adjacent-normal tissue may carry early-stage methylation drift not yet visible histologically. Documented as a v0.4+ next-validation-step: replicate calibration on a population-normal prostate methylation cohort if/when surfaced. For v0.3 ship, TCGA adjacent-normal is the substrate-matched control we have — same as cardio used.

---

## Pre-locked outcomes

Per CHK-2.1 (all outcomes pre-locked, none added post-hoc):

### O1 — `PROSTATEREF_CALIBRATION_SEALED`

CHK-3.1A passes on ≥190/210 samples (≥90%); CHK-3.1B passes on ≥200/210 samples (≥95%, since atlas subset is small ~2,603 CpGs); CHK-3.1C passes (no duplicate probeIDs in bridged matrix); per-tile healthy-floor A-score distributions (mean, sd, n, q2.5, q5, q50, q95, q97.5) sealed for all 6 tiles (BE, EC, Fib, LE, Leu, SM); CHK-3.1B q5 threshold sealed.

**Outcome:** ProstateRef enters prostate-epic v0.3 atlases_run with calibration anchor VAL-117.

### O2 — `PROSTATEREF_CALIBRATION_PARTIAL`

CHK-3.1A or CHK-3.1B passes on 75-90%/85-95% of samples (substrate edge case). Calibration sealed but flagged as partial; v0.3 atlases_run inclusion deferred pending platform-specific re-calibration on a second healthy cohort.

### O3 — `PROSTATEREF_TISSUE_FLOOR_DOMINATED`

Per-tile healthy-floor A-scores all cluster within 0.46-0.51 across the calibration cohort with within-cohort tissue discrimination max < 0.02 (analog to VAL-111 HeartRef pattern). Seals as gene-promoter-atlas-family floor finding. Atlas → atlases_deferred for prostate-epic next version with explicit unblock dependency. Logged as DISC-PROSTATE-NNN finding propagating to LESSONS_LEARNED.md per CCL-043 + DISC-CARDIO-004 generalization.

### O4 — `PROSTATEREF_BRIDGE_FAILURE`

CHK-3.1A or CHK-3.1B fails on >25% of samples; or CHK-3.1C dedup fails (duplicate probeIDs in bridged matrix); or smoke test produces all-NaN tiles or all-zero CpG-intersection failures. Bridge engineering bug — defer atlas, log DISC-PROSTATE finding, propagate to LESSONS_LEARNED.md.

### O5 — `PROSTATEREF_UNEXPECTED`

Anything else not anticipated in O1-O4. Per CCL-032 (data integrity → biology → framework), classify as O5 if data integrity is uncertain or result contradicts expected gene-promoter-atlas behavior. Convene with Heath before sealing direction.

---

## Pre-locked thresholds (CHK-2.1)

Per VAL-106/107/112/113 cardio precedent on the same calibration cohort:

| Threshold | Pre-locked value | Source |
|---|---|---|
| CHK-3.1A f_extreme baseline | ≥ 50.0% | VAL-106 sealed at 55.87% ± 2.44% |
| CHK-3.1A f_middle ceiling | ≤ 12.0% | VAL-106 sealed at 7.42% ± 0.75% |
| CHK-3.1A pass rate | ≥ 190/210 (≥90%) | Same envelope as VAL-106 |
| CHK-3.1B atlas-subset coverage | ≥ 95% per sample | Cardio precedent |
| CHK-3.1C dedup | 0 duplicate probeIDs in bridged matrix | Hard gate |
| Tissue-floor-dominated threshold | within-cohort tile range < 0.02 | VAL-111 precedent (HeartRef cluster 0.46-0.51) |

---

## Reproducibility triple (CHK-7.6)

### Source code
`val117_prostateref_calibrate.py` — Python 3 stdlib + numpy. Loads bridged ProstateRef CSV, loads TCGA β files, computes per-sample CHK-3.1A on full genome + per-sample CHK-3.1B on atlas subset + per-tile A-scores using H_min(secretory) = 0.843264 for BE/LE (epithelial), H_min(stromal) = 0.862950 for Fib/SM, H_min(immune) = 0.838889 for Leu, H_min(stromal) = 0.862950 for EC.

### Inputs
1. `episcore_prostateref_cpg_bridged.csv` — 2,603 CpGs × 6 cell types + weight; SHA-256 `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2`; 145 KB
2. TCGA-KIRC + TCGA-PRAD adjacent-normal sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/`; 210 files; ~5 MB each

### Environment
- Python 3.12
- numpy 2.4.4
- No pandas dependency (csv stdlib only)
- Expected runtime: ~30 seconds for n=210 cohort
- Expected memory: ~500 MB peak (210 β vectors of ~485k CpGs each held simultaneously)

### Expected headline output
- `VAL-117_calibration_results.json` — per-tile mean, sd, n, q2.5, q5, q50, q95, q97.5
- `VAL-117_per_sample_calibration.csv` — per-sample CHK-3.1A f_extreme, f_middle, n_subset_valid, per-tile A-scores
- `VAL-117_outcome.md` — sealed outcome class

---

## RNG seed

20260420 (cookbook standard).

---

## SHA-256 of this prereg

To be computed at seal time and recorded in VAL-117_PREREG_SEAL.txt before val117 script reads any β files.

---

## Pre-registered audit chain

This prereg seals against the v0.5 TODO Phase B requirement and Guardrail #11 (calibration before testing is the inviolable order). val117 script execution begins ONLY after this prereg.md is sealed and SHA-hashed. Outcome sealed against pre-locked thresholds above.

**No outcome may be added post-hoc. No threshold may be relaxed post-hoc. No exception under CCL-041.**

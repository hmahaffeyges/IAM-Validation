# VAL-12X (provisional ID) — Phase B Calibration of BoccellatoStomachRef v1 on VAL-106 Cohort

**Prereg version:** v1.0-DRAFT (awaiting Heath sign-off before seal)
**Date drafted:** 2026-05-02
**Card:** gastric+esophageal-epic v0.1 sprint, Phase B
**Prereg type:** Type 2 calibration (per-tile healthy-floor distributions, frozen artifact for run-everything)
**SUPERSEDES:** none — first VAL on this atlas

---

## 1. VAL identification

- Provisional VAL ID: **VAL-12X** (final ID assigned at GitHub seal time, expected next sequential after the most recent VAL in `Biological_Physics/validation_runs/`)
- Atlas under calibration: **BoccellatoStomachRef_HM450 v1** (`boccellato_stomachref_HM450_v1.csv`, SHA-256 `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`, 380,467 CpGs after HM450-platform restriction). The original EPIC-built atlas (`boccellato_stomachref_v1.csv`, SHA `fbe1dbfdec...`) is retained in atlas_vault for provenance; the HM450-restricted derivative is the operational atlas for TCGA HM450 sesame Level 3 calibration. See BoccellatoStomachRef README amendment section for the EPIC→HM450 restriction methodology and CHK-2.17 history.
- Cohort: TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (the standing VAL-106 healthy substrate)
- Substrate: TCGA HM450 sesame Level 3 (the calibrated EDEAR substrate per CCL-048)

## 2. Hypothesis (pre-locked)

**Primary hypothesis:** When BoccellatoStomachRef v1 is applied to non-stomach healthy adjacent-normal tissue (kidney, prostate), all 6 gastric tiles (Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff) read at similar baseline A-scores reflecting "non-stomach" methylation distance. Per-tile A-score distributions are sealed as the healthy-floor reference for future patient scoring.

**Operational floor convention:** Per-tile q5 of the A-score distribution becomes the **healthy-floor threshold** below which a patient's tile A-score is flagged as anomalously gastric-similar (i.e. potential gastric tissue contribution to ccfDNA or paired-tissue readout).

**Type 2 calibration artifact (per atlas calibration typology doc):** the resulting per-tile thresholds (mean, sd, q5, q95) are sealed in the card JSON's `chk_3_1_thresholds_per_substrate.boccellato_stomachref_HM450_v1` block and loaded at run-everything scoring time. EDEAR's official application loads these once and never re-calibrates.

## 3. Pre-locked decision criteria (CHK-2.1)

Outcomes use magnitude-based thresholds with direction labels per CHK-2.7. The "comparison type" here is **diseased-tissue-vs-healthy-cross-reference** for atlas-application context but the calibration cohort itself is healthy-only (no disease signal expected); the relevant question is per-tile separation across the cohort.

### O1_TILES_DIFFERENTIATING_HEALTHY_FLOORS_SEALED (target)
- All 6 BoccellatoStomachRef tiles produce non-degenerate A-score distributions on the n=210 healthy cohort: mean SD per tile ≥ 0.005 (i.e. NOT all samples reading identical A-scores)
- Maximum within-cohort tile-range across the 6 tiles ≥ 0.02 (some discrimination present even on non-stomach tissue, expected from gastric atlas's secondary signature on systemic methylation drift)
- All 6 q5 thresholds sealed in card JSON

### O2_PARTIAL_FLOORS
- 4-5 tiles produce non-degenerate distributions; 1-2 tiles collapse to single-value (likely indicating those region/state combinations have insufficient distinct CpGs surviving the EPIC-to-HM450 panel intersection)
- Per-tile q5 still seals where calculable; collapsed tiles flagged

### O3_TISSUE_FLOOR_DOMINATED
- All 6 tiles read identically at the healthy-tissue substrate floor (within-cohort tile-range < 0.02 across all tiles); the atlas does not discriminate at all on non-stomach tissue
- This is the cardio-LL-005 HeartRef precedent — gene-promoter-style atlases sometimes collapse to substrate-floor on non-target tissue
- Per-tile q5 thresholds still sealed; cookbook-wide implication is that BoccellatoStomachRef is a stomach-tissue-only atlas (not a systemic methylation drift detector)
- This is acceptable v0.1 outcome — just sets expectations

### O4_BRIDGE_FAILURE
- < 80% of CpGs in the atlas produce valid A-scores due to platform mismatch / CpG coverage failure / data-integrity failure
- Routes to CCL-040 deferral pathway or CHK-3.1B repair pathway

### O5_UNEXPECTED
- Anything else, e.g. all tiles read NEGATIVE on healthy substrate (calibration artifact suggesting atlas requires β-shift normalization)
- O6 if data integrity flagged at CHK-3.1A or CHK-3.1B per CCL-041

**No bidirectional cancellation outcome applies** here per CCL-031 — calibration on healthy cohort produces a baseline distribution, not a case-vs-control direction test.

## 4. Pre-locked stratifications

- **Cohort split sealed:** within-VAL-106, kidney (TCGA-KIRC adjacent-normal, n=119) vs prostate (TCGA-PRAD adjacent-normal, n=91) per-tile A-score distributions reported separately as a CHK-3.2 sub-check. Difference between tissue means in anchor-SD units flagged if >1 SD.
- **Sex stratification:** TCGA-KIRC + TCGA-PRAD ratio is heavily male-skewed (PRAD all-male; KIRC ~70% male). Sex stratification declared but underpowered for female arm; documented in outcome.

## 5. CHK-3.1A and CHK-3.1B substrate gates (pre-locked)

- **CHK-3.1A:** TCGA HM450 sesame Level 3 substrate threshold per VAL-106: f_extreme ≥ 50.5%, f_middle ≤ 9.0% (full-genome). Per-sample gate: PASS if both thresholds met. Sample is excluded if either fails.
- **CHK-3.1B per-sample atlas-CpG-intersection coverage threshold:** ≥ 80% (per substrate floor for TCGA HM450K sesame Level 3, per CHK-2.8 / VAL-117 amendment precedent). Sample is excluded if BoccellatoStomachRef CpG intersection coverage falls below 80%.
- **CHK-3.1C:** PASS — atlas has zero duplicate CpG IDs (verified at build).

**Pre-flight CHK-2.17 cohort-substrate-coverage check (BLOCKING for prereg seal):** before sealing this prereg, sample 5-10 random TCGA-KIRC + TCGA-PRAD adjacent-normal β files, compute per-sample BoccellatoStomachRef CpG intersection coverage. If mean coverage <90% OR q5 < 80%, flag as bridge-failure risk and re-evaluate atlas suitability before sealing.

**CHK-2.17 pre-flight execution history (2026-05-02):**

*First run* (against EPIC-version atlas `boccellato_stomachref_v1.csv` SHA `fbe1dbfdec...`, 738,115 CpGs): **GATE FAIL.** Mean coverage 49.26%, min 48.77%. Root cause: EPIC 850K vs HM450 platform mismatch — only 380,467 of 738,115 atlas CpGs (51.55%) exist on the HM450 platform. Identical to VAL-117 ProstateRef amendment precedent. Atlas was rebuilt as HM450-restricted derivative.

*Second run* (against HM450-restricted atlas `boccellato_stomachref_HM450_v1.csv` SHA `f5a620a93a...`, 380,467 CpGs): **GATE PASS.** Mean valid-β coverage 95.56% (target ≥90%), min 94.62% (target ≥80%), max 96.17%. Probe-level coverage 100% on all 5 samples (every CpG in the restricted atlas is on HM450 by construction). Within-atlas bimodality: f_extreme 53-57%, f_middle 7-8% — consistent with TCGA HM450 sesame Level 3 platform threshold (extreme ≥50.5%, middle ≤9%).

Sample list (preflight_samples.json, RNG seed 20260502): TCGA-KIRC TCGA-BP-5183-11A, TCGA-KIRC TCGA-CZ-5451-11A, TCGA-KIRC TCGA-CZ-5468-11A, TCGA-KIRC TCGA-CJ-4903-11A, TCGA-PRAD TCGA-G9-6353-11A. All 5 SHA-256 hashes verified against VAL-106 cohort manifest.

Pre-flight artifacts: `CHK-2.17_PREFLIGHT_RESULTS.json` (first run, FAIL) + `CHK-2.17_PREFLIGHT_RESULTS_HM450.json` (second run, PASS).

## 6. Methodology

### Scoring procedure (per-sample)
For each TCGA-KIRC / TCGA-PRAD adjacent-normal sample (n=210):
1. Load sample β-vector (HM450 sesame Level 3, 485,512 CpGs).
2. CHK-3.1A full-genome bimodality check — fail-fast if substrate gate fails.
3. Intersect sample CpGs with BoccellatoStomachRef_HM450 380,467 CpGs → typically ~360-366K CpG overlap (~95% mean valid-β coverage on TCGA HM450 sesame Level 3, verified 2026-05-02 in CHK-2.17_PREFLIGHT_RESULTS_HM450.json on 5 random VAL-106 samples).
4. CHK-3.1B per-sample coverage gate (≥80% of atlas CpGs available in this sample's measured CpGs).
5. For each of 6 tiles, compute A-score = `mean( |sample_β - tile_ref_β| )` over the intersection CpGs.
6. Output per-sample 6-tile A-score vector.

### Sealed numbers per tile (computed at execution)
- mean ± sd
- q5, q25, median, q75, q95
- Per-tissue (KIRC vs PRAD) means

### Atlas-family-fitness summary
- Within-cohort tile-range distribution on healthy substrate (median, max, 95th pctile)
- Most-discriminating tile pair on healthy substrate (largest |Δmean|)

## 7. CHK-3.2 cross-cohort baseline check

This calibration VAL is the **anchor cohort** for BoccellatoStomachRef. Subsequent VALs (VAL-12X+2 STAD, VAL-12X+3 ESCA, etc.) will perform CHK-3.2 against the per-tile means and SDs sealed here. Mismatches >1 anchor-SD flagged; >3 anchor-SDs invalidate cross-cohort statistic.

## 8. Run-everything mandate compliance

Per Heath sign-off 2026-04-26 + calibration typology doc: this VAL produces a Type 2 frozen artifact. EDEAR's official application loads the per-tile thresholds at startup and applies them to every patient IDAT without re-calibrating. CHK-3.2 is mandatory at every subsequent run. Multi-disease detection patterns surfaced via run-everything depend on this baseline being platform-correct + preprocessing-correct.

## 9. CHK-7.6 reproducibility triple

- **Source code:** `val12X_boccellato_calibration.py` (to be authored; standard NumPy + pandas + scipy.stats; ~120 lines)
- **Inputs:**
  - `boccellato_stomachref_HM450_v1.csv` (SHA `f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390`, 380,467 CpGs, HM450-restricted derivative of `boccellato_stomachref_v1.csv` SHA `fbe1dbfdec...`)
  - VAL-106 cohort sample files (TCGA-KIRC + TCGA-PRAD adjacent-normal β matrices, paths inherited from VAL-106 manifest)
- **Environment:** Python 3.x, NumPy, pandas, scipy.stats. Standard library otherwise. ~5-15 min runtime, ~3 GB memory peak.
- **Expected output:** `val12X_boccellato_calibration_results.json` containing per-tile mean/SD/quantile distributions + per-tissue stratification + atlas-family-fitness diagnostics.

## 10. Test 2 placeholder

Per CCL-030/CHK-2.5: Test 2 (lymphoid vs myeloid sub-panel test) is BLOCKED on OQ-2026-01 immune-atlas staging. This calibration VAL does not test bidirectional cancellation; the placeholder is documented but no Test 2 claim arises here.

## 11. Failure recovery paths

- **CHK-3.1A failure**: route to CCL-040 deferral; do not score this VAL until raw IDAT reprocessing completes.
- **CHK-3.1B failure (coverage <80%)**: re-evaluate the EPIC-to-HM450 bridge; consider rebuilding atlas with HM450-restricted CpG subset.
- **All-tiles-collapse outcome (O3)**: card v0.1 still ships; atlas is stomach-tissue-only detector; CCL-039 documents the gene-promoter-vs-purified-cell-type discriminating-power difference.

---

## Awaiting Heath sign-off

Before sealing this prereg with PREREG_SEAL.txt + git commit:
- ✅ Atlas built and SHA-256 sealed (HM450-restricted v1: `f5a620a93a...`, EPIC source: `fbe1dbfdec...`)
- ✅ Pre-flight CHK-2.17 cohort-substrate-coverage check on TCGA-KIRC+PRAD adjacent-normal — FIRST RUN (EPIC atlas) FAIL at 49.26%, atlas restricted to HM450 platform, SECOND RUN (HM450-restricted atlas) PASS at 95.56% mean / 94.62% min coverage
- ⏳ Heath sign-off on outcome thresholds (O1-O5 above)
- ⏳ Heath sign-off on cohort split (KIRC vs PRAD as separate strata, sex stratification declared underpowered)
- ⏳ Heath sign-off on the EPIC-to-HM450 atlas restriction approach (precedent: VAL-117 ProstateRef amendment)
- ⏳ Final VAL ID assignment from sequential numbering after most-recent GitHub VAL

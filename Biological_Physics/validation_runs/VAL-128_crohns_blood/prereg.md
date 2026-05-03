# VAL-128 — Phase C Run-Everything on GSE87650 Crohn's Disease Blood Methylation

**Sprint:** gastric+esophageal-epic v0.1 sprint, Phase C (disease cohort scoring)
**Card target:** gastric+esophageal-epic v0.1 (Crohn's pathway language amendment)
**VAL ID:** VAL-128
**Cohort:** GSE87650 GPL13534 (HM450) sorted-cell sub-experiment — n=240 samples (60 monocytes + 59 CD4 + 56 CD8 + 65 whole-blood-companion). The 384 whole-blood samples from the Ventham main cohort are stored in a separate supplementary file (`GSE87650_processedMethCombinedWb.txt.gz`, 1.4 GB compressed) NOT in the GPL13534 series matrix; queued for v0.2 expansion.
**Source paper:** Ventham NT et al. "Integrative Epigenome-Wide Analysis Shows That DNA Methylation May Mediate Genetic Risk In Inflammatory Bowel Disease." Nat Commun 2016;7:13507. PMID 27886173.
**Substrate:** Illumina HumanMethylation450 (GPL13534)
**RNG seed:** 20260502
**Prereg version:** v1.0

**SEALED BEFORE β DATA OBSERVED.** CCL-041 compliance.

---

## 1. Hypothesis (pre-locked, BIDIRECTIONAL per CHK-2.7 + Heath sign-off)

This is the **subordinate exploratory** Crohn's-pathway VAL of the gastric+esophageal-epic v0.1 sprint, motivated by Heath's stepbrother Marcus's history (Crohn's → ileostomy in both Marcus and his father Jim, prior to Marcus's HCC). The Crohn's pathway language amendment to gastric+esophageal-epic v0.1 known_limitations and to hcc-epic v0.3.1 known_limitations depends on this VAL's findings.

**Stage 1 hypothesis (Xu-538 architectural drift in Crohn's blood):**
- **CD vs healthy:** modest |d| ≥ 0.2 expected in whole blood (chronic inflammation produces measurable architectural drift signature, but blood substrate has lower baseline drift than solid tumor). Direction expected POSITIVE per cycling-class precedent for chronic-inflammation-driven cell-turnover acceleration.
- **UC vs healthy:** similar magnitude to CD; UC vs CD direct comparison not strongly directional pre-locked.

**Stage 2 hypothesis (cell-of-origin tile pattern in Crohn's blood):**
- **BoccellatoStomachRef gastric tiles** read NULL in blood (gastric tiles orthogonal to blood substrate). If they read non-NULL, that's a CCL-class observation about gut-blood epigenetic communication.
- **EsoRef Epi_* + OEref Basal** read NULL in blood (squamous-tile orthogonality to blood substrate).
- **Loyfer Upper_GI** reads NULL in blood (cell-of-origin atlas applied to non-target tissue).
- **Loyfer immune tiles (B-cells, CD4T, CD8T, NK, Mono, Neu)** read meaningful structure (these ARE the cell-of-origin atlases for blood substrate).

**Stage 3 hypothesis (immune sub-composition in CD/UC blood):**
- **Salas IDOL 6-cell:** CD and UC patients expected to show shifted immune cell composition vs healthy — increased neutrophil signature, B-cell/CD4 proportional shifts. Magnitude direction not pre-locked.
- **UniLIFE 19-cell fine-grained:** activated CD4 vs naive, regulatory T (aTreg), monocyte subsets — expect IBD-pattern shifts.

**Cell-type-stratified hypothesis (key pre-lock):**
- **Sorted CD4 + CD8 + Monocytes** datasets are the cleanest substrate; expect tightest disease-vs-healthy d.
- **Whole blood** (mixture) shows attenuated d due to mixture-effect dilution.
- **Pre-locked test:** sorted-cell d should be ≥ 1.5x whole-blood d.

**Bidirectionality:** all outcome thresholds use magnitude-based |d| with explicit direction labels per CHK-2.7.

---

## 2. Pre-locked decision criteria (CHK-2.1 + CHK-2.7 + CHK-4.11)

### Stage 1 outcomes (per cell-type stratum)
- **O1_STAGE1_PASS_POSITIVE/NEGATIVE**: |d_unpaired| ≥ 0.5 with explicit direction
- **O2_STAGE1_PARTIAL**: 0.2 ≤ |d_unpaired| < 0.5
- **O3_STAGE1_NULL**: |d_unpaired| < 0.2

### Stage 2 outcomes (per atlas, per tile)
- `{tile}_DIFFERENTIATING_POSITIVE/NEGATIVE`: |d| ≥ 0.5 with direction
- `{tile}_PARTIAL`: 0.2 ≤ |d| < 0.5
- `{tile}_NULL`: |d| < 0.2

### Crohn's-pathway language outcome (PRIMARY scientific test)
- **O1_CROHNS_LANGUAGE_SUPPORTED**: CD vs healthy produces |d| ≥ 0.5 on Stage 1 OR on ≥1 Stage 3 immune tile in interpretable direction. Card v0.1 known_limitations adds a Crohn's-pathway-detectable note.
- **O2_CROHNS_LANGUAGE_PARTIAL**: 0.2 ≤ |d| < 0.5 on Stage 1 + Stage 3 — language hedged ("subclinical signature, exploratory")
- **O3_NO_CROHNS_SIGNATURE**: |d| < 0.2 — language not added; documented null

### CD vs UC discrimination outcome (secondary)
- **O1_CD_UC_DISCRIMINATION**: |d_CD−UC| ≥ 0.4 on at least 1 atlas/tile (suggests CD and UC produce distinguishable methylation signatures)
- **O2_NO_DISCRIMINATION**: |d_CD−UC| < 0.2 on all atlases (CD and UC indistinguishable from blood methylation, expected for v0.1)

---

## 3. Pre-locked stratifications (CHK-2.2) — ALL DATA AVAILABLE FROM GEO

### Cell type × diagnosis (sorted-cell sub-experiment, n=240)

| Cell type | CD | UC | HC | Total |
|-----------|---:|---:|---:|------:|
| wh blood (sorted-companion control) | 19 | 21 | 25 | 65 |
| Monocytes | 20 | 20 | 20 | 60 |
| CD4 | 20 | 19 | 20 | 59 |
| CD8 | 18 | 19 | 19 | 56 |
| **Total** | 77 | 79 | 84 | 240 |

Note: the Ventham main whole-blood cohort (n=384, with CD=103, UC=101, HL=105, HS=75) is queued for v0.2 expansion — its β data lives in `GSE87650_processedMethCombinedWb.txt.gz` (separate supplementary file, 1.4 GB compressed), not in the standard series matrix.

### Sex
M: 352, F: 272.

### Smoking status (where reported, n=381 of 624)
Never: 171, Ex: 107, Current: 99, Don't know: 4.

### Age
Continuous, age at sample.

---

## 4. Run-everything atlas stack

Same 8 calibrated atlases as VAL-126/127:

| Stage | Atlas | Tiles |
|-------|-------|------:|
| 1 | Xu-538 panel | 1 |
| 2 | Layered Moss+Loyfer 25-tile | 25 |
| 2 | BoccellatoStomachRef_HM450 | 6 |
| 2 | EpiSCORE EsoRef bridged | 8 |
| 2 | EpiSCORE OEref bridged | 9 |
| 2 | Caggiano CelFiE TIM | 19 |
| 3 | Salas IDOL Blood.EPIC (450K variant) | 6 |
| 3 | UniLIFE Guo 2025 | 19 |

Total 1+67+25 = 93 A-scores per IDAT. Run-everything mandate per Heath sign-off.

---

## 5. CHK-3.1A / CHK-3.1B / CHK-3.1C (per VAL-126/VAL-127 precedent)

- **CHK-3.1A:** Documented, not gated on disease/cross-substrate cohort. GSE87650 is published HM450 data preprocessed by source authors (Ventham et al. 2016); substrate baseline expected to differ from TCGA-KIRC+PRAD anchor due to (a) blood vs tissue substrate, (b) different preprocessing pipeline (Ventham used minfi+SWAN; TCGA uses sesame).
- **CHK-3.1B:** Per-sample atlas-CpG-coverage ≥ 0.80; pass rate ≥ 95% per atlas.
- **CHK-3.1C:** All 8 atlases verified PASS at calibration time.

---

## 6. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING)

Sample 5 random GSE87650 β columns (1 per cell-type × dx if possible). Verify per-sample coverage ≥80% on each calibrated atlas. **Pre-flight executes BEFORE prereg seal goes hot.**

---

## 7. Comparison strategy

- **Primary statistic:** Welch unpaired d on Crohn's (CD) vs healthy (HL+HC+HS pooled) per cell-type stratum
- **Secondary:** UC vs healthy per cell-type stratum
- **Tertiary:** CD vs UC per cell-type stratum (subtype discrimination)
- **Exploratory:** sorted-cell d vs whole-blood d (mixture-attenuation test)

---

## 8. CHK-3.2 cross-cohort baseline check

For every atlas/tile: compare GSE87650 healthy mean A-score vs VAL-106 anchor mean. Document baseline shift (blood substrate vs tissue substrate; Ventham preprocessing vs sesame). Not gated.

---

## 9. CHK-7.6 reproducibility triple

- **Source code:** `val128_crohns_blood.py` (mirrors VAL-126/127 scorer with GSE87650 series-matrix β reader)
- **Inputs:**
  - GSE87650-GPL13534_series_matrix.txt.gz (909 MB compressed, 2.3 GB decompressed)
  - 8 calibrated atlases
  - VAL-123/124/125 calibration_results.json
- **Environment:** Python 3.x, NumPy, scipy.stats.
- **Expected output:** `VAL-128_results.json`, `VAL-128_per_sample.csv`, `outcome.md`

---

## 10. Specimen pathway compliance (CHK-2.4)

Specimen: peripheral blood (whole blood + sorted CD4/CD8/monocytes). This is BLOOD substrate not tissue. Xu-538 panel was originally validated on Sister Study buffy-coat blood per Xu 2020 — direct substrate match.

---

## 11. Test 2 placeholder

CCL-030 / CHK-2.5: Test 2 (lymphoid vs myeloid sub-panel) BLOCKED on OQ-2026-01.

---

## 12. Marcus-pathway language note (NOT scope of this VAL)

VAL-128 outcome will inform the Crohn's-pathway language amendment to:
- gastric+esophageal-epic v0.1 known_limitations: "Crohn's-pathway methylation signature: sub-clinical/clinical detection capability per VAL-128 result."
- hcc-epic v0.3.1 amendment: same language for the Crohn's→HCC pathway concern raised by Marcus's case.

The amendment language depends entirely on whether VAL-128 fires O1, O2, or O3.

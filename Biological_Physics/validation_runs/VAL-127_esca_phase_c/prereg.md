# VAL-127 — Phase C Run-Everything on TCGA-ESCA Esophageal Carcinoma

**Sprint:** gastric+esophageal-epic v0.1 sprint, Phase C (disease cohort scoring)
**Card target:** gastric+esophageal-epic v0.1
**VAL ID:** VAL-127
**Cohort:** TCGA-ESCA HM450 — n=185 primary tumor + n=16 paired adjacent-normal + n=1 metastatic
**Substrate:** TCGA HM450K sesame Level 3
**RNG seed:** 20260502
**Prereg version:** v1.0

**SEALED BEFORE β DATA OBSERVED.** CCL-041 compliance.

---

## 1. Hypothesis (pre-locked, BIDIRECTIONAL per CHK-2.7)

**Headline test:** ESCC (squamous cell carcinoma, n=90) vs EAC (esophageal adenocarcinoma per ESCA_CIN, n=74) cell-of-origin tile-pattern divergence. This is the first-of-kind multi-atlas subtype-discrimination test enabled by the gastric+esophageal-epic v0.1 sprint's atlas stack (BoccellatoStomachRef columnar-lineage + EsoRef squamous-stratified).

**Stage 1 hypothesis:** Both ESCC and EAC show |d_unpaired| ≥ 0.5 vs control on Xu-538 architectural drift; magnitude POSITIVE.

**Stage 2 ESCC subtype-specific hypothesis:**
- **EsoRef Epi_basal/stratified/suprabasal/upper** read NEGATIVE direction in ESCC tumor (squamous cell-of-origin de-differentiation per VAL-118 LE-flip pattern)
- **OEref Basal** reads NEGATIVE in ESCC (oral squamous reference cross-validates ESCC squamous lineage)
- **BoccellatoStomachRef tiles** read NULL or POSITIVE on ESCC (columnar reference orthogonal to squamous tumor)
- **Loyfer Head_and_neck_larynx** reads NEGATIVE on ESCC (squamous mucosa cell-of-origin tile)

**Stage 2 EAC subtype-specific hypothesis (Barrett's columnar):**
- **BoccellatoStomachRef tiles (Antrum/Corpus/Fundus × undiff/diff)** read POSITIVE direction in EAC tumor (Barrett's metaplasia = columnar-lineage shift; gastric reference is the closest available proxy)
- **EsoRef Epi_* tiles** read NULL or POSITIVE on EAC (squamous tiles orthogonal to columnar tumor; this MIRRORS the cross-tissue-overread test in VAL-126 STAD)
- **Loyfer Upper_GI** reads structure on EAC (mixed bulk-tissue reference)

**Subtype-discrimination claim:** ESCC and EAC produce DIFFERENT cell-of-origin tile-pattern signatures. ESCC = squamous-tile pattern (EsoRef negative, OEref negative). EAC = columnar-tile pattern (Boccellato pattern). Quantified by per-tile d-difference between subtypes.

**Stage 3 immune hypothesis:** Both subtypes show immune signature; ESCC expected stronger neutrophil-dominated TIL pattern, EAC expected stronger B-cell/plasma cell pattern (Barrett's-driven chronic inflammation). BIDIRECTIONAL.

---

## 2. Pre-locked decision criteria (CHK-2.1 + CHK-2.7)

Same outcome class structure as VAL-126:
- **O1_DIFFERENTIATING_POSITIVE/NEGATIVE** if |d| ≥ 0.5 with explicit direction label
- **O2_PARTIAL** if 0.2 ≤ |d| < 0.5
- **O3_NULL** if |d| < 0.2

**Subtype-discrimination outcome:** O1_SUBTYPE_DISCRIMINATION_POSITIVE fires if ANY tile produces |d_ESCC − d_EAC| ≥ 1.0 in the discriminating direction (i.e., the same tile reads opposite directions or one subtype's tile pattern is ≥1.0 magnitude larger than the other's).

---

## 3. Pre-locked stratifications (CHK-2.2) — ALL DATA PULLED

n=185 tumor:

### Subtype (cBioPortal SUBTYPE)
| SUBTYPE | n |
|---------|--:|
| ESCA_ESCC (Squamous Cell Carcinoma) | 90 |
| ESCA_CIN (Adenocarcinoma — Chromosomal Instability) | 74 |
| ESCA_MSI | 2 |
| ESCA_POLE | 2 |
| ESCA_GS | 1 |
| NotReported | 16 |

### Histology cross-validation (primary_diagnosis)
| Diagnosis | n |
|-----------|--:|
| Squamous cell carcinoma NOS | 87 |
| Adenocarcinoma NOS | 83 |
| Squamous cell carcinoma keratinizing | 5 |
| Other | 10 |

### Smoking history (TCGA exposures)
- tobacco_smoking_status: tracked, exploratory
- alcohol_history: tracked, exploratory  
- Both are major ESCC risk factors; small effect sizes expected at TCGA cohort granularity

### Demographics
- Sex (binarized male/female)
- Age (continuous)
- AJCC pathologic stage I/II/III/IV

### MSI score
- MSI_SCORE_MANTIS, MSI_SENSOR_SCORE — continuous from cBioPortal

---

## 4. Run-everything atlas stack

Same 8-atlas stack as VAL-126:
1. Xu-538 panel (Stage 1)
2. Layered Moss+Loyfer 25-tile (Stage 2)
3. BoccellatoStomachRef_HM450 6 tiles (Stage 2; columnar reference for EAC)
4. EpiSCORE EsoRef bridged 8 tiles (Stage 2; squamous reference for ESCC — primary)
5. EpiSCORE OEref bridged 9 tiles (Stage 2; oral squamous confirmatory for ESCC)
6. Caggiano TIM 19 tiles (Stage 2)
7. Salas IDOL 450K 6 cell types (Stage 3)
8. UniLIFE Guo 2025 19 cell types (Stage 3)

Total: 1 + 67 + 25 = 93 A-scores per IDAT.

---

## 5. CHK-3.1A / CHK-3.1B / CHK-3.1C (per VAL-118/VAL-126 precedent)

- **CHK-3.1A:** Documented, not gated on tumor/cross-substrate cohort. Hard fail only on single-tone (all-NaN, median β > 0.95).
- **CHK-3.1B:** Per-sample atlas-CpG-coverage ≥ 0.80, pass rate ≥ 95% per atlas.
- **CHK-3.1C:** All atlases verified PASS at calibration time.

---

## 6. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING)

7 random TCGA-ESCA HM450 β files (5 random tumor + first 2 normals from manifest). Verify per-sample coverage ≥80% on each calibrated atlas.

**Pre-flight executes BEFORE prereg seal goes hot.**

---

## 7. Comparison strategy

- **Primary statistic:** Welch unpaired d. **Stronger statistical power than STAD here** because TCGA-ESCA has 16 adjacent-normals (vs STAD's 2). Within-cohort comparison is now properly powered.
- Anchor-based d: where calibration anchor available (Boccellato VAL-123, EsoRef VAL-124, OEref VAL-125)
- Within-cohort d: tumor (n=185) vs ESCA adjacent-normal (n=16) — primary for atlases without VAL-106 anchor
- Subtype-stratified d: ESCC (n=90) vs paired/within-cohort, EAC (n=74) vs paired/within-cohort, then ESCC-vs-EAC tile-pattern divergence as headline test

---

## 8. Logged follow-ups

- Cross-tissue overread test (EsoRef on PRAD/KIRC tumor) remains queued for kidney-card
- ESCC-vs-EAC subtype discrimination is THE first-of-kind multi-atlas subtype-stratification claim of this sprint

---

## 9. CHK-7.6 reproducibility triple

- **Source code:** `val127_esca_phase_c.py` (parametrized clone of val126 scorer with esca cohort)
- **Inputs:** 202 TCGA-ESCA HM450 β files (~2.5 GB), 8 calibrated atlases, VAL-123/124/125 anchor distributions
- **Environment:** Python 3.x, NumPy, scipy.stats. ~10 min runtime.
- **Expected output:** `VAL-127_phase_c_results.json`, `outcome.md`

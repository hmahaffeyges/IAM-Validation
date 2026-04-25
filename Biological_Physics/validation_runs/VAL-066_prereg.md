# VAL-066 Pre-Registration — pancreatic-epic Tissue Arm

**Sealed:** 2026-04-25 UTC
**Card:** pancreatic-epic
**Card version target:** v0.1 (first version, accompanies card initial build)
**Cohort:** TCGA-PAAD HM450 matched tumor/adjacent-normal subset
**Platform:** Illumina Infinium HumanMethylation450 (HM450)
**Sample composition:** 10 patients with both Primary Tumor + Solid Tissue Normal samples on HM450, all pancreatic ductal adenocarcinoma (PDAC) histology, ages 49-83, mixed sex (6M/4F), mixed stage (mostly Stage IIB)

## Background

The pancreatic-epic card enters the Cookbook at `cohort_screening_validated` tier anchored by **VAL-046 Rotterdam pancreatic n=182 pre-diagnostic blood signal** (multi-class systemic drift cascade, April 2026). VAL-046 establishes Stage 1 cohort-level pre-diagnostic signal but not per-patient validation. Per the workflow established by hcc-epic v0.1, a card entering at `cohort_screening_validated` tier benefits from a supplementary tissue arm (when cohort is available) to confirm Stage 2 localization at the architectural-disruption level on the same Xu-538 panel. VAL-066 is that supplementary tissue arm for pancreatic-epic, on the only public TCGA-PAAD HM450 matched tumor/normal cohort available.

**Class assignment.** Pancreatic ductal adenocarcinoma cells (PDAC) arise from pancreatic_exocrine duct epithelium, which is in the **secretory class** alongside breast_ductal, prostate_epithelial, and hepatocyte. H_min(secretory) = 0.843264 (G-002 MCMC posterior, R-hat < 1.001). Healthy reference β = 0.745 (Moss 2018 healthy pancreatic_exocrine reference, anchored in pancreatic-epic v0.1 universal_reference block).

## Cohort

**TCGA-PAAD matched tumor/adjacent-normal pairs on HM450:** n = 10 patients, all with both Primary Tumor + Solid Tissue Normal β value files publicly accessible from NIH GDC (no dbGaP application required for Level 3 sesame β values). 

**Composition (clinical metadata pulled from GDC cases endpoint):**
- All 10 patients: pancreatic ductal adenocarcinoma (9 "Infiltrating duct carcinoma NOS" + 1 "Adenocarcinoma NOS")
- Sex: 6 male, 4 female
- Race: 8 white, 1 black, 1 asian
- Age range: 49-83 (mean ~66)
- Smoking status: 4 lifelong non-smoker, 1 reformed smoker (duration unspecified), 5 unknown
- Alcohol history: 1 yes, 1 no, 8 not populated (sparse TCGA-PAAD metadata)
- BMI: not populated for any of the 10 patients in the GDC exposures endpoint
- Stage: 1 Stage IIA, 5 Stage IIB, 1 Stage III, 1 Stage IV, 2 not populated

## Pre-registered hypotheses (sealed before β value access)

**Primary hypothesis H1 (POOLED):** PDAC tumor tissue Xu-538 immune-class A-score is elevated above adjacent-normal pancreatic tissue Xu-538 immune-class A-score, paired Cohen's d > 0.

Rationale: VAL-058 (prostate, secretory, n=238) paired d = +0.497; VAL-060 (breast, secretory, n=86) paired d = +0.675; VAL-064 non-viral HCC (secretory, n=34) paired d = +0.664. All three secretory-class tissue arms show positive direction at d ≈ +0.5 to +0.7. Pancreatic-exocrine is the same architectural class.

**Secondary hypothesis H2 (DIRECTION):** Per-CpG direction preservation rate (fraction of Xu-538 CpGs where 10-patient majority sign of (β_tumor − β_normal) is positive) > 50%, consistent with predominantly hypermethylation-direction signal seen in other secretory tumors.

**Outcome thresholds:**

### O1: TISSUE_VALIDATED
Paired d > +0.3 AND lower 95% CI > 0 AND H2 PASS. Card enters Cookbook at `cohort_screening_validated + tissue_arm_validated` modifier flag tier (analogous to hcc-epic v0.2).

### O2: TISSUE_VALIDATED_WEAKER
Paired d in [0, +0.3] with lower 95% CI > 0 (positive direction confirmed but small effect at n=10). Card enters at `cohort_screening_validated` tier without tissue_arm_validated modifier. VAL-066 documents the result as direction-confirmed-but-magnitude-modest pending larger cohort.

### O3: TISSUE_NULL
Paired d straddles zero (lower 95% CI ≤ 0). Card enters at `cohort_screening_validated` tier; tissue arm documented as inconclusive at n=10.

### O4: TISSUE_INVERTED
Paired d < 0 with upper 95% CI < 0 (negative direction confirmed). UNEXPECTED — no other secretory-class tissue arm has shown this. Convene with Heath before card direction. Flag for further investigation analogous to crc-epic CRC blood inversion (CCL-019 — direction depends on class+compartment, not just disease).

### O5: UNEXPECTED
Any pattern not matching O1-O4. Report numbers honestly; convene with Heath.

## Methods (frozen pre-data)

### Cohort
- Public TCGA-PAAD HM450 Level 3 sesame β value files from NIH GDC.
- 10 matched pairs, patient IDs frozen at prereg seal:
  - TCGA-FZ-5919, TCGA-FZ-5920, TCGA-FZ-5922, TCGA-FZ-5923, TCGA-FZ-5924,
  - TCGA-FZ-5926, TCGA-HZ-A9TJ, TCGA-IB-7651, TCGA-IB-7652, TCGA-YB-A89D

### Panel
- Xu-538 immune panel (canonical SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`, file-bytes verified at runtime).

### Scoring class
- Pancreatic_exocrine = secretory class.
- H_min(secretory) = 0.843264.
- Reference β = 0.745 (Moss 2018 hepatocyte/pancreatic_exocrine, secretory class universal reference).

### QC threshold
- Minimum 400,000 valid HM450 β values per sample (~82% probe coverage).
- Minimum 400 valid Xu-538 panel CpGs per sample (~74% panel coverage).

### RNG seed
- 20260425. Used for any tie-breaking in CI calculations; analysis is deterministic.

### Statistical procedure
- A-score per sample = mean over Xu-538 CpGs of `H(β)/H_min(secretory)`, where `H(β) = -β log2(β) - (1-β) log2(1-β)`.
- Paired Cohen's d: d = mean(ΔA_paired) / sd(ΔA_paired) where ΔA_paired = A_tumor − A_normal per patient.
- 95% CI on paired d: standard formula `d ± 1.96 × sqrt(1/n + d²/(2n))`.
- Hedges small-sample correction `1 - 3/(4(n-1) - 1)` reported alongside raw paired d.
- p-values reported but inference based on effect size and CI bounds, not p-thresholds (n=10 is small).

### Stratified analyses (planned, with explicit underpowered caveats)
- **Sex stratification:** 6M vs 4F. Underpowered but report direction.
- **Smoking stratification:** 4 lifelong non-smoker vs 1 reformed smoker (5 unknown excluded). Highly underpowered, report direction only.
- **Stage stratification:** Stage IIA+IIB combined (n=6) vs Stage III+IV (n=2). Highly underpowered, report direction only.
- **Race stratification:** 8 white vs 2 non-white. Highly underpowered.

### Mandatory covariate honest reporting
Per the Master README rule that "BMI for pancreatic" is a mandatory covariate, VAL-066 will explicitly report that **BMI is not populated for any of the 10 patients** in the GDC exposures endpoint. This is a documented data gap that pancreatic-epic v0.1 carries forward to v0.2+.

### What this run does not do
- No diabetes (T2D) stratification — TCGA-PAAD does not populate diabetes status reliably.
- No family history stratification — TCGA-PAAD family_histories endpoint is sparse for PDAC.
- No claims about pre-diagnostic detection — TCGA-PAAD samples are at-diagnosis tumor tissue.
- No claim that n=10 is sufficient for deployment-grade tissue arm — it is supplementary, exploratory tissue confirmation supporting the VAL-046 cohort-level Stage 1 anchor.
- No deconvolution of tumor cell fraction — TCGA β values are bulk tumor tissue including stromal admixture; this is the standard for all Cookbook tissue-arm runs (VAL-058, VAL-060, VAL-061, VAL-062, VAL-063, VAL-064).

## Reproducibility anchors

- Pre-registration SHA-256: (computed at seal)
- Cohort SHA: (computed from sorted patient ID list at seal)
- Xu-538 panel SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6 (file-bytes)
- β matrix SHA: (computed at run)
- Results SHA: (computed at output)

## Deliverables

1. `val066_pancreatic_epic_tcga_paad.py` — reproducible Python 3 stdlib script
2. `VAL-066_prereg.md` — this document, sealed
3. `VAL-066_outcome.md` — outcome doc with per-patient, primary, and stratified results
4. `VAL-066_results.json` — primary + stratified
5. `PAAD_matched_manifest.json` — sample-to-patient map
6. `PAAD_clinical.json` — clinical metadata used for stratification

GitHub destination: `Biological_Physics/validation_runs/`
Cookbook destination: `pancreatic-epic_README.md` v0.1 + `pancreatic-epic_card_v0.1.json` + `GAPE_Evidence_Report_UPDATED.html` + `README_MASTER_v2.1.md` + `LESSONS_LEARNED.md` (if any new CCL emerges)

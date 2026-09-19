# VAL-067 Pre-Registration — pancreatic-epic Large-Cohort Tissue Case-Control

**Sealed:** 2026-04-25 UTC
**Card:** pancreatic-epic
**Card version target:** v0.1 (joins VAL-066 TCGA-PAAD n=5 paired and VAL-068 GSE74071 multi-substrate as the three tissue-arm anchors for pancreatic-epic v0.1)
**Cohort:** GSE49149 (Mishra et al. / Wood lab) — 167 PDAC tumors + 29 adjacent non-tumor pancreatic tissue samples on Illumina HumanMethylation 450K
**Platform:** GPL13534 (HM450)
**Reference:** Pubmed 24500968 and 26909576

## Background

VAL-066 ran the TCGA-PAAD HM450 matched tumor/normal cohort and yielded n=5 effective matched pairs after QC (two patients failed QC, three patients dropped at amendment for missing solid-tissue-normal HM450 files). At n=5 the primary paired d = +1.18 [+0.04, +2.32] is positive but the lower 95% CI just barely excludes zero, and per-CpG direction preservation came in at 46.9% positive vs 53.1% negative — exactly the bidirectional pattern CCL-027 flags as cancellation risk. **VAL-067 expands the cohort dramatically: 196 samples on the same HM450 platform from a separate large PDAC methylation study.**

GSE49149 has unmatched-pair design (167 tumors + 29 adjacent normal, not patient-paired) — this trades patient-pairing for sample size. The unpaired Cohen's d on this sample size will be much more statistically informative than the n=5 paired d from VAL-066. Combined with VAL-068 (GSE74071 multi-substrate, n=28) and the prior VAL-046 Rotterdam pre-diagnostic blood anchor (n=182), the pancreatic-epic v0.1 card will rest on four independent cohorts spanning the cohort-screening, tissue, multi-substrate, and pre-diagnostic dimensions.

## Class assignment

Pancreatic ductal adenocarcinoma cells (PDAC) arise from pancreatic_exocrine duct epithelium = **secretory class**. H_min(secretory) = 0.843264. Reference β = 0.745.

## Pre-registered hypotheses (sealed before β value access)

**Primary H1:** PDAC tumor tissue Xu-538 immune-class A-score is elevated above adjacent non-tumor pancreatic tissue Xu-538 immune-class A-score, unpaired Cohen's d > +0.3 with lower 95% CI > 0.

**Secondary H2:** Per-CpG direction preservation rate (fraction of Xu-538 CpGs where mean (β_tumor − β_normal) is positive) > 50%, consistent with predominantly hypermethylation-direction signal in other secretory tumors.

**Tertiary H3 (CCL-027 mandatory check):** Document the per-CpG positive-vs-negative direction split explicitly. If the split deviates from the breast-epic / hcc-epic / prostate-epic pattern of >55% positive, log it as a bidirectional-cancellation flag for pancreatic-epic v0.x and propagate to Stage 1 design.

## Pre-registered outcome decision matrix

### O1: TISSUE_VALIDATED_LARGE_COHORT
H1 PASS (unpaired d > +0.3, lower CI > 0) AND H2 PASS (>50% positive direction). Card pancreatic-epic enters Cookbook with `cohort_screening_validated + tissue_arm_validated` modifier flag tier (analogous to hcc-epic v0.2).

### O2: TISSUE_VALIDATED_WEAKER
Unpaired d in [0, +0.3] with lower CI > 0. Direction confirmed at large n but magnitude is modest.

### O3: TISSUE_NULL_LARGE_COHORT
Unpaired d straddles zero. PDAC does not produce a uniform Xu-538 immune-class architectural signal even at n=196.

### O4: TISSUE_INVERTED_LARGE_COHORT
Unpaired d < 0 with upper CI < 0. UNEXPECTED — would parallel CRC peripheral immune inversion (CCL-019). Convene before card direction.

### O5: UNEXPECTED
Any pattern not matching O1-O4. For example, large pooled magnitude but per-CpG split inverted (the VAL-066 pattern). Convene with Heath.

## Methods

- Public GSE49149 series matrix from NCBI GEO public access.
- Xu-538 immune panel (canonical SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`, file-bytes verified at runtime).
- Pancreatic_exocrine = secretory class; H_min(secretory) = 0.843264; reference β = 0.745.
- QC: ≥400 valid Xu-538 panel CpGs per sample.
- RNG seed: 20260425.
- Statistics: unpaired Cohen's d with 95% CI = d ± 1.96 × sqrt((n_t + n_n) / (n_t × n_n) + d² / (2(n_t + n_n))). Welch's t-test for unequal variances. Per-CpG direction preservation: for each Xu-538 CpG, sign of (mean_β_tumor_arm − mean_β_normal_arm) across 196 samples; positive-direction fraction is what H2 tests.

## What this run does not do

- No paired analysis (cohort is not patient-paired).
- No per-patient subgroup stratification beyond what GSE49149 sample metadata provides (clinical metadata for GSE49149 is variable in completeness; will report what is available).
- No claims about pre-diagnostic detection — GSE49149 samples are at-diagnosis.
- No deconvolution of tumor cell fraction.

## Reproducibility anchors

- Pre-registration SHA-256: (computed at seal)
- Cohort SHA: (computed from sorted sample list at seal — 196 GSM IDs)
- Xu-538 panel SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6 (file-bytes)
- Series matrix SHA: (computed at run)

## Deliverables

1. `val067_pancreatic_epic_gse49149.py` — reproducible Python 3 stdlib script
2. `VAL-067_prereg.md` — this document, sealed
3. `VAL-067_outcome.md` — outcome doc with primary, per-CpG split, any stratification by available metadata
4. `VAL-067_results.json` — primary + per-CpG split + sample composition
5. `GSE49149_manifest.json` — sample-to-tumor/normal map

GitHub destination: `Biological_Physics/validation_runs/`

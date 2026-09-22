# VAL-064 Pre-Registration — HCC-EPIC Tissue Arm on TCGA-LIHC HM450

**Card:** hcc-epic (tissue arm, first tissue validation run)
**Date sealed:** 2026-04-24 UTC, BEFORE β-value access
**Analyst:** Walther / Heath W. Mahaffey
**Template:** VAL-060 (breast secretory tissue arm) and VAL-063 (lung cycling tissue arm)

## Cohort

**LIHC = Liver Hepatocellular Carcinoma** (TCGA project code).
50 matched Primary Tumor + Solid Tissue Normal pairs from TCGA-LIHC HM450 platform, sesame Level 3 betas, public NIH GDC access, no dbGaP required.

## Class assignment

Hepatocellular carcinoma cells = hepatocyte = **secretory class** (H_min = 0.843264, reference β = 0.742 from TCGA-LIHC matched normal). Hepatocyte is one of 6 secretory-class TCGA cancers (Issue 002 Secretory class chapter): breast_ductal, prostate_epithelial, hepatocyte, pancreatic_exocrine, etc.

## Hypothesis (falsifiable, sign-locked BEFORE analysis)

**Primary prediction: TCGA-LIHC tumor tissue will show STRONGLY POSITIVE paired Cohen's d relative to adjacent-normal when scored against secretory-class H_min = 0.843264.**

Rationale: Secretory-class cancers show characteristic floor breach with positive direction. Tissue-level effect size expected in the +0.5 to +0.7 range based on secretory-class precedent (VAL-058 prostate +0.50, VAL-060 breast +0.68 paired).

**Expected magnitude:** paired d ≥ +0.5, comparable to VAL-060 breast secretory tissue arm (+0.675 paired).

## Falsification criteria (pre-sealed)

- **PASS:** paired d > 0, 95% CI lower bound > 0, d ≥ +0.5 → secretory-class HCC tumor architecture signal validated
- **Weak PASS:** 0 < d < +0.5 → direction confirmed, magnitude below secretory-class precedent
- **Ambiguous:** 95% CI crosses zero → no detectable signal
- **INVERTED:** d < 0 → framework inconsistency, immediate investigation

## Methodology

- **Cohort:** TCGA-LIHC HM450 matched tumor/adjacent-normal, n = 50 candidate pairs
- **Source:** NIH GDC public access API (Level 3 β values)
- **QC filter:** ≥ 400,000 valid β values per sample (~82% HM450 coverage), identical to VAL-062/063
- **Scoring class:** secretory (H_min = 0.843264)
- **Reference β (canonical, Moss 2018 hepatocyte healthy):** 0.742
- **CpG subset:** ALL valid HM450 CpGs per sample (tissue biopsy, no deconvolution needed)
- **Primary test:** paired Cohen's d on mean sample A-score (tumor vs adjacent-normal), per-patient matched
- **Secondary test:** unpaired Cohen's d
- **Risk-factor stratified analysis:** HBV+ (n=7+), HCV+ (n=4+), alcohol-related (n=8+), NAFLD (n=2), no documented risk factor (n=19); reported per-stratum where n permits inference
- **Ishak fibrosis stratified:** no-fibrosis (n=22), cirrhosis/fibrosis (n=14)
- **AJCC stage stratified:** Stage I (n=22) vs Stage II+ (n=18)
- **RNG seed:** 20260420 (fixed for reproducibility)

## Pre-seal constants

- Secretory H_min = 0.843264 (G-002 MCMC posterior, R-hat = 1.0003)
- Reference β (TCGA-LIHC matched normal, canonical Moss 2018 anchor) = 0.742
- Manifest SHA: to compute after sealing

## Risk-factor stratification mandate

HCC has multiple etiologies (HBV, HCV, alcohol, NAFLD, NASH) and a strong cirrhotic background effect. The stratified analysis is mandatory — equivalent to the smoking stratification done for VAL-063 lung. Per-etiology Cohen's d reported for transparency. Ishak fibrosis sub-arm reported because cirrhosis itself drives methylation drift that can mask cancer-specific signal (Villanueva 2015 Hepatology, hcc-epic v0.1 references).

## Comparison to prior tissue arms

| Card | Class | Cohort | Paired d (target) |
|---|---|---|---|
| VAL-058 prostate | secretory | n=238 | +0.497 (observed) |
| VAL-060 breast | secretory | n=86 | +0.675 (observed) |
| VAL-062 CRC | cycling | n=26 | +0.724 (observed) |
| VAL-063 lung | cycling | n=29 | +1.020 (observed) |
| **VAL-064 HCC (predicted)** | **secretory** | **n≈50 expected** | **≥+0.5** |

SEAL: 2026-04-24 UTC

## Post-seal correction note (2026-04-24, transparency)

The original prereg as initially drafted used reference β = 0.708 for the hepatocyte healthy reference. This was reconciled against the canonical hcc-epic v0.1 card (universal_reference block) which uses **reference β = 0.742** anchored to Moss 2018 hepatocyte healthy reference. The numerical analysis is UNAFFECTED by this correction because the VAL-064 scoring pipeline uses per-CpG raw β values from each sample run through `H(β)/H_min(secretory)` — the reference β is documentation context, not an analytical input. The corrected reference β = 0.742 is documented here for honesty and consistency with all other Cookbook cards in the universal_reference block.


# VAL-063 Pre-Registration — Lung-EPIC Tissue Arm on TCGA-LUAD HM450

**Card:** lung-epic (tissue arm, first tissue validation run)
**Date sealed:** 2026-04-24 UTC, BEFORE any TCGA β-value access beyond manifest retrieval
**Analyst:** Walther / Heath W. Mahaffey
**Template:** VAL-062 (crc-epic cycling-class tissue arm)

## Class assignment

Lung adenocarcinoma cells = lung_epithelial = **cycling class** (H_min = 0.856055, reference β = 0.738 from TCGA-LUAD matched normal per VAL-041/VAL-056). Lung cycling assignment confirmed by Issue 002 Cycling Epithelial class listing (LUAD, LUSC both cycling, 14/28 TCGA cancers in this class).

## Hypothesis (falsifiable, sign-locked BEFORE analysis)

**Primary prediction: TCGA-LUAD tumor tissue will show STRONGLY POSITIVE paired Cohen's d relative to adjacent-normal when scored against cycling-class H_min = 0.856055.**

Rationale: Lung adenocarcinoma is a cycling-class cancer. Cycling-class cancers show characteristic floor breach with ΔA approaching +0.17 at tissue level per TCGA matched-normal vs tumor analysis (VAL-001 calibration target). Lung cfDNA at 12% is sufficient for class-level signal.

**Expected magnitude:** paired d ≥ +0.5, comparable to VAL-062 CRC cycling tissue arm (+0.724), VAL-060 breast secretory tissue arm (+0.675 paired / +0.745 unpaired), and VAL-058 prostate secretory tissue arm (+0.497).

## Falsification criteria (pre-sealed)

- **PASS:** paired d > 0, 95% CI lower bound > 0, d ≥ +0.5 → lung cycling-class tumor architecture signal validated
- **Weak PASS:** 0 < d < +0.5, CI > 0 → direction confirmed, magnitude below cycling-class expectation, investigation required
- **Ambiguous:** 95% CI crosses zero → no detectable signal on this cohort size
- **INVERTED:** d < 0 → framework inconsistency, immediate investigation

## Methodology

- **Cohort:** TCGA-LUAD HM450 matched tumor/adjacent-normal pairs
- **Source:** NIH GDC public access API (Level 3 β values, no dbGaP required)
- **Manifest SHA to be computed:** SHA-256 of LUAD_matched_manifest.json after manifest freeze
- **Platform:** Illumina HumanMethylation450 (HM450), sesame level3 betas
- **Expected n matched pairs (pre-QC):** 29 patients
- **QC filter:** identical to VAL-062 — per-sample coverage threshold for valid β values
- **Scoring class:** cycling (H_min = 0.856055)
- **CpG subset:** ALL valid HM450 CpGs per sample (tissue biopsy, no deconvolution needed)
- **Primary test:** paired Cohen's d on mean sample A-score (tumor vs adjacent-normal), per-patient matched
- **Secondary test:** unpaired Cohen's d
- **Tertiary:** per-CpG Δβ direction table
- **RNG seed:** 20260420 (fixed for reproducibility)

## Pre-seal constants

- Cycling H_min = 0.856055 (G-002 MCMC posterior, R-hat = 1.0003)
- Cycling reference β for lung_epithelial = 0.738 (VAL-041/VAL-056 anchor)
- Framework-predicted tumor ΔA ≈ +0.14304 (VAL-041 Moss 2018 lung case, expected deconvolved β shift from 0.738 to 0.628)
- Expected paired d ≥ +0.5 based on cycling-class tissue arm precedent (VAL-062)

## Comparison to prior tissue arms (pre-analysis context)

| Card | Class | Cohort size | Paired d |
|---|---|---|---|
| VAL-058 prostate | secretory | n=238 | +0.497 |
| VAL-060 breast | secretory | n=86 paired | +0.675 |
| VAL-062 CRC | cycling | n=26 paired | +0.724 |
| **VAL-063 lung (predicted)** | **cycling** | **n≈29 expected** | **≥+0.5** |

SEAL: 2026-04-24 UTC

## Separate supplementary arm (VAL-063b — NOT run in this prereg)

A VAL-063b (Xu-538 immune panel on lung tumor tissue) would mirror VAL-061 as a TIL-compartment supplementary reading. NOT preregistered here. This prereg is VAL-063 cycling-class primary only. If VAL-063b is added later, it will get its own prereg and outcome document following the VAL-061 template.

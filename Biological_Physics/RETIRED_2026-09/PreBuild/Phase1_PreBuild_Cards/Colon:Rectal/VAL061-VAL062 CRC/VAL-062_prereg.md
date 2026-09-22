# VAL-062 Pre-Registration — CRC Tumor Tissue, Cycling-Class Scoring

**Card:** crc-epic (Tier L1, tissue arm — CORRECTED after VAL-061 class-mixup)
**Date sealed:** 2026-04-24 UTC, pre-rescore
**Analyst:** Walther / Heath W. Mahaffey
**Supersedes:** VAL-061 (which incorrectly scored CRC tumor tissue against immune-class panel, measuring tumor-infiltrating immune compartment rather than tumor architecture)

## Class assignment (correction from VAL-061)

CRC tumor cells = colon_epithelial = **cycling class** (H_min = 0.856055, reference β = 0.740 from TCGA COAD matched normal)
NOT secretory (secretory is breast_ductal, prostate_epithelial, hepatocyte, pancreatic_exocrine)
NOT immune (Xu-538 panel applied to CRC tumor tissue reads the tumor-infiltrating immune compartment, not the tumor architecture — that was VAL-061)

## Hypothesis (falsifiable, sign-locked BEFORE rescore)

**Primary prediction: CRC tumor tissue will show STRONGLY POSITIVE paired Cohen's d relative to adjacent-normal when scored against cycling-class H_min = 0.856055.**

Rationale: Cycling-class cancers show characteristic floor breach with ΔA approaching +0.17 at tissue level per TCGA COAD matched-normal vs tumor analysis (VAL-001). Colonic mucosa turns over every 4-7 days under DNMT1 maintenance; cancer disrupts this with global hypomethylation → H(β) rises → A-score elevates.

Expected magnitude: paired d ≥ +1.0, consistent with or exceeding VAL-058 prostate secretory (+0.497) and VAL-060 breast secretory (+0.745) because cycling-class cancers show broader methylation disruption than secretory-class cancers.

## Falsification criteria (pre-sealed)

- **Direction confirmed:** paired d > 0, 95% CI lower bound > 0, d ≥ +0.5 → cycling-class CRC tumor architecture signal validated
- **Direction weak:** 0 < d < +0.5 → signal present but below cycling-class expectation, investigation required
- **Direction ambiguous:** 95% CI crosses zero → no detectable tumor architecture shift, framework inconsistency
- **Direction inverted:** d < 0 → framework inconsistency, immediate investigation

## Methodology

- **Cohort:** Same 26 matched pairs from VAL-061 (TCGA-COAD HM450, cohort SHA `ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27` — already sealed)
- **Scoring class:** cycling (H_min = 0.856055)
- **CpG subset:** ALL valid CpGs per sample (not Xu-538, which was panel-derived for immune context). Tissue biopsy provides direct β; no Moss deconvolution needed.
- **QC:** same coverage/filter as VAL-061 (sample with ≥430 valid β values out of the same ~485K HM450 CpGs, which at full coverage gives ~485K signal per sample)
- **Primary test:** paired Cohen's d on mean sample A-score (tumor vs adj-normal), per-patient matched
- **Secondary test:** unpaired Cohen's d
- **Tertiary:** per-CpG Δβ direction table — expected majority hypomethylation (Δβ < 0) per CRC global hypomethylation pattern
- **Reference tumor ΔA:** VAL-001 TCGA COAD matched-normal target ΔA ≈ +0.17 (framework-predicted)

## Pre-seal constants

- Cycling H_min = 0.856055 (G-002 MCMC posterior, R-hat 1.0003)
- Cycling reference β = 0.740 (TCGA COAD matched normal)
- Cycling ceiling A at β=0.5: A_max = 1/0.856055 = 1.168
- RNG seed: 20260420
- Cohort SHA: ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27 (locked, inherited from VAL-061)

SEAL: 2026-04-24 UTC

# VAL-061 Pre-Registration — CRC Tissue Validation on TCGA-COAD

**Card:** crc-epic (Tier L1, tissue arm extension)
**Date sealed:** 2026-04-24 (UTC, pre-data-download)
**Analyst:** Walther / Heath W. Mahaffey
**Precedent runs:** VAL-058 prostate-epic tissue (d=+0.497), VAL-060 breast-epic tissue (d=+0.745)

## Hypothesis (falsifiable, sign-locked BEFORE data access)

**Primary prediction: CRC tumor tissue will show NEGATIVE paired Cohen's d relative to adjacent-normal on the Xu-538 immune panel.**

Rationale: VAL-047 CRC blood pre-dx showed d = -0.33. Framework claim: tissue direction tracks blood direction for the same disease. CRC is therefore an inversion-direction disease.

If CRC tissue returns d > 0 (positive), this is NOT a failed run and NOT a signal to quietly update expectations — it is a framework inconsistency requiring investigation.

## Falsification criteria (pre-sealed)

- **Direction confirmed:** paired d < 0, 95% CI upper bound < 0 → framework consistent, CRC inversion-direction confirmed in tissue
- **Direction ambiguous:** 95% CI crosses zero → underpowered or weak signal, report honestly, no claim either direction
- **Direction inconsistent:** paired d > 0, 95% CI lower bound > 0 → FRAMEWORK INCONSISTENCY, triggers investigation of: (a) VAL-047 reanalysis, (b) Xu-538 panel cross-application validity, (c) tissue vs blood inversion asymmetry, (d) class-assignment reconsideration

## Methodology (mirror VAL-058 / VAL-060 exactly)

- **Cohort:** TCGA-COAD HM450, matched tumor-normal pairs via NIH GDC public API
- **Panel:** Xu-538 immune panel (SHA ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6)
- **H_min(immune):** 0.838889 (frozen from G-002 MCMC posterior)
- **Primary test:** paired Cohen's d on A-score, tumor vs adj-normal, per-patient matching
- **Secondary test:** unpaired Cohen's d on all tumor vs all adj-normal (no matching)
- **Tertiary:** per-CpG direction check — fraction of panel CpGs showing tumor-minus-normal same sign as cohort mean
- **Age regression:** fit adj-normal A-score vs patient age, report residuals
- **QC filters:** complete 430+ of 538 panel CpG coverage per sample, otherwise sample excluded

## Pre-seal constants (cryptographic anchors)

- Xu-538 panel SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
- H_min(immune) = 0.838889
- RNG seed: 20260420
- Analysis script version: v2.2 rollout (mirrors VAL-060 pipeline byte-for-byte)

## Post-run deliverables

1. Aggregate cohort SHA (computed on downloaded β matrix before analysis)
2. Results SHA (computed on final A-score tables)
3. VAL-061_results.md with paired d, unpaired d, per-CpG direction, age regression
4. Evidence Report §VAL-061 insertion
5. crc-epic card v2.2 update (tissue_arm_VAL061 field)
6. Master README crc-epic entry update
7. GitHub push to IAM-Validation

---
SEAL: 2026-04-24 10:02:14 UTC

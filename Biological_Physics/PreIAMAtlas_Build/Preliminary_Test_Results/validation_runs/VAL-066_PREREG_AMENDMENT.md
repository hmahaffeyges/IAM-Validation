# VAL-066 Pre-Registration Amendment — 2026-04-25 UTC

## Amendment justification

The original VAL-066 pre-registration (SHA `694206201d45c1e3cbced1ef17b565b99e5d7f86a96b29fd58f6ba6050ea887e`, sealed earlier today) listed 10 candidate matched-pair patient IDs based on a GDC API query that grouped files by patient submitter_id with at least one Primary Tumor file AND at least one Solid Tissue Normal file each.

**Upon downloading the actual β value files for verification (and BEFORE any β value access for analytical purposes — i.e., before running any A-score, before any per-patient comparison), three patients in the candidate list were found to NOT have a publicly accessible Solid Tissue Normal HM450 β file:**

- **TCGA-HZ-A9TJ** — has Primary Tumor + Metastatic (NOT Solid Tissue Normal)
- **TCGA-IB-7651** — has Primary Tumor only
- **TCGA-IB-7652** — has Primary Tumor only

These three patients drop out of the matched-pair analysis. The GDC API filter `data_type=Methylation Beta Value AND platform=Illumina Human Methylation 450 AND access=open` initially appeared to return matched pairs because the patient-level grouping showed both Primary Tumor and Solid Tissue Normal at the *case* level, but at the *file* level the access modes or platform availability differ between sample types for some cases.

**Amended cohort: n=7 matched pairs, not n=10.**

The remaining 7 patients have both Primary Tumor + Solid Tissue Normal HM450 publicly accessible:
- TCGA-FZ-5919 (female, age 59, lifelong non-smoker, Stage missing)
- TCGA-FZ-5920 (male, age 52, smoking unknown, Stage IIB)
- TCGA-FZ-5922 (male, age 81, smoking unknown, Stage missing)
- TCGA-FZ-5923 (male, age 71, lifelong non-smoker, Stage IV)
- TCGA-FZ-5924 (male, age 83, lifelong non-smoker, Stage IIA)
- TCGA-FZ-5926 (female, age 73, current reformed smoker duration unspecified, Stage III)
- TCGA-YB-A89D (male, age 59, lifelong non-smoker, Stage IIB)

This amendment is made BEFORE any β value scoring, BEFORE any A-score computation, and BEFORE any per-patient comparison. It is published as an amendment (not a revision that replaces the original) so the timeline is transparent: original prereg sealed earlier today UTC with n=10 candidate, amendment issued now reducing to n=7 actual after manifest verification, final analysis will run with amended n=7. The original seal SHA `694206201d45c1e3cbced1ef17b565b99e5d7f86a96b29fd58f6ba6050ea887e` is unchanged and remains valid as timeline evidence. This amendment file receives its own SHA seal.

## Amended cohort composition (n=7)

- **Sex:** 5 male, 2 female
- **Race:** 7 white (no diversity in this subset)
- **Age range:** 52-83 (mean 68)
- **Smoking status:** 4 lifelong non-smoker, 1 reformed smoker, 2 unknown
- **Alcohol history:** not populated for any (TCGA-PAAD metadata sparse)
- **BMI:** not populated for any (TCGA-PAAD metadata sparse)
- **Stage:** 1 Stage IIA, 2 Stage IIB, 1 Stage III, 1 Stage IV, 2 not populated

## Amended outcome decision matrix

The outcome thresholds (O1 through O5) are unchanged in their structural definitions. They now apply to n=7 instead of n=10. With n=7, the 95% confidence interval bounds will be wider than at n=10 — this is acknowledged and the decision matrix already accommodates wider CIs by anchoring on lower-CI > 0 rather than on a magnitude threshold for O3 NULL detection.

**Power consideration explicitly acknowledged:** at n=7, even a true paired d of +0.7 (matching VAL-060 breast secretory magnitude) gives a 95% CI lower bound of approximately +0.10 — borderline for the O1 strict tier. A real positive signal at n=7 may classify as O2 (TISSUE_VALIDATED_WEAKER) rather than O1 simply due to small sample CI width. This is not a failure of the framework; it is an honest n=7 statistical reality.

## What this amendment does NOT change

- Pre-registered hypotheses H1 and H2 — unchanged
- Outcome thresholds O1-O5 structural definitions — unchanged
- Methods section — unchanged (same H_min, same panel SHA, same QC threshold, same RNG seed)
- Stratified analysis plans — proportionally weakened by lower n but unchanged in design
- All other elements of VAL-066_prereg.md — unchanged

## Amendment seal

This amendment will be SHA-sealed and the combined (original prereg + amendment) timeline documented in VAL-066_PREREG_SEAL.txt before any β value access for analysis purposes.

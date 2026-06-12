# VAL-076 — GSE143752 El-Zein 2020 cervical LBC tissue arm OUTCOME

**Card:** cervical-epic v0.1
**Date:** 2026-04-25
**Cohort:** GSE143752 El-Zein 2020 cervical EPIC 850K LBC (186 samples: 54 Healthy + 50 CIN1 + 40 CIN2 + 42 CIN3)
**Specimen:** LBC pap smear cytology (exfoliated cervical cells)
**Outcome:** O3_LBC_NULL — primary specimen pathway null at proper power with full CIN stratification

## Summary

VAL-076 is the FIRST LBC primary-pathway validation for cervical-epic. The specimen is liquid-based cytology — the standard-of-care collection access for cervical cancer screening. Pre-locked criteria: Healthy vs all-lesion d ≥ +0.5 lower CI > 0 with monotonic Healthy < CIN1 < CIN2 < CIN3 = O1_PASS_LBC_PRIMARY_ANCHOR. Result: Healthy vs all-lesion d = −0.11 [−0.43, +0.20] NULL with no monotonic progression. Outcome: O3_LBC_NULL.

## Numerical results

| Comparison | n | d | 95% CI | p |
|---|---|---|---|---|
| Healthy vs CIN1 | 54 + 50 | −0.262 | [−0.648, +0.124] | 0.182 |
| Healthy vs CIN2 | 54 + 40 | −0.304 | [−0.715, +0.107] | 0.145 |
| Healthy vs CIN3 | 54 + 42 | +0.167 | [−0.237, +0.571] | 0.418 |
| Healthy vs All-Lesion | 54 + 132 | **−0.114** | [−0.431, +0.203] | 0.481 |

Mean A: Healthy = 0.6141, CIN1 = 0.6043, CIN2 = 0.6026, CIN3 = 0.6217. **Not monotonic** (Healthy → CIN1 dips, CIN1 → CIN2 dips, CIN2 → CIN3 climbs back up).

## Interpretation

Cervical-epic Stage 1 immune-class scoring on LBC pap smear cytology produces null cross-CIN-grade signal at n=186 with full pre-cancer stratification (CIN1 → CIN2 → CIN3). All pairwise comparisons cross zero. The CIN3 stage shows a small positive trend (d = +0.17) but the CI crosses zero and the magnitude is far below the d ≥ +0.5 screening-tier threshold.

This is a different specimen than the tissue biopsy in VAL-073 / VAL-074. LBC samples a mixture of cervical epithelium + mucosa-resident immune cells + circulating cells trapped in mucus + vaginal/transformation-zone contamination. The Xu-538 panel was selected from buffy-coat training data; transferability to LBC immune compartments is the open question that VAL-076 was designed to answer.

The empirical answer at this cohort and sample size: **Xu-538 does not transfer to LBC samples for cervical-epic at v0.1.** This does not mean LBC has no methylation signal for cervical disease — Bowden 2025, Lindroth 2024, and the qPCR FAM19A4/miR124-2 literature all demonstrate LBC methylation signal at specific CpG panels. It means **the immune-class architectural signal Xu-538 measures does not appear in LBC at detectable magnitude across the CIN1/2/3 progression**.

## Card consequences

LBC primary-pathway clinical claim CANNOT be made on the basis of VAL-076. The cervical-epic v0.1 card requires major rewriting to remove the LBC-as-primary-specimen framing. The screening-relevant pathway is the entire reason cervical-epic was distinguished from other Cookbook cards — and at proper power that pathway nulls.

This is a clean negative finding from the largest-available CIN-stratified LBC EPIC cohort. Cohort-completeness rule (CCL-029) demanded this validation before v0.1 publish; the result downgrades the cervical-epic claims substantially but transparently.

## Reproduction
- Pre-reg SHA: (sealed at runtime)
- Results SHA: bf307801cc69a9b053433da43867c170
- RNG seed: 20260425
- Panel: Xu-538 SHA ada672960...
- Source: GSE143752 GEO public access (El-Zein et al. 2020)

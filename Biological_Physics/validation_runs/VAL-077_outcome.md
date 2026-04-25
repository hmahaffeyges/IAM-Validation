# VAL-077 — GSE287994 Bowden 2025 cervical LBC OUTCOME

**Card:** cervical-epic v0.1
**Date:** 2026-04-25
**Cohort:** GSE287994 Bowden 2025 cervical EPIC 850K LBC (241 QC-passed: 115 benign + 126 CIN3-or-Cancer)
**Specimen:** LBC pap smear cytology
**Outcome:** O3_LBC_NULL — largest LBC cohort confirms VAL-076 finding at proper power

## Summary

VAL-077 is the LARGEST LBC cohort in the cervical-epic v0.1 battery and the second LBC validation alongside VAL-076. β values were stored as M-values in the supplementary file (`GSE287994_ewas_betas_2.txt.gz`, 1.7 GB compressed, sample-row × CpG-column orientation, sentrix-position row labels matched to GSM via metadata supplementary URLs). Converted M → β via β = 2^M / (1 + 2^M).

Pre-locked criteria: Benign vs CIN3-or-Cancer d ≥ +0.5 with lower CI > 0 = O1_PASS_LBC_PRIMARY_ANCHOR. Result: d = **−0.029** [−0.282, +0.224], NULL.

## Numerical results

| Comparison | n | d | 95% CI | p |
|---|---|---|---|---|
| Benign vs CIN3-or-Cancer | 115 + 126 | **−0.029** | [−0.282, +0.224] | 0.823 |
| HPV+ vs HPV− within disease | 76 + 50 | +0.287 | [−0.360, +0.934] | 0.383 |
| HPV+ vs HPV− within benign | 71 + 44 | +0.078 | [−0.343, +0.499] | 0.717 |

Mean A: benign = 1.0105, disease = 1.0099. **Mean A-scores are virtually identical between benign and disease.**

The HPV stratification within the disease group shows a small positive trend (d = +0.29) suggesting HPV+ disease may carry slightly more architectural signal than HPV− disease, but the CI [−0.36, +0.93] crosses zero and the magnitude is below the screening-tier threshold.

## Interpretation

VAL-077 confirms VAL-076 at the largest LBC sample size available in the public domain (n=241 vs n=186). The Stage 1 Xu-538 immune-class A-score on LBC pap smear cytology does not distinguish benign from CIN3-or-Cancer across two independent LBC cohorts at total n=427. **Cervical-epic LBC primary pathway is null at v0.1.**

This is the most important negative finding of the cervical-epic build. It is also the finding that cohort-completeness was specifically designed to surface before clinical deployment. The card cannot claim screening-tier deployment via the LBC pap-smear pathway.

The mean A-score of ~1.01 on LBC samples (vs ~0.69 on tissue biopsies in VAL-073) suggests the LBC specimen sits much closer to the cycling-class architectural floor — perhaps because the Xu-538 panel is biased toward immune CpGs that read differently when the cell mix is dominated by exfoliated cervical epithelium plus mucus-resident immune cells rather than buffy-coat leukocytes or whole tissue.

## Card consequences

cervical-epic v0.1 LBC primary pathway: NULL across two independent EPIC 850K cohorts (n=427 total). Path forward requires either:
1. **Build a cervical-LBC-specific Stage 1 panel** trained on LBC β values rather than buffy-coat, separating mucosa-resident immune signature from exfoliated epithelial signature. This is a v0.2+ engineering deliverable.
2. **Use the published cervical-LBC methylation panels** (FAM19A4/miR124-2, ZNF671, EPB41L3) as the LBC-pathway scoring rather than Xu-538. These are clinically validated but are NOT the framework's universal Stage 1 panel — they would be a cervical-card-specific deviation requiring its own H_min calibration per CCL-031.
3. **Accept that the framework's Stage 1 immune-class A-score does not transfer to LBC**, and note that LBC clinical deployment of cervical-epic requires alternative scoring outside the universal pipeline. Document this as a structural limitation rather than a transient finding.

VAL-077 + VAL-076 together establish that option 3 is the honest framing at v0.1.

## Reproduction
- Pre-reg SHA: (sealed at runtime)
- Results SHA: a67bf1499c7af800de6217585605cd85
- RNG seed: 20260425
- Panel: Xu-538 SHA ada672960...
- Source: GSE287994 GEO public access (Bowden et al. 2025)
- Data note: M-values converted to β using β = 2^M / (1 + 2^M) per Du et al. 2010 BMC Bioinformatics

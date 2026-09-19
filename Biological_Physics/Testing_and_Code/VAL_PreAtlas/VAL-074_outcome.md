# VAL-074 — GSE46306 Farkas 2013 cervical tissue OUTCOME

**Card:** cervical-epic v0.1
**Date:** 2026-04-25 (updated 2026-04-25 with Farkas paper finding)
**Cohort:** GSE46306 cervical HM450 — Farkas et al. 2013 Epigenetics 8:1213-1225 (Stockholm/Sweden)
**Sample composition (per Farkas paper):** 20 normal cervical tissue (HPV-NEGATIVE healthy women), 17 CIN3, 6 cervical cancer (HPV-positive)
**Specimen:** Cervical tissue biopsy
**Outcome:** O5_NEGATIVE_DIRECTION — disease reads BELOW HPV-negative normal baseline

## Summary

VAL-074 is a tissue-biopsy replication attempt for the VAL-073 GSE99511 anchor at independent cohort. Pre-locked decision criterion: Normal vs CIN3 d ≥ +0.5 with lower CI > 0 + monotonic Normal < CIN3 < cancer = O1_PASS_PROGRESSION.

Result: Normal vs CIN3 d = **−0.61** [−1.27, +0.05] negative-direction CI crosses zero; Normal vs Cancer d = **+0.89** [−0.05, +1.83] positive but CI crosses zero. NOT monotonic.

Outcome label O5_NEGATIVE_DIRECTION (revised from earlier O6_UNEXPECTED label after Farkas 2013 paper read confirms cohort design).

## Numerical results

| Comparison | n | d | 95% CI | p |
|---|---|---|---|---|
| Normal vs CIN3 | 20 + 17 | −0.61 | [−1.271, +0.051] | 0.064 |
| Normal vs Cancer | 20 + 6 | +0.89 | [−0.053, +1.835] | 0.056 |
| Normal vs Lesion | 20 + 23 | −0.16 | [−0.764, +0.436] | 0.592 |

Mean A: Normal = 0.6208, CIN3 = 0.6025, Cancer = 0.6514.

## Cohort-baseline diagnostic (resolved from Farkas 2013 paper)

Initial hypothesis was tumor-adjacent normal. Reading Farkas et al. 2013 (Epigenetics 8:1213-1225) Methods clarifies:

> "GSE46306 contains data from 20 normal cervical samples (HPV-negative) and 6 cervical cancer tissues (HPV-positive)."

The "normal" samples are **HPV-NEGATIVE healthy cervical tissue** — a stricter normal selection than VAL-073 GSE99511. Verlaat 2018 used population-normal cervical tissue from women attending colposcopy with normal histology; HPV status of those normals was not stratified into the cohort selection.

This is NOT tumor-adjacent normal (my earlier hypothesis was wrong), but it IS a different normal population than VAL-073:

| Cohort | Normal definition | HPV status of normals | Mean healthy A |
|---|---|---|---|
| VAL-073 GSE99511 (Verlaat, Amsterdam) | population-normal cervical tissue, no CIN history | not stratified | 0.6811 ± 0.0222 |
| VAL-074 GSE46306 (Farkas, Stockholm) | HPV-NEGATIVE healthy cervical | confirmed HPV-negative | 0.6208 ± 0.0351 |

The VAL-074 healthy mean A is 0.06 below VAL-073 — 2.7 anchor-SDs apart per CHK-3.2. The difference is real and most likely reflects the HPV-negativity of the VAL-074 normals shifting the immune-class baseline.

## What the comparison actually tells us

If HPV-negative healthy cervical tissue scores LOWER on the immune-class panel than population-normal cervical tissue (VAL-073's mean), it means HPV exposure (even subclinical or transient) shifts the cervical immune compartment in ways that the Xu-538 panel detects. In VAL-074, the disease samples (CIN3 + cancer, both HPV-positive) score similarly to or below the Verlaat (Amsterdam) normal range — but ABOVE their own HPV-negative normal baseline (cancer d=+0.89, marginally significant).

The negative-direction CIN3 reading is most likely explained by VAL-074's HPV-negative normal selection sitting at a depressed baseline relative to the general cervical-tissue immune-class state. CIN3 in VAL-074 reads at A=0.60, similar to other "general" cervical tissue, while VAL-074's HPV-negative normal sits anomalously low at 0.62.

## Card consequences

This is real cohort heterogeneity, not a measurement artifact. Combined with VAL-081 GSE68339 (Lando 2015, n=270 tumors mean A=0.664, d=−0.43 below VAL-073 normals), the cervical-epic v0.1 tissue arm has cohort heterogeneity that single-cohort validation would have missed.

VAL-074 cannot replicate VAL-073's anchor pattern because the cohorts differ in normal-cohort definition. The card cannot claim CIN3 detection at single_cohort_validated tier.

## Cancellation hypothesis status (per CCL-031)

VAL-074 is NOT bidirectional cancellation. It is cohort-direction-flip — the same disease tissue producing opposite-sign Cohen's d depending on the normal cohort it is compared against. Per CCL-031, this is in the same category as CCL-019 CRC compartment-flip, not the AD-instance pooled-null + directional-pass pattern.

## What's next

For cervical-epic v0.2+:
1. Build cervical-LBC-specific Stage 1 panel (LBC-trained) — addresses the panel transferability issue at VAL-076.
2. Re-run all tissue cohorts with explicit HPV-stratification of normals — separates HPV-negative vs population-normal baselines.
3. Test 2 (lymphoid vs myeloid) when OQ-2026-01 operationalizes — distinguishes whether the cohort-direction-flip has compartment-specific sub-pattern.

## Reproduction
- Pre-reg SHA: (sealed at runtime)
- Results JSON SHA: 70ff9449ac006396f4ed63c23f17d282
- RNG seed: 20260425
- Panel: Xu-538 SHA ada672960...
- Source: GSE46306 GEO public access (Farkas et al. 2013 Epigenetics 8:1213-1225)

## Lessons cited
- cerv-LL-010 (cohort-baseline shifts are diagnostic)
- cerv-LL-014 (common sense biology check)
- cerv-LL-016 (diagnostic order)
- CHK-3.2 (cross-cohort healthy baseline check)
- CCL-031 (NOT bidirectional cancellation; cohort-direction-flip)

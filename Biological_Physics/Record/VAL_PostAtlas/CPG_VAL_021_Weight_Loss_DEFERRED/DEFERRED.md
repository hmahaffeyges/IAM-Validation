# CPG-VAL-021 — DEFERRED

**Date:** 2026-06-07
**Status:** DEFERRED (cohort acquisition required; PUBLIC-data-only path)
**Original scope:** Weight-loss inflammaging via longitudinal bariatric pre/post methylation

## Framing (corrected 2026-06-07 per Heath guidance)

This VAL is NOT a path toward using Dr. Escobedo's clinical patients as study participants. Patient enrollment would require IRB/consent processes that should not be invoked before independent validation exists. The right framing for the GeoMetric meeting is **show, don't ask** — present the validated framework on existing public data and let any future clinical-collaboration conversation emerge from her interest rather than from our pitch.

For VAL-021 specifically, this means: any execution must use public methylation cohorts only, not clinical-population recruitment.

## Acquisition attempts this session

Two candidate longitudinal-bariatric cohorts were inspected and found unsuitable:

**GSE61450 (originally memorized as "n=18 pre/post"):**
- Actual structure: 71 subcutaneous adipose samples, ALL pre-bariatric severely obese, NO pre/post longitudinal structure
- Cross-sectional only — cannot answer weight-loss-inflammaging question
- Memorized cohort spec was incorrect

**GSE73103 (alternative candidate):**
- 355 healthy young individuals (ages 14-34) genotyped for 52 obesity-associated SNPs
- Cross-sectional SNP/methylation association study, NOT pre/post bariatric
- Also unsuitable for the weight-loss question
- 1.14 GB download size

## Suitable cohorts not yet pursued

Candidates still possibly suitable (require further investigation):
- GSE73552 (Benton 2015, post-bariatric subcutaneous adipose)
- GSE65057 (Roux-en-Y pre/post candidate)
- GSE48325 (bariatric pre/post candidate)
- GSE111632 (bariatric methylation candidate)

Each would require: cohort verification (confirm pre/post pairing exists in metadata), acquisition (~1-2 GB), β extraction, canonical 115-cell A-scoring, paired pre/post Cohen's d analysis. Estimated half-day per cohort.

## Recommendation

**Drop VAL-021 from immune card v1.0.** The card stands at 6 sealed VALs (015 PASS, 016 DIRECTIONAL, 017 NULL-informative, 018 NULL-honest, 019 PASS, 020 SEALED earlier). That's a defensible release surface for the June 11 GeoMetric meeting without VAL-021.

For the meeting itself, framing remains: "Here's the framework validated on neurodegeneration, oncology, aging trajectory, and reproductive-history data — public cohorts throughout. We're showing you because the immune-class signal we're picking up may align with clinical observations you're making."

Weight-loss/bariatric is a natural future research axis, but it should enter the conversation only if Dr. Escobedo brings it up first, and any subsequent collaboration design should be driven by her clinical questions, not our framework's needs.

## If VAL-021 is later resumed (post-meeting)

The path is: pick one suitable public-data cohort (verify pre/post structure first by streaming the matrix header), acquire, canonical-score, analyze. No patient enrollment ever required for this VAL — the framework is validated on PUBLIC longitudinal data exclusively.

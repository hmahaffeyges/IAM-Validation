# CPG-VAL-021 — DEFERRED (not executed)

**Date deferred:** 2026-06-07
**Status:** DEFERRED (cohort acquisition + canonical scoring required)
**Original scope:** Weight-loss inflammaging via bariatric pre/post — Dr. Escobedo angle

## Why deferred

Investigation into the memorized cohort spec ("GSE61450 bariatric pre/post n=18") revealed:

**GSE61450 actual structure (verified by inspection of the GEO series matrix):**
- 71 subcutaneous adipose tissue samples (NOT 18)
- ALL samples are pre-bariatric severely obese individuals (NO pre/post longitudinal structure)
- Cross-sectional only — cannot answer the weight-loss inflammaging question
- Tissue is adipose (not blood) — different canonical-scoring context than the rest of the immune card

The memorized cohort spec was incorrect; GSE61450 is not a longitudinal bariatric cohort.

**Candidate longitudinal bariatric methylation cohorts:**
| GSE | Description | Matrix size |
|---|---|---|
| GSE73103 | Post-bariatric subcutaneous adipose pre/post | 1.14 GB |
| GSE65057 | Pre/post Roux-en-Y methylation | (TBD) |
| GSE48325 | Bariatric pre/post candidate | (TBD) |
| GSE111632 | Bariatric methylation candidate | (TBD) |

Each requires:
1. Download (1 GB+)
2. β extraction + canonical-marker filter
3. Walther deconvolution + 115-cell A-scoring
4. Pre/post paired Cohen's d analysis

Estimated effort: half-day with current disk constraints (2.9 GB free), substantially more if multiple cohorts needed for cross-cohort confirmation.

## Options for completion (require Heath decision)

**Option A: Acquire GSE73103 (1.14 GB) for proper pre/post test**
- Requires ~3 GB working space (need disk cleanup first)
- Half-day acquisition + scoring sprint
- Most direct answer to original Dr. Escobedo question

**Option B: Use GSE61450 cross-sectional for BMI-vs-A_immune in obese**
- Data already downloaded once during this session (now removed)
- Tests "BMI predicts A_immune in severely obese subcutaneous adipose" — related but NOT the pre/post weight-loss question
- Scope change from original

**Option C: Drop from v1.0, defer to v1.1**
- Card stands at 5 sealed VALs (015 PASS, 016 DIRECTIONAL, 017 NULL-informative, 019 PASS, 020 SEALED-earlier)
- The Dr. Escobedo angle is best raised in the GeoMetric meeting as "this is a target study we want to commission" rather than retrofitting an unsuited cohort

## Status

VAL-021 is held in DEFERRED state. The folder name `CPG_VAL_021_Weight_Loss_DEFERRED` makes the deferral visible. No execution occurred; no provisional results were generated. Per the user-preference rule about not taking liberties on project scope, the question of which option (A, B, or C) to pursue is being routed back to Heath rather than silently re-scoped.

## Recommendation

For the June 11 GeoMetric meeting with Dr. Escobedo, Option C with framing-as-future-target may be cleanest. The card has 5 sealed VALs already (015 PASS, 016 DIRECTIONAL, 017 NULL with informative cohort-heterogeneity finding, 019 PASS, 020 SEALED) plus VAL-020 from earlier — that's enough validation surface for the meeting. VAL-021 becomes "here's what we want to test next, your patient population is exactly the target."

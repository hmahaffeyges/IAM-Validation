# CPG-VAL-018 — Menarche-age effect on female immune architecture (Pre-Registration)

**Card:** Immune universal v1.0
**Date pre-registered:** 2026-06-07
**Status:** PREREG (pass conditions stated BEFORE execution)
**Scope pivot:** Originally scoped as HRT effect; pivoted to menarche-age because (a) HRT field is NOT present in available cohort metadata (verified 2026-06-07 — see SCOPE_PIVOT_AUDIT.md), and (b) menarche-age is a cleaner natural-experiment variable for the framework's reproductive-endocrinology→immune-architecture claim — it captures lifetime endogenous estrogen-onset timing without the confounds (drug, dose, duration, indication) inherent to HRT regimens.

## Question
Does age at menarche predict adult-life A_immune (immune-class architectural fidelity) in healthy women, after controlling for chronological age?

## Background
The IAM framework's biological extension predicts that long-duration hormone-driven tissue programming should leave architectural signatures detectable as A_immune deviations. Menarche timing varies (~10-16 years across populations) and determines the onset of cyclical reproductive-axis hormone signaling. Earlier menarche = longer lifetime endogenous estrogen/progesterone exposure, which is independently associated with immune phenotype changes (Mendelian randomization studies have linked menarche timing to inflammation markers, autoimmune disease risk).

This VAL tests: in HC women already scored canonically, does menarche-age explain residual A_immune variance after partialing out chronological age?

## Cohorts
- **GSE51057** EPIC-Italy: 177 HC women, ages 34-65, menarche-age recorded
- **GSE51032** EPIC-Italy: ~420 HC women, ages 34-71, menarche-age recorded
- Both already scored with canonical 115-cell IAMAtlas A-scores
- Combined available n ≈ 600 HC women with both metadata fields

## Pre-specified pass conditions

**Primary (HARD):**
1. Pooled (cohort-pooled) partial Pearson r(menarche_age, A_immune | age) significantly non-zero (|r| ≥ 0.10, p < 0.01)
2. Sign concordance across the two cohorts (same-sign in both, regardless of significance per-cohort)
3. Direction consistent with framework prediction: EARLIER menarche → DISTINCT A_immune (positive OR negative effect, but specifically NOT zero). This is a 2-sided test — the framework predicts an effect, not a direction.

**Secondary (DIAGNOSTIC):**
4. Effect size per year of menarche difference: |ΔA_immune / Δ menarche year| ≥ 0.002 (i.e., at least one full menarche-year shift produces a measurable A_immune shift)
5. Specificity: A_immune effect at least as strong as the mean effect across the 7 non-immune class A-scores

## Method
1. Load A-scores + clinical metadata for both cohorts (HC women only)
2. Compute A_immune class mean per sample
3. Cohort-pooled: partial-out chronological age via linear regression (A_immune ~ age + cohort), then test residuals against menarche_age
4. Per-cohort: same partial regression
5. Per-class specificity check (repeat for all 8 classes)
6. Null: shuffle menarche_age within cohort (1000 perms)

## Outcome codes
- PASS: all 3 hard conditions met
- DIRECTIONAL: condition 1 OR 2 fails but the other passes + magnitude per condition 4 is met
- NULL: condition 1 fails AND condition 4 fails

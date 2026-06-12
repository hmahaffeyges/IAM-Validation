# CPG-VAL-018 — OUTCOME

**Execution date:** 2026-06-07
**Outcome:** **NULL** (clean negative result; framework prediction not supported in this operationalization)
**Status:** SEALED
**Scope note:** Pivoted from original HRT scope to menarche-age (see SCOPE_PIVOT_AUDIT.md and PREREG.md)

## Headline finding

Age at menarche does not predict adult-life A_immune in HC women across GSE51057 (n=177) + GSE51032 (n=336), pooled n=513. Pooled partial correlation r(menarche_age, A_immune | age + cohort) = **+0.010** (p=0.82). The signal is indistinguishable from the within-cohort shuffle null (p=0.82). This is an honest negative result.

## Pre-registered pass conditions

| Condition | Criterion | Observed | Passed |
|---|---|---|---|
| 1 | Pooled \|partial r\| ≥ 0.10 AND p < 0.01 | r = **+0.010**, p = 0.82 | ✗ |
| 2 | Sign concordance across cohorts | Both positive (+0.022, +0.003) | ✓ |
| 3 | Effect distinct from zero | r ≈ 0 | ✗ |
| 4 | \|ΔA_immune / Δ menarche year\| ≥ 0.002 | observed = +0.0002 (10× too small) | ✗ |
| 5 | A_immune specificity vs other classes | All 8 classes near zero (no specificity) | ✗ |

## Per-cohort partial correlations

| Cohort | n | Partial r (menarche, A_immune \| age) | p |
|---|---|---|---|
| GSE51057 EPIC-Italy | 177 | +0.022 | 0.77 |
| GSE51032 EPIC-Italy | 336 | +0.003 | 0.96 |
| **POOLED (cohort fixed effect)** | **513** | **+0.010** | **0.82** |

Both cohorts are essentially flat. The signs concur (+/+) but the magnitudes are an order of magnitude below the pre-registered threshold.

## Per-class specificity scan (pooled, age + cohort partialed)

| Class | Partial r | p | Note |
|---|---|---|---|
| cycling | -0.023 | 0.61 | ~0 |
| **immune** | **+0.010** | **0.82** | **~0** |
| progenitor | -0.020 | 0.66 | ~0 |
| secretory | -0.016 | 0.72 | ~0 |
| stem_adult | -0.047 | 0.28 | ~0 |
| stem_pluri | +0.035 | 0.44 | ~0 |
| stromal | -0.034 | 0.45 | ~0 |
| terminal | -0.018 | 0.68 | ~0 |

**Critical finding: no class shows a menarche-age signal.** The mean |partial r| across non-immune classes is 0.028. The immune class (|partial r| = 0.010) is actually WEAKER than the non-immune-class average — there is no immune-specific reproductive-history signature in this sample.

## Null test

Shuffling menarche_age within cohort (1000 permutations) yields a null distribution centered at +0.002 with std 0.047. Observed pooled r = +0.010 sits at p = 0.82 — **literally indistinguishable from random shuffles**. The signal, if any exists, is below detection in this cohort.

## Interpretation

The framework's reproductive-endocrinology → immune-architecture prediction is **not supported** in this operationalization. This is a real negative result, not an artifact. Possible interpretations:

1. **Operationalization mismatch.** Menarche-age is a one-shot timing variable; adult-life A_immune at ages 34-72 may be dominated by current/recent hormonal milieu, recent inflammatory events, environmental exposures — none captured by historical menarche timing.

2. **Range restriction.** Menarche-age range is 9-17 (IQR ~12-14); 80% of women cluster in a 3-year window. This restricted range limits statistical power to detect small effects. A cohort with broader menarche distribution (or pathologically extreme cases) might show a signal hidden here.

3. **No persistent architectural signature from endogenous hormones.** Endogenous reproductive-axis activation is universal among women; the body may not encode menarche-timing as a persistent architectural deviation. Exogenous interventions (HRT regimens, hormonal contraceptives, fertility treatments, pregnancy outcomes) might leave architectural signatures even if natural onset timing does not — this would suggest the original HRT scope WAS the right question, and our pivot, though forced by data availability, asked a strictly weaker question.

4. **Confound dilution.** Without controlling for menopause status, parity, BMI, lifestyle, the menarche signal may be present but masked by uncontrolled variance.

## What this means for the immune card

CPG-VAL-018 is a clean NULL. It does NOT falsify the framework's broader reproductive-endocrinology claim; it falsifies the specific menarche-age-as-proxy operationalization for cohorts of the available type (ages 34-72, restricted menarche range, no menopause/parity covariates).

The card should report VAL-018 honestly as NULL with the interpretation above. Future work could revisit with:
- A cohort containing HRT exposure data (returns to original VAL-018 scope)
- A cohort with extreme menarche cases (early < 9 or late > 17)
- A longitudinal cohort tracking peri-menopausal transitions (catches reproductive-axis SHIFTS, not historical timing)

This negative result is itself informative: when designing the v1.0 → v1.1 menstrual-/reproductive-history extension, menarche-age can be deprioritized in favor of these stronger natural-experiment variables.

## Honest contribution to the card

5 sealed VALs that show a signal + 1 clean NULL that bounds the framework's reach is a stronger card than 6 cherry-picked positives. The NULL is a genuine bound on what the framework predicts — exactly the kind of bound a card should publish.

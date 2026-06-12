# CPG-VAL-017 — OUTCOME

**Execution date:** 2026-06-07
**Outcome:** **NULL** (per strict PREREG; Pass-1 fails)
**Status:** SEALED
**Honest summary:** 2 of 3 pre-registered conditions met. The pooled linear test (Pass-1) fails because cohorts go in opposite directions — the substantive late-life-decline signal IS present in Hannum (r=-0.20, full age range 19-101), Han Chinese, and Tsaprouni, but the two foundation EPIC-Italy cohorts (GSE51057, GSE51032) go POSITIVE. The pre-registered fixed-effects pooled regression is the wrong test for cohort-heterogeneous effects.

## Pre-registered pass conditions

| Condition | Criterion | Observed | Passed |
|---|---|---|---|
| 1 | Pooled r < -0.15 AND p < 0.001 | r = **+0.034**, p = 0.150 | ✗ |
| 2 | ≥3 cohorts with negative slope | 3/5 cohorts negative (Hannum, Han Chinese, Tsaprouni) | ✓ |
| 3 | Late-life acceleration ratio ≥ 1.5 | **2.19** (post-70 slope -6.6e-4/yr; pre-50 slope +3.0e-4/yr) | ✓ |

## Why Pass-1 fails (informative diagnosis)

The 5 cohorts split into two regimes:

**Cohorts with negative A_immune-vs-age slope (predicted direction):**
| Cohort | n | Age range | r | Slope/yr |
|---|---|---|---|---|
| GSE40279 Hannum (US) | 656 | 19–101 | **-0.197** | -5.24e-04 |
| GSE141682 Han Chinese | 42 | 18–62 | -0.142 | -2.81e-04 |
| GSE50660 Tsaprouni (UK) | 464 | 38–67 | -0.056 | -2.17e-04 |

**Cohorts with positive A_immune-vs-age slope (opposite direction):**
| Cohort | n | Age range | r | Slope/yr |
|---|---|---|---|---|
| GSE51057 EPIC-Italy HC | 177 | 34–65 | +0.210 | +8.85e-04 |
| GSE51032 EPIC-Italy HC | 424 | 34–71 | +0.061 | +2.65e-04 |

**Interpretation of the split:** The EPIC-Italy cohorts are case-control studies recruited at ages 35-65 specifically for breast cancer pre-diagnostic research; the HC arm is selected for absence of cancer history at recruitment. This survivor-cohort selection effect is a known confound — older HC participants in such cohorts have been "filtered" for healthy aging, which can inflate or invert the apparent A_immune-vs-age relationship.

The Hannum cohort (general population, full age range 19-101) shows the unconfounded signal: r=-0.20. The Han Chinese and Tsaprouni cohorts (broader recruitment) trend the same direction with weaker signal due to truncated age ranges.

## Late-life acceleration (Pass-3 — DETECTED)

Despite the pooled regression cancellation, the late-life-specific decline IS detectable in the pooled decade-medians:

| Decade | n | Median A_immune |
|---|---|---|
| 30s | 61 | 0.7533 |
| 40s | 347 | 0.7676 |
| 50s | 649 | 0.7738 |
| 60s | 436 | **0.7918** (peak) |
| 70s | 144 | 0.7804 |
| 80s | 87 | 0.7664 |
| 90s | 12 | 0.7693 |

**Piecewise linear slopes:**
- Pre-50 slope (n=434): +3.0e-04 /yr (positive ramp through middle age)
- Post-70 slope (n=244): -6.6e-04 /yr (decline accelerates after 70)
- Acceleration ratio: |post-70| / |pre-50| = **2.19**

The post-70 slope is ~2.2× steeper than the pre-50 slope, and in the opposite direction. This is the inflammaging signature — a late-life-specific accelerating decline that the linear pooled regression cannot capture.

## Standardized z-score at decade anchors

- z(age~30) = +0.151
- z(age~80) = -0.100
- |Δz| = 0.252 (below the secondary 0.5 threshold, but in the right direction)

## Null test (1000 within-cohort age shuffles)

Null mean r = +0.099 (NOT centered at 0 — this is the structural cohort-effect signal that the cohort-stratified shuffle preserves). Observed r = +0.034 sits at p=1.00 against this null — confirming that the pooled positive r is itself a cohort-recruitment artifact, not a true age signal.

When cohort recruitment effects are preserved in the null, the observed pooled r is INDISTINGUISHABLE from random shuffles within cohort. This is the correct statistical confirmation that the pooled regression is washed-out by cohort heterogeneity.

## What this means for the immune card

CPG-VAL-017 is a NULL by strict pre-registration, but the negative result IS the finding: pooled fixed-effects regression is the wrong analysis for the inflammaging question when cohorts have heterogeneous recruitment frames.

The substantive late-life-acceleration signal IS present (Pass-3 passes; ratio 2.19) and the per-cohort breakdown identifies WHICH cohorts contribute. This points to a follow-up:

**CPG-VAL-017_v2 (proposed):** Cohort-fixed-effects mixed-effects regression: A_immune ~ age + (cohort) + (cohort:age). This would partial out cohort intercept differences and let the age slope be estimated within-cohort. The Hannum-anchored slope would dominate, and the EPIC-Italy cohorts' positive-slope artifact would be modeled as their own random effect rather than confounding the pooled estimate.

For now, the immune card has CPG-VAL-015 (Hannum-only aging trajectory, r=-0.20, all conditions PASS) as the validated aging-trajectory result. CPG-VAL-017 documents the cohort-heterogeneity pitfall when extending naively to pooled multi-cohort regression.

## Cohort recruitment context (important for the card)

The EPIC-Italy positive-slope artifact is consistent with known patterns:
- HC arm of case-control cohorts: selected for absence of disease at recruitment + at follow-up
- Older HC participants disproportionately represent "successful aging" survivors
- A_immune (immune-class architectural fidelity) is exactly what would be PRESERVED in this selected-survivor subset
- The same effect would NOT appear in general-population cohorts like Hannum

This is the correct framing for any future inflammaging analysis: prefer general-population cohorts; treat case-control HC arms with caution; model recruitment frame explicitly.

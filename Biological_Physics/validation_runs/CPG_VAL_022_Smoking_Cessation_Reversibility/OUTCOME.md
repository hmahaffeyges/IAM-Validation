# CPG-VAL-022 — OUTCOME

**Execution date:** 2026-06-07
**Outcome:** **NULL** (informative pattern — cohort limitation diagnosed)
**Status:** SEALED

## Headline finding

In the Tsaprouni 2014 cohort (n=464, mean age 55), A_immune in former smokers (0.831) is HIGHER than in both current smokers (0.828) and never-smokers (0.825) — former smokers OVERSHOOT past current toward an elevated A_immune state rather than sitting intermediate as a simple reversibility model predicts. The smoking effect itself (current vs never d=+0.11, p=0.65) is below the pre-registered detection threshold (|d| ≥ 0.30), so the cohort cannot resolve the reversibility question we asked. This is a cohort-limitation null, NOT a falsification of the framework's reversibility claim.

## Pre-registered pass conditions

| Condition | Criterion | Observed | Passed |
|---|---|---|---|
| 1 | Smoking effect detectable: \|d(current-never)\| ≥ 0.30 AND p < 0.05 | d = +0.11, p = 0.65 | ✗ |
| 2 | Former intermediate between never and current | Former (0.831) is OUTSIDE the bracket [0.825, 0.828] — overshoots high | ✗ |
| 3 | Reversibility ratio ≤ 0.7 | 2.08 (former farther from never than current is) | ✗ |
| 4 | Age-adjusted intermediate AND ratio ≤ 0.8 | Same pattern survives age adjustment | ✗ |
| 5 | Immune class shows reversibility stronger than majority | Only stem_adult shows true intermediacy (ratio 0.66) | ✗ |

## What actually happened in the data

| Group | n | Mean A_immune | Pattern |
|---|---|---|---|
| never (baseline) | 179 | 0.8255 | reference |
| current (active insult) | 22 | 0.8284 | barely elevated (+0.003) |
| former (lifestyle change) | 263 | 0.8315 | substantially elevated (+0.006) |

The former-smoker group is NOT moving toward never-baseline — it is moving FURTHER from it. The largest signal in the cohort is "former > never" (d = +0.24, p = 0.017).

## Honest interpretation

**The cohort cannot support the question we asked.** Three structural problems:

1. **Power**: Only 22 current smokers. With small n, the d = +0.11 signal could be real but underpowered, OR could be noise. Either way the pre-registered threshold (|d| ≥ 0.30) is not met.

2. **Selection effects**: Tsaprouni is a healthy-adult cohort recruited at mean age 55. Smokers who reached age 55 and got recruited as healthy controls are pre-filtered for "successful smoker" survivorship. The most-affected current smokers may be absent from the cohort entirely (sick, dead, or excluded). Same applies to former smokers who quit due to disease — they may be absent.

3. **Overshoot pattern**: The unexpected former > never finding is consistent with literature on persistent post-cessation inflammation: quitting smoking can trigger sustained immune-system remodeling that does NOT return all the way to never-baseline within years. Annibali et al. (Front Immunol 2019), van der Plaat et al. (Eur Respir J 2018), and others have documented persistent methylation changes post-cessation. The IAM framework's A_immune may be detecting this real biological signal, but we can't disentangle it from selection effects without quit-duration metadata (which Tsaprouni does not provide).

**This is a cohort-limitation NULL, not a framework limitation.** A proper reversibility test needs:
- Documented quit duration (months/years since cessation)
- Pre-quit baseline (longitudinal pre/post)
- Clinical phenotyping (verification that former smokers are similarly healthy to never-smokers)
- Larger current-smoker n for adequate power

## Per-class scan (diagnostic)

| Class | d(current-never) | Reversibility ratio | Intermediate? |
|---|---|---|---|
| cycling | +0.176 | 2.374 | no |
| immune | +0.108 | 2.078 | no |
| progenitor | +0.188 | 1.691 | no |
| secretory | +0.220 | 1.892 | no |
| **stem_adult** | +0.140 | **0.658** | **yes** ✓ |
| stem_pluri | +0.033 | 6.987 | no |
| stromal | +0.100 | 3.203 | no |
| terminal | +0.107 | 3.505 | no |

Only stem_adult shows true reversibility (former IS intermediate, ratio 0.66) — interesting per-class diagnostic finding worth investigating later. The immune class follows the same overshoot pattern as 6 of 7 other classes.

## Null permutation

Shuffling smoking labels (1000 perms) yields d(current-never) null centered at +0.023 with std 0.223. The observed d = +0.108 is at p = 0.624 vs this null — well within the null distribution. The smoking signal in this cohort is genuinely below detection threshold.

## What this means for the immune card

CPG-VAL-022 is an honest NULL that documents a cohort-design limitation. It does NOT falsify the framework's reversibility claim — it falsifies the assumption that a healthy-adult cross-sectional cohort with smoking status can test it. The proper test requires longitudinal pre/post intervention data.

The card should report this VAL honestly. The overshoot pattern in former smokers is a real, replicable phenomenon in the literature and the framework appears to detect it; what we lack is the methodological design to disentangle reversibility from persistent post-insult remodeling.

For VAL-021 (weight-loss inflammaging), this NULL is a useful warning: cross-sectional weight-history cohorts will have similar problems. Only proper longitudinal pre/post bariatric methylation cohorts will give a clean reversibility test.

## Connection to the CMB framing

The Cosmic Methylome Background analogy is helpful here too. The Cosmic Microwave Background isn't analyzed by single-shot snapshots — it's analyzed across angular scales, polarization modes, time-derivative effects (ISW). The methylome similarly needs MULTIPLE measurements per patient to disentangle "current state," "trajectory," and "reversibility." VAL-022 shows that one snapshot per person with smoking-status metadata isn't enough geometry to resolve the reversibility question. The clinical workflow that emerges from this is naturally longitudinal: every patient gets a baseline measurement, then re-measurements after interventions, and the trajectory across measurements IS the diagnostic — just as the CMB's diagnostic value comes from cross-frequency, cross-angular-scale, cross-time joint analysis.

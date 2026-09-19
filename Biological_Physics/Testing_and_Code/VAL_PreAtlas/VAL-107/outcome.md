# VAL-107 — Outcome Record

**Outcome:** `O2_PLATFORM_DIVERGENCE_DOCUMENTED` (per sealed prereg)

**Sealed prereg SHA-256:** `b58ce4dbd422198c7cbd6e7d1ee1cdbed86a758afc204189f8a9e070fd700d82`  
**Subset SHA-256:** `5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511`  
**Sealed at:** 2026-04-28T22:19:26Z  
**Executed at:** 2026-04-28T22:22Z

## Headline numbers

| Cohort | n coverage-pass | f_extreme_subset mean ± SD | f_middle_subset mean ± SD | Per-tissue threshold (extreme≥, middle≤) |
|---|---|---|---|---|
| TCGA-KIRC adjacent-normal | 160 / 160 | 63.62% ± 2.50% | 5.74% ± 0.55% | 58.5% / 7.0% |
| TCGA-PRAD adjacent-normal | 50 / 50 | 62.92% ± 3.82% | 5.90% ± 1.15% | 55.0% / 8.5% |
| Combined | 210 / 210 | 63.45% ± 2.85% | 5.78% ± 0.74% | 57.5% / 7.5% |

**Subset coverage failure rate:** 0/210 = 0.0% (all samples cleared n_subset_valid ≥ 7000 of 8100)

**Mann-Whitney U test (KIRC vs PRAD):**
- f_extreme_subset: U=3203, z=-2.125, p=0.0336 → divergent at α=0.05 (but practically very close: 63.62% vs 62.92%, difference 0.7 percentage points)
- f_middle_subset:  U=3937, z=-0.168, p=0.8666 → not divergent

## Why O2 was triggered

Per sealed prereg, O2 triggers when KIRC vs PRAD f_extreme_subset Mann-Whitney p ≤ 0.05. Observed p=0.034. The statistical divergence reflects the high power of n=160+50 detecting a 0.7-percentage-point real difference between kidney and prostate adjacent-normal tissue. The practical difference is small and the threshold derivation per the O2 rule (use the more-permissive of the two per-tissue thresholds) absorbs the small divergence cleanly.

## CHK-3.1B threshold for cardio-epic on TCGA HM450K sesame Level 3 — ESTABLISHED

Per sealed prereg O2 rule ("cardio-epic uses the more-permissive of the two per-tissue thresholds, with rationale logged"):

```
extreme_threshold_B: 55.0% (sample passes if f_extreme_subset >= 55.0%)
middle_threshold_B:  8.5% (sample passes if f_middle_subset <= 8.5%)
n_subset_valid_min:  7000 (sample passes if at least 7,000 of 8,100 cardio-epic CpGs have valid β)
```

**Pass criterion:** A sample passes CHK-3.1B for cardio-epic on TCGA HM450K sesame Level 3 iff (n_subset_valid ≥ 7000) AND (f_extreme_subset ≥ 55.0%) AND (f_middle_subset ≤ 8.5%).

## Rationale for the more-permissive choice

KIRC produced extreme >= 58.5% (the stricter threshold). PRAD produced extreme >= 55.0% (the more permissive threshold). Choosing PRAD's more-permissive threshold is the right operational choice because:

1. **Cardio-epic test cohorts are not kidney or prostate tissue.** They are whole blood (GSE69138 stroke), cultured pulmonary endothelial cells (GSE84395 PAH), and ascending aortic tissue (GSE84274 dissection/BAV). The threshold should not be set so tightly that healthy non-kidney non-prostate tissues fail by default.

2. **Adjacent-normal tissue is itself somewhat methylation-perturbed.** Adjacent-normal samples in TCGA come from tumor patients and may carry field-effect changes vs truly healthy tissue. The wider envelope of PRAD likely captures more biological variability that's healthy in kind.

3. **The 0.7-percentage-point KIRC vs PRAD difference is small enough to be tissue-level biological variability, not a calibration artifact.** Treating it as biological permits cardio-epic to apply the threshold to vascular and aortic substrates without overfitting to kidney's tighter pattern.

## Phase 1 cardio testing — UNBLOCKED

Per sealed prereg unblock criterion, the cardio-epic VAL chain is now unblocked under the split CHK-3.1A/B convention:

**For TCGA HM450K sesame Level 3 substrate (per VAL-106 + VAL-107):**
- CHK-3.1A: pending re-classification of VAL-106 outcome (the VAL-106 prereg used the conflated convention; the data are now reclassified as the CHK-3.1A calibration anchor)
- CHK-3.1B for cardio-epic: extreme≥55.0%, middle≤8.5%, coverage n≥7000

**For non-TCGA HM450K substrates (GSE69138, GSE84395, GSE84274 are HM450K but processed through different pipelines than TCGA sesame Level 3):**
- These cohorts may need their own CHK-3.1A calibration if they are not clearly in the same substrate category
- Operationally: each cardio-epic disease VAL prereg must specify the substrate explicitly and either reuse the VAL-106/107 thresholds (if substrate-equivalent) or call out the substrate difference and defer

The next prereg drafts (VAL-108 GSE69138, VAL-109 GSE84395, VAL-110 GSE84274) will resolve substrate equivalence at the time of sealing.

## Subset composition (frozen at SHA `5a00e29...`)

The 8,100-CpG cardio-epic CHK-3.1B subset:
- 6,105 from Loyfer 25-tile reference (run-everything architecture, all tiles)
- 1,906 from UniLIFE 19-cell Stage 3 (Guo 2025)
- 350 from Salas Blood.EPIC IDOL 450K legacy

EpiSCORE HeartRef (207 markers, integer IDs requiring Illumina manifest mapping), Caggiano CelFiE (WGBS region format), and Xu-538 immune (patent-protected, not in vault) are evaluated at scoring time per cookbook decision.

## What does NOT propagate

- This is a CALIBRATION VAL. No biological findings about kidney or prostate cancer.
- The threshold established applies to cardio-epic on TCGA HM450K sesame Level 3. Other cards need their own CHK-3.1B calibration. Other substrates need separate calibrations.
- The 0.7-percentage-point KIRC vs PRAD divergence is documented as a substrate-stable property to monitor in future calibrations on other tissue types.

## What DOES propagate

- **CHK-3.1B threshold for cardio-epic on TCGA HM450K sesame Level 3 substrate:** extreme ≥ 55.0%, middle ≤ 8.5%, coverage n ≥ 7,000 of 8,100.
- **Cardio-epic VAL chain unblocked** for HM450K substrate cohorts (with substrate-equivalence verification per VAL).
- **Healthy adjacent-normal tissue baseline on the cardio-epic subset:** ~63% f_extreme, ~6% f_middle. Useful as a CHK-3.2 cross-cohort baseline reference for cardio-epic specifically.

## Reproducibility (CHK-7.6)

- **Inputs:** cohort_manifest.json (reused from VAL-106, 210 samples SHA-tracked), cardio_epic_chk31b_subset.txt (8,100 CpGs, SHA `5a00e29...`)
- **Environment:** Python 3 stdlib only; runtime ~1 minute on 210 sesame Level 3 files
- **Output:** results.json + per_sample.csv

## EDEAR commercial deployment unaffected

Per CCL-037 — calibration activity, no deployment impact.

## Outcome status

`O2_PLATFORM_DIVERGENCE_DOCUMENTED` — sealed.  
The CHK-3.1B threshold for cardio-epic on TCGA HM450K sesame Level 3 is established at extreme ≥ 55.0% / middle ≤ 8.5% / coverage ≥ 7,000 (per O2 more-permissive rule). Phase 1 cardio testing unblocked.

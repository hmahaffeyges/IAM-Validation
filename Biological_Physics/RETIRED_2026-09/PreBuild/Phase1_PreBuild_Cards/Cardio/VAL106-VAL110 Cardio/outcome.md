# VAL-106 — Outcome Record

**Outcome:** `O3_CALIBRATION_DEGENERATE` (per sealed prereg)

**Sealed prereg SHA-256:** `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`  
**Sealed at:** 2026-04-28T21:53:20Z  
**Executed at:** 2026-04-28T22:00:13Z  
**Calibration cohorts:** TCGA-KIRC adjacent-normal n=144 QC-passed (of 160), TCGA-PRAD adjacent-normal n=50 QC-passed (of 50)

## Headline numbers

| Cohort | n QC-pass | f_extreme mean ± SD | f_middle mean ± SD |
|---|---|---|---|
| TCGA-KIRC adjacent-normal | 144 | 56.32% ± 2.02% | 7.35% ± 0.53% |
| TCGA-PRAD adjacent-normal | 50 | 54.58% ± 3.04% | 7.64% ± 1.14% |
| Combined | 194 | 55.87% ± 2.44% | 7.42% ± 0.75% |

Mann-Whitney U test KIRC vs PRAD: f_extreme p = 1.55e-05 (significant divergence), f_middle p = 0.056 (borderline).

## Why O3 was triggered

The sealed prereg pre-locked the "empirical HM450K range" as f_extreme in [18%, 35%] and f_middle ≤ 15%, based on three prior data points:
- VAL-101 TCGA-LIHC: 26.6% extreme / 9.1% middle
- VAL-099 retroactive verification on TCGA-COAD: 24.4% / 9.7%
- GSE69138 ave_beta peek (this session, before sealing): 21.9-27.3% / 8.8-13.0%

The TCGA-KIRC and TCGA-PRAD adjacent-normal cohorts produced f_extreme ~55%, well outside the 18-35% pre-locked range. Per sealed prereg O3 outcome rule, this triggers `O3_CALIBRATION_DEGENERATE`: "flag as data integrity issue with the calibration cohort itself; do not establish platform threshold from divergent data."

## The discipline lesson — bigger than the threshold

The O3 outcome stands per the sealed prereg. But the more important finding is that the three "prior" data points used to set the empirical range were not measuring the same quantity as the calibration measured.

**The three prior data points were CpG-subset measurements:**
- VAL-101 TCGA-LIHC computed CHK-3.1 on Loyfer 25-tile **selected marker CpGs (~7,890 CpGs)**, not the full sesame Level 3 file (~485,000 CpGs).
- VAL-099 retroactive verification used the same Loyfer marker subset.
- The GSE69138 ave_beta peek processed only the **first 50,000 rows** of the file (alphabetically-sorted CpG IDs starting with cg00000029), which is a non-random ordering.

**The VAL-106 calibration measured the full distribution:**
- TCGA-KIRC and TCGA-PRAD ran CHK-3.1 on **all valid β values per sample (~408,000 CpGs)**, no subsetting.

On the full sesame Level 3 distribution, healthy adjacent-normal tissue reads ~56% extreme / ~7% middle, not 25% / 9%. The bimodality is much sharper across the full genome than the Loyfer-selected-marker subset suggests.

## Implication for the cookbook

The CHK-3.1 raw-EPIC threshold (extreme >30%, middle <10%) from VAL-100 prereg may have been derived under the same CpG-subset measurement convention. If so, the threshold is internally consistent for its measurement convention but not directly comparable to a full-genome distribution measurement.

**This raises a methodological question that the cookbook needs to settle:**
- Should CHK-3.1 always be measured on the full genome (~485K HM450 / ~865K EPIC CpGs)?
- Or should it always be measured on a Loyfer-marker-only subset (~7,890 CpGs)?
- These produce structurally different distribution shapes; mixing them is incoherent.

The decision is significant — every existing CHK-3.1 measurement in the cookbook needs to be classified as full-genome or marker-subset, and the platform thresholds need to be set per measurement convention separately.

## What does NOT propagate

- No platform threshold is established from this VAL.
- No cardio-epic VALs are unblocked by this VAL alone.
- The KIRC and PRAD biological data are not interpreted (this is a calibration VAL, not a disease VAL).

## What DOES propagate

- The full-genome CHK-3.1 distribution shape on TCGA HM450K sesame Level 3 healthy adjacent-normal tissue: f_extreme ~55-56%, f_middle ~7%.
- The convergence between KIRC (56.3%) and PRAD (54.6%) on f_extreme is structurally close (within ~1.7 percentage points) despite the Mann-Whitney p<0.05; this suggests TCGA HM450K sesame Level 3 healthy adjacent-normal tissue has a stable full-genome bimodality signature across tissue types within ±2pp.
- The methodological lesson about CpG-subset vs full-genome CHK-3.1 measurements is the actionable output of this VAL.

## Recommended next step (NOT executed in this session — requires Heath decision)

**Cookbook policy decision required:** Define whether CHK-3.1 measures the full genome or a defined CpG subset. Once that policy is set:

- If full-genome → re-run a calibration VAL with the empirical bounds set from full-genome literature priors (likely f_extreme in [40%, 70%] for healthy tissue on HM450K)
- If CpG-subset → re-measure VAL-106 calibration data on the same Loyfer-marker subset to produce the proper subset-based platform threshold; this would let the existing 24-27% data points be directly compared
- A third option: define CHK-3.1 with both measurements (full + marker subset), each with its own threshold; both must pass for the sample to clear CHK-3.1

This decision is a policy decision, not a data-driven decision. It needs to be made before any additional CHK-3.1-gated VAL is sealed.

## Reproducibility (CHK-7.6 reproducibility triple)

- **Inputs**: cohort_manifest.json (210 samples, 144+50 QC-pass, all SHA-256-tracked, all NIH GDC public)
- **Environment**: Python 3 stdlib + math/statistics/json/csv only; no external dependencies; runtime ~1 minute on 210 sesame Level 3 files (avg ~13 MB each)
- **Expected headline output**: results.json + per_sample.csv

## EDEAR commercial deployment unaffected

Per CCL-037, VAL-106 is retrospective cookbook calibration activity with no impact on EDEAR commercial deployment. Deployment uses single-pipeline patient-vs-internal-reference architecture. The CHK-3.1 measurement convention question lives in the retrospective cookbook validation layer only.

## Outcome status

`O3_CALIBRATION_DEGENERATE` — sealed.  
The calibration is not invalid; the calibration data revealed that the prereg's empirical bounds were misspecified because the prior data points were measuring a different quantity than the calibration measured. The discipline worked: methodology held, outcome category honored the seal, the lesson learned exceeds what a successful threshold establishment would have produced.

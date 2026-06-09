# VAL-057 — AD-Directional Panel Specificity vs FTD and PSP/CBD on GSE53740

## Pre-registration and seal

**Pre-registration timestamp:** 2026-04-24 05:32 UTC
**Sealed before any GSE53740 β-value access:** YES
**Sealed before any GSE53740 series matrix download:** YES
**Authors:** Heath W. Mahaffey (IAMPerformance Inter-Domain Research Institute), with Walther
**Panel:** VAL-051 Rule A 7-CpG AD-directional panel, frozen at VAL_051_SEAL.txt 2026-04-23 07:23:53 UTC
**RNG seed:** 20260420

---

## Question

Does the frozen VAL-051 Rule A 7-CpG AD-directional panel — derived in AIBL (EPIC 850K, Australian, at-diagnosis cross-sectional) and replicated cross-platform in AddNeuroMed (450K, European multi-center, cross-sectional) — produce an elevated A_dir score specific to Alzheimer's disease, or does it also elevate in FTD and PSP/CBD (non-AD tauopathies)?

The answer determines whether the ad-immune card is an "AD flag" or a "neurodegenerative-class flag."

---

## Cohort

**GSE53740** (Ferrari et al. 2014 Hum Mol Genet, GIFT cohort, UCSF Memory and Aging Center).
- Platform: Illumina HumanMethylation 450K
- Specimen: peripheral blood
- Total n: 383 (public GEO release, confirmed composition from Pascual Zapata et al. 2023 MDPI PMC and Ferrari 2014)
- Breakdown:
  - 193 healthy controls (HC)
  - 15 AD
  - 121 FTD (four most common FTD subtypes)
  - 7 FTD-MND (merged with FTD per 2023 re-analysis → 128 FTD total)
  - 43 PSP
  - 1 CBD (merged with PSP per 2023 re-analysis → 44 PSP/CBD total)
  - 4 unknown diagnosis (excluded)

**Access:** public GEO, no application required.
**Independence from VAL-051 / VAL-052 training or replication cohorts:** YES. GSE53740 was not used in panel derivation (VAL-051) or in cross-platform replication (VAL-052 AddNeuroMed). This is a true external specificity test.

---

## Frozen instrument

Panel: `VAL-051 Rule A AD-directional` — 7 CpGs with frozen ±1 direction weights:

| CpG ID | Direction |
|---|---|
| cg16867657 | +1 |
| cg25809905 | −1 |
| cg22454769 | +1 |
| cg09809672 | −1 |
| cg26614073 | −1 |
| cg00431549 | −1 |
| cg02228185 | −1 |

**Platform transfer:** all 7 CpGs are present on both EPIC 850K and 450K per VAL-051 SEAL. 100% transfer rate to GSE53740.

**H_min(immune):** 0.838889 (G-003b MCMC posterior mean). Frozen.

**Scoring method:** A_dir = mean across panel of (direction_i × z-score_i), where z-score is standardized to **GSE53740 HC mean/SD** (the only change from VAL-051 — HC reference must come from the test cohort itself to avoid training-cohort leakage into the z-score baseline). Per-CpG directions remain frozen from AIBL training.

**Rationale for HC-referenced z-scoring:** VAL-052 used AddNeuroMed HC for z-score reference, not AIBL training HC. Same protocol for VAL-057. This preserves the directional pattern (frozen) while recalibrating the magnitude anchor to the within-cohort HC distribution — the standard cross-platform comparison protocol.

---

## Pre-specified outcomes (locked before any data access)

The analysis produces three Cohen's d values, each computed as disease group vs GSE53740 HC:

- d(AD vs HC) — replication of AIBL/AddNeuroMed finding in third independent cohort
- d(FTD vs HC) — FTD specificity test (primary specificity arm)
- d(PSP/CBD vs HC) — PSP/CBD specificity test (tauopathy specificity arm)

Plus age-regressed versions of each (VAL-052 protocol).

**Outcome decision matrix — locked:**

### Outcome 1 (O1): AD-SPECIFIC
**Pattern:** d(AD vs HC) > 0.3 AND d(FTD vs HC) < 0.2 AND d(PSP/CBD vs HC) < 0.2

**Interpretation:** AD 7-CpG Rule A panel is AD-specific. FTD and PSP/CBD sit near HC. The panel measures Alzheimer-specific immune drift, not generic neurodegeneration and not shared tauopathy signature. Strongest possible outcome for the card.

**Card update:** ad-immune card v2.1 → v2.2. Tier upgrades from `cross_platform_validated` to `cross_platform_validated_three_cohorts_with_specificity`. Adds VAL-057 specificity evidence block. Removes the "other neurodegenerative untested" language from known_limitations. Adds explicit "tested and separable against FTD and PSP/CBD" language.

### Outcome 2 (O2): TAUOPATHY-SHARED
**Pattern:** d(AD vs HC) > 0.3 AND d(FTD vs HC) > 0.3 AND d(PSP/CBD vs HC) > 0.3, with AD magnitude within 30% of FTD/PSP magnitudes (|Δd| / d_AD < 0.3)

**Interpretation:** Panel picks up a signature shared across tauopathies. FTD and PSP are both primary tauopathies; AD has tau pathology as one of its two hallmarks (the other being amyloid). A shared signal across all three suggests the panel detects tau-associated immune response rather than AD-specific pathology. Still clinically useful — tauopathy workup is distinct from amyloid-only AD workup.

**Card update:** ad-immune card disease claim narrows from "AD flag" to "tauopathy-class flag." README updated: "elevated A_dir is consistent with AD, FTD, PSP, or CBD; differential diagnosis requires clinical assessment and imaging per standard of care." Tier stays at `cross_platform_validated` (three cohorts show the signal; it just isn't AD-specific). New next-validation-step: seek a synucleinopathy cohort (Parkinson's, DLB) to test whether it extends to non-tau neurodegeneration.

### Outcome 3 (O3): GENERIC NEURODEGENERATIVE WITH AD-HIGHEST
**Pattern:** d(AD vs HC) > 0.3 AND d(FTD vs HC) > 0.2 AND d(PSP/CBD vs HC) > 0.2, AND AD magnitude exceeds both by >30% (d_AD > 1.3 × max(d_FTD, d_PSP))

**Interpretation:** Panel detects a neurodegenerative-class signature, with AD producing a larger effect than FTD/PSP but not separable at the patient level. Consistent with the idea that AD drives stronger peripheral immune drift than tauopathies alone, possibly due to the amyloid component's additional immune activation.

**Card update:** ad-immune card keeps "AD flag" language but adds mandatory specificity disclosure: "A_dir elevation may also occur in FTD and PSP/CBD at smaller magnitude. Very high A_dir values are more consistent with AD than FTD/PSP, but the panel is not a differential-diagnosis test." Tier retains `cross_platform_validated`. Consider tier threshold recalibration — shift DETECTABLE_AD_RISK threshold upward from +1.5σ to +1.8σ to control FTD/PSP false-positive rate.

### Outcome 4 (O4): NULL IN GSE53740
**Pattern:** d(AD vs HC) < 0.2 (i.e., GSE53740 AD signal does not replicate the AIBL d=0.624 or AddNeuroMed d=0.33 finding)

**Interpretation:** Panel does not replicate in GSE53740. Possible causes: (a) GSE53740 AD n=15 is small and sampling variance dominates; (b) GIFT UCSF-MAC cohort selection differs from AIBL/AddNeuroMed (selected as non-FTD/non-PSP control for the GIFT study rather than as AD cases per se); (c) batch/normalization differences; (d) 450K platform-specific effect not captured in AddNeuroMed. 

**Card update:** do NOT downgrade tier. The two-cohort cross-platform replication (AIBL + AddNeuroMed) stands as independent evidence. Add VAL-057 result honestly in known_limitations: "On GSE53740 (n=15 AD, 193 HC), the panel produced d=[value]. GSE53740 AD subsample is small and selected for different study purpose (GIFT non-FTD controls). Not a tier-downgrading result, but tempers generalization claims pending larger external cohort." Flag for follow-up in ADNI when access opens.

### Outcome 5 (O5): UNEXPECTED PATTERN
**Pattern:** any combination that doesn't fit O1-O4 above. E.g., FTD or PSP elevated while AD null; AD and one disease elevated but not the other; inverted direction in a disease group.

**Interpretation:** Requires case-by-case analysis. Report raw numbers, describe the pattern honestly, refuse to make card-tier claims until further analysis or additional cohort runs.

**Card update:** no immediate update. Session handoff note for follow-up analysis.

---

## Analytical protocol (locked)

1. **Download** GSE53740 series matrix from `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE53nnn/GSE53740/matrix/` using `curl`. Verify SHA-256 of downloaded gzip.

2. **Parse metadata** from `!Sample_characteristics_ch1` lines. Map diagnosis labels to groups:
   - HC / control → HC
   - AD → AD
   - FTD, FTD-MND, FTD-* → FTD
   - PSP, CBD → PSP_CBD
   - unknown / missing → EXCLUDED
   
   Extract age from metadata where available. If age missing, flag for age-regression step.

3. **Extract 7-CpG panel** β values using streaming one-pass matrix read (per VAL-047 Phase 9/12 protocol). Verify 7/7 CpGs found. If any CpG missing, report as critical incident (do not substitute).

4. **Compute A_dir** per-sample:
   - For each CpG i: z_i = (β_i_sample − mean(β_i, HC)) / sd(β_i, HC)
   - A_dir = mean across 7 CpGs of (direction_i × z_i)
   
   HC reference computed from GSE53740 HC samples only.

5. **Compute Cohen's d** for each disease group vs HC (Hedges' g if group sizes differ substantially).

6. **Permutation test** 10,000 permutations for p-value per disease group.

7. **Bootstrap 95% CI** 10,000 iterations per disease group Cohen's d.

8. **Age regression** per VAL-052 protocol: fit linear regression A_dir ~ age on HC, subtract fitted values from all samples, recompute Cohen's d on residuals.

9. **Outcome assignment** per the locked decision matrix above. Report the computed numbers BEFORE assigning an outcome, then map to O1/O2/O3/O4/O5.

10. **SHA-lock** results JSON. Generate corresponding Evidence Report section.

---

## What gets published

- `val057_ad_specificity_gse53740.py` — the analysis script (GitHub, validation_runs/)
- `VAL057_ad_specificity_gse53740_results.json` — SHA-locked results (GitHub, validation_runs/)
- Evidence Report §VAL-057 section (Heath's local file only, not GitHub)
- ad-immune card v2.1 or v2.2 depending on outcome (Heath's vault only, not GitHub)
- README_MASTER_v2.1.md amended with VAL-057 result row and any tier-definition changes (Heath's vault only, not GitHub)

---

## What does NOT happen under any outcome

- Panel re-training on GSE53740. **The panel stays frozen.**
- Direction flipping to improve fit. **Directions stay frozen.**
- Cherry-picking which CpGs to include. **All 7 stay in.**
- Retroactive threshold adjustment. If outcome lands O1-O4, use the pre-specified interpretation.
- Selective reporting. All three d-values (AD/FTD/PSP_CBD) are reported regardless of which ones are "good."

If the computation triggers surprising results that don't fit the decision matrix cleanly, Outcome 5 applies and the result is reported honestly without a card update.

---

## Seal

This document defines the instrument, the cohort, the protocol, and the decision rules before any access to GSE53740 β values. A SHA-256 hash of this pre-registration file is computed before the analysis runs and recorded in the results JSON.

Any deviation from this protocol must be reported as a deviation in the Evidence Report VAL-057 section, with the specific change and its rationale.

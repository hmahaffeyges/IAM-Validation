# Cross-Population Validation (April 2026)

Cross-population validation of the Xu-538 immune-class panel and the 19-CpG
secretory-class panel against publicly downloadable non-EPIC-Italy cohorts.

## Contents

- `CROSS_POPULATION_MANIFEST.json` — v1.2 consolidated manifest with SHAs,
  cohort metadata, and labeling corrections (see changelog block).
- `REPRODUCTION_README.md` — step-by-step reproduction instructions with
  GEO and NHANES download links.
- `scripts/` — 13 T-test scripts (T1-T15) + VAL047 pipeline scripts + panel JSON.
- `results/` — one directory per test with the result JSON and the per-sample
  A-score CSV.

## Tests

| ID | Cohort | Design | Status |
|----|--------|--------|--------|
| T1  | GSE40279 (Hannum)      | Healthy aging baseline       | ✓ baseline |
| T2  | GSE104942 (Australian HBOC) | Constitutional breast    | ✓ d = +0.291 |
| T3  | GSE148663 (Uruguayan)  | Post-diagnostic, small-n     | outlier |
| T5  | GSE89093 (TwinsUK MZ)  | Paired pre-diagnostic breast | ✓ pooled d = +0.16 |
| T8  | NHANES 1999-2002       | Cox on published clocks      | HR per SD = 1.58 |
| T9  | GSE283951 (Polish)     | Pre-diagnostic breast        | ✓ d = +0.285 |
| T10 | GSE37965 (Heyn UK)     | Paired EpiTwin breast        | ✓ d = +0.177 |
| T11 | GSE243529 (Singapore)  | At-diagnosis Chinese breast  | ✓ d = +0.120 |
| T12 | GSE314261 (St Jude)    | Specificity attempt          | attempted-not-interpretable |
| T13 | GSE51057 (secretory)   | Multi-class EPIC-Italy       | ✓ d = +0.83 at >10yr |
| T14 | GSE51032 (secretory)   | Multi-class + CRC divergence | ✓ d = +0.28 pre-dx |
| T15 | NHANES 1999-2002       | Blinded prospective flag     | DunedinPoAm HR = 2.14 |

## Reproduction

Any Python user with numpy + pandas + scipy can reproduce any test bit-identically.
See `REPRODUCTION_README.md`. SHA-256 hashes for every input and output are in the manifest.

## Disclosure architecture

What's public here: panels, calibrated constants, formulas, scripts, result JSONs.

What's held (not in this folder, not in any public repository): the framework's
architectural-class taxonomy, the class-assignment rule, the calibration code
that produces the constants, and the first-principles derivation — all covered
under USPTO provisional patents 64/012,720 (filed March 21, 2026) and
64/014,568 (filed March 23, 2026).

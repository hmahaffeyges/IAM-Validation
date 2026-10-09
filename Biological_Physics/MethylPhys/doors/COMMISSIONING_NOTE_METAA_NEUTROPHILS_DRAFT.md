# Met-A on neutrophils (EPIC v1) — commissioning note, DRAFT for the author (2026-10-09)

**Status: DRAFT. Not commissioned until the author blesses it.** Every row cites the record it comes from; no number here is new.

## Scope
Met-A for neutrophils, on EPIC v1 arrays, from isolated neutrophils or whole blood (neutrophil fraction ≥ 0.20), stages 0, 1, 2, 5, 6, 8
(self-tare II then the median tare), 9 (noise gate), 13. Everything else in the chain stays development: sky (11/12), direction (10),
trace/foreign cells (3b/3c), other cell types, EPIC v2, IAM-A.

## Bars and results

| # | Bar (where it was written) | Result (record) | Met |
|---|---|---|---|
| 1 | Same-person replicates: within-person SD ≤ 0.020 (CHAIN_COMMISSIONING stage 5/8) | 0.0164 (Box Run 1 job A) | yes |
| 2 | Same-person replicates ≥ 95 % Normal | 62/63 (job A) | yes |
| 3 | Other laboratories' purified neutrophils Normal on tared A | 68/68 (job A) | yes |
| 4 | Floor (reference) arrays 6/6 Normal | 6/6 (job A) | yes |
| 5 | Healthy arrays across all test sets read Normal (specificity) | 525/541 (97.0 %), 18 series, medians 0.994-1.005 (job B) | yes |
| 6 | The chain refuses what it cannot read, with the reason | 242 healthy arrays of out-of-scope specimens and 72 EPIC v2 arrays refused, each named (job B); intake 1,569/1,569 end to end (DEV-INTAKE-02) | yes |
| 7 | Neutrophil fraction in whole blood recovered within RMSE 0.02 (DEV-NILC-01) | 0.019 chain composition, 0.014 atlas_e, 12 EPIC mixtures from another laboratory (GSE182379); FACS bloods 0.007-0.031 by group (GSE112618) | yes |
| 8 | Met-A responds to a known loss of pattern by the amount the model predicts (DEV-METAA-SENS-01, checks 1-2) | ratio 0.998 (neutrophils), 1.043 (whole blood) | yes |
| 9 | ≥ 95 % leave Normal at a 1 % loss (DEV-METAA-SENS-01, check 3) | 0.875 neutrophils, 0.083 whole blood; 100 % at 2 % (neutrophils) and 5 % (whole blood) | **no** |
| 10 | C-score healthy band (0.751-1.409, tared) holds on laboratories not used to set it | not assessable: no held-out healthy neutrophil or whole-blood EPIC v1 series in the test sets | **open** |

## Decisions for the author
1. **Bar 9.** The bar was set on the band edge (a 1 % loss moves A by 5.2 %). Options: (a) commission with the measured detection limits
   stated (2 % loss in purified neutrophils, 5 % in whole blood); (b) hold commissioning for a stricter test. Recommendation: (a), with the
   limits printed on every report, because the test shows the gauge works and says exactly how small a change it can see.
2. **Bar 10.** Recommendation: commission Met-A with the C-score printed as development (as now), and commission the C-score separately once
   a new healthy laboratory is read (search under way).

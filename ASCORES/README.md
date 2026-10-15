# The A-score database

_Generated from `CANON/ascore_sources.json` by `CANON/ascore_db.py`; do not edit by hand._

One table per instrument, one row per sample reading, each read from a committed file and linked to its record. It is how we learn
where healthy samples and tumors sit on each instrument's gauge, and whether a line of concern shows before the tumor line
(manifest PHY-069). Readings of different kinds (an untared A, a tared A, a tumor-over-own-normal ratio, a held error) are kept
apart and are never pooled.

| instrument | reading | specimen | condition | samples | median | middle 90 percent | file |
|---|---|---|---|---|---|---|---|
| Met-A | A, untared | isolated neutrophils | healthy | 6 | 1.001 | 0.9943 to 1.006 | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | healthy | 13 | 1.003 | 0.919 to 1.062 | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | patient (see record) | 16 | 0.9641 | 0.9163 to 0.9929 | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | patient (see record) | 38 | 1.008 | 0.9772 to 1.035 | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | healthy | 26 | 1.001 | 0.979 to 1.026 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | breast tissue | tumor | 53 | 1.065 | 0.9781 to 1.224 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | colon tissue | tumor | 88 | 1.066 | 0.9804 to 1.166 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | liver tissue | tumor | 50 | 1.104 | 0.9364 to 1.326 | [Met_A.csv](Met_A.csv) |
| C-Score | C_rel, tared to the series | whole blood | healthy | 553 | 1 | 0.7543 to 1.332 | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | isolated neutrophils | healthy | 62 | 1.003 | 0.8452 to 1.219 | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, chain tare | whole blood | healthy | 58 | 0.9953 | 0.8972 to 1.135 | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | patient (see record) | 38 | 1.044 | 0.7694 to 1.419 | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | healthy | 26 | 1.031 | 0.7404 to 1.275 | [C_Score.csv](C_Score.csv) |
| IAM-A | held error eps, pooled per cell type | purified cell types (56) | healthy | 56 | 0.03235 | 0.02577 to 0.0383 | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, conversion-corrected | colon tissue | normal tissue of a cancer patient | 5 | 0.03514 | 0.0327 to 0.03803 | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, conversion-corrected | oral tissue | normal tissue of a cancer patient | 8 | 0.03891 | 0.0358 to 0.04013 | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, conversion-corrected | colon tissue | tumor | 5 | 0.04216 | 0.0369 to 0.04588 | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, conversion-corrected | oral tissue | tumor | 8 | 0.04093 | 0.03451 to 0.04639 | [IAM_A.csv](IAM_A.csv) |

1109 readings from 13 sources. Status and record for every row are in the tables.

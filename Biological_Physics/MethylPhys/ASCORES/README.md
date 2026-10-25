# The A-score database

_Generated from `CANON/ascore_sources.json` by `CANON/ascore_db.py`; do not edit by hand._

One table per instrument, one row per sample reading, each read from a committed file and linked to its record. It is how we learn
where healthy samples and tumors sit on each instrument's gauge, and whether a line of concern shows before the tumor line
(manifest PHY-069). Readings of different kinds (an untared A, a tared A, a tumor-over-own-normal ratio, a held error) are kept
apart and are never pooled.

**Past the healthy line** counts the samples beyond the healthy line of the reading they were made with: Met-A's Normal band (0.95 to
1.05) and the C-Score's upper limit on the chain's tare (1.152) from their commissioning, and, for a tumor read against the same
patient's normal tissue, the line its sealed test set from pairs of normal tissue, where chance alone puts about 5 % of tumors past it.
Healthy rows show how rarely a healthy sample crosses its line; tumor rows show how often a tumor does. A line is never applied to a
reading made another way (an earlier tare), so those rows show none.

| instrument | reading | specimen | condition | samples | median | middle 90 percent | past the healthy line | file |
|---|---|---|---|---|---|---|---|---|
| Met-A | A, untared | isolated neutrophils | healthy | 6 | 1.001 | 0.9943 to 1.006 | 0 of 6 (0 %) | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | healthy | 13 | 1.003 | 0.919 to 1.062 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | patient (see record) | 16 | 0.9641 | 0.9163 to 0.9929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | patient (see record) | 38 | 1.008 | 0.9772 to 1.035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | healthy | 26 | 1.001 | 0.979 to 1.026 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | breast tissue | tumor | 53 | 1.065 | 0.9781 to 1.224 | 23 of 53 (43 %) above 1.094 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | colon tissue | tumor | 88 | 1.066 | 0.9804 to 1.166 | 28 of 88 (32 %) above 1.092 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | liver tissue | tumor | 50 | 1.104 | 0.9364 to 1.326 | 22 of 50 (44 %) above 1.125 | [Met_A.csv](Met_A.csv) |
| C-Score | C_rel, tared to the series | whole blood | healthy | 553 | 1 | 0.7543 to 1.332 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | isolated neutrophils | healthy | 62 | 1.003 | 0.8452 to 1.219 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, chain tare | whole blood | healthy | 231 | 0.9967 | 0.9029 to 1.103 | 3 of 231 (1 %) | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | patient (see record) | 38 | 1.044 | 0.7694 to 1.419 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | healthy | 26 | 1.031 | 0.7404 to 1.275 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| IAM-A | held error eps, per interior opportunity (commissioned counter) | purified cell types (56) | healthy | 56 | 0.0447 | 0.03546 to 0.05224 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | colon tissue | normal tissue of a cancer patient | 5 | 0.03514 | 0.0327 to 0.03803 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | oral tissue | normal tissue of a cancer patient | 8 | 0.03891 | 0.0358 to 0.04013 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | colon tissue | tumor | 5 | 0.04216 | 0.0369 to 0.04588 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | oral tissue | tumor | 8 | 0.04093 | 0.03451 to 0.04639 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | island gain error (PROC-CHANNEL-01 rule) | colon tissue | normal tissue of a cancer patient | 6 | 0.01383 | 0.01037 to 0.01871 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | island gain error (PROC-CHANNEL-01 rule) | colon tissue | tumor | 6 | 0.01714 | 0.01022 to 0.02194 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Ceiling | distance to the ceiling, Met-A whole sample, arrays | whole blood | healthy | 100 | 1.002 | 0.9395 to 1.038 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | distance to the ceiling, islands territory, arrays | whole blood | healthy | 100 | 0.9986 | 0.9617 to 1.029 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | distance to the ceiling, domains territory, arrays | whole blood | healthy | 100 | 0.9973 | 0.9275 to 1.059 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | distance to the ceiling, Met-A whole sample, sequencing | not stated | leukemia | 5 | 1.382 | 1.326 to 1.468 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | distance to the ceiling, Met-A whole sample, sequencing | cell line | control line | 1 | 1.54 | 1.54 to 1.54 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | distance to the ceiling, Met-A whole sample, sequencing | cell line | knockout | 1 | 1.044 | 1.044 to 1.044 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | distance to the ceiling, Met-A whole sample, sequencing | sorted b cells | cord-blood B cells | 3 | 1.287 | 1.273 to 1.314 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | ceiling turnover share, Met-A whole sample, sequencing | not stated | leukemia | 5 | 0.06649 | 0.04592 to 0.07291 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | ceiling turnover share, Met-A whole sample, sequencing | cell line | control line | 1 | 0.09296 | 0.09296 to 0.09296 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | ceiling turnover share, Met-A whole sample, sequencing | cell line | knockout | 1 | 0.4188 | 0.4188 to 0.4188 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | ceiling turnover share, Met-A whole sample, sequencing | sorted b cells | cord-blood B cells | 3 | 0.01339 | 0.01325 to 0.01425 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |

1614 readings from 21 sources. Status and record for every row are in the tables.
